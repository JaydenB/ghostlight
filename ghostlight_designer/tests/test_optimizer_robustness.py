"""Tests for the "smarter optimizer" additions:

* geometry_penalties — thickness + aperture-vs-radius barriers
* curvature parameterization — R↔C round-trip + scipy-space packing
* preserve_radius_signs — MF-level toggle
* MF right-click menu — checkable toggle wired through
* End-to-end regression — a run with penalties still converges cleanly
"""
from __future__ import annotations

import math
import pathlib

import pytest

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QMenu

import ghostlight

from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
from ghostlight_designer.optimization_panel.data import (
    GoalEntry,
    GoalKind,
    MeritFunction,
)
from ghostlight_designer.optimization_panel.geometry_penalties import (
    APERTURE_RADIUS_MARGIN,
    MIN_THICKNESS_MM,
    PENALTY_WEIGHT,
    _aperture_penalty,
    _thickness_penalty,
    evaluate_geometry_penalties,
    n_geometry_residuals,
)
from ghostlight_designer.optimization_panel.nodes import MeritFunctionNode
from ghostlight_designer.optimization_panel.optimizer import OptimizationRun
from ghostlight_designer.optimization_panel.variables import (
    CURVATURE_BOUND_ABS,
    apply_variables_scipy,
    curvature_to_radius,
    pack_bounds_scipy,
    radius_to_curvature,
    VariableRef,
)
from ghostlight_designer.project import Project


# ---------------------------------------------------------------------------
# geometry_penalties — pure math
# ---------------------------------------------------------------------------


class _FakeSurface:
    """Minimal ``ghostlight.Surface`` stand-in for penalty math tests.

    Real surfaces are C++-backed; constructing them ad-hoc is expensive
    and requires the full OpticalSystem chain. A duck-typed struct
    exercises the penalty math against the exact attribute set the
    helpers read.
    """
    def __init__(
        self, *,
        thickness=1.0,
        radius=50.0,
        semi_aperture=10.0,
        is_active=True,
        is_stop=False,
        form=int(ghostlight.SurfaceForm.SPHERE),
    ):
        self.thickness = thickness
        self.radius = radius
        self.semi_aperture = semi_aperture
        self.is_active = is_active
        self.is_stop = is_stop
        self.form = form


def test_thickness_penalty_zero_when_valid():
    surf = _FakeSurface(thickness=1.0)
    assert _thickness_penalty(surf) == 0.0


def test_thickness_penalty_grows_linearly_with_shortfall():
    below = _FakeSurface(thickness=MIN_THICKNESS_MM - 0.1)
    p = _thickness_penalty(below)
    assert p == pytest.approx(PENALTY_WEIGHT * 0.1)


def test_thickness_penalty_maxes_negative_thickness():
    """Negative thickness (surface behind previous) — big residual."""
    negative = _FakeSurface(thickness=-2.0)
    p = _thickness_penalty(negative)
    assert p == pytest.approx(PENALTY_WEIGHT * (MIN_THICKNESS_MM - -2.0))


def test_thickness_penalty_skips_muted_surface():
    surf = _FakeSurface(thickness=-5.0, is_active=False)
    assert _thickness_penalty(surf) == 0.0


def test_aperture_penalty_zero_when_valid():
    surf = _FakeSurface(radius=100.0, semi_aperture=20.0)
    assert _aperture_penalty(surf) == 0.0


def test_aperture_penalty_grows_when_semi_ap_exceeds_margin():
    R = 10.0
    max_r = APERTURE_RADIUS_MARGIN * R
    r = max_r + 0.5
    surf = _FakeSurface(radius=R, semi_aperture=r)
    p = _aperture_penalty(surf)
    assert p == pytest.approx(PENALTY_WEIGHT * 0.5)


def test_aperture_penalty_uses_absolute_radius():
    """Negative radius (concave) — |R| governs the sag constraint."""
    R = -10.0
    max_r = APERTURE_RADIUS_MARGIN * abs(R)
    r = max_r + 0.5
    surf = _FakeSurface(radius=R, semi_aperture=r)
    p = _aperture_penalty(surf)
    assert p == pytest.approx(PENALTY_WEIGHT * 0.5)


def test_aperture_penalty_skips_flat_surface():
    """R = 0 (ghostlight's flat sentinel) has no sag equation."""
    surf = _FakeSurface(radius=0.0, semi_aperture=100.0)
    assert _aperture_penalty(surf) == 0.0


def test_aperture_penalty_skips_stop_surface():
    surf = _FakeSurface(is_stop=True, radius=1.0, semi_aperture=100.0)
    assert _aperture_penalty(surf) == 0.0


def test_aperture_penalty_skips_non_spherical_forms():
    """Aspheric / cylindrical sag rules differ, so they are not penalised."""
    surf = _FakeSurface(
        form=int(ghostlight.SurfaceForm.ASPHERE), radius=1.0, semi_aperture=100.0,
    )
    assert _aperture_penalty(surf) == 0.0


def test_evaluate_geometry_penalties_on_sample_lens_is_all_zero(
    sample_lens_path: pathlib.Path,
):
    """The bundled sample lens is a valid design — penalties must be 0."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    penalties = evaluate_geometry_penalties(system)
    assert penalties  # non-empty
    for p in penalties:
        assert p == 0.0, f"unexpected non-zero penalty on healthy lens: {p}"


def test_evaluate_geometry_penalties_fires_on_negative_thickness(
    sample_lens_path: pathlib.Path,
):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    system.surfaces[0].thickness = -2.0
    penalties = evaluate_geometry_penalties(system)
    # First half of the returned list is thickness penalties in surface
    # order; surface 0's should now be positive.
    n_surf = len(list(system.surfaces))
    assert penalties[0] > 0.0
    # Sanity: other surfaces still zero.
    for other in penalties[1:n_surf]:
        assert other == 0.0


def test_n_geometry_residuals_matches_actual_length(
    sample_lens_path: pathlib.Path,
):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    assert n_geometry_residuals(system) == len(
        evaluate_geometry_penalties(system)
    )


# ---------------------------------------------------------------------------
# Curvature parameterization — R↔C round-trip
# ---------------------------------------------------------------------------


def test_radius_curvature_round_trip():
    for R in (50.0, 1.5, 500.0, -47.0, -0.9, -1000.0):
        C = radius_to_curvature(R)
        R2 = curvature_to_radius(C)
        assert R2 == pytest.approx(R, rel=1e-9)


def test_flat_surface_maps_to_zero_curvature():
    """R = 0 is ghostlight's flat convention; must round-trip through C = 0."""
    assert radius_to_curvature(0.0) == 0.0
    assert curvature_to_radius(0.0) == 0.0


def test_pack_bounds_scipy_marks_radius_variables_as_curvature(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    uuids = list(project.system.surface_ids)
    project.set_variable_flag(uuids[0], "radius")
    project.set_variable_flag(uuids[0], "thickness")
    from ghostlight_designer.optimization_panel.variables import collect_variables
    vars_ = collect_variables(project)
    x0, lb, ub, is_curvature = pack_bounds_scipy(vars_, project.system)
    # Two entries: order is by (surface_index, attr-sort) — radius comes
    # before thickness alphabetically inside the same surface.
    assert len(x0) == 2
    # is_curvature marks radius True, thickness False.
    radius_pos = next(
        i for i, v in enumerate(vars_) if v.attr == "radius"
    )
    thickness_pos = next(
        i for i, v in enumerate(vars_) if v.attr == "thickness"
    )
    assert is_curvature[radius_pos] is True
    assert is_curvature[thickness_pos] is False
    # Radius entry is packed in curvature space with default |C| < 1.0.
    assert x0[radius_pos] == pytest.approx(
        radius_to_curvature(project.system.surfaces[0].radius)
    )
    assert lb[radius_pos] == -CURVATURE_BOUND_ABS
    assert ub[radius_pos] == CURVATURE_BOUND_ABS


def test_pack_bounds_scipy_preserve_radius_signs_narrows_to_positive_half(
    qapp, sample_lens_path: pathlib.Path,
):
    """A positive-R variable optimized with preserve_signs=True gets
    a strictly-positive C interval so scipy can't cross zero."""
    project = Project()
    project.load(str(sample_lens_path))
    uuids = list(project.system.surface_ids)
    # Surface 0 has R = +47 (verified during Slice 1 debugging).
    project.set_variable_flag(uuids[0], "radius")
    from ghostlight_designer.optimization_panel.variables import collect_variables
    vars_ = collect_variables(project)
    x0, lb, ub, _ = pack_bounds_scipy(
        vars_, project.system, preserve_radius_signs=True,
    )
    assert lb[0] > 0.0, "positive-R half-line should exclude zero"
    assert ub[0] > 0.0


def test_pack_bounds_scipy_preserve_radius_signs_narrows_to_negative_half(
    qapp, sample_lens_path: pathlib.Path,
):
    """Negative-R start → strictly-negative C interval."""
    project = Project()
    project.load(str(sample_lens_path))
    uuids = list(project.system.surface_ids)
    # Surface 2 has R = -360 in the sample doublet.
    project.set_variable_flag(uuids[2], "radius")
    from ghostlight_designer.optimization_panel.variables import collect_variables
    vars_ = collect_variables(project)
    x0, lb, ub, _ = pack_bounds_scipy(
        vars_, project.system, preserve_radius_signs=True,
    )
    assert ub[0] < 0.0, "negative-R half-line should exclude zero"
    assert lb[0] < 0.0


def test_apply_variables_scipy_inverts_curvature_only(
    qapp, sample_lens_path: pathlib.Path,
):
    """Radius entries get C→R inversion; other attrs pass through."""
    project = Project()
    project.load(str(sample_lens_path))
    uuids = list(project.system.surface_ids)
    project.set_variable_flag(uuids[0], "radius")
    project.set_variable_flag(uuids[0], "thickness")
    from ghostlight_designer.optimization_panel.variables import collect_variables
    vars_ = collect_variables(project)
    _, _, _, is_curvature = pack_bounds_scipy(vars_, project.system)
    # Write curvature=0.02 (i.e. R=50) and thickness=7.5.
    values = [0.0] * len(vars_)
    radius_pos = next(i for i, v in enumerate(vars_) if v.attr == "radius")
    thickness_pos = next(i for i, v in enumerate(vars_) if v.attr == "thickness")
    values[radius_pos] = 0.02
    values[thickness_pos] = 7.5
    apply_variables_scipy(vars_, project.system, values, is_curvature)
    assert project.system.surfaces[0].radius == pytest.approx(50.0)
    assert project.system.surfaces[0].thickness == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# MeritFunction field + right-click toggle
# ---------------------------------------------------------------------------


def test_merit_function_default_preserve_radius_signs_is_false():
    mf = MeritFunction.make(name="X")
    assert mf.preserve_radius_signs is False


def test_mf_right_click_menu_shows_preserve_signs_toggle(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        # Add a merit function so the tree has an MF row to right-click.
        mf = MeritFunction.make(name="X")
        body.model.add_merit_function(mf)
        # Reach into the model for its MF node index.
        mf_idx = body.model.index(0, 0, QModelIndex())
        node = mf_idx.internalPointer()
        assert isinstance(node, MeritFunctionNode)
        menu = QMenu(body)
        body._populate_mf_menu(menu, node)
        titles = [a.text() for a in menu.actions()]
        assert "Preserve Radius Signs" in titles
        act = next(
            a for a in menu.actions() if a.text() == "Preserve Radius Signs"
        )
        assert act.isCheckable()
        assert act.isChecked() is False
        # Trigger → MF field flips.
        act.trigger()
        assert mf.preserve_radius_signs is True
        menu.deleteLater()
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# End-to-end — smart optimizer still converges cleanly on the sample lens
# ---------------------------------------------------------------------------


def _wait_for_run(qapp, run, timeout_s=45.0):
    import time
    captured: dict = {}
    run.runFinished.connect(lambda r: captured.setdefault("result", r))
    if run.is_finished and run.result is not None:
        return run.result
    deadline = time.monotonic() + timeout_s
    while "result" not in captured and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return captured.get("result")


def test_end_to_end_curvature_space_still_converges(
    qapp, sample_lens_path: pathlib.Path,
):
    """Regression: after switching to curvature-space + adding penalty
    residuals, the standard EFL-hit test still converges.

    Uses the same setup as ``test_end_to_end_optimization_with_injected_variable``
    but exercises the new machinery (pack_bounds_scipy, penalties, apply_variables_scipy).
    """
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer import lens_metrics as lm

    project = Project()
    project.load(str(sample_lens_path))
    current_efl = lm._effective_focal_length_on_axis(project.system, "y")
    target_efl = current_efl + 5.0
    mf = MeritFunction.make(
        name="EFL Hit (curvature)",
        goals=[
            GoalEntry.make(kind=GoalKind.EFL, target=target_efl, weight=1.0),
        ],
        max_iters=40,
    )
    uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(uuid, "radius")
    try:
        run = OptimizationRun(project, mf, project.system_setup)
        run.start()
        result = _wait_for_run(qapp, run)
        assert result is not None
        assert result.status == "ok", (
            f"got status={result.status} msg={result.message!r}"
        )
        final_efl = lm._effective_focal_length_on_axis(
            run.virtual_system, "y",
        )
        assert final_efl is not None
        # Must have moved toward the target.
        assert abs(final_efl - target_efl) < abs(current_efl - target_efl)
    finally:
        project.clear_variable_flag(uuid, "radius")


def test_end_to_end_thickness_penalty_prevents_negative_thickness(
    qapp, sample_lens_path: pathlib.Path,
):
    """A merit function that would prefer a very thin element still keeps
    thickness above MIN_THICKNESS_MM because of the barrier.

    We flag surface 0's thickness with a goal that pushes it toward 0.
    Without the penalty scipy would drive it negative; with the penalty
    it stops at the barrier.
    """
    pytest.importorskip("scipy.optimize", exc_type=ImportError)

    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(uuid, "thickness")
    # Merit function: hit EFL = 0 (pathological). scipy will try
    # every knob including driving thickness bad. The penalty keeps
    # the thickness ≥ MIN_THICKNESS_MM.
    mf = MeritFunction.make(
        name="Bogus",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=0.0, weight=1.0)],
        max_iters=30,
    )
    try:
        run = OptimizationRun(project, mf, project.system_setup)
        run.start()
        result = _wait_for_run(qapp, run)
        assert result is not None
        # Regardless of goal outcome, thickness stayed valid.
        final_t = run.virtual_system.surfaces[0].thickness
        assert final_t >= MIN_THICKNESS_MM - 1e-3, (
            f"thickness slipped through barrier: {final_t}"
        )
    finally:
        project.clear_variable_flag(uuid, "thickness")


def test_end_to_end_preserve_signs_pins_radius_direction(
    qapp, sample_lens_path: pathlib.Path,
):
    """A positive-R surface's radius stays positive across a run with
    ``preserve_radius_signs`` on, even when the goal would prefer the
    opposite sign."""
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer import lens_metrics as lm

    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])
    # Surface 0 starts at R = +47.
    assert project.system.surfaces[0].radius > 0.0
    project.set_variable_flag(uuid, "radius")
    # A very short-EFL target will push scipy hard on this radius —
    # far enough that without the sign-pin it can cross zero.
    current_efl = lm._effective_focal_length_on_axis(project.system, "y")
    mf = MeritFunction.make(
        name="Push hard",
        goals=[
            GoalEntry.make(
                kind=GoalKind.EFL, target=current_efl - 40.0, weight=1.0,
            ),
        ],
        max_iters=60,
        preserve_radius_signs=True,
    )
    try:
        run = OptimizationRun(project, mf, project.system_setup)
        run.start()
        result = _wait_for_run(qapp, run)
        assert result is not None
        final_R = run.virtual_system.surfaces[0].radius
        assert final_R > 0.0, (
            f"surface 0 radius crossed zero despite preserve_radius_signs: "
            f"{final_R}"
        )
    finally:
        project.clear_variable_flag(uuid, "radius")
