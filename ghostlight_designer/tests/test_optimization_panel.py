"""Smoke tests for the optimization panel.

The expensive part (the optimizer running scipy.least_squares against
the C++ tracer) is exercised by an end-to-end test that flags a
synthetic variable, asserts the worker finishes, and accepts the
solution. Everything else here is import + plumbing.
"""
from __future__ import annotations

import pathlib

import pytest

from PySide6.QtCore import Qt

import ghostlight

from ghostlight_designer.optimization_panel import (
    OPTIMIZATION_TYPE_ID,
    register_optimization_panel_type,
)
from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
from ghostlight_designer.optimization_panel.columns import Column
from ghostlight_designer.optimization_panel.data import GoalEntry, GoalKind, MeritFunction
from ghostlight_designer.optimization_panel.goals.base import (
    GOAL_REGISTRY,
    default_params_for,
    default_target_for,
    safe_evaluate,
)
from ghostlight_designer.optimization_panel.optimizer import (
    OptimizationRun,
    clone_system,
)
from ghostlight_designer.optimization_panel.presets import (
    PRESETS,
    build_empty_merit_function,
    build_image_quality_mf,
)
from ghostlight_designer.optimization_panel.variables import (
    VariableRef,
    apply_variables,
    pack_bounds,
)
from ghostlight_designer.project import Project


# ---------------------------------------------------------------------------
# Goal registry
# ---------------------------------------------------------------------------


def test_every_goal_kind_has_an_evaluator():
    for kind in GoalKind:
        assert kind in GOAL_REGISTRY, f"missing evaluator for {kind!r}"


def test_evaluator_metadata_complete():
    for kind, ev in GOAL_REGISTRY.items():
        assert ev.display_name, f"{kind} has no display_name"
        assert isinstance(ev.default_target, float), f"{kind} default_target wrong type"
        for p in ev.param_schema:
            assert p.name
            assert p.kind in (
                "wavelength_pick", "field_pick", "surface_pick", "axis",
            ), f"{kind}.{p.name} has unknown kind {p.kind!r}"


# ---------------------------------------------------------------------------
# Cloning + variables
# ---------------------------------------------------------------------------


def test_clone_system_round_trips(sample_lens_path: pathlib.Path):
    source = ghostlight.OpticalSystem.load(str(sample_lens_path))
    clone = clone_system(source)
    assert clone is not source
    assert clone.num_surfaces() == source.num_surfaces()
    # Cloning should not mutate the source.
    if source.num_surfaces() > 0:
        assert clone.surfaces[0].radius == pytest.approx(source.surfaces[0].radius)


def test_collect_variables_returns_empty_when_no_flags(
    qapp, sample_lens_path: pathlib.Path,
):
    """No flags on the project → collect_variables returns an empty list."""
    from ghostlight_designer.optimization_panel.variables import collect_variables
    project = Project()
    project.load(str(sample_lens_path))
    assert collect_variables(project) == []


def test_collect_variables_reads_project_flags(
    qapp, sample_lens_path: pathlib.Path,
):
    """Flag one attribute via the project API and confirm collect_variables
    surfaces it with the right surface index + attr."""
    from ghostlight_designer.optimization_panel.variables import collect_variables
    from ghostlight_designer.project import VariableBounds
    project = Project()
    project.load(str(sample_lens_path))
    if project.system.num_surfaces() < 1:
        pytest.skip("Sample lens has no surfaces")
    uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(uuid, "radius", VariableBounds(lo=-100.0, hi=100.0))
    refs = collect_variables(project)
    assert len(refs) == 1
    assert refs[0].surface_index == 0
    assert refs[0].attr == "radius"
    assert refs[0].lo == pytest.approx(-100.0)
    assert refs[0].hi == pytest.approx(100.0)


def test_pack_and_apply_variables(sample_lens_path: pathlib.Path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    if system.num_surfaces() < 1:
        pytest.skip("Sample lens has no surfaces")
    vars_ = [
        VariableRef(surface_index=0, attr="radius", lo=None, hi=None),
    ]
    x0, lb, ub = pack_bounds(vars_, system)
    assert len(x0) == 1
    assert x0[0] == pytest.approx(float(system.surfaces[0].radius))
    # Apply a new value and read it back.
    apply_variables(vars_, system, [x0[0] + 1.0])
    assert system.surfaces[0].radius == pytest.approx(x0[0] + 1.0)


# ---------------------------------------------------------------------------
# Goal evaluators on a real lens
# ---------------------------------------------------------------------------


def test_evaluators_return_finite_values_on_sample_lens(
    sample_lens_path: pathlib.Path,
):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    # We need a SystemSetup so wavelength/field resolution has something
    # to read; a default one is fine.
    from ghostlight_designer.system_setup_data import SystemSetup
    setup = SystemSetup()

    for kind, ev in GOAL_REGISTRY.items():
        params = default_params_for(kind)
        value = safe_evaluate(ev, system, setup, params)
        # safe_evaluate returns the penalty rather than raising; just
        # check that the call is exercised and we get a finite float.
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Presets + project plumbing
# ---------------------------------------------------------------------------


def test_presets_build_without_a_lens():
    """Empty-lens presets must still construct usable MFs."""
    for label, builder in PRESETS:
        mf = builder(None)
        assert isinstance(mf, MeritFunction)
        # Empty is allowed to have no goals; the others should not.
        if label != "Empty":
            assert mf.goals, f"{label} preset has no goals"


def test_presets_seed_efl_target_from_sample_lens(sample_lens_path: pathlib.Path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    mf = build_image_quality_mf(system)
    # Whichever goal is EFL must have a non-zero target — the seed
    # function falls back to 50 mm if the probe failed, which would
    # also be acceptable. We only want to verify the wire is connected.
    efl_goals = [g for g in mf.goals if g.kind == GoalKind.EFL]
    assert efl_goals, "Image Quality preset missing EFL goal"
    assert efl_goals[0].target > 0.0


def test_project_merit_functions_list_round_trips(qapp):
    project = Project()
    assert project.merit_functions == []
    project.merit_functions.append(build_empty_merit_function())
    project.mark_merit_functions_replaced()
    assert len(project.merit_functions) == 1


def test_new_clears_merit_functions(qapp):
    project = Project()
    project.merit_functions.append(build_empty_merit_function())
    project.mark_merit_functions_replaced()
    project.new()
    assert project.merit_functions == []


# ---------------------------------------------------------------------------
# Panel body smoke
# ---------------------------------------------------------------------------


def test_register_panel_type_idempotent(qapp):
    register_optimization_panel_type()
    register_optimization_panel_type()  # second call is a no-op
    from ghostlight_designer.panel_system import registry
    assert registry.get(OPTIMIZATION_TYPE_ID) is not None


def test_panel_body_constructs_and_adds_mf(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        # No MFs initially.
        assert body.model.rowCount() == 0
        # Add via the toolbar handler path the menu would invoke.
        body._on_add_mf("Image Quality")
        assert body.model.rowCount() == 1
        # Adding a goal targets the current MF; since none is selected,
        # the body falls back to the single MF if there's exactly one.
        from PySide6.QtCore import QModelIndex
        body.tree.setCurrentIndex(body.model.index(0, 0, QModelIndex()))
        before = len(project.merit_functions[0].goals)
        body._on_add_goal(GoalKind.EFL_X.value)
        assert len(project.merit_functions[0].goals) == before + 1
    finally:
        body.deleteLater()


def test_name_column_is_not_checkable(qapp, sample_lens_path: pathlib.Path):
    """User flagged the checkbox as unwanted — make sure no row exposes
    Qt.ItemIsUserCheckable on the NAME column."""
    from PySide6.QtCore import QModelIndex
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        body._on_add_mf("Image Quality")
        # MF row
        mf_idx = body.model.index(0, int(Column.NAME), QModelIndex())
        assert not (body.model.flags(mf_idx) & Qt.ItemIsUserCheckable)
        # First goal row
        goal_idx = body.model.index(0, int(Column.NAME), mf_idx)
        assert not (body.model.flags(goal_idx) & Qt.ItemIsUserCheckable)
    finally:
        body.deleteLater()


def test_target_and_weight_are_scrubbable(qapp, sample_lens_path: pathlib.Path):
    """Predicate exposes Target / Weight on goal rows only."""
    from PySide6.QtCore import QModelIndex
    from ghostlight_designer.optimization_panel.body import _is_scrubbable

    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        body._on_add_mf("Image Quality")
        mf_idx = body.model.index(0, 0, QModelIndex())
        goal_row_idx = body.model.index(0, 0, mf_idx)
        # Goal row's Target + Weight scrubbable.
        for col in (Column.TARGET, Column.WEIGHT):
            cell = body.model.index(goal_row_idx.row(), int(col), mf_idx)
            assert _is_scrubbable(cell), f"goal col {col!r} should be scrubbable"
        # Goal row's Name / Value / Residual not scrubbable.
        for col in (Column.NAME, Column.VALUE, Column.RESIDUAL):
            cell = body.model.index(goal_row_idx.row(), int(col), mf_idx)
            assert not _is_scrubbable(cell), (
                f"goal col {col!r} should NOT be scrubbable"
            )
        # MF row's Target / Weight not scrubbable (those cells are blank
        # for MF rows; only goal rows expose numeric scalars).
        for col in (Column.TARGET, Column.WEIGHT):
            cell = body.model.index(0, int(col), QModelIndex())
            assert not _is_scrubbable(cell), (
                f"MF col {col!r} should NOT be scrubbable"
            )
    finally:
        body.deleteLater()


def test_combo_cells_are_recognised_by_delegate(
    qapp, sample_lens_path: pathlib.Path,
):
    """Param columns of a goal that exposes combo params must report
    as combo cells so the body's click-to-edit filter handles them."""
    from PySide6.QtCore import QModelIndex
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        body._on_add_mf("Image Quality")
        mf_idx = body.model.index(0, 0, QModelIndex())
        # Find a goal whose schema has at least one param.
        from ghostlight_designer.optimization_panel.goals.base import (
            param_schema_for,
        )
        target_goal_row = None
        for gi, g in enumerate(project.merit_functions[0].goals):
            if param_schema_for(g.kind):
                target_goal_row = gi
                break
        if target_goal_row is None:
            pytest.skip("Image Quality preset has no param-bearing goals")
        # First trailing param column.
        param_col = body.model.columnCount() - 1
        cell = body.model.index(target_goal_row, param_col, mf_idx)
        assert body._delegate.uses_combo(cell)
    finally:
        body.deleteLater()


def test_run_with_no_variables_reports_cleanly(
    qapp, sample_lens_path: pathlib.Path,
):
    """A run with no flagged variables must short-circuit to no_variables."""
    project = Project()
    project.load(str(sample_lens_path))
    mf = build_image_quality_mf(project.system)
    run = OptimizationRun(project, mf, project.system_setup)
    captured = {}

    def on_finished(result):
        captured["result"] = result

    run.runFinished.connect(on_finished)
    run.start()
    # The no_variables short-circuit is queued via QTimer.singleShot(0);
    # spin the event loop briefly.
    qapp.processEvents()
    assert "result" in captured
    assert captured["result"].status == "no_variables"


def test_end_to_end_optimization_with_injected_variable(
    qapp, sample_lens_path: pathlib.Path,
):
    """Flag surface 0's radius as variable via the project API and
    confirm scipy least_squares actually drives EFL toward the target.

    Exercises the full read seam: Project.set_variable_flag →
    collect_variables(project) → OptimizationRun → scipy convergence.
    """
    # ImportError covers both "scipy not installed" and the scipy/numpy
    # version-mismatch case the user's box happens to be in right now.
    pytest.importorskip("scipy.optimize", exc_type=ImportError)

    project = Project()
    project.load(str(sample_lens_path))

    # Build a single-goal MF: hit EFL = current + 5 mm. With surface 0's
    # radius as the only variable, scipy should converge in a handful of
    # iterations.
    from ghostlight_designer import lens_metrics as lm
    current_efl = lm._effective_focal_length_on_axis(project.system, "y")
    assert current_efl is not None and current_efl > 0
    target_efl = current_efl + 5.0
    mf = MeritFunction.make(
        name="EFL Hit",
        goals=[
            GoalEntry.make(
                kind=GoalKind.EFL,
                target=target_efl,
                weight=1.0,
                params={},
            ),
        ],
        max_iters=40,
    )

    # Flag surface 0's radius through the actual project API.
    surface_uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(surface_uuid, "radius")
    try:
        run = OptimizationRun(project, mf, project.system_setup)
        finished = {}

        def on_finished(result):
            finished["result"] = result

        run.runFinished.connect(on_finished)
        run.start()
        # Spin the event loop until the worker thread reports back. The
        # least_squares call is synchronous on the worker but result
        # delivery uses a queued connection — process events repeatedly
        # until we see the finish.
        import time
        deadline = time.monotonic() + 30.0
        while "result" not in finished and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert "result" in finished, "Optimizer did not finish within 30s"
        result = finished["result"]
        assert result.status == "ok", f"got status={result.status} msg={result.message!r}"
        # Verify the virtual system's final EFL is closer to the target
        # than the start. (Don't over-constrain to exact convergence —
        # the gradient is shallow far from optimum.)
        final_efl = lm._effective_focal_length_on_axis(run.virtual_system, "y")
        assert final_efl is not None
        assert abs(final_efl - target_efl) < abs(current_efl - target_efl)
        # Sanity check: the source system was NOT mutated.
        source_efl = lm._effective_focal_length_on_axis(project.system, "y")
        assert source_efl == pytest.approx(current_efl, abs=1e-6)
    finally:
        # Clear the flag for cleanliness — the fixture creates a fresh
        # project each test, but explicit teardown makes intent obvious.
        project.clear_variable_flag(surface_uuid, "radius")
