"""Tests for the Seidel-bar-chart evaluation panel.

Covers:
* Spec clamping (degenerate inputs are coerced to safe values).
* Paraxial trace sanity — Lagrange invariant is constant through the
  system and is zero only when the chief ray field is zero.
* Stop surface (c = 0) contributes nothing to S_I … S_V at any sane
  field — confirms the bar chart leaves it visibly blank.
* The example doublet has non-zero spherical + axial chromatic at
  field 5° (real-world signature of an uncorrected doublet at this
  scale).
* Body integration with EvaluationPanelBody (auto-update gate,
  status, sync-from-system-setup).
* View menu mirrors the spot / field-diagram order.
* Registration tags the type with ``category="Evaluations"``.
"""
from __future__ import annotations


import numpy as np
import pytest

from ghostlight_designer.evaluation_panels.seidel.body import SeidelBody
from ghostlight_designer.evaluation_panels.seidel.compute import (
    _collect_surfaces,
    _paraxial_trace,
    _trace_marginal_and_chief,
    compute_seidel,
)
from ghostlight_designer.evaluation_panels.seidel.spec import (
    DEFAULT_WAVELENGTHS_NM,
    SeidelSpec,
)
from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import Field, Wavelength

from _corpus import EXAMPLE_DOUBLET


def _make_body(qapp, isolated_settings):
    project = Project()
    body = SeidelBody(project, isolated_settings)
    return project, body


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_spec_clamp_negative_field_pinned_to_zero():
    s = SeidelSpec(field_deg=-3.0).clamp()
    assert s.field_deg == 0.0


def test_spec_clamp_empty_wavelengths_falls_back():
    s = SeidelSpec(wavelengths_nm=()).clamp()
    assert s.wavelengths_nm == DEFAULT_WAVELENGTHS_NM


def test_spec_clamp_primary_index_bounded():
    s = SeidelSpec(
        wavelengths_nm=(486.13, 587.56),
        primary_wavelength_index=99,
    ).clamp()
    assert s.primary_wavelength_index == 1


def test_spec_primary_wavelength_picks_index():
    s = SeidelSpec(
        wavelengths_nm=(486.13, 587.56, 656.27),
        primary_wavelength_index=2,
    ).clamp()
    assert s.primary_wavelength_nm == pytest.approx(656.27)


# ---------------------------------------------------------------------------
# Compute on the real example doublet
# ---------------------------------------------------------------------------


def _load_lens():
    lens_path = EXAMPLE_DOUBLET
    if not lens_path.exists():
        pytest.skip("sample lens not present")
    project = Project()
    project.load(str(lens_path))
    return project


def test_compute_returns_expected_shape():
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0)
    result = compute_seidel(project.system, spec)
    # The example doublet has 5 refracting surfaces + 1 image plane,
    # so the Seidel arrays must be length 5.
    n_refracting = project.system.num_surfaces() - 1
    assert result.surface_indices.shape == (n_refracting,)
    for arr_name in (
        "spherical_per_surface",
        "coma_per_surface",
        "astigmatism_per_surface",
        "petzval_per_surface",
        "distortion_per_surface",
        "axial_color_per_surface",
        "lateral_color_per_surface",
    ):
        assert getattr(result, arr_name).shape == (n_refracting,)


def test_compute_stop_contributes_zero_to_seidel():
    """The stop is flat (c = 0). Every Seidel per-surface contribution
    must therefore be exactly 0 there — if it isn't, the formulas have
    a stray ``y · Δ(u/n)`` term that wasn't zeroed by ``c``."""
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0)
    result = compute_seidel(project.system, spec)
    # Find the stop surface in the surface_indices array.
    stop_idx = None
    for k, sys_idx in enumerate(result.surface_indices):
        if bool(project.system.surfaces[int(sys_idx)].is_stop):
            stop_idx = k
            break
    assert stop_idx is not None, "example doublet should have an is_stop surface"
    for arr in (
        result.spherical_per_surface,
        result.coma_per_surface,
        result.astigmatism_per_surface,
        result.petzval_per_surface,
        result.distortion_per_surface,
    ):
        assert arr[stop_idx] == pytest.approx(0.0, abs=1e-12)


def test_compute_axis_field_yields_zero_chief_aberrations():
    """On-axis (field = 0) the chief ray collapses onto the marginal
    direction, so coma, distortion, and the lateral-colour terms must
    all sum to 0 — they all carry an Ā factor that's identically zero
    on-axis."""
    project = _load_lens()
    spec = SeidelSpec(field_deg=0.0)
    result = compute_seidel(project.system, spec)
    assert result.coma_per_surface.sum() == pytest.approx(0.0, abs=1e-10)
    assert result.distortion_per_surface.sum() == pytest.approx(0.0, abs=1e-10)
    assert result.lateral_color_per_surface.sum() == pytest.approx(0.0, abs=1e-10)
    # Lagrange invariant is zero at zero field (no chief tilt).
    assert result.lagrange_invariant == pytest.approx(0.0, abs=1e-10)


def test_compute_offaxis_has_nontrivial_spherical_and_chromatic():
    """A simple doublet has visible spherical + axial colour. If either
    sum is zero we've collapsed something we shouldn't have."""
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0)
    result = compute_seidel(project.system, spec)
    assert abs(float(result.spherical_per_surface.sum())) > 1e-4
    assert abs(float(result.axial_color_per_surface.sum())) > 1e-4
    # And the Lagrange invariant is non-zero at 5° field.
    assert abs(result.lagrange_invariant) > 1e-3


def test_lagrange_invariant_is_constant_through_system():
    """Trace the marginal and chief rays directly and verify that
    ``n·u·ȳ − n·ū·y`` is constant from surface to surface. A non-
    constant Lagrange means the paraxial trace has lost track of the
    Snell invariant — a silent killer for the Seidel sums."""
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0).clamp()
    surfaces = _collect_surfaces(project.system, spec.primary_wavelength_nm)
    marg, chief, H = _trace_marginal_and_chief(
        surfaces, pupil_radius_mm=12.5, field_deg=spec.field_deg,
    )
    y_m, u_m, _ = marg
    y_c, u_c, _ = chief
    invariants = []
    for k in range(len(surfaces)):
        nb = surfaces[k].n_before
        invariants.append(nb * (u_m[k] * y_c[k] - u_c[k] * y_m[k]))
    invariants = np.asarray(invariants)
    # All within 1e-9 of H.
    assert np.allclose(invariants, H, atol=1e-9)


def test_chief_ray_passes_through_aperture_stop_center():
    """The chief-ray launch height is chosen so that y_chief = 0 at the
    stop. Verify directly — anything else means the linear-system
    inverse has a bug."""
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0).clamp()
    surfaces = _collect_surfaces(project.system, spec.primary_wavelength_nm)
    _, chief, _ = _trace_marginal_and_chief(
        surfaces, pupil_radius_mm=12.5, field_deg=spec.field_deg,
    )
    y_c = chief[0]
    stop_idx = next(k for k, s in enumerate(surfaces) if s.is_stop)
    assert y_c[stop_idx] == pytest.approx(0.0, abs=1e-9)


def test_marginal_ray_lands_at_pupil_radius_at_stop():
    project = _load_lens()
    spec = SeidelSpec(field_deg=5.0).clamp()
    surfaces = _collect_surfaces(project.system, spec.primary_wavelength_nm)
    pupil_r = 12.5
    marg, _, _ = _trace_marginal_and_chief(
        surfaces, pupil_radius_mm=pupil_r, field_deg=spec.field_deg,
    )
    y_m = marg[0]
    stop_idx = next(k for k, s in enumerate(surfaces) if s.is_stop)
    assert y_m[stop_idx] == pytest.approx(pupil_r, abs=1e-9)


# ---------------------------------------------------------------------------
# Body integration
# ---------------------------------------------------------------------------


def test_body_constructs_without_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        assert body.auto_update_local is True
        assert body.settings_visible is True
        assert not hasattr(body, "toolbar")
    finally:
        body.deleteLater()


def test_body_compute_returns_none_without_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        body._is_active = True
        assert body.compute() is None
    finally:
        body.deleteLater()


def test_body_compute_returns_result_for_real_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        result = body.compute()
        assert result is not None
        body.apply_result(result)
        text = body.status_label.text()
        # Status surfaces ΣS_I and the Lagrange invariant.
        assert "S_I" in text
        assert "H =" in text
    finally:
        body.deleteLater()


def test_sync_from_system_setup_takes_max_abs_field_tilt(qapp, isolated_settings):
    """Same convention as field-diagrams: largest |tilt_y_deg|."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        seq = project.system_setup.sequences[0]
        seq.source.wavelengths.wavelengths = [
            Wavelength(486.13), Wavelength(587.56), Wavelength(656.27),
        ]
        seq.source.wavelengths.primary_index = 1
        seq.source.fields = [
            Field("F0", 0.0, 0.0),
            Field("F1", 0.0, 7.5),
            Field("F2", 0.0, -12.0),  # |12| > |7.5|
        ]
        seq.source.aperture_radius = 5.0

        body.sync_from_system_setup()
        assert body.spec.field_deg == pytest.approx(12.0)
        assert body.spec.primary_wavelength_index == 1
        assert body.spec.pupil_radius_mm == pytest.approx(5.0)
        assert body.spec.wavelengths_nm == (486.13, 587.56, 656.27)
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Menus + registration
# ---------------------------------------------------------------------------


def test_view_menu_action_order_matches_other_evaluation_panels(qapp, isolated_settings):
    """Seidel must use the same View menu order as spot / field diagram
    so users don't have to relearn the layout when switching panels."""
    from ghostlight_designer.evaluation_panels.seidel.menus import build_menus
    project, body = _make_body(qapp, isolated_settings)
    try:
        menus = build_menus(body, project)
        assert len(menus) == 1
        view_menu = menus[0]
        assert view_menu.title() == "&View"
        labels = [a.text() for a in view_menu.actions() if not a.isSeparator()]
        assert labels == [
            "&Auto-Update",
            "&Refresh",
            "&Sync from System Setup",
            "Show &Settings Sidebar",
            "Reset Spec to &Defaults",
        ]
    finally:
        body.deleteLater()


def test_seidel_type_has_evaluations_category(qapp, isolated_settings):
    from ghostlight_designer.panel_system import registry
    from ghostlight_designer.evaluation_panels.seidel import (
        SEIDEL_TYPE_ID,
        register_seidel_panel_type,
    )
    register_seidel_panel_type(isolated_settings)
    t = registry.get(SEIDEL_TYPE_ID)
    assert t is not None
    assert t.category == "Evaluations"
    assert t.display_name == "Seidel Bar Chart"
