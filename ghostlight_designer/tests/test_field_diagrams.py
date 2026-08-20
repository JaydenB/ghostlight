"""Tests for the ``field_diagrams`` evaluation panel.

Covers:
* Spec clamping (degenerate inputs are coerced to safe values).
* Sagittal / tangential focus on a real lens — at axis the two foci
  agree, off-axis they diverge.
* Distortion is zero at axis by construction and non-zero off-axis.
* Lateral chromatic columns are populated for every wavelength.
* The body integrates with EvaluationPanelBody (auto-update gate,
  status, sync-from-system-setup).
* View menu mirrors the spot-diagram order.
* Registration tags the type with ``category="Evaluations"``.
"""
from __future__ import annotations


import numpy as np
import pytest

from ghostlight_designer.evaluation_panels.field_diagrams.body import FieldDiagramBody
from ghostlight_designer.evaluation_panels.field_diagrams.compute import (
    compute_field_diagrams,
)
from ghostlight_designer.evaluation_panels.field_diagrams.spec import (
    DEFAULT_WAVELENGTHS_NM,
    FieldDiagramSpec,
)
from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import Field, Wavelength

from _corpus import EXAMPLE_DOUBLET


def _make_body(qapp, isolated_settings):
    project = Project()
    body = FieldDiagramBody(project, isolated_settings)
    return project, body


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_spec_clamp_minimum_field_samples():
    s = FieldDiagramSpec(field_samples=0).clamp()
    assert s.field_samples >= 2


def test_spec_clamp_empty_wavelengths_falls_back():
    s = FieldDiagramSpec(wavelengths_nm=()).clamp()
    assert s.wavelengths_nm == DEFAULT_WAVELENGTHS_NM


def test_spec_clamp_primary_index_bounded():
    s = FieldDiagramSpec(
        wavelengths_nm=(486.13, 587.56),
        primary_wavelength_index=99,
    ).clamp()
    assert s.primary_wavelength_index == 1  # last valid


def test_spec_primary_wavelength_property_picks_index():
    s = FieldDiagramSpec(
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


def test_compute_axis_field_has_zero_distortion():
    project = _load_lens()
    spec = FieldDiagramSpec(max_field_deg=10.0, field_samples=6)
    result = compute_field_diagrams(project.system, spec)
    # Field index 0 is on-axis. By construction (paraxial reference fit
    # is anchored at the smallest non-zero field, treating that as
    # distortion-free), the axial sample reports 0% — anything else
    # would mean the anchor logic broke.
    assert result.distortion_pct[0] == pytest.approx(0.0, abs=1e-9)


def test_compute_axis_sag_equals_tan():
    """On the optical axis there's no astigmatism — sagittal and
    tangential foci must coincide."""
    project = _load_lens()
    spec = FieldDiagramSpec(max_field_deg=10.0, field_samples=6, rays_per_fan=5)
    result = compute_field_diagrams(project.system, spec)
    sag0 = result.sagittal_defocus_mm[0]
    tan0 = result.tangential_defocus_mm[0]
    assert not np.isnan(sag0)
    assert not np.isnan(tan0)
    # Tight tolerance — the two fans hit the same focal point on axis.
    assert sag0 == pytest.approx(tan0, abs=1e-6)


def test_compute_offaxis_distortion_nonzero():
    project = _load_lens()
    spec = FieldDiagramSpec(max_field_deg=14.0, field_samples=6)
    result = compute_field_diagrams(project.system, spec)
    last = result.distortion_pct[-1]
    assert not np.isnan(last)
    # A simple doublet at 14° off-axis has visible distortion (>0.1%).
    assert abs(last) > 0.1


def test_compute_offaxis_astigmatism_separates():
    """At a significant off-axis field, sagittal and tangential foci
    should be measurably different — that's astigmatism by definition."""
    project = _load_lens()
    spec = FieldDiagramSpec(max_field_deg=14.0, field_samples=6, rays_per_fan=5)
    result = compute_field_diagrams(project.system, spec)
    sag = result.sagittal_defocus_mm[-1]
    tan = result.tangential_defocus_mm[-1]
    assert not np.isnan(sag)
    assert not np.isnan(tan)
    # Spread of > 100 µm between sagittal and tangential at 14° is
    # easily exceeded for a simple doublet.
    assert abs(sag - tan) > 0.1


def test_compute_chromatic_columns_populated():
    project = _load_lens()
    spec = FieldDiagramSpec(
        max_field_deg=14.0, field_samples=4,
        wavelengths_nm=(486.13, 587.56, 656.27),
        primary_wavelength_index=1,
    )
    result = compute_field_diagrams(project.system, spec)
    # Every (field, wavelength) cell must be populated for the lateral
    # chromatic plot to draw a curve.
    assert result.chief_y_per_wavelength_mm.shape == (4, 3)
    assert not np.isnan(result.chief_y_per_wavelength_mm).any()
    # The chief-ray landings for different wavelengths must differ
    # off-axis — otherwise lateral chromatic is identically zero and
    # the panel has nothing to show.
    last_field = result.chief_y_per_wavelength_mm[-1]
    spread = float(last_field.max() - last_field.min())
    assert spread > 0.0


# ---------------------------------------------------------------------------
# Body integration with EvaluationPanelBody
# ---------------------------------------------------------------------------


def test_body_constructs_without_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        assert body.auto_update_local is True
        assert body.settings_visible is True
        # Inherits the base-class plumbing (no in-panel toolbar).
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
        # Confirm body.apply_result feeds the canvas without raising
        # and writes a useful status (not "OK", not empty).
        body.apply_result(result)
        text = body.status_label.text()
        assert text != "OK"
        assert text != ""
        assert "D_max" in text
    finally:
        body.deleteLater()


def test_sync_from_system_setup_takes_max_abs_field_tilt(qapp, isolated_settings):
    """The setup carries discrete field tilts; for the field-diagrams
    *range*, we take the largest |tilt_y_deg|."""
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
        assert body.spec.max_field_deg == pytest.approx(12.0)
        assert body.spec.primary_wavelength_index == 1
        assert body.spec.pupil_radius_mm == pytest.approx(5.0)
        assert body.spec.wavelengths_nm == (486.13, 587.56, 656.27)
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Menus + registration
# ---------------------------------------------------------------------------


def test_view_menu_action_order_matches_spot_diagram(qapp, isolated_settings):
    """Field diagrams must use the same View menu order as spot diagram
    so users don't have to relearn the layout when switching panels."""
    from ghostlight_designer.evaluation_panels.field_diagrams.menus import build_menus
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


def test_field_diagrams_type_has_evaluations_category(qapp, isolated_settings):
    from ghostlight_designer.panel_system import registry
    from ghostlight_designer.evaluation_panels.field_diagrams import (
        FIELD_DIAGRAMS_TYPE_ID,
        register_field_diagrams_panel_type,
    )
    register_field_diagrams_panel_type(isolated_settings)
    t = registry.get(FIELD_DIAGRAMS_TYPE_ID)
    assert t is not None
    assert t.category == "Evaluations"
