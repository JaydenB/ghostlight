"""Tests for evaluation panels: shared base class + spot diagram.

Coverage:
* Global View → Auto-Update toggle on :class:`AppSettings` round-trips
  and emits a signal.
* :class:`SpotDiagramSpec` clamps degenerate values.
* :class:`SpotDiagramBody` honours the two-layer auto-update gate.
* "Sync from System Setup" copies wavelengths and fields onto the spec.
* Ray-tracing compute returns plausible shapes for a real lens.
"""
from __future__ import annotations


import numpy as np
import pytest

from ghostlight_designer.evaluation_panels.spot_diagram.body import SpotDiagramBody
from ghostlight_designer.evaluation_panels.spot_diagram.spec import (
    DEFAULT_DEFOCUS_OFFSETS_MM,
    DEFAULT_FIELDS_DEG,
    DEFAULT_WAVELENGTHS_NM,
    SpotDiagramSpec,
)
from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import Field, Wavelength

from _corpus import EXAMPLE_DOUBLET


# ---------------------------------------------------------------------------
# AppSettings: global auto-update flag
# ---------------------------------------------------------------------------


def test_auto_update_defaults_to_true(isolated_settings):
    assert isolated_settings.auto_update_enabled() is True


def test_auto_update_round_trips(isolated_settings):
    isolated_settings.set_auto_update_enabled(False)
    assert isolated_settings.auto_update_enabled() is False
    isolated_settings.set_auto_update_enabled(True)
    assert isolated_settings.auto_update_enabled() is True


def test_auto_update_emits_only_on_change(isolated_settings):
    received: list[bool] = []
    isolated_settings.autoUpdateChanged.connect(received.append)

    # Setting to the same value as default is a no-op.
    isolated_settings.set_auto_update_enabled(True)
    assert received == []

    isolated_settings.set_auto_update_enabled(False)
    isolated_settings.set_auto_update_enabled(False)  # no second signal
    isolated_settings.set_auto_update_enabled(True)
    assert received == [False, True]


# ---------------------------------------------------------------------------
# SpotDiagramSpec
# ---------------------------------------------------------------------------


def test_spec_clamp_rescues_zero_rings():
    """0 rings is legal (just the axial ray) — clamp keeps it.
    1 fan is the minimum; below that we promote to 1."""
    s = SpotDiagramSpec(rings=-3, fans=0).clamp()
    assert s.rings == 0
    assert s.fans == 1


def test_spec_clamp_falls_back_on_empty_wavelengths():
    s = SpotDiagramSpec(wavelengths_nm=()).clamp()
    assert s.wavelengths_nm == DEFAULT_WAVELENGTHS_NM


def test_spec_clamp_falls_back_on_empty_fields():
    s = SpotDiagramSpec(fields_deg=()).clamp()
    assert s.fields_deg == DEFAULT_FIELDS_DEG


def test_spec_clamp_pupil_radius_negative_becomes_auto():
    s = SpotDiagramSpec(pupil_radius_mm=-5.0).clamp()
    assert s.pupil_radius_mm == 0.0  # 0 → auto-from-front-surface


def test_spec_clamp_empty_defocus_falls_back_to_zero():
    s = SpotDiagramSpec(defocus_offsets_mm=()).clamp()
    assert s.defocus_offsets_mm == (0.0,)


# ---------------------------------------------------------------------------
# SpotDiagramBody
# ---------------------------------------------------------------------------


def _make_body(qapp, isolated_settings):
    project = Project()
    body = SpotDiagramBody(project, isolated_settings)
    return project, body


def test_body_constructs_without_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        assert body.auto_update_local is True
        # The settings sidebar is visible by default — same affordance
        # as a render panel's image being visible by default.
        assert body.settings_visible is True
        # No in-panel toolbar — actions live on the View menu instead.
        assert not hasattr(body, "toolbar")
        # No lens → no compute possible.
        assert not body._lens_eligible()
    finally:
        body.deleteLater()


def test_settings_sidebar_toggles(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        # Showing/hiding only affects layout — the spec stays the same.
        original_spec = body.spec
        body.set_settings_visible(False)
        assert body.settings_visible is False
        assert body.spec == original_spec
        body.set_settings_visible(True)
        assert body.settings_visible is True
        assert body.spec == original_spec
    finally:
        body.deleteLater()


def test_reset_spec_to_defaults_replaces_custom_spec(qapp, isolated_settings):
    from ghostlight_designer.evaluation_panels.spot_diagram.spec import SpotDiagramSpec
    project, body = _make_body(qapp, isolated_settings)
    try:
        # Set a non-default spec.
        body._spec = SpotDiagramSpec(rings=1, fans=2, plot_half_extent_mm=10.0)
        body._editor.set_spec(body._spec, emit=False)
        assert body.spec.rings == 1

        body.reset_spec_to_defaults()
        defaults = SpotDiagramSpec()
        assert body.spec == defaults
    finally:
        body.deleteLater()


def test_should_auto_update_gates_on_both_layers(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        # Default: both on.
        assert body._should_auto_update() is True

        # Per-panel off → off.
        body.set_auto_update_local(False)
        assert body._should_auto_update() is False
        body.set_auto_update_local(True)
        assert body._should_auto_update() is True

        # Global off → off, even with per-panel on.
        isolated_settings.set_auto_update_enabled(False)
        assert body._should_auto_update() is False

        # Both off.
        body.set_auto_update_local(False)
        assert body._should_auto_update() is False

        # Both back on.
        isolated_settings.set_auto_update_enabled(True)
        body.set_auto_update_local(True)
        assert body._should_auto_update() is True
    finally:
        body.deleteLater()


def test_view_menu_action_order_matches_render_panels(qapp, isolated_settings):
    """The spot diagram View menu must match the order and grouping of
    the other render panels' View menus so users see the same
    affordances in the same places."""
    from ghostlight_designer.evaluation_panels.spot_diagram.menus import build_menus
    project, body = _make_body(qapp, isolated_settings)
    try:
        menus = build_menus(body, project)
        assert len(menus) == 1
        view_menu = menus[0]
        assert view_menu.title() == "&View"

        # Visible action labels in display order, separators skipped.
        labels = [a.text() for a in view_menu.actions() if not a.isSeparator()]
        assert labels == [
            "&Auto-Update",
            "&Refresh",
            "&Sync from System Setup",
            "Show &Settings Sidebar",
            "&Auto-Fit Scale Now",
            "Reset Spec to &Defaults",
        ]
        # Auto-Update + Show Settings are checkable; the action group
        # of one-shot actions are not.
        checkable = {a.text(): a.isCheckable()
                     for a in view_menu.actions() if not a.isSeparator()}
        assert checkable["&Auto-Update"] is True
        assert checkable["Show &Settings Sidebar"] is True
        assert checkable["&Refresh"] is False
        assert checkable["&Sync from System Setup"] is False
        assert checkable["&Auto-Fit Scale Now"] is False
        assert checkable["Reset Spec to &Defaults"] is False
    finally:
        body.deleteLater()


def test_lens_edit_with_global_off_marks_dirty_pending(qapp, isolated_settings):
    """When the global toggle is off, lens edits must NOT schedule
    a refresh — but the panel must remember to catch up later."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        body._is_active = True
        isolated_settings.set_auto_update_enabled(False)
        body._dirty_pending = False  # reset

        # Lens edit — should be gated.
        body._on_system_modified()
        assert body._dirty_pending is True
        # Debounce timer should NOT be running.
        assert not body._debounce.isActive()

        # Flip global back on; the panel catches up via request_refresh.
        isolated_settings.set_auto_update_enabled(True)
        # request_refresh starts the debounce.
        assert body._debounce.isActive()
    finally:
        body.deleteLater()


def test_sync_from_system_setup_copies_wavelengths_and_fields(
    qapp, isolated_settings,
):
    project, body = _make_body(qapp, isolated_settings)
    try:
        # Replace the project's first sequence wavelengths and fields.
        seq = project.system_setup.sequences[0]
        seq.source.wavelengths.wavelengths = [
            Wavelength(420.0),
            Wavelength(550.0),
        ]
        seq.source.fields = [
            Field("F0", 0.0, 0.0),
            Field("F1", 5.0, 0.0),
            Field("F2", 0.0, 12.0),
        ]
        seq.source.aperture_radius = 4.2

        body.sync_from_system_setup()
        # Spec should now reflect the project's values.
        assert body._spec.wavelengths_nm == (420.0, 550.0)
        assert body._spec.fields_deg == ((0.0, 0.0), (5.0, 0.0), (0.0, 12.0))
        assert body._spec.pupil_radius_mm == pytest.approx(4.2)
    finally:
        body.deleteLater()


def test_compute_returns_result_for_real_lens(qapp, isolated_settings):
    """End-to-end: load the sample lens, run compute(), confirm we get
    the expected shapes back."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        assert body._lens_eligible()

        result = body.compute()
        assert result is not None
        # Default spec: 3 fields × 3 wavelengths × (1 + 4*8) samples.
        n_samples_expected = 1 + body._spec.rings * body._spec.fans
        assert len(result.fields) == len(body._spec.fields_deg)
        for field_result in result.fields:
            assert field_result.xs.shape == (
                len(body._spec.defocus_offsets_mm),
                len(body._spec.wavelengths_nm),
                n_samples_expected,
            )
            # At least the axial sample on the axial field should land
            # cleanly — anything less means the trace is broken.
            assert field_result.valid_mask.sum() > 0
    finally:
        body.deleteLater()


def test_compute_returns_none_when_lens_too_small(qapp, isolated_settings):
    """A bare Project has zero surfaces — compute should bail without
    raising so the canvas can show a placeholder."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        body._is_active = True
        result = body.compute()
        assert result is None
    finally:
        body.deleteLater()


def test_lens_edit_changes_spot_bundle(qapp, isolated_settings):
    """The trace must respond to lens edits — moving the sensor surface
    by 1 mm should produce a measurably different on-axis spot.

    Regression for the "single dot, never changes" report: with the old
    fixed 0.5 mm plot extent the user saw an unchanging single dot.
    Now we both auto-scale the plot AND prove here that the underlying
    RMS spot radius actually moves when the user edits the lens.
    """
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))

        baseline = body.compute()
        assert baseline is not None

        # Move the final surface (image plane) by +1 mm — a "defocus the
        # lens" edit, which the panel must not render as a no-op.
        last_idx = project.system.num_surfaces() - 1
        last = project.system.surfaces[last_idx]
        orig_z = float(last.z)
        last.z = orig_z + 1.0
        try:
            edited = body.compute()
        finally:
            last.z = orig_z
        assert edited is not None

        # Compare on-axis Gaussian-slice RMS — that's what the user
        # sees in the centre subplot.
        def rms(result, fi: int, di: int) -> float:
            fr = result.fields[fi]
            mask = fr.valid_mask[di]
            assert mask.any(), "no valid rays in slice"
            xs = fr.xs[di][mask]
            ys = fr.ys[di][mask]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            return float(np.sqrt(np.mean((xs - cx) ** 2 + (ys - cy) ** 2)))

        # Find the Gaussian (defocus = 0.0) slice index.
        gauss_idx = baseline.defocus_offsets_mm.index(0.0)
        before = rms(baseline, 0, gauss_idx)
        after = rms(edited, 0, gauss_idx)
        assert before != pytest.approx(after, rel=1e-6), (
            f"sensor +1mm did not change the on-axis RMS spot "
            f"(before={before}, after={after}). Trace is not "
            f"responding to lens edits — the panel will look static."
        )
    finally:
        body.deleteLater()


def test_status_survives_apply_result(qapp, isolated_settings):
    """The base class must not overwrite the subclass's
    descriptive status with a useless ``"OK"`` after every apply_result.
    Make sure the subclass message sticks now."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        body._is_active = True
        result = body.compute()
        assert result is not None

        # Mimic the worker → poller path: pretend we were "Computing…"
        # so the fallback would have kicked in if subclass set nothing.
        body.set_status("Computing…")
        body.apply_result(result)
        text = body.status_label.text()
        # Subclass writes a "{N} fields × {M} λ …" line — check it
        # ended up on the label (and that "OK" did NOT overwrite it).
        assert text != "OK"
        assert text != "Computing…"
        assert "field" in text.lower() or "λ" in text or "pupil" in text.lower()
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Canvas helpers: auto-extent + unit selection
# ---------------------------------------------------------------------------


def test_field_auto_extent_grows_with_bundle_spread():
    """Per-field auto-extent should follow the largest distance from
    centroid across the field's defocus slices."""
    from ghostlight_designer.evaluation_panels.spot_diagram.canvas import (
        _field_auto_extent,
    )
    from ghostlight_designer.evaluation_panels.spot_diagram.compute import SpotFieldResult

    # 1 defocus slice, 1 wavelength, 4 samples forming a square of half-
    # width 0.01 mm around (0,0). Max distance from centroid = sqrt(2) * 0.01.
    xs = np.array([[[-0.01, 0.01, 0.01, -0.01]]])
    ys = np.array([[[-0.01, -0.01, 0.01, 0.01]]])
    valid = np.ones_like(xs, dtype=bool)
    fr = SpotFieldResult(0.0, 0.0, xs, ys, valid)
    extent = _field_auto_extent(fr)
    # max_r = sqrt(2)*0.01 ≈ 0.01414, with 1.3x padding ≈ 0.01838 mm
    assert extent == pytest.approx(np.sqrt(2) * 0.01 * 1.3, rel=1e-3)


def test_format_extent_label_switches_units():
    from ghostlight_designer.evaluation_panels.spot_diagram.canvas import (
        _format_extent_label,
    )
    # Tiny spots → µm
    scale, unit, _ = _format_extent_label(0.001)
    assert unit == "µm"
    assert scale == pytest.approx(1000.0)
    # Sub-100µm → still µm
    scale, unit, _ = _format_extent_label(0.05)
    assert unit == "µm"
    # ≥ 0.1 mm → mm
    scale, unit, _ = _format_extent_label(0.5)
    assert unit == "mm"
    assert scale == pytest.approx(1.0)


def test_default_spec_has_multiple_defocus_slices():
    """The default spec should make the focus/defocus comparison
    visible out-of-the-box — single-slice defaults trapped users into
    not seeing focus changes (the "single dot" regression)."""
    spec = SpotDiagramSpec()
    assert len(spec.defocus_offsets_mm) >= 3
    # Must straddle the Gaussian image plane.
    assert any(d < 0.0 for d in spec.defocus_offsets_mm)
    assert any(d == 0.0 for d in spec.defocus_offsets_mm)
    assert any(d > 0.0 for d in spec.defocus_offsets_mm)


def test_default_spec_plot_extent_is_auto():
    """Default extent must be 0 (auto-scale) so the user gets a useful
    picture on any lens without manual tweaking."""
    spec = SpotDiagramSpec()
    assert spec.plot_half_extent_mm == 0.0


# ---------------------------------------------------------------------------
# Panels-menu category grouping
# ---------------------------------------------------------------------------


def test_spot_lands_at_sensor_plane_not_last_lens_surface(qapp, isolated_settings):
    """**Critical regression.** The image plane is the virtual z=0 plane,
    not ``system.surfaces[-1]`` (which is the back of the last lens
    element, typically z ≈ -30 mm for a photographic lens). An earlier
    version of compute() read landings from ``events[-1].hit_point``,
    which is the LAST LENS SURFACE — rays hadn't converged there, so
    the "spot" was the bundle's lens-back diameter (millimetres) and
    barely changed under lens edits.

    Verify that on-axis Gaussian (defocus=0) spots are tightly
    clustered near (0, 0) for a real focused doublet, i.e. orders of
    magnitude smaller than the bundle diameter at the lens back.
    """
    from ghostlight_designer.evaluation_panels.spot_diagram.compute import (
        compute_spot_diagram,
    )
    import ghostlight

    project = Project()
    lens_path = EXAMPLE_DOUBLET
    if not lens_path.exists():
        pytest.skip("sample lens not present")
    project.load(str(lens_path))

    spec = SpotDiagramSpec(
        wavelengths_nm=(587.56,),
        fields_deg=((0.0, 0.0),),
        rings=2, fans=8,
        pupil_radius_mm=2.0,
        defocus_offsets_mm=(0.0,),
        plot_half_extent_mm=0.0,
    )
    result = compute_spot_diagram(project.system, spec)
    fr = result.fields[0]
    mask = fr.valid_mask[0]
    assert mask.any(), "no rays reached the sensor for axial field"
    xs = fr.xs[0][mask]
    ys = fr.ys[0][mask]
    radii = np.hypot(xs, ys)
    # If we were (incorrectly) reading the last-lens-surface hit, the
    # rays would be spread across the pupil radius (~2 mm). The sensor
    # landing for an on-axis collimated bundle through a doublet is in
    # the tens-of-microns range. 0.5 mm is a generous ceiling that
    # rules out the lens-back regression while staying lens-agnostic.
    assert float(np.max(radii)) < 0.5, (
        f"On-axis spot too large ({float(np.max(radii))} mm). Are we "
        f"reading the last lens surface again instead of the z=0 sensor?"
    )
    # And while we're here, confirm the result.sensor_z is 0 (the
    # convention this fix encodes).
    assert result.sensor_z_mm == 0.0


def test_scale_lock_persists_across_renders(qapp, isolated_settings):
    """The body must capture the first render's auto-fit and reuse it
    on subsequent renders so a lens edit visibly changes spot SIZE.

    Without this lock the canvas re-auto-fits every render and the
    user can't see whether their edit made the lens better or worse —
    that's the user-reported "doesn't update as I'd expect" bug.
    """
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))

        # First render captures the lock.
        r1 = body.compute()
        assert r1 is not None
        body.apply_result(r1)
        assert body._reference_extents is not None
        first_lock = dict(body._reference_extents)
        assert first_lock, "reference extents should be populated"

        # Mutate the lens — that grows / shrinks the bundle. The lock
        # must NOT be replaced by the new auto-fit.
        last_idx = project.system.num_surfaces() - 1
        last = project.system.surfaces[last_idx]
        orig_z = float(last.z)
        last.z = orig_z + 2.0
        try:
            r2 = body.compute()
            body.apply_result(r2)
            assert body._reference_extents == first_lock, (
                "scale lock must persist across renders; got a different "
                "map after a lens edit"
            )
        finally:
            last.z = orig_z

        # Explicit auto-fit drops the lock.
        body.auto_fit_scale_now()
        # auto_fit_scale_now clears the lock; force_refresh_now is
        # async-ish (kicks the worker), so just check the lock got
        # cleared regardless of whether a fresh apply has happened.
        assert body._reference_extents is None
    finally:
        body.deleteLater()


def test_scale_lock_resets_on_lens_swap(qapp, isolated_settings):
    """A brand-new lens load must invalidate the old scale lock — the
    new geometry has unrelated spot sizes."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = EXAMPLE_DOUBLET
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        r = body.compute()
        body.apply_result(r)
        assert body._reference_extents is not None

        # Reload the lens — counts as a new system → lock should reset.
        project.load(str(lens_path))
        assert body._reference_extents is None
    finally:
        body.deleteLater()


def test_spot_diagram_type_has_evaluations_category(qapp, isolated_settings):
    """Registration must mark the type with category='Evaluations' so
    the Panels submenu groups it alongside any future evaluation panels,
    mirroring the existing 'Renderers' grouping."""
    from ghostlight_designer.panel_system import registry
    from ghostlight_designer.evaluation_panels.spot_diagram import (
        SPOT_DIAGRAM_TYPE_ID,
        register_spot_diagram_panel_type,
    )
    register_spot_diagram_panel_type(isolated_settings)
    t = registry.get(SPOT_DIAGRAM_TYPE_ID)
    assert t is not None
    assert t.category == "Evaluations"
