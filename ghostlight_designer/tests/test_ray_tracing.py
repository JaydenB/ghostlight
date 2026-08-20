"""Tests for the SystemSetup → RayBundle tracing pipeline + viewport wiring."""
from __future__ import annotations

import math

import pytest

import ghostlight
from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import (
    ApertureType,
    Distribution,
    DistributionType,
    Field,
    FieldType,
    Sequence,
    Source,
    SourceType,
    SystemSetup,
    Wavelength,
    WavelengthContainer,
)
from ghostlight_designer import ray_tracing as rt


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


def test_is_ready_to_trace_empty_system_is_false():
    system = ghostlight.OpticalSystem()
    assert rt.is_ready_to_trace(system) is False


def test_is_ready_to_trace_loaded_doublet_is_true(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    assert rt.is_ready_to_trace(system) is True


# ---------------------------------------------------------------------------
# Stop resolution — spec: Auto → is_stop surface → fallback to last surface
# ---------------------------------------------------------------------------


def test_resolve_stop_uses_is_stop_surface_when_auto(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence()  # stop_surface=None → Auto
    expected = next(
        (i for i, s in enumerate(system.surfaces) if s.is_stop), None
    )
    if expected is None:
        pytest.skip("sample lens has no is_stop surface")
    assert rt.resolve_stop_index(system, seq) == expected


def test_resolve_stop_falls_back_to_last_surface_without_is_stop(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    # Clear is_stop everywhere so we exercise the fallback.
    for s in system.surfaces:
        if s.is_stop:
            s.is_stop = False
    seq = Sequence()
    assert rt.resolve_stop_index(system, seq) == system.num_surfaces() - 1


def test_resolve_stop_uses_explicit_override(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence(stop_surface=1)
    assert rt.resolve_stop_index(system, seq) == 1


# ---------------------------------------------------------------------------
# Pupil sampling
# ---------------------------------------------------------------------------


def test_single_ray_samples_returns_one_origin():
    d = Distribution(type=DistributionType.SINGLE_RAY, ray_count=1)
    assert rt.pupil_samples(d) == [(0.0, 0.0)]


@pytest.mark.parametrize(
    "dtype", [
        DistributionType.Y_FAN, DistributionType.X_FAN, DistributionType.XY_FAN,
        DistributionType.RING, DistributionType.RANDOM,
    ]
)
def test_distribution_samples_are_within_unit_disk(dtype):
    d = Distribution(type=dtype, ray_count=7)
    samples = rt.pupil_samples(d)
    assert len(samples) > 0
    for u, v in samples:
        assert u * u + v * v <= 1.0 + 1e-6


def test_random_samples_are_deterministic():
    d1 = Distribution(type=DistributionType.RANDOM, ray_count=16)
    d2 = Distribution(type=DistributionType.RANDOM, ray_count=16)
    assert rt.pupil_samples(d1) == rt.pupil_samples(d2)


# ---------------------------------------------------------------------------
# Field → angle mapping
# ---------------------------------------------------------------------------


def test_angle_field_type_maps_degrees_to_radians(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence(field_type=FieldType.ANGLE)
    fld = Field(tilt_x_deg=0.0, tilt_y_deg=10.0)
    ax, ay = rt._angle_from_field(system, seq, fld)
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(math.radians(10.0))


# ---------------------------------------------------------------------------
# build_ray_bundles end-to-end
# ---------------------------------------------------------------------------


def test_build_ray_bundles_empty_when_not_ready():
    system = ghostlight.OpticalSystem()
    setup = SystemSetup()
    assert rt.build_ray_bundles(system, setup) == []


def test_build_ray_bundles_default_setup_yields_one_bundle_per_field(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    bundles = rt.build_ray_bundles(system, setup)
    assert len(bundles) == len(setup.sequences[0].source.fields)


def test_build_ray_bundles_uses_all_wavelengths(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    setup.sequences[0].source.distribution = Distribution(
        type=DistributionType.SINGLE_RAY, ray_count=1
    )
    setup.sequences[0].source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(486.13), Wavelength(587.56), Wavelength(656.27)],
        primary_index=1,
    )
    bundles = rt.build_ray_bundles(system, setup)
    assert bundles, "expected at least one bundle"
    # Every bundle: 1 sample × 3 wavelengths
    for b in bundles:
        assert len(b.paths) == len(b.wavelengths_nm)
        assert sorted(set(b.wavelengths_nm)) == [486.13, 587.56, 656.27]


def test_build_ray_bundles_field_type_none_skips_sequence(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    setup.sequences[0].field_type = FieldType.NONE
    assert rt.build_ray_bundles(system, setup) == []


def test_build_ray_bundles_point_source_origin_is_shared(sample_lens_path):
    """Point sources collapse all origins of a field to one location."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    seq = setup.sequences[0]
    seq.source.type = SourceType.POINT_SOURCE
    seq.source.distribution = Distribution(type=DistributionType.XY_FAN, ray_count=3)
    seq.source.fields = [Field("Axial", 0.0, 0.0)]
    seq.source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(587.56)], primary_index=0
    )
    bundles = rt.build_ray_bundles(system, setup)
    assert bundles
    origins = bundles[0].origins
    assert origins, "expected origins for point source bundle"
    # Every kept ray came from the same point source location.
    first = (origins[0].x, origins[0].y, origins[0].z)
    for o in origins[1:]:
        assert (o.x, o.y, o.z) == first


def test_build_ray_bundles_plane_wf_origins_spread_across_pupil(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    seq = setup.sequences[0]
    seq.source.type = SourceType.PLANE_WF
    seq.source.distribution = Distribution(type=DistributionType.XY_FAN, ray_count=3)
    seq.source.fields = [Field("Axial", 0.0, 0.0)]
    seq.source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(587.56)], primary_index=0
    )
    bundles = rt.build_ray_bundles(system, setup)
    assert bundles
    origins = bundles[0].origins
    xs = {round(o.x, 6) for o in origins}
    ys = {round(o.y, 6) for o in origins}
    assert len(xs) >= 2 and len(ys) >= 2  # spread across both axes


def test_marginal_pupil_search_returns_positive_radii(sample_lens_path):
    """The marginal-ray search must find a non-zero unvignetted pupil for
    a real lens on-axis."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence()
    fld = Field("Axial", 0.0, 0.0)
    source = Source()
    rx, ry, _center = rt._marginal_pupil_radii(system, seq, fld, source, 587.56)
    assert rx > 0.0
    assert ry > 0.0


def test_marginal_pupil_search_respects_stop_size(sample_lens_path):
    """Shrinking the stop's semi_aperture must shrink the unvignetted pupil.

    This is the core 'stop drives the distribution' contract: with a tighter
    stop, the marginal-ray search has to produce a tighter pupil.
    """
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence()
    fld = Field("Axial", 0.0, 0.0)
    source = Source()
    stop_idx = rt.resolve_stop_index(system, seq)
    assert stop_idx is not None
    original = float(system.surfaces[stop_idx].semi_aperture)
    if original <= 0.0:
        pytest.skip("stop has zero semi_aperture")

    rx_full, _, _ = rt._marginal_pupil_radii(system, seq, fld, source, 587.56)
    try:
        system.surfaces[stop_idx].semi_aperture = original * 0.5
        rx_half, _, _ = rt._marginal_pupil_radii(system, seq, fld, source, 587.56)
    finally:
        system.surfaces[stop_idx].semi_aperture = original
    assert rx_half < rx_full


def test_aperture_type_none_skips_marginal_search(sample_lens_path):
    """``ApertureType.NONE`` bypasses the stop-based marginal search and uses
    ``source.aperture_radius`` directly. Setting a value clearly different
    from the lens's natural unvignetted pupil proves the search isn't being
    consulted at all."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence(aperture_type=ApertureType.NONE)
    fld = Field("Axial", 0.0, 0.0)
    source = Source(aperture_radius=0.123)
    rx, ry, (cx, cy) = rt._pupil_radii_for_field(system, seq, fld, source, 587.56)
    assert rx == pytest.approx(0.123)
    assert ry == pytest.approx(0.123)
    # The optical-axis launch is used as the chief ray.
    assert cx == 0.0
    assert cy == 0.0


def test_aperture_type_none_does_not_discard_rays_at_stop(sample_lens_path):
    """``NONE`` traces independently of the stop, so the build loop must
    keep every traced ray rather than filtering out the ones that would
    be vignetted at the stop."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    seq = setup.sequences[0]
    seq.aperture_type = ApertureType.NONE
    # A radius much larger than the stop's semi_aperture; under FROM_STOP
    # the rays at the rim would be discarded.
    seq.source.aperture_radius = 100.0
    seq.source.distribution = Distribution(type=DistributionType.XY_FAN, ray_count=5)
    seq.source.fields = [Field("Axial", 0.0, 0.0)]
    seq.source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(587.56)], primary_index=0
    )
    bundles = rt.build_ray_bundles(system, setup)
    assert bundles
    # The XY-Fan distribution emits 2N-1 unique samples. With ray_count=5
    # that's 9 samples × 1 wavelength = 9 rays. Without the stop filter
    # we expect all of them to survive.
    assert len(bundles[0].paths) == 9


def test_off_axis_field_produces_a_bundle(sample_lens_path):
    """Off-axis fields used to vanish — the on-axis marginal search probed
    a region where every tilted ray missed the stop, so the binary search
    collapsed to zero.  With chief-ray centring the bundle must survive.
    """
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence(field_type=FieldType.ANGLE)
    fld = Field("Off-axis", 0.0, 10.0)
    source = Source()
    rx, ry, (cx, cy) = rt._pupil_radii_for_field(system, seq, fld, source, 587.56)
    assert rx > 0.0
    assert ry > 0.0
    # The chief ray for a Y-tilted field has to land off the optical axis
    # at the launch plane — a centred bundle would mean the centring code
    # was a no-op.
    assert abs(cy) > 1e-3


def test_chief_ray_lands_at_stop_center(sample_lens_path):
    """The found chief-ray launch point must trace to (≈0, ≈0) at the stop.

    This is the central correctness contract of the centring routine: the
    bundle is centred on a ray that actually passes through the stop's
    centre, not on the optical-axis launch.
    """
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    seq = Sequence(field_type=FieldType.ANGLE)
    fld = Field("Off-axis", 0.0, 10.0)
    source = Source()
    stop_idx = rt.resolve_stop_index(system, seq)
    assert stop_idx is not None
    upper = rt._search_upper_bound(system, seq, source)
    center = rt._find_chief_ray_center(
        system, seq, fld, source, 587.56, stop_idx, upper
    )
    assert center is not None
    cx, cy = center
    hit = rt._stop_hit_position(system, seq, fld, source, cx, cy, 587.56, stop_idx)
    assert hit is not None
    # Stop semi-aperture for the sample doublet is on the order of mm; a
    # successful chief-ray fit lands within a small fraction of that.
    sx, sy = hit
    stop_r = float(system.surfaces[stop_idx].semi_aperture)
    assert math.hypot(sx, sy) < max(0.05, stop_r * 0.05)


def test_multiple_off_axis_fields_each_produce_their_own_bundle(sample_lens_path):
    """Three fields (0°, 10°, 20°) must each yield a bundle whose centre
    lands at a distinct, non-axial chief-ray position."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    setup.sequences[0].source.distribution = Distribution(
        type=DistributionType.SINGLE_RAY, ray_count=1
    )
    setup.sequences[0].source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(587.56)], primary_index=0
    )
    setup.sequences[0].source.fields = [
        Field("Axial",   0.0,  0.0),
        Field("Mid",     0.0, 10.0),
        Field("Max FOV", 0.0, 20.0),
    ]
    bundles = rt.build_ray_bundles(system, setup)
    # One bundle per field, in field order.
    assert len(bundles) == 3
    # Axial bundle's single ray launches near (0, 0); off-axis bundles
    # launch at distinct non-zero y offsets.
    ys = [b.origins[0].y for b in bundles]
    assert abs(ys[0]) < 0.1
    assert ys[1] != pytest.approx(ys[0], abs=0.1)
    assert ys[2] != pytest.approx(ys[1], abs=0.1)


def test_all_traced_rays_clear_the_stop(sample_lens_path):
    """Bundles produced under ``FROM_STOP`` must contain only rays that pass
    the stop cleanly — that's the user-visible promise of 'use the stop to
    figure out the distribution'. Rays may still vignette at later
    surfaces; the viewport shows that, by design."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    setup = SystemSetup()
    setup.sequences[0].source.distribution = Distribution(
        type=DistributionType.XY_FAN, ray_count=5
    )
    setup.sequences[0].source.wavelengths = WavelengthContainer(
        wavelengths=[Wavelength(587.56)], primary_index=0
    )
    bundles = rt.build_ray_bundles(system, setup)
    assert bundles
    seq = setup.sequences[0]
    stop_idx = rt.resolve_stop_index(system, seq)
    assert stop_idx is not None
    for bundle in bundles:
        for path in bundle.paths:
            stop_event = next(
                (ev for ev in path.events
                 if int(getattr(ev, "surface_index", -1)) == stop_idx),
                None,
            )
            assert stop_event is not None, "ray failed before reaching stop"
            assert stop_event.status == ghostlight.TraceStatus.OK, \
                "ray was vignetted at the stop"


# ---------------------------------------------------------------------------
# ViewportPanelBody integration: rays default on, react to all 3 signals
# ---------------------------------------------------------------------------


def test_viewport_panel_pushes_rays_on_load(qapp, sample_lens_path, monkeypatch):
    from ghostlight_viewport import LensViewport
    from ghostlight_designer.viewport_panel import ViewportPanelBody

    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    pushed: list = []
    monkeypatch.setattr(
        LensViewport, "set_trace_results",
        lambda self, bundles: pushed.append(("set", len(bundles))),
    )
    monkeypatch.setattr(
        LensViewport, "clear_trace_results",
        lambda self: pushed.append(("clear",)),
    )

    project = Project()
    body = ViewportPanelBody(project)
    try:
        pushed.clear()
        project.load(str(sample_lens_path))
        # At least one push (set or clear) occurred; default setup has fields
        # so a real lens should produce bundles.
        assert any(p[0] == "set" and p[1] > 0 for p in pushed)
    finally:
        body.deleteLater()


def test_viewport_panel_retraces_on_setup_change(qapp, sample_lens_path, monkeypatch):
    from ghostlight_viewport import LensViewport
    from ghostlight_designer.viewport_panel import ViewportPanelBody

    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    calls: list = []
    monkeypatch.setattr(
        LensViewport, "set_trace_results",
        lambda self, bundles: calls.append(len(bundles)),
    )
    monkeypatch.setattr(
        LensViewport, "clear_trace_results", lambda self: None,
    )

    project = Project()
    project.load(str(sample_lens_path))
    body = ViewportPanelBody(project)
    try:
        calls.clear()
        # Mutate the setup and emit the signal as the model would.
        project.system_setup.sequences[0].source.distribution.ray_count = 5
        project.mark_system_setup_modified()
        assert calls, "expected a retrace on systemSetupChanged"
    finally:
        body.deleteLater()


def test_viewport_panel_toggle_hides_and_restores_rays(
    qapp, sample_lens_path, monkeypatch
):
    from ghostlight_viewport import LensViewport
    from ghostlight_designer.viewport_panel import ViewportPanelBody

    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    events: list = []
    monkeypatch.setattr(
        LensViewport, "set_trace_results",
        lambda self, bundles: events.append(("set", len(bundles))),
    )
    monkeypatch.setattr(
        LensViewport, "clear_trace_results",
        lambda self: events.append(("clear",)),
    )

    project = Project()
    project.load(str(sample_lens_path))
    body = ViewportPanelBody(project)
    try:
        events.clear()
        body.set_show_rays(False)
        assert ("clear",) in events
        events.clear()
        body.set_show_rays(True)
        # Coming back on should push fresh bundles (set, not clear).
        assert any(e[0] == "set" for e in events)
    finally:
        body.deleteLater()


def test_viewport_sensor_mirrors_system_setup_dimensions(qapp, monkeypatch):
    """Picking a sensor preset (or typing custom mm) should immediately
    push a SensorSpec matching the System Setup values into the viewport."""
    from ghostlight_viewport import LensViewport, SensorSpec
    from ghostlight_designer.viewport_panel import ViewportPanelBody

    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_trace_results", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "clear_trace_results", lambda *a, **k: None)

    pushed: list = []
    monkeypatch.setattr(
        LensViewport, "set_sensor",
        lambda self, spec: pushed.append((spec.half_w, spec.half_h)),
    )

    project = Project()
    # Initial push from __init__: matches the default Super 35 sensor.
    body = ViewportPanelBody(project)
    try:
        assert pushed, "expected an initial sensor push"
        s = project.system_setup.sensor
        assert pushed[-1] == (s.width_mm / 2.0, s.height_mm / 2.0)

        # Change the sensor to Full Frame and verify the next push matches.
        project.system_setup.sensor.width_mm = 36.0
        project.system_setup.sensor.height_mm = 24.0
        project.mark_system_setup_modified()
        assert pushed[-1] == (18.0, 12.0)
    finally:
        body.deleteLater()


def test_view_menu_includes_rays_toggle(qapp, sample_lens_path, monkeypatch):
    from ghostlight_viewport import LensViewport
    from ghostlight_designer.viewport_panel import ViewportPanelBody
    from ghostlight_designer.viewport_panel.menus import build_menus

    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_trace_results", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "clear_trace_results", lambda *a, **k: None)

    project = Project()
    body = ViewportPanelBody(project)
    try:
        menus = build_menus(body, project)
        view_menu = menus[0]
        titles = [a.text() for a in view_menu.actions() if not a.isSeparator()]
        assert any("Ray" in t for t in titles)
        ray_action = next(a for a in view_menu.actions() if "Ray" in a.text())
        assert ray_action.isCheckable()
        assert ray_action.isChecked() is True  # default ON per the spec
    finally:
        body.deleteLater()
