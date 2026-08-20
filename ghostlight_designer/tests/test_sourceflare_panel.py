"""Unit tests for the sourceflare panel body — exercises the parts that
don't actually launch a GPU render (shape sampling, widget gating,
serial/pending bookkeeping, overlay math).
"""
from __future__ import annotations

import gc
import math

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from ghostlight_designer.project import Project
from ghostlight_designer.sourceflare_panel.body import (
    SHAPE_CIRCLE,
    SHAPE_POINT,
    SHAPE_POLYGON,
    SHAPE_RECT,
    SHAPE_SQUARE,
    SourceFlarePanelBody,
    build_shape_offsets,
)
from ghostlight_designer.system_setup_data import SensorSettings

from _corpus import EXAMPLE_DOUBLET


def _make_body(qapp, isolated_settings):
    project = Project()
    body = SourceFlarePanelBody(project, isolated_settings)
    return project, body


def _destroy(body) -> None:
    """Destroy a body deterministically inside the owning test.

    The body's spinboxes carry scrubber plumbing (hidden QTreeView +
    adapter model); letting bodies pile up in the deleteLater queue and
    die mid-way through a *later* test's event processing corrupts the
    heap on Windows/PySide6. Flushing DeferredDelete here keeps each
    teardown inside its own test.
    """
    body.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    # Force a Python GC so the scrubber wrappers (hidden QTreeView + adapter
    # model per spinbox) release their C++ objects now, instead of piling up to
    # interpreter shutdown where the cumulative teardown corrupts the heap on
    # Windows/PySide6 (0xc0000374). See feedback_scrubber_teardown.
    gc.collect()


def _example_lens_path():
    return EXAMPLE_DOUBLET


# ---------------------------------------------------------------------------
# Shape sampling
# ---------------------------------------------------------------------------

def test_point_shape_is_single_full_weight_sample():
    pts = build_shape_offsets(SHAPE_POINT, 5.0, 5.0, 64)
    assert pts.shape == (1, 3)
    assert pts[0, 0] == 0.0 and pts[0, 1] == 0.0 and pts[0, 2] == 1.0


@pytest.mark.parametrize(
    "shape", [SHAPE_CIRCLE, SHAPE_RECT, SHAPE_SQUARE, SHAPE_POLYGON]
)
def test_extended_shapes_average_to_unit_weight(shape):
    pts = build_shape_offsets(shape, 2.0, 1.0, 48)
    assert pts.shape == (48, 3)
    assert pts[:, 2].sum() == pytest.approx(1.0, abs=1e-5)


def test_offsets_bounded_by_angular_size():
    """Samples must stay inside the declared angular extents."""
    pts = build_shape_offsets(SHAPE_RECT, 4.0, 2.0, 128)
    assert np.abs(pts[:, 0]).max() <= math.radians(2.0) + 1e-9
    assert np.abs(pts[:, 1]).max() <= math.radians(1.0) + 1e-9

    pts = build_shape_offsets(SHAPE_CIRCLE, 3.0, 3.0, 128)
    assert np.hypot(pts[:, 0], pts[:, 1]).max() <= math.radians(1.5) + 1e-9

    # Polygon: bounded by circumradius = half the across-corners W.
    pts = build_shape_offsets(SHAPE_POLYGON, 4.0, 0.0, 128, n_sides=6)
    assert np.hypot(pts[:, 0], pts[:, 1]).max() <= math.radians(2.0) + 1e-9


def test_rotation_rotates_offsets():
    """A rotated rect's offsets differ from the unrotated ones but keep
    the same per-sample radius (rigid rotation)."""
    base = build_shape_offsets(SHAPE_RECT, 4.0, 1.0, 64, rotation_deg=0.0)
    rot = build_shape_offsets(SHAPE_RECT, 4.0, 1.0, 64, rotation_deg=30.0)
    assert not np.allclose(base[:, :2], rot[:, :2])
    np.testing.assert_allclose(np.hypot(base[:, 0], base[:, 1]),
                               np.hypot(rot[:, 0], rot[:, 1]), atol=1e-7)


def test_circle_ignores_rotation():
    base = build_shape_offsets(SHAPE_CIRCLE, 3.0, 3.0, 64, rotation_deg=0.0)
    rot = build_shape_offsets(SHAPE_CIRCLE, 3.0, 3.0, 64, rotation_deg=45.0)
    np.testing.assert_array_equal(base, rot)


def test_square_ignores_height():
    a = build_shape_offsets(SHAPE_SQUARE, 2.0, 99.0, 32)
    b = build_shape_offsets(SHAPE_SQUARE, 2.0, 0.5, 32)
    np.testing.assert_array_equal(a, b)


def test_samples_clamped():
    assert build_shape_offsets(SHAPE_CIRCLE, 1.0, 1.0, 0).shape[0] == 1
    assert build_shape_offsets(SHAPE_CIRCLE, 1.0, 1.0, 10**9).shape[0] == 1024


# ---------------------------------------------------------------------------
# Widget gating per shape
# ---------------------------------------------------------------------------

def test_shape_widgets_enable_state(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        # (W, H, Rot, Sides, Samples) enabled flags per shape.
        cases = {
            SHAPE_POINT: (False, False, False, False, False),
            SHAPE_CIRCLE: (True, False, False, False, True),
            SHAPE_RECT: (True, True, True, False, True),
            SHAPE_SQUARE: (True, False, True, False, True),
            SHAPE_POLYGON: (True, False, True, True, True),
        }
        for shape, (w_on, h_on, rot_on, sides_on, samples_on) in cases.items():
            body._shape_combo.setCurrentIndex(body._shape_combo.findData(shape))
            assert body._spin_size_w.isEnabled() is w_on, shape
            assert body._spin_size_h.isEnabled() is h_on, shape
            assert body._spin_rot.isEnabled() is rot_on, shape
            assert body._spin_sides.isEnabled() is sides_on, shape
            assert body._spin_samples.isEnabled() is samples_on, shape
    finally:
        _destroy(body)


def test_shape_label_formats(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._shape = SHAPE_POINT
        assert body._shape_label() == "point"
        body._shape = SHAPE_CIRCLE
        body._size_w_deg = 0.53
        assert body._shape_label() == "circle 0.53°"
        body._shape = SHAPE_RECT
        body._size_w_deg, body._size_h_deg = 3.0, 1.5
        body._rotation_deg = 0.0
        assert body._shape_label() == "rect 3.00×1.50°"
        body._rotation_deg = 45.0
        assert body._shape_label() == "rect 3.00×1.50° @45°"
        body._shape = SHAPE_POLYGON
        body._n_sides = 6
        body._size_w_deg = 2.0
        assert body._shape_label() == "6-gon 2.00° @45°"
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Overlay math
# ---------------------------------------------------------------------------

def test_overlay_centered_source_is_symmetric(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        # (angle_x, angle_y, screen_x, screen_y, J, half_w, half_h) — the
        # snapshot the panel takes from the renderer's source-map solve. J is
        # d(landing mm)/d(angle rad), row-major, and a plain diagonal here
        # stands in for a lens with no cross-coupling.
        body._overlay_map = (0.0, 0.0, 0.5, 0.5,
                             (24.0, 0.0, 0.0, 18.0), 12.0, 9.0)
        body._sx = body._sy = 0.5

        # Circle → many-point outline whose centroid sits on the source.
        body._shape = SHAPE_CIRCLE
        body._size_w_deg = 2.0
        body._update_overlay()
        pts = body._canvas._outline
        assert pts is not None and len(pts) == 48
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        assert cx == pytest.approx(0.5, abs=1e-3)
        assert cy == pytest.approx(0.5, abs=1e-3)

        # Rect → 4-corner outline.
        body._shape = SHAPE_RECT
        body._size_h_deg = 1.0
        body._update_overlay()
        assert len(body._canvas._outline) == 4

        # Polygon → n_sides-corner outline.
        body._shape = SHAPE_POLYGON
        body._n_sides = 5
        body._update_overlay()
        assert len(body._canvas._outline) == 5

        # Point → nothing.
        body._shape = SHAPE_POINT
        body._update_overlay()
        assert body._canvas._outline is None
    finally:
        _destroy(body)


def test_overlay_rotation_changes_outline(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._overlay_map = (0.0, 0.0, 0.5, 0.5,
                             (24.0, 0.0, 0.0, 24.0), 12.0, 12.0)
        body._sx = body._sy = 0.5
        body._shape = SHAPE_RECT
        body._size_w_deg, body._size_h_deg = 4.0, 1.0
        body._rotation_deg = 0.0
        body._update_overlay()
        flat = list(body._canvas._outline)
        body._rotation_deg = 45.0
        body._update_overlay()
        rotated = list(body._canvas._outline)
        assert any(abs(a[0] - b[0]) > 1e-4 or abs(a[1] - b[1]) > 1e-4
                   for a, b in zip(flat, rotated))
    finally:
        _destroy(body)


def test_overlay_hidden_before_first_dispatch(qapp, isolated_settings):
    """Without an angle→screen snapshot there is nothing to draw."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        assert body._overlay_map is None
        body._update_overlay()
        assert body._canvas._outline is None
    finally:
        _destroy(body)


def test_dispatch_builds_the_overlay_snapshot(qapp, isolated_settings, monkeypatch):
    """A dispatch must actually POPULATE _overlay_map from the solve.

    The tests above inject the snapshot to exercise _update_overlay's math,
    which leaves the code that BUILDS it untested — and that code is wrapped
    in a log-and-continue guard, so anything wrong in it (a bad name, a
    changed binding signature) degrades to "no outline" without failing a
    single test.  It shipped exactly that way once.  Assert the guard did not
    fire, and that the snapshot means what _update_overlay reads it as.
    """
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        body._is_active = True

        class FakeThread:
            def __init__(self, *_a, **_k):
                pass

            def start(self):
                pass

        monkeypatch.setattr(
            "ghostlight_designer.sourceflare_panel.body.threading.Thread", FakeThread,
        )
        body.request_render()
        qapp.processEvents()

        assert body._overlay_map is not None, (
            "dispatch left the overlay snapshot unset — the build guard fired"
        )
        ax, ay, sx, sy, jac, half_w, half_h = body._overlay_map
        assert len(jac) == 4
        assert all(math.isfinite(v) for v in (ax, ay, sx, sy, half_w, half_h))
        assert all(math.isfinite(v) for v in jac)

        # The snapshot's base screen position IS the requested source: that is
        # what welds the outline to the marker, and it is the alignment the
        # traced map exists to get right.
        assert sx == pytest.approx(body._sx, abs=2e-3)
        assert sy == pytest.approx(body._sy, abs=2e-3)

        # Non-degenerate, and oriented +x → +x: _update_overlay divides by
        # half_w/half_h and would silently mirror or collapse the outline.
        assert half_w > 0.0 and half_h > 0.0
        assert jac[0] > 0.0 and jac[3] > 0.0
        assert abs(jac[0] * jac[3] - jac[1] * jac[2]) > 1e-6
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Dispatch bookkeeping (no GPU)
# ---------------------------------------------------------------------------

def test_render_height_tracks_sensor_aspect(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        project.system_setup.sensor = SensorSettings(
            width_mm=16.0, height_mm=9.0, preset_name="Custom",
        )
        width, height, half_w, half_h = body._resolve_render_dims()
        assert height == round(width * 9.0 / 16.0)
        assert half_w == pytest.approx(8.0)
        assert half_h == pytest.approx(4.5)
    finally:
        _destroy(body)


def test_hidden_panel_does_not_schedule_render(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        assert body._lens_eligible()
        for _ in range(30):
            project.mark_modified()
        qapp.processEvents()
        assert body._is_active is False
        assert body._busy is False
        assert body._dirty_pending is True
    finally:
        _destroy(body)


def test_burst_collapses_to_one_launch_and_bumps_serial(
    qapp, isolated_settings, monkeypatch
):
    """A scrub burst launches one worker; every request bumps the serial
    so an in-flight progressive pass knows to abort between chunks."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        body._is_active = True
        serial_before = body._serial

        launch_count = {"n": 0}

        class FakeThread:
            def __init__(self, *_a, **_k):
                launch_count["n"] += 1

            def start(self):
                pass

        monkeypatch.setattr(
            "ghostlight_designer.sourceflare_panel.body.threading.Thread",
            FakeThread,
        )

        for _ in range(50):
            project.mark_modified()
        qapp.processEvents()

        assert launch_count["n"] == 1, f"expected 1 launch, got {launch_count['n']}"
        assert body._busy is True
        assert body._pending is True
        assert body._serial - serial_before == 50
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Matte box (inline source control)
# ---------------------------------------------------------------------------

def test_matte_box_off_by_default_and_toggles_enable_state(qapp, isolated_settings):
    """The matte box ships off (byte-identical render) and its blade /
    distance spinboxes are only editable once enabled."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        assert body._matte.enabled is False
        assert body._spin_matte_w.isEnabled() is False
        assert body._spin_matte_h.isEnabled() is False
        assert body._spin_matte_z.isEnabled() is False

        body._matte_check.setChecked(True)
        assert body._matte.enabled is True
        assert body._spin_matte_w.isEnabled() is True
        assert body._spin_matte_h.isEnabled() is True
        assert body._spin_matte_z.isEnabled() is True
    finally:
        _destroy(body)


def test_matte_controls_hidden_by_default_and_view_menu_toggle(qapp, isolated_settings):
    """The inline matte-box controls start hidden (per-panel clutter opt-in)
    and the View menu's visibility toggle shows/hides them without touching
    the matte box's own enabled state."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        assert body.matte_controls_visible is False
        assert body._matte_container.isHidden() is True

        body.set_matte_controls_visible(True)
        assert body.matte_controls_visible is True
        assert body._matte.enabled is False

        body.set_matte_controls_visible(False)
        assert body.matte_controls_visible is False
    finally:
        _destroy(body)


def test_matte_box_spinboxes_feed_state(qapp, isolated_settings):
    """Editing the width / height / distance spins updates the immutable
    matte-box snapshot the renderer reads."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._matte_check.setChecked(True)
        body._spin_matte_w.setValue(42.0)
        body._spin_matte_h.setValue(21.0)
        body._spin_matte_z.setValue(75.0)
        assert body._matte.half_w_mm == pytest.approx(42.0)
        assert body._matte.half_h_mm == pytest.approx(21.0)
        assert body._matte.z_front_mm == pytest.approx(75.0)
    finally:
        _destroy(body)


def test_matte_box_snapshot_reaches_worker(qapp, isolated_settings, monkeypatch):
    """The matte-box state is snapshotted onto the dispatch thread's args
    (position 10 of _worker's signature), so the render sees it."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))

        captured = {}

        class FakeThread:
            def __init__(self, *_a, **kw):
                captured["args"] = kw.get("args")

            def start(self):
                pass

        monkeypatch.setattr(
            "ghostlight_designer.sourceflare_panel.body.threading.Thread",
            FakeThread,
        )

        body._is_active = True
        # Configure the matte box with launches suppressed, then dispatch once
        # so exactly one FakeThread captures the fully-set matte snapshot.
        body._auto_render = False
        body._matte_check.setChecked(True)
        body._spin_matte_w.setValue(40.0)
        body._spin_matte_h.setValue(25.0)
        body._spin_matte_z.setValue(80.0)
        body._auto_render = True
        body.request_render()

        matte = captured["args"][10]
        assert matte.enabled is True
        assert matte.half_w_mm == pytest.approx(40.0)
        assert matte.half_h_mm == pytest.approx(25.0)
        assert matte.z_front_mm == pytest.approx(80.0)
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Panel defaults (Mid preset, desqueeze, auto-expose on lens load)
# ---------------------------------------------------------------------------

def test_mid_preset_sits_between_draft_and_high_with_the_extra_layers_off():
    from ghostlight_designer.render_common import (
        DRAFT_PRESET, HIGH_PRESET, MID_PRESET,
    )

    assert DRAFT_PRESET.width_px <= MID_PRESET.width_px <= HIGH_PRESET.width_px
    assert DRAFT_PRESET.ray_grid < MID_PRESET.ray_grid < HIGH_PRESET.ray_grid
    assert (DRAFT_PRESET.spectral_samples < MID_PRESET.spectral_samples
            < HIGH_PRESET.spectral_samples)
    # The point of Mid: sampling only, no extra layers and no HURB.
    assert MID_PRESET.starburst is False
    assert MID_PRESET.veil is False
    assert MID_PRESET.gate is False
    assert MID_PRESET.hurb is False
    assert MID_PRESET.clamp() == MID_PRESET


def test_panel_defaults_to_mid_preset_and_desqueeze_on(qapp, isolated_settings):
    from ghostlight_designer.render_common import MID_PRESET

    _project, body = _make_body(qapp, isolated_settings)
    try:
        assert body.settings == MID_PRESET
        assert body.desqueeze is True
        # Source starts off-axis (upper right), and the widgets agree.
        assert (body._sx, body._sy) == (0.75, 0.25)
        assert body._spin_x.value() == pytest.approx(0.75)
        assert body._spin_y.value() == pytest.approx(0.25)
        body.recenter_source()  # "Recenter" still means centre
        assert (body._sx, body._sy) == (0.5, 0.5)
        # Menu action exists and switches back to Mid from another preset.
        body.apply_preset_high()
        assert body.settings != MID_PRESET
        body.apply_preset_mid()
        assert body.settings == MID_PRESET
    finally:
        _destroy(body)


def test_high_carries_hurb_but_none_of_the_extra_layers():
    """HURB refines the ghost pass itself rather than compositing a layer over
    it, so it belongs to the plain quality ladder; the three whole-frame extras
    do not."""
    from ghostlight_designer.render_common import HIGH_PRESET

    assert HIGH_PRESET.hurb is True
    assert HIGH_PRESET.starburst is False
    assert HIGH_PRESET.veil is False
    assert HIGH_PRESET.gate is False
    assert HIGH_PRESET.clamp() == HIGH_PRESET


def test_high_plus_is_high_plus_the_three_extra_layers():
    """High+ must never be a worse ghost render than High — same sampling, same
    HURB, with only the extra layers added on top."""
    from dataclasses import replace

    from ghostlight_designer.render_common import HIGH_PLUS_PRESET, HIGH_PRESET

    assert HIGH_PLUS_PRESET.starburst is True
    assert HIGH_PLUS_PRESET.veil is True
    assert HIGH_PLUS_PRESET.gate is True
    assert HIGH_PLUS_PRESET.hurb is True
    assert HIGH_PLUS_PRESET.clamp() == HIGH_PLUS_PRESET
    # Sampling is inherited verbatim.
    assert HIGH_PLUS_PRESET.width_px == HIGH_PRESET.width_px
    assert HIGH_PLUS_PRESET.ray_grid == HIGH_PRESET.ray_grid
    assert HIGH_PLUS_PRESET.spectral_samples == HIGH_PRESET.spectral_samples
    # ...and the layer switches are the ONLY difference between the two, so a
    # future sampling tweak to High can't silently skip High+.
    assert replace(
        HIGH_PLUS_PRESET,
        starburst=HIGH_PRESET.starburst,
        starburst_engine=HIGH_PRESET.starburst_engine,
        starburst_grid=HIGH_PRESET.starburst_grid,
        veil=HIGH_PRESET.veil,
        gate=HIGH_PRESET.gate,
    ) == HIGH_PRESET


def test_apply_preset_high_plus_swaps_settings(qapp, isolated_settings):
    from ghostlight_designer.render_common import HIGH_PLUS_PRESET, HIGH_PRESET

    _project, body = _make_body(qapp, isolated_settings)
    try:
        body.apply_preset_high()
        assert body.settings == HIGH_PRESET
        body.apply_preset_high_plus()
        assert body.settings == HIGH_PLUS_PRESET
    finally:
        _destroy(body)


def test_loading_a_lens_auto_exposes_the_first_completed_frame(
    qapp, isolated_settings
):
    """A newly loaded lens resets exposure to 0 and re-meters itself once
    its first render finishes — the user never lands on a black frame."""
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        assert body._auto_expose_pending is True
        body._exposure_stops = 0.0

        # A bright frame, pushed straight through the result queue so the
        # test never touches the GPU. Partial frames must NOT meter (the
        # worker bakes the dispatch-time stops into each chunk's image).
        hwc = np.full((4, 4, 3), 0.001, dtype=np.float32)
        body._results.put((body._epoch, None, hwc, hwc, 1, 2, False))
        body._poll_results()
        assert body._exposure_stops == 0.0
        assert body._auto_expose_pending is True

        body._results.put((body._epoch, None, hwc, hwc, 2, 2, True))
        body._poll_results()
        assert body._exposure_stops > 0.0
        assert body._auto_expose_pending is False

        # One-shot: a later render keeps the metered exposure.
        stops = body._exposure_stops
        body._results.put((body._epoch, None, hwc * 100.0, hwc * 100.0, 2, 2, True))
        body._poll_results()
        assert body._exposure_stops == stops
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Exposure snapping — half-stop grid with a +90 st cap (this panel only)
# ---------------------------------------------------------------------------

def _drive_spin(spin, text):
    """Type ``text`` wholesale and interpret it, as Enter would."""
    spin.lineEdit().setText(text)
    spin.interpretText()
    return spin.value()


def test_exposure_dialog_snaps_to_half_stops_with_a_90_stop_cap(
    qapp, isolated_settings
):
    """Typed numbers, typed calculations, and programmatic writes (the
    Ctrl+MMB scrubber's path) all land on the 0.5 st grid, and the range
    tops out at +90 st."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._exposure_stops = 1.3
        body.open_exposure_dialog()
        spin = body._cc_dialog._spin
        assert spin.maximum() == pytest.approx(90.0)
        assert spin.singleStep() == pytest.approx(0.5)
        # The opening value itself is snapped onto the grid.
        assert spin.value() == pytest.approx(1.5)
        # Typed plain numbers round on commit...
        assert _drive_spin(spin, "3.8") == pytest.approx(4.0)
        # ...as do typed calculations...
        assert _drive_spin(spin, "1.2*2") == pytest.approx(2.5)
        # ...and setValue, which is how the scrubber writes.
        spin.setValue(7.3)
        assert spin.value() == pytest.approx(7.5)
        # The snapped edits flowed back into the panel state.
        assert body._exposure_stops == pytest.approx(7.5)
    finally:
        _destroy(body)


def test_auto_expose_rounds_onto_the_half_stop_grid(
    qapp, isolated_settings, monkeypatch
):
    from ghostlight_designer.sourceflare_panel import body as body_mod

    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._latest_hwc = np.full((4, 4, 3), 0.5, dtype=np.float32)
        monkeypatch.setattr(body_mod.vt, "meter_auto_stops", lambda _img: 7.3)
        body.auto_expose()
        assert body._exposure_stops == pytest.approx(7.5)
        # The cap holds when the meter overshoots it.
        monkeypatch.setattr(body_mod.vt, "meter_auto_stops", lambda _img: 123.4)
        body.auto_expose()
        assert body._exposure_stops == pytest.approx(90.0)
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Value scrubber on the panel spinboxes
# ---------------------------------------------------------------------------

def test_scrubber_attached_to_every_spinbox(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        spins = [
            body._spin_x, body._spin_y, body._spin_size_w, body._spin_size_h,
            body._spin_rot, body._spin_sides, body._spin_samples,
            body._spin_matte_w, body._spin_matte_h, body._spin_matte_z,
        ]
        assert len(body._scrubbers) == len(spins)
        # Every scrubber targets a distinct panel spinbox.
        assert {s._spinbox for s in body._scrubbers} == set(spins)
    finally:
        _destroy(body)


def test_ctrl_mmb_opens_scrubber_popup(qapp, isolated_settings):
    """Ctrl+MiddleButton over a panel spinbox opens a ScrubPopup — i.e.
    the scrubber is actually wired, not just constructed."""
    from ghostlight_designer.value_scrubber import ScrubPopup

    _project, body = _make_body(qapp, isolated_settings)
    try:
        trig = next(s for s in body._scrubbers if s._spinbox is body._spin_size_w)
        ev = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(5.0, 5.0), QPointF(5.0, 5.0),
            Qt.MiddleButton, Qt.MiddleButton, Qt.ControlModifier,
        )
        consumed = trig.eventFilter(body._spin_size_w, ev)
        assert consumed is True
        popups = [w for w in QApplication.topLevelWidgets()
                  if isinstance(w, ScrubPopup)]
        assert popups, "Ctrl+MMB should open a ScrubPopup"
    finally:
        for w in QApplication.topLevelWidgets():
            if isinstance(w, ScrubPopup):
                w.close()
        qapp.processEvents()
        _destroy(body)
