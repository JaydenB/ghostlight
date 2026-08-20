"""Tests for the designer-wide ACES 2.0 view transform.

Covers the pure pipeline (known-value ACES 2.0 mid-grey, stops linearity,
metering, processor caching), the AppSettings persistence + broadcast signal,
and the panel re-display-without-render broadcast path.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from ghostlight_designer import viewtransform as vt


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _default_spec() -> vt.ViewTransformSpec:
    display, view = vt.resolve_default_display_view("")
    return vt.ViewTransformSpec(config_key="", display=display, view=view)


def test_default_config_is_aces_2() -> None:
    """The bundled builtin config must be an ACES 2.0 studio config."""
    name = vt.default_config_name()
    assert "aces-v2.0" in name and name.startswith("studio-config")


def test_midgrey_known_value() -> None:
    """ACEScg 0.18 mid-grey through the ACES 2.0 SDR sRGB view.

    Measured on OCIO 2.5.2 / studio-config-v4.0.0_aces-v2.0: 0.3492 (8-bit 89).
    This is the regression anchor — a wrong config or a broken input-space
    resolution would move it.
    """
    disp = vt.apply_view(np.full((1, 1, 3), 0.18, np.float32), 0.0, _default_spec())
    val = float(disp[0, 0, 0])
    assert val == pytest.approx(0.3492, abs=0.005)
    assert int(round(val * 255.0)) == 89


def test_stops_are_a_linear_premultiply() -> None:
    """Exposure in stops is 2**stops applied in linear before the transform,
    so apply(2x, s) == apply(x, s+1) exactly."""
    x = np.array([[[0.05, 0.1, 0.2]]], np.float32)
    a = vt.apply_view(x * 2.0, 0.0, _default_spec())
    b = vt.apply_view(x, 1.0, _default_spec())
    assert np.allclose(a, b, atol=1e-5)


def test_black_and_negatives_clamp_to_zero_display() -> None:
    """Black stays black; small CMF-lobe negatives resolve to display 0
    (not NaN / not negative)."""
    spec = _default_spec()
    black = vt.apply_view(np.zeros((2, 2, 3), np.float32), 0.0, spec)
    assert np.all(black == 0.0)
    neg = vt.apply_view(np.full((2, 2, 3), -0.01, np.float32), 0.0, spec)
    assert np.all(np.isfinite(neg))
    assert np.all(neg <= 0.0 + 1e-6)


def test_apply_view_preserves_shape_and_is_copy() -> None:
    """apply_view returns a fresh (H, W, 3) float32 array and never mutates
    the caller's scene-linear frame."""
    src = np.random.rand(5, 7, 3).astype(np.float32)
    original = src.copy()
    out = vt.apply_view(src, 0.5, _default_spec())
    assert out.shape == (5, 7, 3)
    assert out.dtype == np.float32
    assert np.array_equal(src, original)  # input untouched


def test_meter_auto_stops() -> None:
    """Empty frame -> 0 stops; a filled frame -> log2 of the p99 meter."""
    assert vt.meter_auto_stops(np.zeros((4, 4, 3), np.float32)) == 0.0
    frame = np.full((8, 8, 3), 0.01, np.float32)
    expected = float(np.log2(vt.compute_exposure_scale(frame)))
    assert vt.meter_auto_stops(frame) == pytest.approx(expected)


def test_to_qimage_rounds_and_owns_buffer() -> None:
    from PySide6.QtGui import QImage

    disp = np.zeros((2, 3, 3), np.float32)
    disp[0, 0] = [0.5, 1.0, 0.0]   # 0.5*255+0.5 = 128 (round, not 127 truncate)
    disp[1, 2] = [2.0, -1.0, 0.25]  # clamps to [0,1]
    img = vt.to_qimage(disp)
    assert isinstance(img, QImage)
    assert img.format() == QImage.Format_RGB888
    assert img.width() == 3 and img.height() == 2
    assert img.pixelColor(0, 0).red() == 128
    assert img.pixelColor(0, 0).green() == 255
    assert img.pixelColor(2, 1).red() == 255   # clamped high
    assert img.pixelColor(2, 1).green() == 0    # clamped low


def test_processor_is_cached_per_spec() -> None:
    spec = _default_spec()
    p1 = vt.get_processor(spec)
    p2 = vt.get_processor(spec)
    assert p1 is p2  # same (config, display, view) -> cached instance


def test_available_views_lists_default_display() -> None:
    views = dict(vt.available_views(""))
    disp, view = vt.resolve_default_display_view("")
    assert disp in views
    assert view in views[disp]


def test_bad_config_raises_view_transform_error() -> None:
    spec = vt.ViewTransformSpec(
        config_key="/nonexistent/path/to.ocio", display="x", view="y"
    )
    with pytest.raises(vt.ViewTransformError):
        vt.get_processor(spec)


# ---------------------------------------------------------------------------
# AppSettings persistence + broadcast
# ---------------------------------------------------------------------------


def test_settings_display_view_roundtrip_and_signal(isolated_settings) -> None:
    fired = []
    isolated_settings.viewTransformChanged.connect(lambda: fired.append(1))

    assert isolated_settings.view_display_view() == ("", "")
    isolated_settings.set_view_display_view("sRGB - Display", "Raw")
    assert isolated_settings.view_display_view() == ("sRGB - Display", "Raw")
    assert len(fired) == 1

    # No-op when unchanged.
    isolated_settings.set_view_display_view("sRGB - Display", "Raw")
    assert len(fired) == 1


def test_settings_config_change_clears_display_view(isolated_settings) -> None:
    fired = []
    isolated_settings.viewTransformChanged.connect(lambda: fired.append(1))
    isolated_settings.set_view_display_view("sRGB - Display", "Raw")
    fired.clear()

    isolated_settings.set_view_ocio_config("$OCIO")
    assert isolated_settings.view_ocio_config() == "$OCIO"
    # Switching config invalidates the old (display, view).
    assert isolated_settings.view_display_view() == ("", "")
    assert len(fired) == 1


def test_spec_from_settings_fills_config_defaults(isolated_settings) -> None:
    """An unset (display, view) resolves to the active config's defaults."""
    spec = vt.spec_from_settings(isolated_settings)
    default_display, default_view = vt.resolve_default_display_view("")
    assert spec.config_key == ""
    assert spec.display == default_display
    assert spec.view == default_view


# ---------------------------------------------------------------------------
# Panel broadcast: view-transform change re-displays without re-rendering
# ---------------------------------------------------------------------------


def test_view_transform_change_redisplays_without_render(qapp, isolated_settings):
    from ghostlight_designer.project import Project
    from ghostlight_designer.sourceflare_panel.body import SourceFlarePanelBody

    project = Project()
    body = SourceFlarePanelBody(project, isolated_settings)
    try:
        # Simulate a cached rendered frame (scene-linear ACEScg).
        body._latest_hwc = np.full((16, 16, 3), 0.2, np.float32)

        calls = {"set_image": 0}
        real_set_image = body._canvas.set_image

        def spy(img):
            calls["set_image"] += 1
            return real_set_image(img)

        body._canvas.set_image = spy

        # Broadcast a designer-wide view-transform change.
        isolated_settings.set_view_display_view("sRGB - Display", "Raw")
        qapp.processEvents()

        assert calls["set_image"] >= 1        # re-displayed
        assert body._busy is False            # WITHOUT dispatching a render
        assert body._canvas._image is not None
    finally:
        # The body's spinboxes carry scrubber plumbing (hidden QTreeView +
        # adapter model); flush DeferredDelete here so they can't die during a
        # later test's event processing and corrupt the heap on Windows/PySide6.
        body.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        gc.collect()
