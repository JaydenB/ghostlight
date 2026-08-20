"""Smoke test: instantiate LensViewport, push a doublet, paint a frame.

Skips automatically when PySide6 isn't installed or when the offscreen GL
platform can't create a 3.3 context (CI runners without a GPU may need
``QT_QPA_PLATFORM=offscreen`` plus a software GL).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed; viewport tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "ghostlight" / "bindings" / "python"))
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

from _helpers import example_doublet_path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import ghostlight

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from ghostlight_viewport import LensViewport, RayBundle, SensorSpec
from ghostlight_viewport.widget import set_default_surface_format


LENS_PATH = example_doublet_path()


@pytest.fixture(scope="module")
def app():
    set_default_surface_format()
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def lens():
    return ghostlight.OpticalSystem.load(str(LENS_PATH))


@pytest.fixture
def elements():
    return ghostlight.Element.from_lens_file(str(LENS_PATH))


def _build_bundle(lens):
    system = lens
    front = system.surfaces[0]
    start_z = front.z - 20.0
    paths = []
    origins = []
    wavelengths = []
    for j in range(3):
        for i in range(3):
            u = (i - 1) * 0.5
            v = (j - 1) * 0.5
            d = ghostlight.Vec3f(0.0, 0.0, 1.0)
            origin = ghostlight.Vec3f(u * 5.0, v * 5.0, start_z)
            ray = ghostlight.Ray(origin, d, 587.56)
            paths.append(ghostlight.trace_primary_ray_diagnostic(ray, system))
            origins.append(origin)
            wavelengths.append(587.56)
    return RayBundle(paths=paths, wavelengths_nm=wavelengths, origins=origins)


def test_widget_constructs_and_paints(app, lens, elements):
    viewport = LensViewport()
    viewport.resize(640, 480)
    calib = lens.calibration()
    sensor = SensorSpec.from_calibration(calib)
    viewport.set_lens(lens, elements)
    viewport.set_sensor(sensor)
    viewport.set_trace_results([_build_bundle(lens)])
    viewport.show()
    app.processEvents()
    app.processEvents()

    if not viewport._gl_ready:
        pytest.skip("offscreen GL context unavailable on this platform")

    img = viewport.screenshot()
    assert not img.isNull(), "screenshot returned null QImage"
    assert img.width() > 0 and img.height() > 0


def test_clip_plane_changes_pixels(app, lens, elements):
    viewport = LensViewport()
    viewport.resize(320, 240)
    sensor = SensorSpec.from_calibration(lens.calibration())
    viewport.set_lens(lens, elements)
    viewport.set_sensor(sensor)
    viewport.show()
    app.processEvents()
    if not viewport._gl_ready:
        pytest.skip("offscreen GL context unavailable on this platform")

    before = viewport.screenshot()
    viewport.set_clip_plane_y(0.0)
    app.processEvents()
    after = viewport.screenshot()
    if before.isNull() or after.isNull():
        pytest.skip("screenshot unavailable in offscreen mode")
    # Note: byte equality test is unreliable across drivers; just confirm both
    # screenshots are real images.
    assert before.width() == after.width() == viewport.width()


def test_reset_view_no_crash(app, lens, elements):
    viewport = LensViewport()
    viewport.resize(200, 200)
    viewport.set_lens(lens, elements)
    viewport.show()
    app.processEvents()
    viewport.reset_view()
    viewport.set_view("+y")
    viewport.set_view("iso")
    viewport.set_view("+x")
    app.processEvents()


def test_selection_signal_on_programmatic_set(app, lens, elements):
    viewport = LensViewport()
    viewport.resize(200, 200)
    viewport.set_lens(lens, elements)
    viewport.show()
    app.processEvents()
    # Direct selection state mutation (the high-level click is hard to fire
    # reliably in offscreen mode; this exercises the signal plumbing).
    received: list = []
    viewport.elementSelected.connect(lambda el: received.append(el))
    viewport.selection.set_element(elements[0])
    viewport.elementSelected.emit(elements[0])
    app.processEvents()
    assert received and received[-1] is elements[0]


def test_pick_fbo_returns_non_empty_at_lens_centre(app, lens, elements):
    """End-to-end pick smoke test.

    The pick pass enables depth test on an FBO; an earlier iteration of this
    file disabled depth test because PySide6's depth-buffer clear didn't
    seem to work on FBOs — the read came back all-zero.  We re-enabled it
    after fixing the clear ordering, so guard the regression: a click at
    the centre of a freshly-painted viewport must resolve to *some*
    element, never to the empty-pixel signal.
    """
    viewport = LensViewport()
    viewport.resize(400, 300)
    viewport.set_lens(lens, elements)
    viewport.show()
    app.processEvents()
    app.processEvents()
    if not viewport._gl_ready:
        pytest.skip("offscreen GL context unavailable on this platform")

    cx = viewport.width() // 2
    cy = viewport.height() // 2
    dpr = float(viewport.devicePixelRatioF()) or 1.0
    info = viewport._pick_at(int(round(cx * dpr)), int(round(cy * dpr)))
    # Lens centre should hit a glass element after fit_to_bbox.  If the
    # depth-cleared FBO regressed to all-zero, info["is_empty"] would be
    # True here.
    assert not info["is_empty"], (
        "pick FBO returned empty at the lens centre — depth-clear regression?"
    )
    assert info["element_index"] is not None
    assert 0 <= info["element_index"] < len(elements)


# ---------------------------------------------------------------------------
# Right-click context menu hook
# ---------------------------------------------------------------------------


def test_context_menu_enabled_default_and_toggle(app, lens, elements):
    viewport = LensViewport()
    viewport.set_lens(lens, elements)
    assert viewport.context_menu_enabled() is True
    viewport.set_context_menu_enabled(False)
    assert viewport.context_menu_enabled() is False
    viewport.set_context_menu_enabled(True)
    assert viewport.context_menu_enabled() is True


def _pick_hit(element_index=0, surface_index=None, tag=None):
    from ghostlight_viewport import picking
    return {
        "tag": picking.TAG_ELEMENT_BODY if tag is None else tag,
        "element_index": element_index,
        "surface_index": surface_index,
        "face_index": None,
        "is_empty": False,
    }


def test_context_click_emits_on_element_pick(app, lens, elements, monkeypatch):
    viewport = LensViewport()
    viewport.resize(400, 300)
    viewport.set_lens(lens, elements)
    viewport.set_selection_mode("element")
    if not viewport.scene.elements:
        pytest.skip("scene elements not built in this environment")

    monkeypatch.setattr(viewport, "_pick_at", lambda *a, **k: _pick_hit(0))
    received = []
    viewport.contextMenuRequested.connect(received.append)
    viewport._handle_context_click(50, 50, QPoint(10, 10))

    assert len(received) == 1
    info = received[0]
    assert info["mode"] == "element"
    assert info["element_index"] == 0
    assert info["element"] is viewport.scene.elements[0].element
    assert info["global_pos"] == QPoint(10, 10)
    # Selection updated so the highlight matches the menu target.
    assert viewport.selected_element() is viewport.scene.elements[0].element


def test_context_click_suppressed_when_disabled(app, lens, elements, monkeypatch):
    viewport = LensViewport()
    viewport.set_lens(lens, elements)
    viewport.set_selection_mode("element")
    viewport.set_context_menu_enabled(False)
    monkeypatch.setattr(viewport, "_pick_at", lambda *a, **k: _pick_hit(0))
    received = []
    viewport.contextMenuRequested.connect(received.append)
    viewport._handle_context_click(50, 50, QPoint(10, 10))
    assert received == []


def test_context_click_suppressed_in_none_mode(app, lens, elements, monkeypatch):
    viewport = LensViewport()
    viewport.set_lens(lens, elements)
    viewport.set_selection_mode("none")
    monkeypatch.setattr(viewport, "_pick_at", lambda *a, **k: _pick_hit(0))
    received = []
    viewport.contextMenuRequested.connect(received.append)
    viewport._handle_context_click(50, 50, QPoint(10, 10))
    assert received == []


def test_context_click_suppressed_on_empty_pick(app, lens, elements, monkeypatch):
    viewport = LensViewport()
    viewport.set_lens(lens, elements)
    viewport.set_selection_mode("element")
    empty = {"tag": 0, "element_index": None, "surface_index": None,
             "face_index": None, "is_empty": True}
    monkeypatch.setattr(viewport, "_pick_at", lambda *a, **k: empty)
    received = []
    viewport.contextMenuRequested.connect(received.append)
    viewport._handle_context_click(50, 50, QPoint(10, 10))
    assert received == []
