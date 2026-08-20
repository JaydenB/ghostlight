"""ViewportToolbar overlay — cutaway midpoint logic + selection-mode stub.

Verifies the toolbar exists as a child of `LensViewport`, that picking a
cutaway axis sets the clip plane at the bbox midpoint with the correct
sign convention, that "none" clears both planes, and that the
selection-mode button emits without side-effects (it's stubbed out for
future picking integration).
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest


pytest.importorskip("PySide6", reason="PySide6 not installed; viewport tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from ghostlight_viewport import LensViewport
from ghostlight_viewport.viewport_toolbar import ViewportToolbar
from ghostlight_viewport.widget import set_default_surface_format


@pytest.fixture(scope="module")
def app():
    set_default_surface_format()
    instance = QApplication.instance() or QApplication([])
    yield instance


def _set_bbox(viewport: LensViewport, mn, mx) -> None:
    viewport.scene.bbox_min = np.asarray(mn, dtype=np.float32)
    viewport.scene.bbox_max = np.asarray(mx, dtype=np.float32)


def test_toolbar_is_child_of_viewport(app):
    viewport = LensViewport()
    children = viewport.findChildren(QWidget)
    assert any(isinstance(c, ViewportToolbar) for c in children)


def test_default_cutaway_is_none(app):
    viewport = LensViewport()
    assert viewport.cutaway_mode() == "none"
    assert viewport.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 0.0, 0.0, 0.0)


def test_set_cutaway_x_sets_midpoint_plane(app):
    viewport = LensViewport()
    _set_bbox(viewport, [2.0, -4.0, 0.0], [8.0, 4.0, 50.0])
    viewport.set_cutaway_mode("x")
    # mid_x = 5.0; set_x(-5.0) then a_invert=True ⇒ resolved (-1, 0, 0, 5)
    assert viewport.clip_state.uniform_vec4() == (-1.0, 0.0, 0.0, 5.0)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.a_invert is True


def test_set_cutaway_y_sets_midpoint_plane(app):
    viewport = LensViewport()
    _set_bbox(viewport, [-10.0, -2.0, 0.0], [10.0, 6.0, 50.0])
    viewport.set_cutaway_mode("y")
    # mid_y = 2.0; set_y(-2.0) ⇒ vec4_b = (0, 1, 0, -2)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 1.0, 0.0, -2.0)
    assert viewport.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)


def test_set_cutaway_xy_sets_both_planes(app):
    viewport = LensViewport()
    _set_bbox(viewport, [2.0, -2.0, 0.0], [8.0, 6.0, 50.0])
    viewport.set_cutaway_mode("xy")
    # X side is inverted to keep the +X half; Y side is not.
    assert viewport.clip_state.uniform_vec4() == (-1.0, 0.0, 0.0, 5.0)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 1.0, 0.0, -2.0)
    assert viewport.clip_state.a_invert is True
    assert viewport.clip_state.b_invert is False


def test_set_cutaway_none_clears_both_planes(app):
    viewport = LensViewport()
    _set_bbox(viewport, [2.0, -2.0, 0.0], [8.0, 6.0, 50.0])
    viewport.set_cutaway_mode("xy")
    assert viewport.clip_state.uniform_vec4() != (0.0, 0.0, 0.0, 0.0)
    viewport.set_cutaway_mode("none")
    assert viewport.cutaway_mode() == "none"
    assert viewport.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.a_invert is False
    assert viewport.clip_state.b_invert is False


def test_switching_y_after_x_clears_x_invert(app):
    viewport = LensViewport()
    _set_bbox(viewport, [2.0, -4.0, 0.0], [8.0, 4.0, 50.0])
    viewport.set_cutaway_mode("x")
    assert viewport.clip_state.a_invert is True
    viewport.set_cutaway_mode("y")
    # Y-only must not inherit the X inversion flag.
    assert viewport.clip_state.a_invert is False
    assert viewport.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.uniform_vec4_b() == (0.0, 1.0, 0.0, 0.0)


def test_selection_mode_programmatic_set_updates_mode_and_clears_selection(app):
    viewport = LensViewport()
    assert viewport.selection_mode() == "element"
    viewport.set_selection_mode("surface")
    assert viewport.selection_mode() == "surface"
    # Programmatic mode change doesn't touch clip state, and selection is
    # cleared (no element was selected to begin with, but state stays clean).
    assert viewport.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)
    assert viewport.selected_element() is None
    assert viewport.selected_surface() is None


def test_selection_mode_user_change_clears_existing_selection(app):
    """Switching mode via the toolbar menu clears any current pick."""
    class _Sentinel:
        pass

    viewport = LensViewport()
    sentinel = _Sentinel()
    viewport.selection.element = sentinel
    viewport.selection.surface = 3

    elements_emitted: list = []
    surfaces_emitted: list = []
    viewport.elementSelected.connect(elements_emitted.append)
    viewport.surfaceSelected.connect(surfaces_emitted.append)

    # Drive the toolbar's own signal (what a real menu click would do).
    viewport._toolbar.selection.set_value("none", emit=True)

    assert viewport.selection_mode() == "none"
    assert viewport.selected_element() is None
    assert viewport.selected_surface() is None
    assert elements_emitted == [None]
    assert surfaces_emitted == [None]


def test_toolbar_positioned_left_of_gizmo(app):
    viewport = LensViewport()
    viewport.resize(800, 600)
    viewport._position_toolbar()
    tb = viewport._toolbar
    # Gizmo reserves 120 + 16 = 136 px on the right edge.
    assert tb.x() + tb.width() <= 800 - 136
    assert tb.y() == 16
    assert tb.width() > 0 and tb.height() > 0


def test_set_cutaway_unknown_raises(app):
    viewport = LensViewport()
    with pytest.raises(ValueError):
        viewport._apply_cutaway("diagonal")
