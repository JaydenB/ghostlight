"""Scene tint / alpha drop for muted elements.

Verifies the viewport scene reads `Element.is_muted` during rebuild and
applies the desaturated tint + low alpha that signals "still here but
inactive" to the user.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed; viewport tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
# The compiled `ghostlight` extension is resolved by conftest.py, which only
# prepends the in-tree bindings package when it actually holds a built
# extension. Repeating that insert here unconditionally would shadow an
# installed ghostlight-optics wheel with a stub source package.
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

from _helpers import example_doublet_path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ghostlight

from ghostlight_viewport.scene import Scene, _desaturate


LENS_PATH = example_doublet_path()


def _load_system() -> ghostlight.OpticalSystem:
    return ghostlight.OpticalSystem.load(str(LENS_PATH))


def _glass_element(system: ghostlight.OpticalSystem) -> ghostlight.Element:
    return next(el for el in system.elements if el.kind == ghostlight.ElementKind.GLASS)


def test_desaturate_no_op_at_zero_strength():
    rgb = (0.6, 0.2, 0.9)
    assert _desaturate(rgb, 0.0) == pytest.approx(rgb)


def test_desaturate_fully_collapses_to_grey():
    rgb = (0.6, 0.2, 0.9)
    out = _desaturate(rgb, 1.0)
    assert out[0] == pytest.approx(out[1])
    assert out[1] == pytest.approx(out[2])


def test_scene_active_element_has_full_alpha():
    system = _load_system()
    scene = Scene()
    scene.rebuild(system, list(system.elements))
    glass = next(se for se in scene.elements if not se.is_stop)
    # Default for glass elements is alpha = 0.75 (see scene.py).
    assert glass.alpha == pytest.approx(0.75)


def test_scene_active_element_not_marked_muted():
    system = _load_system()
    scene = Scene()
    scene.rebuild(system, list(system.elements))
    for se in scene.elements:
        assert se.muted is False


def test_scene_muted_element_carries_muted_flag():
    """The widget branches on SceneElement.muted to skip fill / cap /
    depth-prepass passes so muted elements render outline-only. The
    scene rebuild must surface the flag directly, not just via the
    alpha fallback."""
    system = _load_system()
    glass = _glass_element(system)
    glass.set_muted(system, True)

    scene = Scene()
    scene.rebuild(system, list(system.elements))
    muted_se = next(
        se for se in scene.elements
        if se.element.element_id == glass.element_id
    )
    assert muted_se.muted is True


def test_scene_muted_element_drops_alpha_and_desaturates_tint():
    """Fallback styling for consumers that don't honour the muted flag."""
    system = _load_system()
    glass = _glass_element(system)
    # Active baseline first — capture the tint.
    baseline_scene = Scene()
    baseline_scene.rebuild(system, list(system.elements))
    active_se = next(
        se for se in baseline_scene.elements
        if se.element.element_id == glass.element_id
    )
    active_tint = active_se.tint

    # Mute and rebuild.
    glass.set_muted(system, True)
    muted_scene = Scene()
    muted_scene.rebuild(system, list(system.elements))
    muted_se = next(
        se for se in muted_scene.elements
        if se.element.element_id == glass.element_id
    )

    assert muted_se.alpha < active_se.alpha
    assert muted_se.alpha == pytest.approx(0.2)
    # Tint should match _desaturate(active_tint, 0.35) — same hue, lower
    # chroma. Comparing component-wise tolerates the 0.35 strength constant
    # being tuned later.
    expected = _desaturate(active_tint, 0.35)
    assert muted_se.tint == pytest.approx(expected)
