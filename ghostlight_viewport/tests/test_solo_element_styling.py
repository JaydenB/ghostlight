"""Scene tint/alpha boost for elements containing ghost-solo'd surfaces.

Verifies the viewport scene reads the solo surface set during rebuild and
applies the accent-blended tint + near-opaque alpha that signals "ghosts
you're filtering involve this element". Mirrors the mute styling test.
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

from ghostlight_viewport.scene import Scene, _accent_blend


LENS_PATH = example_doublet_path()


def _load_system() -> ghostlight.OpticalSystem:
    return ghostlight.OpticalSystem.load(str(LENS_PATH))


def _first_glass_element_indices(system: ghostlight.OpticalSystem) -> list[int]:
    glass = next(el for el in system.elements if el.kind == ghostlight.ElementKind.GLASS)
    return glass.resolve_surfaces(system)


# ---------------------------------------------------------------------------
# _accent_blend math
# ---------------------------------------------------------------------------

def test_accent_blend_no_op_at_zero_strength():
    rgb = (0.5, 0.5, 0.5)
    accent = (1.0, 0.0, 0.0)
    assert _accent_blend(rgb, accent, 0.0) == pytest.approx(rgb)


def test_accent_blend_full_strength_returns_accent():
    rgb = (0.5, 0.5, 0.5)
    accent = (1.0, 0.0, 0.0)
    assert _accent_blend(rgb, accent, 1.0) == pytest.approx(accent)


def test_accent_blend_half_is_average():
    rgb = (0.5, 0.5, 0.5)
    accent = (1.0, 0.0, 0.0)
    out = _accent_blend(rgb, accent, 0.5)
    assert out == pytest.approx((0.75, 0.25, 0.25))


# ---------------------------------------------------------------------------
# Scene rebuild integration
# ---------------------------------------------------------------------------

def test_scene_default_no_solo_no_highlight():
    system = _load_system()
    scene = Scene()
    scene.rebuild(system, list(system.elements))
    for se in scene.elements:
        # Default glass alpha is 0.75; the solo highlight bumps to 0.95.
        if not se.is_stop:
            assert se.alpha == pytest.approx(0.75)


def test_scene_solo_indices_boost_alpha():
    system = _load_system()
    solo_indices = set(_first_glass_element_indices(system)[:1])
    scene = Scene()
    scene.rebuild(
        system,
        list(system.elements),
        ghost_solo_surface_indices=solo_indices,
    )
    glass_el_index = next(
        i for i, el in enumerate(system.elements)
        if el.kind == ghostlight.ElementKind.GLASS
    )
    se = scene.elements[glass_el_index]
    assert se.alpha == pytest.approx(0.95)


def test_scene_solo_does_not_change_non_solo_elements():
    """Elements with no solo'd surfaces keep their default look."""
    system = _load_system()
    solo_indices = set(_first_glass_element_indices(system)[:1])
    scene = Scene()
    scene.rebuild(
        system,
        list(system.elements),
        ghost_solo_surface_indices=solo_indices,
    )
    # The Front Doublet has the solo'd surface. The Rear Singlet doesn't.
    rear = next(
        se for se in scene.elements
        if se.element.name == "Rear Singlet"
    )
    assert rear.alpha == pytest.approx(0.75)


def test_scene_mute_wins_over_solo():
    """Muting an element overrides the solo highlight — the user
    explicitly turned it off."""
    system = _load_system()
    glass = next(el for el in system.elements if el.kind == ghostlight.ElementKind.GLASS)
    glass.set_muted(system, True)

    solo_indices = set(glass.resolve_surfaces(system)[:1])
    scene = Scene()
    scene.rebuild(
        system,
        list(system.elements),
        ghost_solo_surface_indices=solo_indices,
    )
    se = next(
        s for s in scene.elements
        if s.element.element_id == glass.element_id
    )
    # Muted alpha (0.2), not solo alpha (0.95).
    assert se.alpha == pytest.approx(0.2)
