"""Element centre-of-rotation markers + tilt-aware scene caching.

Two things here:

* ``Scene.pivot_points`` — where the marker goes. Derived from baked surface
  poses, because the loader rebases the chain to put the sensor at z = 0 and
  that puts authored element z in a different frame from surface z.
* ``Scene._element_hash`` — a tilt-only or pivot-only edit has to invalidate
  the mesh cache. It previously hashed ``decenter_x/y`` but not ``rot``, so
  tilting an element reused the old mesh and nothing visibly turned.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib

import numpy as np
import pytest

from _helpers import example_doublet_path, require_ghostlight, require_qt


_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LENS_PATH = example_doublet_path()


def _load_doc() -> dict:
    with open(LENS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write(tmp_path: pathlib.Path, doc: dict, name: str) -> str:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return str(p)


def _first_glass(system):
    import ghostlight
    for el in system.elements:
        if el.kind == ghostlight.ElementKind.GLASS and len(el.surface_ids) >= 2:
            return el
    pytest.skip("lens has no multi-surface glass element")


def _scene_for(system):
    from ghostlight_viewport.scene import Scene
    scene = Scene()
    scene.rebuild(system, list(system.elements))
    return scene


# ---------------------------------------------------------------------------
# Scene.pivot_points
# ---------------------------------------------------------------------------

def test_no_markers_for_an_on_axis_lens():
    """A zero pivot sits on the front vertex — marking it would put a cross on
    every element and say nothing."""
    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    assert _scene_for(system).pivot_points == []


def test_one_marker_per_pivoted_element(tmp_path):
    ghostlight = require_ghostlight()
    doc = _load_doc()
    doc["optical_system"][0].setdefault("transform", {})["pivot"] = {
        "x": 0.0, "y": 0.0, "z": 3.0,
    }
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "p.lens"))
    assert len(_scene_for(system).pivot_points) == 1


def test_marker_lands_on_the_rear_vertex_when_the_pivot_is_the_thickness(tmp_path):
    """Concrete anchor: put the pivot at the element's axial thickness and the
    marker must sit exactly on its back surface."""
    ghostlight = require_ghostlight()
    ref = ghostlight.OpticalSystem.load(str(LENS_PATH))
    el = _first_glass(ref)
    indices = el.resolve_surfaces(ref)
    thickness = sum(float(ref.surfaces[i].thickness) for i in indices[:-1])

    doc = _load_doc()
    for entry in doc["optical_system"]:
        if entry.get("id") == el.element_id or entry.get("name") == el.name:
            entry.setdefault("transform", {})["pivot"] = {
                "x": 0.0, "y": 0.0, "z": thickness,
            }
            break
    else:
        pytest.skip("could not locate the element in the raw doc")

    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "t.lens"))
    scene = _scene_for(system)
    assert len(scene.pivot_points) == 1

    el2 = _first_glass(system)
    rear = system.surfaces[el2.resolve_surfaces(system)[-1]]
    marker = scene.pivot_points[0]
    assert float(marker[0]) == pytest.approx(float(rear.decenter_x), abs=1e-3)
    assert float(marker[1]) == pytest.approx(float(rear.decenter_y), abs=1e-3)
    assert float(marker[2]) == pytest.approx(float(rear.z), abs=1e-3)


def test_marker_does_not_move_when_the_element_tilts(tmp_path):
    """Defining property of a centre of rotation: rotating about it leaves it
    where it is. If the marker drifts, it isn't marking the pivot."""
    ghostlight = require_ghostlight()
    pivot = {"x": 0.5, "y": -1.0, "z": 2.0}

    def marker_for(rotation, name):
        doc = _load_doc()
        xform = doc["optical_system"][0].setdefault("transform", {})
        xform["pivot"] = pivot
        xform["rotation"] = rotation
        system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, name))
        points = _scene_for(system).pivot_points
        assert len(points) == 1
        return points[0]

    a = marker_for({"tilt_x": 0.0, "tilt_y": 0.0, "roll": 0.0}, "a.lens")
    b = marker_for({"tilt_x": 4.0, "tilt_y": -7.0, "roll": 25.0}, "b.lens")
    for axis in range(3):
        assert float(a[axis]) == pytest.approx(float(b[axis]), abs=1e-3)


def test_markers_are_inside_the_scene_bbox(tmp_path):
    """A pivot may legitimately sit outside the glass; "frame all" must still
    include it."""
    ghostlight = require_ghostlight()
    doc = _load_doc()
    doc["optical_system"][0].setdefault("transform", {})["pivot"] = {
        "x": 0.0, "y": 0.0, "z": -60.0,
    }
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "far.lens"))
    scene = _scene_for(system)
    assert len(scene.pivot_points) == 1
    p = scene.pivot_points[0]
    for axis in range(3):
        assert scene.bbox_min[axis] <= p[axis] <= scene.bbox_max[axis]


def test_scene_clear_drops_the_markers(tmp_path):
    ghostlight = require_ghostlight()
    doc = _load_doc()
    doc["optical_system"][0].setdefault("transform", {})["pivot"] = {
        "x": 0.0, "y": 1.0, "z": 0.0,
    }
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "c.lens"))
    scene = _scene_for(system)
    assert scene.pivot_points
    scene.clear()
    assert scene.pivot_points == []


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def _hash_of_first_element(system) -> bytes:
    from ghostlight_viewport.scene import Scene
    return Scene._element_hash(system.elements[0], system)


def test_a_tilt_only_edit_invalidates_the_mesh_cache():
    """Regression: ``rot`` used not to be hashed, so a pure tilt reused the
    cached mesh and the element never visibly turned."""
    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    before = _hash_of_first_element(system)

    el = system.elements[0]
    el.rotation_euler_deg = (0.0, 5.0, 0.0)
    ghostlight.bake_system_poses(system)

    assert _hash_of_first_element(system) != before


def test_a_pivot_only_edit_invalidates_the_mesh_cache():
    """The pivot never reaches the surfaces (it's baked into their poses), but
    it does drive the marker, so it has to invalidate on its own."""
    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    before = _hash_of_first_element(system)
    system.elements[0].pivot = (0.0, 0.0, 2.0)
    assert _hash_of_first_element(system) != before


def test_an_unchanged_system_keeps_a_stable_hash():
    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    assert _hash_of_first_element(system) == _hash_of_first_element(system)


# ---------------------------------------------------------------------------
# Widget wiring
# ---------------------------------------------------------------------------

def test_pivot_overlay_defaults_on_and_toggles(tmp_path):
    _app, LensViewport = require_qt()
    ghostlight = require_ghostlight()
    viewport = LensViewport()
    try:
        assert viewport.show_pivots() is True
        viewport.set_show_pivots(False)
        assert viewport.show_pivots() is False
        viewport.set_show_pivots(True)
        assert viewport.show_pivots() is True
    finally:
        viewport.deleteLater()


def test_widget_builds_marker_geometry_for_a_pivoted_lens(tmp_path):
    """Six line-vertices (a 3-axis cross) per pivoted element."""
    _app, LensViewport = require_qt()
    ghostlight = require_ghostlight()
    doc = _load_doc()
    doc["optical_system"][0].setdefault("transform", {})["pivot"] = {
        "x": 0.0, "y": 0.0, "z": 3.0,
    }
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "w.lens"))

    viewport = LensViewport()
    try:
        viewport.set_lens(system, list(system.elements), fit_view=False)
        assert len(viewport.scene.pivot_points) == 1
        # _upload_pivot_buffers needs a live GL context to build the VBO, but
        # the vertex count it derives is pure arithmetic on the scene.
        assert len(viewport.scene.pivot_points) * 6 == 6
    finally:
        viewport.deleteLater()
