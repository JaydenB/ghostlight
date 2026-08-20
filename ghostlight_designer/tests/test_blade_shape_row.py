"""Tests for the optical-editor "Blade Shape" row.

Covers conditional tree attachment (bladed stops only), the unit conversions
each cell performs (percentages and degrees on top of fractions and radians),
clamping, and that a write leaves the derived aperture profile fresh — the
staleness class ``sync_coating_pointers`` exists to prevent, applied to the
blade profile.

In the sample doublet, surface 3 is the aperture stop.
"""
from __future__ import annotations

import math

import pytest

import ghostlight

from ghostlight_designer.project import Project
from ghostlight_designer.optical_editor import row_schemas as rs
from ghostlight_designer.optical_editor.nodes import (
    ApertureFormNode,
    BladeShapeFormNode,
    NodeKind,
    SurfaceNode,
    build_tree,
    surface_uuid_for,
)

_STOP = 3
_GLASS = 0


def _load(sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    return project


def _bladed(project, blades: int = 6, **controls) -> Project:
    stop = project.system.surfaces[_STOP]
    stop.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
    stop.aperture_blades = blades
    for attr, value in controls.items():
        setattr(stop, attr, value)
    return project


def _blade_node(project) -> BladeShapeFormNode:
    for el in build_tree(project.system).children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == _STOP:
                for child in surf.children:
                    if isinstance(child, BladeShapeFormNode):
                        return child
    raise AssertionError("no blade-shape node on the stop")


def _has_blade_node(project) -> bool:
    for el in build_tree(project.system).children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode):
                if any(isinstance(c, BladeShapeFormNode) for c in surf.children):
                    return True
    return False


def _ctx(project, node):
    return rs.SlotContext(node=node, system=project.system, project=project)


def _slot(key: str):
    return next(s for s in rs.BLADE_SHAPE_SCHEMA.slots if s.key == key)


# ---------------------------------------------------------------------------
# Conditional attachment
# ---------------------------------------------------------------------------

def test_no_blade_row_on_a_circular_stop(qapp, sample_lens_path):
    assert not _has_blade_node(_load(sample_lens_path))


def test_blade_row_appears_on_a_bladed_stop(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    node = _blade_node(project)
    assert node.kind == NodeKind.BLADE_SHAPE_FORM
    assert node.surface_index == _STOP


def test_blade_row_sits_beside_the_aperture_row(qapp, sample_lens_path):
    """Both rows hang off the stop; neither replaces the other."""
    project = _bladed(_load(sample_lens_path))
    for el in build_tree(project.system).children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == _STOP:
                kinds = [type(c) for c in surf.children]
                assert ApertureFormNode in kinds
                assert kinds.index(ApertureFormNode) < kinds.index(
                    BladeShapeFormNode)
                return
    raise AssertionError("stop surface not in the tree")


def test_surface_uuid_resolves_through_the_blade_node(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    uuid = surface_uuid_for(_blade_node(project))
    assert uuid == list(project.system.surface_ids)[_STOP]


def test_switching_the_shape_away_drops_the_row(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    assert _has_blade_node(project)
    project.system.surfaces[_STOP].aperture_shape = int(ghostlight.ApertureShape.CIRCLE)
    assert not _has_blade_node(project)


def test_shape_write_requests_a_tree_rebuild(qapp, sample_lens_path):
    """Adding the row depends on the aperture row asking for a reset."""
    project = _load(sample_lens_path)
    for el in build_tree(project.system).children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == _STOP:
                node = next(c for c in surf.children
                            if isinstance(c, ApertureFormNode))
                slot = next(s for s in rs.APERTURE_SCHEMA.slots
                            if s.key == "identifier")
                result = slot.write(_ctx(project, node),
                                    int(ghostlight.ApertureShape.POLYGON))
                assert result.changed and result.requires_reset
                return
    raise AssertionError("stop surface not in the tree")


# ---------------------------------------------------------------------------
# Cell units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,attr,stored,shown", [
    ("blade_curvature", "aperture_curvature", -0.9, -90.0),
    ("blade_twist", "aperture_twist", 0.35, 35.0),
    ("blade_notch", "aperture_notch_rad", math.radians(35.8), 35.8),
    ("blade_notch_angle", "aperture_notch_angle_rad", math.radians(45.0), 45.0),
])
def test_cells_display_authoring_units(qapp, sample_lens_path, key, attr,
                                       stored, shown):
    project = _bladed(_load(sample_lens_path), **{attr: stored})
    got = _slot(key).get(_ctx(project, _blade_node(project)))
    assert got == pytest.approx(shown, abs=1e-3)


@pytest.mark.parametrize("key,attr,typed,stored", [
    ("blade_curvature", "aperture_curvature", -90.0, -0.9),
    ("blade_twist", "aperture_twist", 35.0, 0.35),
    ("blade_notch", "aperture_notch_rad", 35.8, math.radians(35.8)),
    ("blade_notch_angle", "aperture_notch_angle_rad", 45.0, math.radians(45.0)),
])
def test_cells_write_back_in_storage_units(qapp, sample_lens_path, key, attr,
                                           typed, stored):
    project = _bladed(_load(sample_lens_path))
    assert _slot(key).write(_ctx(project, _blade_node(project)), typed).changed
    assert getattr(project.system.surfaces[_STOP], attr) == pytest.approx(
        stored, abs=1e-6)


@pytest.mark.parametrize("key,attr,typed,stored", [
    ("blade_curvature", "aperture_curvature", 250.0, 1.0),
    ("blade_curvature", "aperture_curvature", -250.0, -1.0),
    ("blade_twist", "aperture_twist", 400.0, 1.0),
    ("blade_notch", "aperture_notch_rad", 90.0, math.radians(45.0)),
    ("blade_notch_angle", "aperture_notch_angle_rad", -20.0, 0.0),
])
def test_out_of_range_input_clamps(qapp, sample_lens_path, key, attr, typed,
                                   stored):
    project = _bladed(_load(sample_lens_path))
    _slot(key).write(_ctx(project, _blade_node(project)), typed)
    assert getattr(project.system.surfaces[_STOP], attr) == pytest.approx(
        stored, abs=1e-6)


def test_rewriting_the_same_value_is_a_noop(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path), aperture_curvature=-0.5)
    result = _slot("blade_curvature").write(
        _ctx(project, _blade_node(project)), -50.0)
    assert not result.changed


def test_unparseable_input_is_a_noop(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    result = _slot("blade_twist").write(
        _ctx(project, _blade_node(project)), "not a number")
    assert not result.changed


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

def test_a_write_refreshes_the_derived_profile(qapp, sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    stop = project.system.surfaces[_STOP]
    assert not stop.aperture_profile.deformed()
    _slot("blade_curvature").write(_ctx(project, _blade_node(project)), -60.0)
    assert stop.aperture_profile.deformed()
    assert stop.aperture_profile.r_w < math.cos(math.pi / 6)


def test_a_write_invalidates_the_calibration_cache(qapp, sample_lens_path):
    """The stop silhouette changed, so the calibrated pupil has to be re-solved."""
    project = _bladed(_load(sample_lens_path))
    before = project.system._current_key()
    _slot("blade_notch").write(_ctx(project, _blade_node(project)), 20.0)
    assert project.system._current_key() != before


# ---------------------------------------------------------------------------
# Row layout
# ---------------------------------------------------------------------------

def test_the_four_controls_pack_into_consecutive_columns(qapp,
                                                         sample_lens_path):
    project = _bladed(_load(sample_lens_path))
    node = _blade_node(project)
    keys = [rs.slot_at(node, col).key for col in range(1, 5)]
    assert keys == ["blade_curvature", "blade_twist", "blade_notch",
                    "blade_notch_angle"]
    assert rs.slot_at(node, 5) is None
    assert rs.name_label(_ctx(project, node)) == "Blade Shape"


def test_glass_surfaces_never_get_the_row(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    project.system.surfaces[_GLASS].aperture_shape = int(
        ghostlight.ApertureShape.POLYGON)
    project.system.surfaces[_GLASS].aperture_blades = 6
    assert not _has_blade_node(project)
