"""Tests for ``flip_element`` — reverses an element's geometry without
moving any of its neighbours."""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.project import Project


def _surfaces_of(project: Project, element: ghostlight.Element):
    """Return the live Surface wrappers of ``element`` in trace order."""
    return [project.system.surfaces[i] for i in element.resolve_surfaces(project.system)]


# ---------------------------------------------------------------------------
# Core flip semantics
# ---------------------------------------------------------------------------


def test_flip_reverses_surface_order_uuid_wise(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    uuids_before = list(doublet.surface_ids)

    assert element_actions.flip_element(project, doublet) is True

    # Re-fetch via name (Element instances are fresh after the JSON reload).
    doublet_after = next(el for el in project.system.elements if el.name == "Front Doublet")
    assert list(doublet_after.surface_ids) == list(reversed(uuids_before))


def test_flip_reverses_materials(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    mats_before = list(doublet.material_glasses)

    element_actions.flip_element(project, doublet)
    doublet_after = next(el for el in project.system.elements if el.name == "Front Doublet")
    assert list(doublet_after.material_glasses) == list(reversed(mats_before))


def test_flip_negates_radii(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    # Capture radii by UUID so we can match after the reload assigns new wrappers.
    radius_before = {
        uuid: project.system.surfaces[
            list(project.system.surface_ids).index(uuid)
        ].radius
        for uuid in doublet.surface_ids
    }

    element_actions.flip_element(project, doublet)

    for uuid, r_before in radius_before.items():
        idx = list(project.system.surface_ids).index(uuid)
        assert project.system.surfaces[idx].radius == pytest.approx(-r_before)


def test_flip_keeps_every_surface_vertex_in_place(qapp, sample_lens_path):
    """No surface in the flipped element moves in axial space — each
    POSITION keeps its old thickness (and hence its successor's z), only
    surface IDs / radii / materials shuffle."""
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")

    # Snapshot positional thicknesses (NOT keyed by UUID — the UUIDs move
    # to new positions during the flip; we're asserting the POSITIONS
    # themselves don't change their thickness value).
    positional_thicknesses_before = [
        s.thickness for s in _surfaces_of(project, doublet)
    ]
    positional_z_before = [s.z for s in _surfaces_of(project, doublet)]

    element_actions.flip_element(project, doublet)

    doublet_after = next(el for el in project.system.elements if el.name == "Front Doublet")
    surfaces_after = _surfaces_of(project, doublet_after)
    for i, s in enumerate(surfaces_after):
        assert s.thickness == pytest.approx(
            positional_thicknesses_before[i], abs=1e-4
        ), f"position {i} thickness changed across flip"
        assert s.z == pytest.approx(
            positional_z_before[i], abs=1e-4
        ), f"position {i} vertex z changed across flip"


# ---------------------------------------------------------------------------
# Neighbour-invariance — the whole point of the "preserve relative pos-z" rule
# ---------------------------------------------------------------------------


def test_flip_does_not_move_neighbour_elements(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    pos_before = {el.name: el.position[2] for el in project.system.elements}

    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    element_actions.flip_element(project, doublet)

    pos_after = {el.name: el.position[2] for el in project.system.elements}
    # Aperture Stop and Rear Singlet must NOT have moved.
    assert pos_after["Aperture Stop"] == pytest.approx(pos_before["Aperture Stop"])
    assert pos_after["Rear Singlet"] == pytest.approx(pos_before["Rear Singlet"])


def test_flip_does_not_change_neighbour_surface_thicknesses(qapp, sample_lens_path):
    """Surfaces in elements adjacent to the flipped one keep their thickness."""
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    doublet_uuids = set(doublet.surface_ids)
    thickness_before = {
        uuid: float(project.system.surfaces[i].thickness)
        for i, uuid in enumerate(project.system.surface_ids)
        if uuid not in doublet_uuids
    }

    element_actions.flip_element(project, doublet)

    for uuid, t_before in thickness_before.items():
        idx = list(project.system.surface_ids).index(uuid)
        assert project.system.surfaces[idx].thickness == pytest.approx(
            t_before, abs=1e-4
        ), f"neighbour surface {uuid} thickness changed across flip"


def test_flip_preserves_sensor_position(qapp, sample_lens_path):
    """Chain still ends at sensor=0 after the flip, regardless of which
    element was flipped."""
    project = Project()
    project.load(str(sample_lens_path))
    rear = next(el for el in project.system.elements if el.name == "Rear Singlet")
    element_actions.flip_element(project, rear)
    last = project.system.surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0, abs=1e-4)


def test_flip_last_element_keeps_back_focal_distance(qapp, sample_lens_path):
    """Flipping the last element keeps its back focal distance, so the
    chain length and sensor placement are unchanged."""
    project = Project()
    project.load(str(sample_lens_path))
    rear = next(el for el in project.system.elements if el.name == "Rear Singlet")
    rear_surfaces_before = _surfaces_of(project, rear)
    bfd_before = rear_surfaces_before[-1].thickness

    element_actions.flip_element(project, rear)

    rear_after = next(el for el in project.system.elements if el.name == "Rear Singlet")
    rear_surfaces_after = _surfaces_of(project, rear_after)
    assert rear_surfaces_after[-1].thickness == pytest.approx(bfd_before)


# ---------------------------------------------------------------------------
# Edge cases + undo + selection
# ---------------------------------------------------------------------------


def test_flip_aperture_stop_is_noop(qapp, sample_lens_path):
    """Aperture stops are one surface — there's nothing to reverse, so the
    action returns False without pushing an undo entry."""
    project = Project()
    project.load(str(sample_lens_path))
    stop = next(el for el in project.system.elements if el.name == "Aperture Stop")
    assert element_actions.flip_element(project, stop) is False
    assert project.can_undo is False


def test_flip_unknown_element_returns_false(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    stranger = ghostlight.Element(name="stranger", surface_ids=["a", "b"], material_glasses=["g"])
    assert element_actions.flip_element(project, stranger) is False


def test_flip_pushes_undo_entry(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    element_actions.flip_element(project, doublet)
    assert project.can_undo is True
    assert project.undo_label.startswith("Flip ")


def test_flip_then_undo_restores(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    uuids_before = list(doublet.surface_ids)
    mats_before = list(doublet.material_glasses)
    radii_before = {
        uuid: project.system.surfaces[
            list(project.system.surface_ids).index(uuid)
        ].radius
        for uuid in doublet.surface_ids
    }

    element_actions.flip_element(project, doublet)
    project.undo()

    doublet_restored = next(el for el in project.system.elements if el.name == "Front Doublet")
    assert list(doublet_restored.surface_ids) == uuids_before
    assert list(doublet_restored.material_glasses) == mats_before
    for uuid, r in radii_before.items():
        idx = list(project.system.surface_ids).index(uuid)
        assert project.system.surfaces[idx].radius == pytest.approx(r)


def test_flip_preserves_selection_by_id(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    project.set_selected_element(doublet)
    doublet_id = doublet.element_id

    element_actions.flip_element(project, doublet)

    assert project.selected_element is not None
    assert project.selected_element.element_id == doublet_id


def test_flip_twice_restores_geometry(qapp, sample_lens_path):
    """Flipping the same element twice should round-trip back to the
    original geometry (radii and thicknesses)."""
    project = Project()
    project.load(str(sample_lens_path))
    doublet = next(el for el in project.system.elements if el.name == "Front Doublet")
    radii_before = {
        uuid: project.system.surfaces[
            list(project.system.surface_ids).index(uuid)
        ].radius
        for uuid in doublet.surface_ids
    }
    thicknesses_before = {
        uuid: project.system.surfaces[
            list(project.system.surface_ids).index(uuid)
        ].thickness
        for uuid in doublet.surface_ids
    }
    uuids_before = list(doublet.surface_ids)

    element_actions.flip_element(project, doublet)
    doublet_mid = next(el for el in project.system.elements if el.name == "Front Doublet")
    element_actions.flip_element(project, doublet_mid)

    doublet_after = next(el for el in project.system.elements if el.name == "Front Doublet")
    assert list(doublet_after.surface_ids) == uuids_before
    for uuid, r in radii_before.items():
        idx = list(project.system.surface_ids).index(uuid)
        assert project.system.surfaces[idx].radius == pytest.approx(r)
    for uuid, t in thicknesses_before.items():
        idx = list(project.system.surface_ids).index(uuid)
        assert project.system.surfaces[idx].thickness == pytest.approx(t, abs=1e-4)
