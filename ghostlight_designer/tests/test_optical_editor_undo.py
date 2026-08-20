from __future__ import annotations

import pytest
from PySide6.QtCore import QModelIndex, Qt

import ghostlight

from ghostlight_designer.optical_editor import surface_actions
from ghostlight_designer.optical_editor.columns import Column
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.project import Project


def _first_glass_element_index(project: Project) -> int:
    for i, el in enumerate(project.system.elements):
        if el.kind == ghostlight.ElementKind.GLASS:
            return i
    raise AssertionError("sample lens has no GLASS element")


def _sphere_surface_indices_in_first_glass(project: Project):
    ei = _first_glass_element_index(project)
    el = project.system.elements[ei]
    indices = el.resolve_surfaces(project.system)
    spheres = [
        i for i in indices
        if int(project.system.surfaces[i].form) == int(ghostlight.SurfaceForm.SPHERE)
    ]
    if not spheres:
        pytest.skip("first GLASS element has no SPHERE surface")
    return ei, el, spheres[0]


def test_setData_radius_creates_undo_entry(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, el, surf_index = _sphere_surface_indices_in_first_glass(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)

    original = project.system.surfaces[surf_index].radius
    ok = model.setData(radius_idx, original + 5.0, Qt.EditRole)
    assert ok
    assert project.can_undo is True
    assert project.undo_label == "Set Radius"


def test_setData_no_op_does_not_push(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, el, surf_index = _sphere_surface_indices_in_first_glass(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)
    same_value = project.system.surfaces[surf_index].radius

    ok = model.setData(radius_idx, same_value, Qt.EditRole)
    assert ok is False
    assert project.can_undo is False


def test_undo_after_radius_edit_restores_value(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, el, surf_index = _sphere_surface_indices_in_first_glass(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)

    original = project.system.surfaces[surf_index].radius
    model.setData(radius_idx, original + 5.0, Qt.EditRole)
    assert project.system.surfaces[surf_index].radius == pytest.approx(
        original + 5.0
    )

    project.undo()
    assert project.system.surfaces[surf_index].radius == pytest.approx(original)


def test_form_change_via_action_undo_round_trip(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    ei, el, surf_index = _sphere_surface_indices_in_first_glass(project)
    el_idx = model.index(ei, 0, QModelIndex())
    surface_row = len(el.material_glasses)

    # Form changes go through surface_actions (right-click "Swap Form") so
    # the inline Name-column combo can stay out of the way of routine cell
    # editing. Undo should still restore the prior form.
    assert surface_actions.set_surface_form(
        project, surf_index, int(ghostlight.SurfaceForm.CYLINDRICAL)
    )
    # New form child row appeared (index 0); the always-present coating row
    # is the second child.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    assert model.rowCount(surf_idx) == 2

    project.undo()
    assert int(project.system.surfaces[surf_index].form) == int(
        ghostlight.SurfaceForm.SPHERE
    )
    # Tree reflects the restored state: the form child is gone, leaving only
    # the coating row.
    el_idx = model.index(ei, 0, QModelIndex())
    surf_idx = model.index(surface_row, int(Column.NAME), el_idx)
    assert model.rowCount(surf_idx) == 1


def test_element_rename_label(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    el_id_idx = model.index(0, int(Column.IDENTIFIER), QModelIndex())
    assert model.setData(el_id_idx, "Renamed Element", Qt.EditRole)
    assert project.undo_label == "Rename Element"

    project.undo()
    assert project.system.elements[0].name != "Renamed Element"
