"""Tests for element reorder (move_element + drag-drop model wiring)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QModelIndex, Qt

import ghostlight

from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.optical_editor.model import (
    ELEMENT_ROW_MIME,
    OpticalTreeModel,
)
from ghostlight_designer.optical_editor.nodes import ElementNode
from ghostlight_designer.project import Project


def _names(project: Project) -> list[str]:
    return [el.name for el in project.system.elements]


# ---------------------------------------------------------------------------
# move_element — pure action
# ---------------------------------------------------------------------------


def test_move_element_forward(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    before = _names(project)
    assert len(before) >= 3, "sample lens must have ≥ 3 elements to exercise reorder"

    # Move element at index 0 to position 2 (Qt-row sense → final index 1).
    assert element_actions.move_element(project, 0, 2) is True
    after = _names(project)
    expected = [before[1], before[0], before[2]] + before[3:]
    assert after == expected


def test_move_element_backward(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    before = _names(project)

    # Move last element to position 0.
    src = len(before) - 1
    assert element_actions.move_element(project, src, 0) is True
    after = _names(project)
    expected = [before[-1]] + before[:-1]
    assert after == expected


def test_move_element_noop_returns_false(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    before = _names(project)

    # Dropping at own position → no-op
    assert element_actions.move_element(project, 1, 1) is False
    # Dropping immediately after itself → also no-op (Qt-row 2 = "between
    # index 1 and 2", which leaves element 1 where it is).
    assert element_actions.move_element(project, 1, 2) is False
    assert _names(project) == before
    assert project.can_undo is False


def test_move_element_out_of_range_returns_false(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    n = len(project.system.elements)
    assert element_actions.move_element(project, n + 5, 0) is False
    assert element_actions.move_element(project, -1, 0) is False


def test_move_element_pushes_undo_entry(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    assert element_actions.move_element(project, 0, 2) is True
    assert project.can_undo is True
    assert project.undo_label.startswith("Move ")


def test_move_element_then_undo_restores_order(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    before = _names(project)

    element_actions.move_element(project, 0, 2)
    project.undo()

    assert _names(project) == before


def test_move_element_preserves_selection_by_id(qapp, sample_lens_path):
    """Selection follows the moved element across the JSON-roundtrip rebuild."""
    project = Project()
    project.load(str(sample_lens_path))
    moved = project.system.elements[0]
    moved_id = moved.element_id
    project.set_selected_element(moved)

    assert element_actions.move_element(project, 0, 3) is True

    assert project.selected_element is not None
    assert project.selected_element.element_id == moved_id


def test_move_element_preserves_per_surface_thickness(qapp, sample_lens_path):
    """Dragging an element to a new position must NOT change any surface's
    thickness (the 'relative pos-z' value the user sees in the tree). The
    C++ loader patches inter-element gaps from element absolute positions,
    so ``move_element`` re-derives those positions to make the patcher
    reproduce the snapshotted thicknesses exactly.
    """
    project = Project()
    project.load(str(sample_lens_path))
    thickness_by_uuid = {
        uuid: float(project.system.surfaces[i].thickness)
        for i, uuid in enumerate(project.system.surface_ids)
    }

    # Front Doublet → end. Done naively this drops Rear Singlet's last
    # surface thickness from 30 to -26 because element positions stayed in
    # place while the loader patched off the wrong neighbour.
    element_actions.move_element(project, 0, 3)

    for i, uuid in enumerate(project.system.surface_ids):
        assert project.system.surfaces[i].thickness == pytest.approx(
            thickness_by_uuid[uuid], abs=1e-4
        ), f"surface {uuid} thickness changed across reorder"


def test_move_element_preserves_thickness_when_dropping_at_start(
    qapp, sample_lens_path,
):
    project = Project()
    project.load(str(sample_lens_path))
    thickness_by_uuid = {
        uuid: float(project.system.surfaces[i].thickness)
        for i, uuid in enumerate(project.system.surface_ids)
    }

    element_actions.move_element(
        project, len(project.system.elements) - 1, 0,
    )

    for i, uuid in enumerate(project.system.surface_ids):
        assert project.system.surfaces[i].thickness == pytest.approx(
            thickness_by_uuid[uuid], abs=1e-4
        )


def test_move_element_keeps_surface_count_and_finalized_z(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    n_surfaces = project.system.num_surfaces()

    element_actions.move_element(project, 0, 2)

    assert project.system.num_surfaces() == n_surfaces
    # Chain still ends at sensor=0 after reload (loader runs finalize()).
    last = project.system.surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# OpticalTreeModel drag-and-drop wiring
# ---------------------------------------------------------------------------


def test_model_only_flags_element_rows_as_draggable(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    # Find a GLASS element row (so it has a Material child row).
    glass_ei = next(
        i for i, el in enumerate(project.system.elements)
        if el.kind == ghostlight.ElementKind.GLASS
    )
    el_idx = model.index(glass_ei, 0, QModelIndex())
    assert bool(model.flags(el_idx) & Qt.ItemIsDragEnabled)

    # A child row (material or surface) must NOT be draggable.
    child_idx = model.index(0, 0, el_idx)
    assert not bool(model.flags(child_idx) & Qt.ItemIsDragEnabled)


def test_model_drops_only_allowed_at_root_level(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)

    data = QMimeData()
    data.setData(ELEMENT_ROW_MIME, b"0")

    # Root parent → drops allowed.
    assert model.canDropMimeData(data, Qt.MoveAction, 1, 0, QModelIndex()) is True
    # Onto an element row → not allowed (would mean "drop INTO that element").
    el_idx = model.index(0, 0, QModelIndex())
    assert model.canDropMimeData(data, Qt.MoveAction, -1, 0, el_idx) is False


def test_model_rejects_non_move_actions(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    data = QMimeData()
    data.setData(ELEMENT_ROW_MIME, b"0")
    assert model.canDropMimeData(data, Qt.CopyAction, 1, 0, QModelIndex()) is False


def test_model_mimeData_carries_source_row(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    el_idx = model.index(2, 0, QModelIndex())
    md = model.mimeData([el_idx])
    payload = bytes(md.data(ELEMENT_ROW_MIME)).decode("utf-8")
    assert payload == "2"


def test_model_dropMimeData_reorders_and_returns_false(qapp, sample_lens_path):
    """dropMimeData calls move_element and returns False so Qt doesn't try
    to call removeRows on the source (we already did the full reorder)."""
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    before = _names(project)

    data = QMimeData()
    data.setData(ELEMENT_ROW_MIME, b"0")
    # Drop at row 2 → moves element 0 to between current 1 and 2.
    result = model.dropMimeData(data, Qt.MoveAction, 2, 0, QModelIndex())
    assert result is False
    expected = [before[1], before[0], before[2]] + before[3:]
    assert _names(project) == expected


def test_model_dropMimeData_at_end_appends(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    before = _names(project)

    data = QMimeData()
    data.setData(ELEMENT_ROW_MIME, b"0")
    # row == -1 means "dropped onto parent" → append at end.
    model.dropMimeData(data, Qt.MoveAction, -1, 0, QModelIndex())
    expected = before[1:] + [before[0]]
    assert _names(project) == expected
