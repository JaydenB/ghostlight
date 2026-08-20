"""Project + OpticalEditor surface-selection plumbing.

Tests the Project's new ``selected_surface_index`` API, the
``surfaceSelectionChanged`` signal, and the optical-editor tree's
bidirectional sync with surface selection.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QModelIndex

import ghostlight

from ghostlight_designer.optical_editor import OpticalEditorBody
from ghostlight_designer.optical_editor.nodes import SurfaceNode
from ghostlight_designer.project import Project


def _collect(signal):
    received: list = []
    signal.connect(lambda *args: received.append(args if len(args) != 1 else args[0]))
    return received


def _first_surface_index_of_first_glass_element(project: Project) -> tuple:
    for ei, el in enumerate(project.system.elements):
        if el.kind == ghostlight.ElementKind.GLASS:
            indices = el.resolve_surfaces(project.system)
            if indices:
                return ei, indices[0]
    raise AssertionError("sample lens has no GLASS element with surfaces")


# ---------------------------------------------------------------------------
# Project surface-selection API
# ---------------------------------------------------------------------------


def test_project_starts_with_no_surface_selection(qapp):
    project = Project()
    assert project.selected_surface_index is None


def test_set_selected_surface_index_emits(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ei, si = _first_surface_index_of_first_glass_element(project)
    project.set_selected_element(project.system.elements[ei])

    emits = _collect(project.surfaceSelectionChanged)
    project.set_selected_surface_index(si)
    assert project.selected_surface_index == si
    assert emits == [si]

    # Idempotent
    project.set_selected_surface_index(si)
    assert emits == [si]

    project.set_selected_surface_index(None)
    assert project.selected_surface_index is None
    assert emits == [si, None]


def test_surface_must_belong_to_selected_element(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    # No element selected — surface cannot resolve.
    project.set_selected_surface_index(0)
    assert project.selected_surface_index is None

    ei, si = _first_surface_index_of_first_glass_element(project)
    project.set_selected_element(project.system.elements[ei])
    project.set_selected_surface_index(si)
    assert project.selected_surface_index == si

    # Pick a surface from a different element — should reject.
    other_si = None
    for other_ei, el in enumerate(project.system.elements):
        if other_ei == ei:
            continue
        indices = el.resolve_surfaces(project.system)
        if indices:
            other_si = indices[0]
            break
    if other_si is None:
        pytest.skip("sample lens has only one element with surfaces")
    project.set_selected_surface_index(other_si)
    assert project.selected_surface_index is None


def test_switching_element_revalidates_surface(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ei, si = _first_surface_index_of_first_glass_element(project)
    project.set_selected_element(project.system.elements[ei])
    project.set_selected_surface_index(si)
    assert project.selected_surface_index == si

    # Switch to a different element that doesn't own ``si`` — surface drops.
    for other_ei, el in enumerate(project.system.elements):
        if other_ei == ei:
            continue
        if si not in el.resolve_surfaces(project.system):
            project.set_selected_element(el)
            break
    else:
        pytest.skip("only one element in sample lens")
    assert project.selected_surface_index is None


def test_new_clears_surface_selection(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ei, si = _first_surface_index_of_first_glass_element(project)
    project.set_selected_element(project.system.elements[ei])
    project.set_selected_surface_index(si)

    emits = _collect(project.surfaceSelectionChanged)
    project.new()
    assert project.selected_surface_index is None
    assert emits == [None]


# ---------------------------------------------------------------------------
# Optical-editor tree sync
# ---------------------------------------------------------------------------


def test_clicking_surface_row_pushes_into_project(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        ei, si = _first_surface_index_of_first_glass_element(project)
        # Find the SurfaceNode model index for (ei, si).
        target_index = body._index_for_surface(si)
        assert target_index.isValid()
        node = target_index.internalPointer()
        assert isinstance(node, SurfaceNode)
        assert node.surface_index == si

        sel_model = body.tree.selectionModel()
        from PySide6.QtCore import QItemSelectionModel
        sel_model.setCurrentIndex(
            target_index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )

        assert project.selected_element is project.system.elements[ei]
        assert project.selected_surface_index == si
    finally:
        body.deleteLater()


def test_project_surface_selection_updates_tree_current_row(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        ei, si = _first_surface_index_of_first_glass_element(project)
        # Push surface selection through the project — the tree should
        # land on the matching SurfaceNode row.
        project.set_selected_element(project.system.elements[ei])
        project.set_selected_surface_index(si)

        current = body.tree.selectionModel().currentIndex()
        assert current.isValid()
        node = current.internalPointer()
        assert isinstance(node, SurfaceNode)
        assert node.surface_index == si
    finally:
        body.deleteLater()


def test_clicking_element_row_clears_surface_in_project(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        ei, si = _first_surface_index_of_first_glass_element(project)
        project.set_selected_element(project.system.elements[ei])
        project.set_selected_surface_index(si)
        assert project.selected_surface_index == si

        # Click the element row (not the surface row).
        el_idx = body._index_for_element(project.system.elements[ei])
        from PySide6.QtCore import QItemSelectionModel
        body.tree.selectionModel().setCurrentIndex(
            el_idx,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        assert project.selected_element is project.system.elements[ei]
        assert project.selected_surface_index is None
    finally:
        body.deleteLater()
