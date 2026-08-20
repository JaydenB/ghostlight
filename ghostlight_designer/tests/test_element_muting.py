"""Tests for the Element-muting designer integration.

Backend semantics live in ghostlight/tests/test_surface_muting.py — this
file covers only the designer-side wiring: the undoable action helper,
the tree icon switching, the context-menu label, and the muted row
font hint.
"""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.optical_editor.nodes import ElementNode, build_tree
from ghostlight_designer.project import Project


def _load(qapp, sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    return project


def _glass_element(project: Project) -> ghostlight.Element:
    for el in project.system.elements:
        if el.kind == ghostlight.ElementKind.GLASS:
            return el
    raise AssertionError("sample lens unexpectedly has no glass element")


def _stop_element(project: Project) -> ghostlight.Element:
    for el in project.system.elements:
        if el.kind == ghostlight.ElementKind.STOP:
            return el
    raise AssertionError("sample lens unexpectedly has no stop element")


# ---------------------------------------------------------------------------
# set_element_muted helper
# ---------------------------------------------------------------------------

def test_set_element_muted_toggles_surfaces(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    element = _glass_element(project)
    indices = element.resolve_surfaces(project.system)
    assert all(project.system.surfaces[i].is_active for i in indices)

    ok = element_actions.set_element_muted(project, element, True)
    assert ok is True
    assert all(not project.system.surfaces[i].is_active for i in indices)
    assert element.is_muted(project.system) is True


def test_set_element_muted_pushes_undo(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    element = _glass_element(project)
    before = project.can_undo

    element_actions.set_element_muted(project, element, True)
    assert project.can_undo is True
    assert project.undo_label == "Mute Element"

    # Undo restores active state
    project.undo()
    indices = [
        list(project.system.surface_ids).index(u) for u in element.surface_ids
    ]
    assert all(project.system.surfaces[i].is_active for i in indices)


def test_set_element_muted_no_op_aborts_edit(qapp, sample_lens_path):
    """Muting an already-muted element doesn't push an empty undo entry."""
    project = _load(qapp, sample_lens_path)
    element = _glass_element(project)
    element_actions.set_element_muted(project, element, True)
    undo_count = len(project._undo)

    # Calling again with the same desired state changes nothing — no undo push.
    ok = element_actions.set_element_muted(project, element, True)
    assert ok is False
    assert len(project._undo) == undo_count


def test_set_element_muted_refuses_stop(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    stop = _stop_element(project)
    ok = element_actions.set_element_muted(project, stop, True)
    assert ok is False
    assert stop.is_muted(project.system) is False


# ---------------------------------------------------------------------------
# build_tree picks the right icon
# ---------------------------------------------------------------------------

def test_build_tree_marks_muted_icon(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    element = _glass_element(project)
    element_actions.set_element_muted(project, element, True)

    root = build_tree(project.system)
    el_nodes = [c for c in root.children if isinstance(c, ElementNode)]
    target = next(n for n in el_nodes if n.element.element_id == element.element_id)
    assert target.muted is True
    assert target.icon_name == "node-element-muted"


def test_build_tree_default_icon_when_unmuted(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    root = build_tree(project.system)
    for node in root.children:
        if isinstance(node, ElementNode):
            assert node.muted is False
            expected = (
                "node-stop" if node.element.kind == ghostlight.ElementKind.STOP
                else "node-element"
            )
            assert node.icon_name == expected


def test_build_tree_stop_keeps_stop_icon_even_if_surface_inactive(
    qapp, sample_lens_path
):
    """STOP elements use the stop glyph regardless of muting state — the
    set_element_muted gate prevents muting in the first place, but build_tree
    is also defensive against direct surface mutations."""
    project = _load(qapp, sample_lens_path)
    stop = _stop_element(project)
    # Force-flip the underlying surface; build_tree should still pick stop glyph.
    for uuid in stop.surface_ids:
        idx = list(project.system.surface_ids).index(uuid)
        project.system.surfaces[idx].is_active = False

    root = build_tree(project.system)
    stop_node = next(
        n for n in root.children
        if isinstance(n, ElementNode)
        and n.element.element_id == stop.element_id
    )
    assert stop_node.icon_name == "node-stop"


# ---------------------------------------------------------------------------
# Model FontRole returns italic for muted elements
# ---------------------------------------------------------------------------

def test_model_font_role_italic_for_muted_element(qapp, sample_lens_path):
    from PySide6.QtCore import Qt

    project = _load(qapp, sample_lens_path)
    element = _glass_element(project)
    element_actions.set_element_muted(project, element, True)

    model = OpticalTreeModel(project)
    # Find the row for our element.
    row_count = model.rowCount()
    for r in range(row_count):
        idx = model.index(r, 0)
        node = idx.internalPointer()
        if (
            isinstance(node, ElementNode)
            and node.element.element_id == element.element_id
        ):
            font = model.data(idx, Qt.FontRole)
            assert font is not None
            assert font.italic() is True
            return
    raise AssertionError("muted element row not found in model")


def test_model_font_role_none_for_active_element(qapp, sample_lens_path):
    from PySide6.QtCore import Qt

    project = _load(qapp, sample_lens_path)
    model = OpticalTreeModel(project)
    row_count = model.rowCount()
    for r in range(row_count):
        idx = model.index(r, 0)
        node = idx.internalPointer()
        if isinstance(node, ElementNode) and not node.muted:
            font = model.data(idx, Qt.FontRole)
            assert font is None
            return
    raise AssertionError("no active element found in tree")
