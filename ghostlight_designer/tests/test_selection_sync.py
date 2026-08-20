"""Cross-panel selection sync (whole element only)."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QModelIndex, QItemSelectionModel

import ghostlight

from ghostlight_designer.optical_editor import OpticalEditorBody
from ghostlight_designer.optical_editor.nodes import ElementNode
from ghostlight_designer.project import Project


def _collect(signal):
    received: list = []

    def slot(*args):
        received.append(args if len(args) != 1 else args[0])

    signal.connect(slot)
    return received


# ---------------------------------------------------------------------------
# Project-level API
# ---------------------------------------------------------------------------

def test_default_selection_is_none(qapp):
    p = Project()
    assert p.selected_element is None


def test_set_selected_element_emits_and_stores(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    received = _collect(p.selectionChanged)

    target = p.system.elements[0]
    p.set_selected_element(target)

    assert p.selected_element is target
    assert received == [target]


def test_set_selected_element_idempotent_no_emit(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    target = p.system.elements[0]
    p.set_selected_element(target)

    received = _collect(p.selectionChanged)
    p.set_selected_element(target)
    assert received == []


def test_set_selected_element_foreign_clears(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    target = p.system.elements[0]
    p.set_selected_element(target)
    received = _collect(p.selectionChanged)

    foreign = ghostlight.Element(name="ghost", surface_ids=[])
    p.set_selected_element(foreign)

    assert p.selected_element is None
    assert received == [None]


def test_load_clears_selection(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    target = p.system.elements[0]
    p.set_selected_element(target)

    received = _collect(p.selectionChanged)
    p.load(str(sample_lens_path))
    assert p.selected_element is None
    assert received == [None]


def test_new_clears_selection(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    p.set_selected_element(p.system.elements[0])

    received = _collect(p.selectionChanged)
    p.new()
    assert p.selected_element is None
    assert received == [None]


def test_undo_preserves_selection_when_target_survives(qapp, sample_lens_path):
    """Undo re-resolves the selected element by ``element_id`` against the
    restored system. As long as the element still exists, selection stays
    put so the user can see what changed without losing their A/B context.
    Lost selection on every Ctrl+Z was the original behaviour that made
    radius edits visually confusing — the row + viewport highlight would
    vanish on undo, masking the geometry change."""
    p = Project()
    p.load(str(sample_lens_path))
    with p.edit("Tweak thickness"):
        p.system.surfaces[0].thickness = float(p.system.surfaces[0].thickness) + 0.1

    selected_id = p.system.elements[0].element_id
    p.set_selected_element(p.system.elements[0])

    received = _collect(p.selectionChanged)
    p.undo()

    # Selection survives the reload because the element_id still exists in
    # the restored system. The wrapper is fresh (reload replaces _elements)
    # but identity-wise it's the same picked row.
    assert p.selected_element is not None
    assert p.selected_element.element_id == selected_id
    # No selection-changed signal fires because the *value* didn't change
    # (same element_id resolves to a wrapper we then store) — we only emit
    # when the resolved instance differs from what was already set.
    # In practice ``_resolve_selection_handles`` sees a different Python
    # object so it does emit; that's fine, but the value carries the live
    # post-undo wrapper, not None.
    if received:
        assert all(el is not None for el in received), \
            f"undo must not clear selection while target still exists, got {received}"


def test_undo_radius_edit_preserves_element_and_surface_selection(
    qapp, sample_lens_path,
):
    """The user's reported workflow: select a surface, edit its radius
    through the tree, then Ctrl+Z. Both the element AND the surface
    selection should survive — losing them was making the geometry
    change appear to 'fuck up the viewport' (highlight gone, no
    visual anchor to compare against)."""
    from PySide6.QtCore import QModelIndex, Qt
    from ghostlight_designer.optical_editor.model import OpticalTreeModel
    from ghostlight_designer.optical_editor.columns import Column

    p = Project()
    p.load(str(sample_lens_path))
    model = OpticalTreeModel(p)

    el = p.system.elements[0]
    si = el.resolve_surfaces(p.system)[0]
    selected_element_id = el.element_id
    selected_surface_uuid = p.system.surface_ids[si]
    p.set_selection(el, si)

    # Edit the radius (model.setData mirrors what the tree does).
    el_idx = model.index(0, 0, QModelIndex())
    surface_row = len(el.material_glasses)
    radius_idx = model.index(surface_row, int(Column.RADIUS), el_idx)
    model.setData(radius_idx, 100.0, Qt.EditRole)
    assert p.system.surfaces[si].radius == pytest.approx(100.0)

    p.undo()

    # Radius reverted.
    assert p.system.surfaces[si].radius == pytest.approx(47.07, abs=1e-2)
    # Selection survives.
    assert p.selected_element is not None
    assert p.selected_element.element_id == selected_element_id
    assert p.selected_surface_index is not None
    new_si = p.selected_surface_index
    assert p.system.surface_ids[new_si] == selected_surface_uuid


def test_undo_clears_selection_when_target_vanishes(qapp, sample_lens_path):
    """If the selected element no longer exists in the restored state
    (e.g. you added an element, selected it, then undo removed it),
    selection naturally falls back to None."""
    from ghostlight_designer.optical_editor import element_actions

    p = Project()
    p.load(str(sample_lens_path))
    new_el = element_actions.add_singlet(p)
    p.set_selected_element(new_el)
    assert p.selected_element is new_el

    p.undo()
    # The just-added element is gone from the restored system → selection
    # clears.
    assert p.selected_element is None


# ---------------------------------------------------------------------------
# Viewport panel sync
# ---------------------------------------------------------------------------

def _make_viewport_body(project, monkeypatch):
    """ViewportPanelBody with the GL-touching methods stubbed."""
    from ghostlight_viewport import LensViewport
    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_trace_results", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "clear_trace_results", lambda *a, **k: None)
    from ghostlight_designer.viewport_panel import ViewportPanelBody
    return ViewportPanelBody(project)


def test_viewport_click_writes_to_project(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    body = _make_viewport_body(project, monkeypatch)
    try:
        target = project.system.elements[0]
        # Simulate the picking handler's emit.
        body.viewport.elementSelected.emit(target)
        assert project.selected_element is target
    finally:
        body.deleteLater()


def test_project_selection_pushes_into_viewport(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    body = _make_viewport_body(project, monkeypatch)
    try:
        target = project.system.elements[0]
        project.set_selected_element(target)
        assert body.viewport.selection.element is target
        project.set_selected_element(None)
        assert body.viewport.selection.element is None
    finally:
        body.deleteLater()


def test_viewport_set_selected_element_does_not_re_emit(qapp, sample_lens_path, monkeypatch):
    """Programmatic setter must not fire elementSelected (would loop)."""
    from ghostlight_viewport import LensViewport
    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    project = Project()
    project.load(str(sample_lens_path))
    vp = LensViewport()
    try:
        emitted: list = []
        vp.elementSelected.connect(lambda el: emitted.append(el))
        vp.set_selected_element(project.system.elements[0])
        vp.set_selected_element(None)
        assert emitted == []
    finally:
        vp.deleteLater()


# ---------------------------------------------------------------------------
# Optical editor panel sync
# ---------------------------------------------------------------------------

def _find_element_row_index(body: OpticalEditorBody, element) -> QModelIndex:
    root_count = body.model.rowCount(QModelIndex())
    for ei in range(root_count):
        idx = body.model.index(ei, 0, QModelIndex())
        node = idx.internalPointer()
        if isinstance(node, ElementNode) and node.element is element:
            return idx
    raise AssertionError("element not found in tree")


def test_editor_click_on_element_row_updates_project(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        target = project.system.elements[0]
        el_idx = _find_element_row_index(body, target)
        body.tree.selectionModel().setCurrentIndex(
            el_idx,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        assert project.selected_element is target
    finally:
        body.deleteLater()


def test_editor_click_on_surface_row_selects_owning_element(qapp, sample_lens_path):
    """Clicking a Surface (or any sub-row) selects the parent Element."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        target = project.system.elements[0]
        el_idx = _find_element_row_index(body, target)
        # First child of any Element row that isn't an ElementNode itself
        # — could be a MaterialNode or SurfaceNode; both should resolve up.
        child_idx = body.model.index(0, 0, el_idx)
        assert child_idx.isValid()
        body.tree.selectionModel().setCurrentIndex(
            child_idx,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        assert project.selected_element is target
    finally:
        body.deleteLater()


def test_project_selection_updates_editor_current_row(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        target = project.system.elements[0]
        project.set_selected_element(target)
        current = body.tree.selectionModel().currentIndex()
        assert current.isValid()
        node = current.internalPointer()
        # Walk up to ElementNode (same logic as the body).
        while node is not None and not isinstance(node, ElementNode):
            node = node.parent
        assert isinstance(node, ElementNode)
        assert node.element is target
    finally:
        body.deleteLater()


def test_project_selection_clears_editor_row(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        project.set_selected_element(project.system.elements[0])
        project.set_selected_element(None)
        current = body.tree.selectionModel().currentIndex()
        assert not current.isValid()
    finally:
        body.deleteLater()


def test_two_editors_stay_in_sync(qapp, sample_lens_path):
    """A click in one editor pulls the same selection into a second editor."""
    project = Project()
    project.load(str(sample_lens_path))
    a = OpticalEditorBody(project)
    b = OpticalEditorBody(project)
    try:
        target = project.system.elements[0]
        el_idx_a = _find_element_row_index(a, target)
        a.tree.selectionModel().setCurrentIndex(
            el_idx_a,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        # b should now reflect the same Element as current.
        current_b = b.tree.selectionModel().currentIndex()
        assert current_b.isValid()
        node_b = current_b.internalPointer()
        while node_b is not None and not isinstance(node_b, ElementNode):
            node_b = node_b.parent
        assert isinstance(node_b, ElementNode)
        assert node_b.element is target
    finally:
        a.deleteLater()
        b.deleteLater()


def test_scrub_trigger_selects_dragged_surface(qapp, sample_lens_path):
    """Repro: open a scrub popup over a Surface row that *isn't* the
    current selection. The trigger must surface the scrubbed row in the
    tree's selection before opening the popup, so the post-drag rebuild
    (driven by ``end_compound``'s ``mark_modified``) lands the user back
    on the surface they were editing — not on the parent element row."""
    from PySide6.QtCore import QPoint
    from ghostlight_designer.optical_editor.columns import Column
    from ghostlight_designer.optical_editor.nodes import SurfaceNode

    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        # Pick a surface row whose Radius cell is scrubbable.
        target = project.system.elements[0]
        el_idx = _find_element_row_index(body, target)
        # First child past the materials is the front surface.
        surface_row = len(target.material_glasses)
        radius_idx = body.model.index(surface_row, int(Column.RADIUS), el_idx)
        assert radius_idx.isValid()
        node = body.model.index(surface_row, 0, el_idx).internalPointer()
        assert isinstance(node, SurfaceNode)

        # Force the selection elsewhere first so the popup-open has to
        # actually move it. Selecting the element row clears surface.
        project.set_selected_element(target)
        project.set_selected_surface_index(None)
        assert project.selected_surface_index is None

        # Open via the trigger's helper — exercises the real code path
        # without synthesising a Ctrl+MMB QMouseEvent.
        popup = body._scrub_trigger._open_popup(radius_idx, QPoint(0, 0))
        try:
            # The trigger moved the tree's current row to the scrubbed
            # cell, which pushed surface_index into the project. After
            # the inevitable post-drag model rebuild, _on_model_reset
            # will restore from this stored surface_index.
            assert project.selected_surface_index == node.surface_index
        finally:
            if popup is not None:
                popup.close()
    finally:
        body.deleteLater()


def test_editor_and_viewport_round_trip(qapp, sample_lens_path, monkeypatch):
    """Picking in the viewport highlights the editor row, and vice versa."""
    project = Project()
    project.load(str(sample_lens_path))
    editor = OpticalEditorBody(project)
    vp_body = _make_viewport_body(project, monkeypatch)
    try:
        target = project.system.elements[0]
        # Viewport → editor.
        vp_body.viewport.elementSelected.emit(target)
        current = editor.tree.selectionModel().currentIndex()
        assert current.isValid()
        node = current.internalPointer()
        while node is not None and not isinstance(node, ElementNode):
            node = node.parent
        assert isinstance(node, ElementNode)
        assert node.element is target

        # Editor → viewport.
        other = project.system.elements[-1] if len(project.system.elements) > 1 else target
        if other is target:
            pytest.skip("sample lens only has one element")
        idx = _find_element_row_index(editor, other)
        editor.tree.selectionModel().setCurrentIndex(
            idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        assert vp_body.viewport.selection.element is other
    finally:
        editor.deleteLater()
        vp_body.deleteLater()
