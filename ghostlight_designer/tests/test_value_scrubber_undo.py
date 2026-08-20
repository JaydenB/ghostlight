from __future__ import annotations

import pytest
from PySide6.QtCore import QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QApplication, QTreeView

import ghostlight

from ghostlight_designer.optical_editor.columns import Column
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.value_scrubber import ScrubPopup
from ghostlight_designer.project import Project


def _first_sphere_radius_index(project: Project, model: OpticalTreeModel):
    for ei, el in enumerate(project.system.elements):
        for li, si in enumerate(el.resolve_surfaces(project.system)):
            if int(project.system.surfaces[si].form) != int(ghostlight.SurfaceForm.SPHERE):
                continue
            el_idx = model.index(ei, 0, QModelIndex())
            surface_row = len(el.material_glasses) + li
            return ei, si, model.index(surface_row, int(Column.RADIUS), el_idx)
    pytest.skip("no sphere-radius cell in sample lens")


def _make_popup(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    tree = QTreeView()
    tree.setModel(model)
    ei, si, radius_idx = _first_sphere_radius_index(project, model)
    popup = ScrubPopup(tree, radius_idx, QPoint(0, 0), project)
    return project, model, tree, popup, si


def test_single_drag_produces_one_undo_entry(qapp, sample_lens_path):
    project, model, tree, popup, si = _make_popup(qapp, sample_lens_path)
    try:
        original = project.system.surfaces[si].radius

        # Simulate a drag without going through real mouse events: arm a row,
        # call _begin_drag, fire several _write_value calls, end the drag.
        row = popup._rows[2]  # arbitrary sensitivity row
        popup._begin_drag(row, QPoint(0, 0))
        for v in (original + 1.0, original + 2.0, original + 3.0):
            popup._write_value(v)
        popup._end_drag()

        assert len(project._undo) == 1
        assert project.undo_label.startswith("Scrub Radius")
        project.undo()
        assert project.system.surfaces[si].radius == pytest.approx(original)
    finally:
        popup.close()
        tree.deleteLater()


def test_two_drags_in_same_popup_are_separate_entries(qapp, sample_lens_path):
    project, model, tree, popup, si = _make_popup(qapp, sample_lens_path)
    try:
        original = project.system.surfaces[si].radius
        row = popup._rows[2]

        popup._begin_drag(row, QPoint(0, 0))
        popup._write_value(original + 1.0)
        popup._end_drag()

        popup._begin_drag(row, QPoint(0, 0))
        popup._write_value(original + 2.0)
        popup._end_drag()

        assert len(project._undo) == 2
        project.undo()
        assert project.system.surfaces[si].radius == pytest.approx(original + 1.0)
        project.undo()
        assert project.system.surfaces[si].radius == pytest.approx(original)
    finally:
        popup.close()
        tree.deleteLater()


def test_mid_drag_mmb_repress_does_not_leak_cursor_or_compound(qapp, sample_lens_path):
    """A second ``_begin_drag`` while already dragging (e.g. user MMB-presses
    a different sensitivity row mid-drag) must NOT push a second override
    cursor or open a nested undo compound. Earlier behaviour leaked one
    override per repress, so a single release left the cursor blank — that's
    the "cursor disappears after let-go" symptom."""
    # Drain any pre-existing overrides so this test isn't fooled by them.
    while QApplication.overrideCursor() is not None:
        QApplication.restoreOverrideCursor()

    project, model, tree, popup, si = _make_popup(qapp, sample_lens_path)
    try:
        original = project.system.surfaces[si].radius
        row_a = popup._rows[2]
        row_b = popup._rows[3]

        popup._begin_drag(row_a, QPoint(0, 0))
        popup._write_value(original + 1.0)
        # Mid-drag: user presses MMB on a different sensitivity row to
        # change scrub speed. Must NOT re-enter the drag bookkeeping.
        popup._begin_drag(row_b, QPoint(0, 0))
        popup._write_value(original + 2.0)
        popup._end_drag()

        # Exactly one push/pop pair — no leaked override on the stack.
        assert QApplication.overrideCursor() is None
        # Exactly one compound on the project — no nesting.
        assert project._compound_depth == 0
        assert len(project._undo) == 1
        # New armed row is the one most recently pressed.
        assert popup._armed_row is row_b
    finally:
        popup.close()
        tree.deleteLater()


def test_close_during_drag_closes_compound(qapp, sample_lens_path):
    project, model, tree, popup, si = _make_popup(qapp, sample_lens_path)
    try:
        original = project.system.surfaces[si].radius
        row = popup._rows[2]
        popup._begin_drag(row, QPoint(0, 0))
        popup._write_value(original + 1.0)
        # Close while compound is still open — must not leak.
        popup.close()

        assert project._compound_depth == 0
        # One coalesced undo entry was created by close → end_compound.
        assert len(project._undo) == 1
        project.undo()
        assert project.system.surfaces[si].radius == pytest.approx(original)
    finally:
        tree.deleteLater()
