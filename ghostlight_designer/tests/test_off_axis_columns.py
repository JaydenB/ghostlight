"""Off-axis element placement in the optical design editor.

Covers the three things that can independently break:

* the column strip — new canonical columns exist, element rows fill them,
  everything else stays blank there;
* the ">>>" reveal — per-element view state that must NOT reach the undo
  stack, and must lock open once a row holds a non-zero value;
* the edit path — writes land on ``ghostlight.Element``, get baked down to the
  surfaces immediately, and survive save / undo / redo.

The undo tests are the load-bearing ones. Project snapshots round-trip through
``writer`` → temp .lens → the C++ loader, so a field the writer forgets is
destroyed by the first Ctrl+Z rather than by anything the UI does.
"""
from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QModelIndex, Qt

import ghostlight

from ghostlight_designer.optical_editor import OpticalEditorBody
from ghostlight_designer.optical_editor import row_schemas as schemas
from ghostlight_designer.optical_editor.columns import Column
from ghostlight_designer.optical_editor.delegates import SlotRole
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.optical_editor.row_schemas import SlotEditor
from ghostlight_designer.project import Project


OFF_AXIS_COLUMNS = {
    "pos_x": Column.POS_X,
    "pos_y": Column.POS_Y,
    "rot_x": Column.ROT_X,
    "rot_y": Column.ROT_Y,
    "rot_z": Column.ROT_Z,
    "pivot_x": Column.PIVOT_X,
    "pivot_y": Column.PIVOT_Y,
    "pivot_z": Column.PIVOT_Z,
}


def _loaded(sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    return project


def _element_row(model: OpticalTreeModel, row: int = 0) -> QModelIndex:
    return model.index(row, 0, QModelIndex())


def _first_glass_row(project: Project) -> int:
    for i, el in enumerate(project.system.elements):
        if el.kind == ghostlight.ElementKind.GLASS:
            return i
    raise AssertionError("sample lens has no GLASS element")


def _surface_poses(project: Project) -> list[tuple]:
    return [
        (float(s.decenter_x), float(s.decenter_y), float(s.z),
         tuple(float(v) for v in s.rot))
        for s in project.system.surfaces
    ]


# ---------------------------------------------------------------------------
# Column strip
# ---------------------------------------------------------------------------

def test_canonical_columns_cover_the_off_axis_block():
    headers = {col: label for col, label, _keys in schemas.CANONICAL_COLUMNS}
    assert headers[Column.OFF_AXIS] == "Off Axis"
    assert headers[Column.POS_X] == "Pos X"
    assert headers[Column.POS_Y] == "Pos Y"
    assert headers[Column.ROT_X] == "Rot X"
    assert headers[Column.ROT_Y] == "Rot Y"
    assert headers[Column.ROT_Z] == "Rot Z"
    assert headers[Column.PIVOT_X] == "Pivot X"
    assert headers[Column.PIVOT_Y] == "Pivot Y"
    assert headers[Column.PIVOT_Z] == "Pivot Z"


def test_off_axis_keys_resolve_to_their_columns():
    for key, column in OFF_AXIS_COLUMNS.items():
        assert schemas.canonical_column_for(key) == int(column)
    # The toggle itself is NOT part of the collapsible block — it has to stay
    # visible or there'd be nothing to click to expand.
    assert "off_axis" not in schemas.OFF_AXIS_COLUMN_KEYS
    assert len(schemas.OFF_AXIS_COLUMN_KEYS) == 8


def test_element_rows_expose_every_off_axis_slot(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)

    for key, column in OFF_AXIS_COLUMNS.items():
        idx = model.index(row, int(column), QModelIndex())
        slot = idx.data(SlotRole)
        assert slot is not None, f"element row has no slot in {key}"
        assert slot.key == key
        assert slot.editor == SlotEditor.FLOAT
        assert bool(model.flags(idx) & Qt.ItemIsEditable)


def test_off_axis_slots_are_scrubbable_but_not_optimizer_variables(
    qapp, sample_lens_path
):
    """Scrubbing yes; the amber variable stripe no.

    ``variable_attr`` names an ``ghostlight.Surface`` attribute for the optimizer to
    tune. Element pose isn't one, so setting it would paint a stripe on a cell
    the optimizer can't actually drive.
    """
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)

    for column in OFF_AXIS_COLUMNS.values():
        slot = model.index(row, int(column), QModelIndex()).data(SlotRole)
        assert slot.options.get("scrubbable") is True
        assert "variable_attr" not in slot.options


def test_non_element_rows_are_blank_in_the_off_axis_block(qapp, sample_lens_path):
    """Surfaces / materials / form rows declare no slots out there."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    el_idx = _element_row(model, _first_glass_row(project))

    checked = 0
    for child_row in range(model.rowCount(el_idx)):
        for column in OFF_AXIS_COLUMNS.values():
            idx = model.index(child_row, int(column), el_idx)
            assert idx.data(SlotRole) is None
            assert idx.data(Qt.DisplayRole) == ""
            assert not (model.flags(idx) & Qt.ItemIsEditable)
            checked += 1
    assert checked > 0, "element had no child rows to check"


def test_asphere_columns_still_follow_the_canonical_strip(qapp, sample_lens_path):
    """Widening the strip must push the dynamic trailing columns right, not
    collide with them."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    assert model.columnCount() >= len(schemas.CANONICAL_COLUMNS)
    # Everything past the canonical strip is an unlabelled asphere column.
    for column in range(len(schemas.CANONICAL_COLUMNS), model.columnCount()):
        assert schemas.header_text(column) == ""


# ---------------------------------------------------------------------------
# The ">>>" toggle
# ---------------------------------------------------------------------------

def test_toggle_cell_is_a_button_and_not_editable(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    idx = model.index(_first_glass_row(project), int(Column.OFF_AXIS), QModelIndex())
    slot = idx.data(SlotRole)
    assert slot is not None
    assert slot.editor == SlotEditor.BUTTON
    assert slot.options.get("glyph") == ">>>"
    # A BUTTON cell has no editor widget — it must never enter edit mode.
    assert not (model.flags(idx) & Qt.ItemIsEditable)


def test_toggle_starts_collapsed_and_flips(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    idx = model.index(row, int(Column.OFF_AXIS), QModelIndex())

    assert model.button_state(idx) == (False, False)
    assert model.any_off_axis_revealed() is False

    model.toggle_button(idx)
    assert model.button_state(idx) == (True, False)
    assert model.any_off_axis_revealed() is True

    model.toggle_button(idx)
    assert model.button_state(idx) == (False, False)
    assert model.any_off_axis_revealed() is False


def test_toggle_does_not_create_an_undo_entry(qapp, sample_lens_path):
    """Reveal state is view state. Ctrl+Z must undo lens edits, not columns."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    idx = model.index(_first_glass_row(project), int(Column.OFF_AXIS), QModelIndex())

    assert project.can_undo is False
    model.toggle_button(idx)
    assert project.can_undo is False, "toggling the reveal pushed an undo entry"
    assert project.is_dirty is False, "toggling the reveal dirtied the document"


def test_a_non_zero_value_locks_the_toggle_open(qapp, sample_lens_path):
    """Off-axis data can never be hidden behind a collapsed
    column."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    toggle = model.index(row, int(Column.OFF_AXIS), QModelIndex())

    assert model.button_state(toggle) == (False, False)

    model.setData(model.index(row, int(Column.POS_Y), QModelIndex()), 1.25, Qt.EditRole)

    assert model.button_state(toggle) == (True, True), "row didn't lock open"
    assert model.any_off_axis_revealed() is True
    # ...and clicking does nothing while it's locked.
    model.toggle_button(toggle)
    assert model.button_state(toggle) == (True, True)

    # Zero it again and the row is collapsible once more.
    model.setData(model.index(row, int(Column.POS_Y), QModelIndex()), 0.0, Qt.EditRole)
    assert model.button_state(toggle) == (False, False)


def test_reveal_is_per_element_but_columns_are_global(qapp, sample_lens_path):
    """One revealed row shows the columns for the whole tree; the others just
    render blank there. That's a property of a column view."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    if model.rowCount(QModelIndex()) < 2:
        pytest.skip("sample lens has only one element")

    first = model.index(0, int(Column.OFF_AXIS), QModelIndex())
    second = model.index(1, int(Column.OFF_AXIS), QModelIndex())

    model.toggle_button(first)
    assert model.button_state(first)[0] is True
    assert model.button_state(second)[0] is False
    assert model.any_off_axis_revealed() is True


def test_reveal_state_clears_on_load(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    model.toggle_button(
        model.index(_first_glass_row(project), int(Column.OFF_AXIS), QModelIndex())
    )
    assert model.any_off_axis_revealed() is True

    project.load(str(sample_lens_path))
    assert model.any_off_axis_revealed() is False


def test_body_hides_the_block_until_revealed(qapp, sample_lens_path):
    """The actual view behaviour, not just the model's opinion of it."""
    project = _loaded(sample_lens_path)
    body = OpticalEditorBody(project)
    try:
        for column in OFF_AXIS_COLUMNS.values():
            assert body.tree.isColumnHidden(int(column)) is True
        # The toggle column itself always stays visible.
        assert body.tree.isColumnHidden(int(Column.OFF_AXIS)) is False

        body.model.toggle_button(
            body.model.index(
                _first_glass_row(project), int(Column.OFF_AXIS), QModelIndex()
            )
        )
        for column in OFF_AXIS_COLUMNS.values():
            assert body.tree.isColumnHidden(int(column)) is False
            assert body.tree.columnWidth(int(column)) > 0

        body.model.toggle_button(
            body.model.index(
                _first_glass_row(project), int(Column.OFF_AXIS), QModelIndex()
            )
        )
        for column in OFF_AXIS_COLUMNS.values():
            assert body.tree.isColumnHidden(int(column)) is True
    finally:
        body.deleteLater()


def test_body_reveals_the_block_for_a_lens_that_already_has_values(
    qapp, sample_lens_path
):
    """Opening a file with decenter already in it must show it, with no click."""
    project = _loaded(sample_lens_path)
    row = _first_glass_row(project)
    element = project.system.elements[row]
    x, y, z = element.position
    element.position = (x, y + 2.0, z)

    body = OpticalEditorBody(project)
    try:
        for column in OFF_AXIS_COLUMNS.values():
            assert body.tree.isColumnHidden(int(column)) is False
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,column,value", [
    ("pos_x", Column.POS_X, 1.5),
    ("pos_y", Column.POS_Y, -2.25),
    ("rot_x", Column.ROT_X, 3.0),
    ("rot_y", Column.ROT_Y, -4.5),
    ("rot_z", Column.ROT_Z, 30.0),
    ("pivot_x", Column.PIVOT_X, 0.5),
    ("pivot_y", Column.PIVOT_Y, -0.75),
    ("pivot_z", Column.PIVOT_Z, 2.0),
])
def test_writes_land_on_the_element(qapp, sample_lens_path, key, column, value):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    idx = model.index(row, int(column), QModelIndex())

    assert model.setData(idx, value, Qt.EditRole) is True
    element = project.system.elements[row]
    assert schemas.element_pose_value(element, key) == pytest.approx(value)
    assert idx.data(Qt.EditRole) == pytest.approx(value)


def test_rotation_columns_display_degrees(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    idx = model.index(row, int(Column.ROT_Y), QModelIndex())
    model.setData(idx, 3.5, Qt.EditRole)
    assert idx.data(Qt.DisplayRole) == "3.500°"


def test_a_write_rebakes_the_surface_poses(qapp, sample_lens_path):
    """The whole point of the live bake: the geometry moves now, not on save."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    element = project.system.elements[row]
    indices = element.resolve_surfaces(project.system)

    before = [float(project.system.surfaces[i].decenter_y) for i in indices]
    assert all(v == 0.0 for v in before)

    model.setData(
        model.index(row, int(Column.POS_Y), QModelIndex()), 2.0, Qt.EditRole
    )

    after = [float(project.system.surfaces[i].decenter_y) for i in indices]
    assert all(v == pytest.approx(2.0, abs=1e-5) for v in after), after


def test_a_tilt_swings_the_rear_surface_out(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    element = project.system.elements[row]
    indices = element.resolve_surfaces(project.system)
    if len(indices) < 2:
        pytest.skip("first glass element is a singleton surface")

    model.setData(
        model.index(row, int(Column.ROT_Y), QModelIndex()), 5.0, Qt.EditRole
    )

    surfaces = project.system.surfaces
    # Front vertex is the default centre of rotation, so it stays put...
    assert float(surfaces[indices[0]].decenter_x) == pytest.approx(0.0, abs=1e-5)
    # ...and the rear surface picks up real lateral offset.
    assert abs(float(surfaces[indices[-1]].decenter_x)) > 1e-3
    # The surface frame itself rotated too.
    assert float(surfaces[indices[0]].rot[0]) != pytest.approx(1.0, abs=1e-6)


def test_pivot_moves_the_centre_of_rotation(qapp, sample_lens_path):
    """Same tilt, pivot on the back vertex — now the BACK is what holds still."""
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    element = project.system.elements[row]
    indices = element.resolve_surfaces(project.system)
    if len(indices) < 2:
        pytest.skip("first glass element is a singleton surface")

    surfaces = project.system.surfaces
    rear_before = float(surfaces[indices[-1]].decenter_x)
    thickness = sum(float(surfaces[i].thickness) for i in indices[:-1])

    model.setData(
        model.index(row, int(Column.PIVOT_Z), QModelIndex()), thickness, Qt.EditRole
    )
    model.setData(
        model.index(row, int(Column.ROT_Y), QModelIndex()), 5.0, Qt.EditRole
    )

    assert float(surfaces[indices[-1]].decenter_x) == pytest.approx(
        rear_before, abs=1e-4
    )
    assert abs(float(surfaces[indices[0]].decenter_x)) > 1e-3


def test_a_no_op_write_is_rejected(qapp, sample_lens_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)
    idx = model.index(row, int(Column.POS_X), QModelIndex())
    assert model.setData(idx, 0.0, Qt.EditRole) is False
    assert project.can_undo is False


# ---------------------------------------------------------------------------
# Persistence — the part that silently breaks
# ---------------------------------------------------------------------------

def test_off_axis_values_survive_undo_and_redo(qapp, sample_lens_path):
    """Undo snapshots go writer -> .lens -> C++ loader.

    If the writer ever stops emitting one of these fields, this is the test
    that notices: the value comes back as zero after a redo.
    """
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)

    edits = {
        Column.POS_X: 1.25,
        Column.POS_Y: -0.5,
        Column.ROT_X: 2.0,
        Column.ROT_Y: -3.0,
        Column.ROT_Z: 15.0,
        Column.PIVOT_X: 0.25,
        Column.PIVOT_Y: -0.75,
        Column.PIVOT_Z: 1.5,
    }
    for column, value in edits.items():
        model.setData(model.index(row, int(column), QModelIndex()), value, Qt.EditRole)

    def current() -> dict:
        return {
            column: float(
                model.index(row, int(column), QModelIndex()).data(Qt.EditRole)
            )
            for column in edits
        }

    assert current() == pytest.approx(
        {c: v for c, v in edits.items()}, abs=1e-5
    )

    for _ in edits:
        project.undo()
    assert current() == pytest.approx({c: 0.0 for c in edits}, abs=1e-9)

    for _ in edits:
        project.redo()
    assert current() == pytest.approx({c: v for c, v in edits.items()}, abs=1e-5)


def test_off_axis_values_survive_a_save_round_trip(qapp, sample_lens_path, tmp_path):
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)

    model.setData(model.index(row, int(Column.POS_X), QModelIndex()), 2.0, Qt.EditRole)
    model.setData(model.index(row, int(Column.ROT_Y), QModelIndex()), 4.0, Qt.EditRole)
    model.setData(model.index(row, int(Column.PIVOT_Z), QModelIndex()), 3.0, Qt.EditRole)
    poses = _surface_poses(project)

    out = tmp_path / "saved.lens"
    project.save_as(str(out))

    reopened = Project()
    reopened.load(str(out))
    element = reopened.system.elements[row]
    assert element.position[0] == pytest.approx(2.0, abs=1e-5)
    assert element.rotation_euler_deg[1] == pytest.approx(4.0, abs=1e-5)
    assert element.pivot[2] == pytest.approx(3.0, abs=1e-5)

    # And the geometry the live bake produced matches what the C++ loader
    # reconstructs from the file — the drift guard, at the designer level.
    for got, want in zip(_surface_poses(reopened), poses):
        assert got[0] == pytest.approx(want[0], abs=2e-4)
        assert got[1] == pytest.approx(want[1], abs=2e-4)
        assert got[2] == pytest.approx(want[2], abs=2e-4)
        for k in range(9):
            assert got[3][k] == pytest.approx(want[3][k], abs=2e-4)


def test_an_unrelated_edit_does_not_disturb_off_axis_values(qapp, sample_lens_path):
    """Editing a surface radius must not quietly re-zero the element pose.

    ``finalize()`` runs on spacing edits and only relays z; this is the guard
    that it stays that way.
    """
    project = _loaded(sample_lens_path)
    model = OpticalTreeModel(project)
    row = _first_glass_row(project)

    model.setData(model.index(row, int(Column.POS_Y), QModelIndex()), 1.5, Qt.EditRole)
    el_idx = _element_row(model, row)

    # Find this element's first surface row and nudge its radius.
    for child_row in range(model.rowCount(el_idx)):
        idx = model.index(child_row, int(Column.RADIUS), el_idx)
        slot = idx.data(SlotRole)
        if slot is not None and slot.key == "radius" and idx.data(Qt.EditRole):
            model.setData(idx, float(idx.data(Qt.EditRole)) + 1.0, Qt.EditRole)
            break
    else:
        pytest.skip("no editable radius cell found")

    element = project.system.elements[row]
    assert element.position[1] == pytest.approx(1.5, abs=1e-6)
    indices = element.resolve_surfaces(project.system)
    assert float(project.system.surfaces[indices[0]].decenter_y) == pytest.approx(
        1.5, abs=1e-4
    )
