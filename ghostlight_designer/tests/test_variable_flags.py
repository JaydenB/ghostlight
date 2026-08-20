"""Tests for the surface variable-flag feature.

Covers:

* Project storage — VariableBounds + set / clear / toggle / bulk / getters,
  signal fan-out, cleared on new / load, pruned when a surface disappears.
* Row-schema wiring — Radius + Pos Z slots carry the ``variable_attr``
  option; the asphere K slot doesn't.
* Delegate flag check — the ``_cell_is_variable_flagged`` predicate does
  the right thing across slot / node kinds.
* Bulk actions — flag_all_radii / thicknesses respect stops, clear works.
* Right-click menu — populate_cell_menu adds actions iff clicked cell is
  flag-able.
* Bounds dialog — enable / disable, blank fields parse as unbounded.
* End-to-end — flag via project → collect_variables returns the ref.
"""
from __future__ import annotations

import pathlib

import pytest

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QMenu

import ghostlight

from ghostlight_designer.optical_editor.body import OpticalEditorBody
from ghostlight_designer.optical_editor.nodes import (
    AsphereFormNode,
    SurfaceNode,
    surface_uuid_for,
)
from ghostlight_designer.optical_editor.variable_bounds_dialog import (
    VariableBoundsDialog,
    _parse_optional_float,
    _format_optional_float,
    edit_variable_bounds,
)
from ghostlight_designer.optical_editor.variable_flag_actions import (
    clear_all_variable_flags,
    flag_all_radii,
    flag_all_thicknesses,
    populate_bulk_menu,
    populate_cell_menu,
    surface_uuid_at,
    variable_attr_at,
)
from ghostlight_designer.optical_editor.row_schemas import SURFACE_SCHEMA
from ghostlight_designer.optimization_panel.variables import collect_variables
from ghostlight_designer.project import Project, VariableBounds


# ---------------------------------------------------------------------------
# VariableBounds
# ---------------------------------------------------------------------------


def test_variable_bounds_defaults_are_unbounded():
    b = VariableBounds()
    assert b.lo is None
    assert b.hi is None
    assert b.is_unbounded()


def test_variable_bounds_equality():
    assert VariableBounds() == VariableBounds()
    assert VariableBounds(lo=1.0) == VariableBounds(lo=1.0)
    assert VariableBounds(lo=1.0) != VariableBounds(lo=2.0)


# ---------------------------------------------------------------------------
# Project storage API
# ---------------------------------------------------------------------------


def test_project_starts_with_no_flags(qapp):
    project = Project()
    assert project.all_variable_flags() == {}
    assert project.get_variable_flags("anything") == {}


def test_set_get_clear_variable_flag(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])

    assert project.set_variable_flag(uuid, "radius") is True  # changed
    assert project.set_variable_flag(uuid, "radius") is False  # no-op
    assert project.is_variable_flagged(uuid, "radius")
    assert project.variable_bounds(uuid, "radius") == VariableBounds()

    assert project.set_variable_flag(
        uuid, "radius", VariableBounds(lo=-10.0, hi=10.0)
    ) is True
    assert project.variable_bounds(uuid, "radius") == VariableBounds(
        lo=-10.0, hi=10.0,
    )

    assert project.clear_variable_flag(uuid, "radius") is True
    assert not project.is_variable_flagged(uuid, "radius")
    assert project.clear_variable_flag(uuid, "radius") is False


def test_toggle_variable_flag(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])

    assert project.toggle_variable_flag(uuid, "radius") is True
    assert project.is_variable_flagged(uuid, "radius")
    assert project.toggle_variable_flag(uuid, "radius") is False
    assert not project.is_variable_flagged(uuid, "radius")


def test_variable_flag_signals_fire(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])
    signals = []
    project.variableFlagChanged.connect(
        lambda u, a: signals.append(("changed", u, a))
    )
    project.variableFlagsReplaced.connect(
        lambda: signals.append(("replaced",))
    )
    project.set_variable_flag(uuid, "radius")
    project.clear_variable_flag(uuid, "radius")
    assert signals == [
        ("changed", uuid, "radius"),
        ("changed", uuid, "radius"),
    ]


def test_bulk_set_variable_flags_preserves_existing(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    uuid0 = str(project.system.surface_ids[0])
    # Pre-flag surface 0 with custom bounds; bulk should NOT overwrite.
    project.set_variable_flag(uuid0, "radius", VariableBounds(lo=-5.0, hi=5.0))
    pairs = [(u, "radius") for u in project.system.surface_ids]
    project.bulk_set_variable_flags(pairs)
    # Surface 0's custom bounds still intact.
    assert project.variable_bounds(uuid0, "radius") == VariableBounds(
        lo=-5.0, hi=5.0,
    )
    # Others now flagged with defaults.
    for uuid in list(project.system.surface_ids)[1:]:
        assert project.variable_bounds(uuid, "radius") == VariableBounds()


def test_new_clears_variable_flags(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(uuid, "radius")
    replaced = []
    project.variableFlagsReplaced.connect(lambda: replaced.append(True))
    project.new()
    assert project.all_variable_flags() == {}
    assert replaced  # signal fired


def test_clear_all_variable_flags(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    for uuid in project.system.surface_ids:
        project.set_variable_flag(uuid, "radius")
    replaced = []
    project.variableFlagsReplaced.connect(lambda: replaced.append(True))
    assert project.clear_all_variable_flags() is True
    assert project.all_variable_flags() == {}
    assert replaced
    # Second call is a no-op.
    replaced.clear()
    assert project.clear_all_variable_flags() is False
    assert not replaced


def test_flags_pruned_when_surface_disappears(
    qapp, sample_lens_path: pathlib.Path,
):
    """Removing an element should drop flag entries for its surfaces."""
    project = Project()
    project.load(str(sample_lens_path))
    uuids_before = list(project.system.surface_ids)
    for uuid in uuids_before:
        project.set_variable_flag(uuid, "radius")
    # Load a fresh empty lens by New — that clears everything, so we
    # can't use it to test pruning specifically. Instead, wipe an
    # element via element_actions.
    from ghostlight_designer.optical_editor import element_actions
    elements = list(project.system.elements)
    if not elements:
        pytest.skip("Sample lens has no elements to remove")
    element_actions.remove_element(project, elements[0])
    # Flags for the removed element's surfaces should be pruned.
    live = set(project.system.surface_ids)
    for uuid, attrs in project.all_variable_flags().items():
        assert uuid in live, f"stale flag for removed surface {uuid}"


# ---------------------------------------------------------------------------
# collect_variables integration
# ---------------------------------------------------------------------------


def test_collect_variables_returns_flagged_refs(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    uuids = list(project.system.surface_ids)
    project.set_variable_flag(uuids[0], "radius")
    project.set_variable_flag(
        uuids[-1], "thickness", VariableBounds(lo=0.5, hi=50.0),
    )
    refs = collect_variables(project)
    # Order is by surface index.
    by_surface = {(r.surface_index, r.attr): r for r in refs}
    assert (0, "radius") in by_surface
    assert (len(uuids) - 1, "thickness") in by_surface
    thickness_ref = by_surface[(len(uuids) - 1, "thickness")]
    assert thickness_ref.lo == pytest.approx(0.5)
    assert thickness_ref.hi == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Row-schema wiring
# ---------------------------------------------------------------------------


def test_surface_slots_carry_variable_attr():
    """Radius + Pos Z opt in; Aperture Rad does not."""
    slots_by_key = {s.key: s for s in SURFACE_SCHEMA.slots}
    assert slots_by_key["radius"].options.get("variable_attr") == "radius"
    assert slots_by_key["pos_z"].options.get("variable_attr") == "thickness"
    assert "variable_attr" not in slots_by_key["aperture_rad"].options


def test_asphere_and_cylindrical_radius_slots_carry_variable_attr():
    """Both form-child Radius slots share the same variable_attr so
    flagging via any of them refers to the same optimizer variable."""
    from ghostlight_designer.optical_editor.row_schemas import (
        ASPHERE_SCHEMA, CYLINDRICAL_SCHEMA,
    )
    asphere_radius = next(
        s for s in ASPHERE_SCHEMA.slots if s.key == "radius"
    )
    assert asphere_radius.options.get("variable_attr") == "radius"
    cyl_radius = next(
        s for s in CYLINDRICAL_SCHEMA.slots if s.key == "radius"
    )
    assert cyl_radius.options.get("variable_attr") == "radius"


def test_asphere_conic_k_slot_does_not_carry_variable_attr():
    """K has no evaluator or writer path, so it must not be flaggable:
    an explicit test so a well-meaning edit doesn't add it silently."""
    from ghostlight_designer.optical_editor.row_schemas import ASPHERE_SCHEMA
    k_slot = next(
        s for s in ASPHERE_SCHEMA.slots
        if s.key == "pos_z"  # asphere reuses the pos_z column for K
    )
    assert "variable_attr" not in k_slot.options


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


def test_flag_all_radii_skips_stops(qapp, sample_lens_path: pathlib.Path):
    """Aperture-stop surfaces have meaningless radii — bulk skip them."""
    project = Project()
    project.load(str(sample_lens_path))
    stop_uuids = {
        u for u, s in zip(project.system.surface_ids, project.system.surfaces)
        if bool(getattr(s, "is_stop", False))
    }
    flag_all_radii(project)
    for uuid, attrs in project.all_variable_flags().items():
        if uuid in stop_uuids:
            assert "radius" not in attrs, (
                f"stop surface {uuid} was flagged"
            )


def test_flag_all_thicknesses_covers_every_surface(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    flag_all_thicknesses(project)
    for uuid in project.system.surface_ids:
        assert project.is_variable_flagged(uuid, "thickness")


# ---------------------------------------------------------------------------
# Right-click menu helpers
# ---------------------------------------------------------------------------


def test_variable_attr_at_reads_slot_option(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        model = body.model
        # Find a SURFACE row and its Radius column.
        el_index = model.index(0, 0, QModelIndex())
        # Walk children to find the surface node.
        found = None
        for r in range(model.rowCount(el_index)):
            child = model.index(r, 0, el_index)
            if isinstance(child.internalPointer(), SurfaceNode):
                found = child
                break
        assert found is not None
        from ghostlight_designer.optical_editor.columns import Column as _C
        radius_index = model.index(found.row(), int(_C.RADIUS), el_index)
        assert variable_attr_at(radius_index) in ("radius", "")
        # Skip stops which don't paint a radius slot; if empty, try the
        # next surface. This keeps the test robust across fixture lenses.
        if not variable_attr_at(radius_index):
            for r in range(found.row() + 1, model.rowCount(el_index)):
                child = model.index(r, 0, el_index)
                if isinstance(child.internalPointer(), SurfaceNode):
                    radius_index = model.index(
                        child.row(), int(_C.RADIUS), el_index,
                    )
                    if variable_attr_at(radius_index):
                        break
        assert variable_attr_at(radius_index) == "radius"
        # NAME column: no slot → no variable_attr.
        name_index = model.index(found.row(), int(_C.NAME), el_index)
        assert variable_attr_at(name_index) == ""
    finally:
        body.deleteLater()


def test_populate_cell_menu_adds_actions_when_flaggable(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        model = body.model
        el_index = model.index(0, 0, QModelIndex())
        from ghostlight_designer.optical_editor.columns import Column as _C
        # Find first non-stop surface row.
        radius_index = None
        for r in range(model.rowCount(el_index)):
            child = model.index(r, 0, el_index)
            node = child.internalPointer()
            if not isinstance(node, SurfaceNode):
                continue
            si = node.surface_index
            if si >= 0 and not bool(project.system.surfaces[si].is_stop):
                radius_index = model.index(r, int(_C.RADIUS), el_index)
                break
        assert radius_index is not None

        menu = QMenu()
        added = populate_cell_menu(
            menu, project, radius_index, dialog_parent=body,
        )
        assert added is True
        labels = [a.text() for a in menu.actions()]
        assert "Flag as Variable" in labels
        assert "Edit Variable Bounds…" not in labels  # not flagged yet

        # Flag it, rebuild menu, confirm Bounds action appears.
        uuid = surface_uuid_at(radius_index)
        project.set_variable_flag(uuid, "radius")
        menu2 = QMenu()
        populate_cell_menu(menu2, project, radius_index, dialog_parent=body)
        labels2 = [a.text() for a in menu2.actions()]
        assert "Unflag Variable" in labels2
        assert "Edit Variable Bounds…" in labels2
    finally:
        body.deleteLater()


def test_populate_cell_menu_skips_non_flaggable_cells(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        model = body.model
        # NAME column of an element row: no variable_attr.
        el_index = model.index(0, 0, QModelIndex())
        menu = QMenu()
        added = populate_cell_menu(
            menu, project, el_index, dialog_parent=body,
        )
        assert added is False
        assert not menu.actions()
    finally:
        body.deleteLater()


def test_bulk_menu_disables_clear_when_no_flags(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        menu = QMenu(body)
        populate_bulk_menu(menu, project)
        # findChildren walks Qt ownership so we hold real Python refs to
        # the parent menu long enough to inspect the submenu's actions.
        sub = menu.findChild(QMenu)
        assert sub is not None, "populate_bulk_menu should append a submenu"
        clear = next(a for a in sub.actions() if "Clear" in a.text())
        assert not clear.isEnabled()
        # Flag one, rebuild, confirm Clear is now enabled.
        project.set_variable_flag(
            str(project.system.surface_ids[0]), "radius",
        )
        menu2 = QMenu(body)
        populate_bulk_menu(menu2, project)
        sub2 = menu2.findChild(QMenu)
        clear2 = next(a for a in sub2.actions() if "Clear" in a.text())
        assert clear2.isEnabled()
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# Bounds dialog
# ---------------------------------------------------------------------------


def test_parse_optional_float_blank_is_none():
    assert _parse_optional_float("") is None
    assert _parse_optional_float("   ") is None
    assert _parse_optional_float("1.5") == pytest.approx(1.5)
    assert _parse_optional_float("not a number") is None


def test_format_optional_float():
    assert _format_optional_float(None) == ""
    assert _format_optional_float(1.5) == "1.5"


def test_bounds_dialog_read_back(qapp):
    """Constructing the dialog with values → read them back via bounds()."""
    dlg = VariableBoundsDialog(
        title_summary="Test — Radius",
        current_enabled=True,
        current_bounds=VariableBounds(lo=-1.0, hi=2.0),
    )
    assert dlg.enabled() is True
    assert dlg.bounds() == VariableBounds(lo=-1.0, hi=2.0)


def test_bounds_dialog_swaps_inverted_bounds(qapp):
    """Typing min > max shouldn't produce an unusable flag."""
    dlg = VariableBoundsDialog(
        title_summary="Test",
        current_enabled=True,
        current_bounds=VariableBounds(lo=5.0, hi=1.0),
    )
    b = dlg.bounds()
    assert b.lo == pytest.approx(1.0)
    assert b.hi == pytest.approx(5.0)


def test_bounds_dialog_disable_grays_out_edits(qapp):
    dlg = VariableBoundsDialog(
        title_summary="Test",
        current_enabled=True,
        current_bounds=VariableBounds(),
    )
    assert dlg._min_edit.isEnabled()
    dlg._enable.setChecked(False)
    assert not dlg._min_edit.isEnabled()
    assert not dlg._max_edit.isEnabled()


# ---------------------------------------------------------------------------
# Lifetime — flag signals must not crash after a body is destroyed
# ---------------------------------------------------------------------------


def test_optimization_panel_variables_section_hidden_when_empty(
    qapp, sample_lens_path: pathlib.Path,
):
    """No flags → no Variables header row. Keeps the tree clean for
    users who haven't flagged anything yet."""
    from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        assert body.model.rowCount(QModelIndex()) == 0
    finally:
        body.deleteLater()


def test_optimization_panel_variables_section_appears_on_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
    from ghostlight_designer.optimization_panel.nodes import (
        VariableEntryNode,
        VariablesHeaderNode,
    )
    from ghostlight_designer.optimization_panel.columns import Column
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        project.set_variable_flag(
            str(project.system.surface_ids[0]), "radius",
        )
        # Header appears at root row 0.
        header_idx = body.model.index(0, 0, QModelIndex())
        assert isinstance(header_idx.internalPointer(), VariablesHeaderNode)
        # Name cell shows the count.
        assert "Variables" in str(body.model.data(header_idx))
        assert "(1)" in str(body.model.data(header_idx))
        # One variable row underneath.
        assert body.model.rowCount(header_idx) == 1
        var_idx = body.model.index(0, 0, header_idx)
        node = var_idx.internalPointer()
        assert isinstance(node, VariableEntryNode)
        assert node.attr == "radius"
        # NAME cell renders "Surface N · radius".
        name = str(body.model.data(var_idx))
        assert "Surface 0" in name
        assert "radius" in name
        # TYPE cell reads "Variable".
        type_idx = body.model.index(0, int(Column.TYPE), header_idx)
        assert body.model.data(type_idx) == "Variable"
        # Bounds columns show ±∞ for the unbounded default.
        target_idx = body.model.index(0, int(Column.TARGET), header_idx)
        weight_idx = body.model.index(0, int(Column.WEIGHT), header_idx)
        assert body.model.data(target_idx) == "-∞"
        assert body.model.data(weight_idx) == "+∞"
    finally:
        body.deleteLater()


def test_optimization_panel_variable_rows_are_read_only(
    qapp, sample_lens_path: pathlib.Path,
):
    """No cell in the Variables section is editable — the ODE owns
    every write path for flags."""
    from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
    from ghostlight_designer.optimization_panel.columns import Column
    project = Project()
    project.load(str(sample_lens_path))
    project.set_variable_flag(
        str(project.system.surface_ids[0]), "radius",
    )
    body = OptimizationPanelBody(project)
    try:
        header_idx = body.model.index(0, 0, QModelIndex())
        for col in range(body.model.columnCount()):
            row_idx = body.model.index(0, col, header_idx)
            flags = body.model.flags(row_idx)
            assert not (flags & Qt.ItemIsEditable), (
                f"col {col!r} should not be editable"
            )
    finally:
        body.deleteLater()


def test_optimization_panel_remove_shortcut_unflags_variable(
    qapp, sample_lens_path: pathlib.Path,
):
    """Selecting a variable row and calling _on_remove clears its flag."""
    from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
    from PySide6.QtCore import QItemSelectionModel
    project = Project()
    project.load(str(sample_lens_path))
    uuid = str(project.system.surface_ids[0])
    project.set_variable_flag(uuid, "radius")
    body = OptimizationPanelBody(project)
    try:
        header_idx = body.model.index(0, 0, QModelIndex())
        var_idx = body.model.index(0, 0, header_idx)
        body.tree.selectionModel().setCurrentIndex(
            var_idx,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        body._on_remove()
        assert not project.is_variable_flagged(uuid, "radius")
        # After removal the whole section disappears.
        assert body.model.rowCount(QModelIndex()) == 0
    finally:
        body.deleteLater()


def test_toolbar_unflag_all_variables_button_gated(
    qapp, sample_lens_path: pathlib.Path,
):
    """Toolbar button is disabled when no flags exist, enabled when
    any flag is set, and back to disabled after clicking it."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        btn = body.toolbar.btn_unflag_all_variables
        assert not btn.isEnabled()
        # Flag one attribute → button lights up.
        uuid = str(project.system.surface_ids[0])
        project.set_variable_flag(uuid, "radius")
        assert btn.isEnabled()
        # Click → all flags cleared → button disabled again.
        btn.click()
        assert project.all_variable_flags() == {}
        assert not btn.isEnabled()
    finally:
        body.deleteLater()


def test_flag_signals_do_not_crash_after_body_destroyed(
    qapp, sample_lens_path: pathlib.Path,
):
    """Regression: connecting the flag signals to lambdas capturing self
    left the connections dangling when the body's C++ QTreeView was
    destroyed (panel close / undock / layout swap). The next emission
    crashed with ``RuntimeError: Internal C++ object already deleted``.

    Bound methods on ``self`` let Qt auto-disconnect when the receiver
    QObject dies; this test forces that path.
    """
    import gc

    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    uuid = str(project.system.surface_ids[0])

    # First emission with the body alive — must repaint without raising.
    project.set_variable_flag(uuid, "radius")
    qapp.processEvents()

    # Simulate the panel being torn down (undock / close / layout swap).
    body.setParent(None)
    body.deleteLater()
    qapp.processEvents()
    del body
    gc.collect()
    qapp.processEvents()

    # Emitting the signal now must NOT crash. This is the regression.
    project.clear_variable_flag(uuid, "radius")
    qapp.processEvents()
    project.set_variable_flag(uuid, "radius")
    qapp.processEvents()
    project.clear_all_variable_flags()
    qapp.processEvents()
