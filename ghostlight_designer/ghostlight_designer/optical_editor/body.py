"""Body widget for the ``optical_editor`` panel — tree view of the system."""
from __future__ import annotations

import os
from typing import Optional, Set

from PySide6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QObject, QPoint, QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..coating_catalogue import get_coating_catalogue
from ..material_catalogue import get_catalogue
from ..project import Project
from . import coating_actions, element_actions, surface_actions, variable_flag_actions
from .coating_dialogs import (
    open_coating_data_dialog,
    open_coating_preset_picker,
    open_tint_picker,
)
from .delegates import SlotDelegate, SlotRole
from .nodes import MaterialNode
from . import row_schemas as schemas
from .row_schemas import (
    Slot,
    canonical_column_for,
    convert_material_to_custom,
    convert_material_to_vendor,
    material_designers,
)
from .model import OpticalTreeModel
from .nodes import ElementNode, SurfaceNode
from .toolbar import (
    ADD_ANAMORPHIC_FRONT,
    ADD_APERTURE_STOP,
    ADD_DOUBLET,
    ADD_IMPORT_LENS_BACK,
    ADD_IMPORT_LENS_FRONT,
    ADD_SINGLET,
    ADD_TRIPLET,
    OpticalEditorToolbar,
)
from ..value_scrubber import attach_value_scrubber

# File dialog filter for lens imports — mirrors main_window.LENS_FILTER
# (kept local so the body doesn't depend on the main-window module).
_LENS_FILTER = "Lens files (*.lens);;All files (*)"


_ADD_DISPATCH = {
    ADD_SINGLET: element_actions.add_singlet,
    ADD_DOUBLET: element_actions.add_doublet,
    ADD_TRIPLET: element_actions.add_triplet,
    ADD_APERTURE_STOP: element_actions.add_aperture_stop,
}


def _open_material_glass_picker(parent, index: QModelIndex):
    """Picker for Material rows' Name slot.

    Reads the sibling Designer cell to know which vendor's glasses to
    list, pops a modal selection over the picked vendor's catalogue,
    and returns the chosen ``CatalogueMaterial.key`` (the string that
    ``material_glasses[i]`` stores). ``None`` on cancel / no-op.

    Defensive against three states that would otherwise crash or silently
    swallow the click:
      * vendor is empty (current glass key doesn't resolve in either the
        bundled MaterialCatalogue or the project's bundled catalogue) —
        show a clear "Designer not set" message instead of opening a
        dialog with no items.
      * vendor is "Custom" — Name slot is read-only there, but clicking
        the picker button on a stale repaint can still get here.
      * the dialog parent is ``None`` (delegate didn't capture
        ``option.widget``) — fall through with no parent; PySide6 uses
        the active window as the implicit parent.
    """
    designer_col = canonical_column_for("material_designer")
    if designer_col is None:
        return None
    vendor = index.sibling(index.row(), designer_col).data(Qt.DisplayRole)
    vendor_str = str(vendor) if vendor else ""
    if not vendor_str:
        QMessageBox.information(
            parent,
            "Select Designer first",
            "This material doesn't resolve to any designer in the bundled "
            "catalogue. Pick a Designer from the dropdown before browsing "
            "glasses.",
        )
        return None
    if vendor_str == "Custom":
        return None
    matches = get_catalogue().search(vendor=vendor_str)
    if not matches:
        QMessageBox.information(
            parent,
            "No glasses found",
            f"No glasses are loaded for designer {vendor_str!r}.",
        )
        return None
    items = [m.display_name for m in matches]
    current = str(index.data(Qt.DisplayRole) or "")
    try:
        current_i = items.index(current)
    except ValueError:
        current_i = 0
    chosen, ok = QInputDialog.getItem(
        parent, "Select Glass", f"Vendor: {vendor_str}",
        items, current=current_i, editable=False,
    )
    if not ok:
        return None
    for m in matches:
        if m.display_name == chosen:
            return m.key
    return None


def _slot_is_scrubbable(index) -> bool:
    """Gate Ctrl+MMB on cells whose slot opts in via
    ``options['scrubbable'] = True``. The cell must also be editable —
    that's still owned by the model's flags."""
    if not index.isValid():
        return False
    model = index.model()
    if model is None or not (model.flags(index) & Qt.ItemIsEditable):
        return False
    slot = index.data(SlotRole)
    if not isinstance(slot, Slot):
        return False
    return bool(slot.options.get("scrubbable", False))


class _ComboClickToEdit(QObject):
    """Mouse filter that opens combo-edited cells on a single left-click.

    Without this the user has to: click row to select → click again to
    enter edit mode (and only because we double-click) → click the combo
    arrow → click an option. Four clicks. With it: one click on the cell
    opens both the editor and the dropdown (the delegate's
    ``setEditorData`` schedules ``showPopup``), then one click on the
    option commits and closes. The filter only intercepts combo-edit
    cells — other cell types fall through to the standard double-click
    behaviour so drag-to-scrub still works on numeric cells.
    """

    def __init__(self, tree: QTreeView) -> None:
        super().__init__(tree)
        self._tree = tree
        tree.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):  # type: ignore[override]
        tree = self._tree
        try:
            viewport = tree.viewport()
        except RuntimeError:
            return False
        if obj is not viewport:
            return False
        if ev.type() != QEvent.MouseButtonPress:
            return False
        if ev.button() != Qt.LeftButton:
            return False
        index = tree.indexAt(ev.position().toPoint())
        if not index.isValid():
            return False
        delegate = tree.itemDelegate(index)
        if not hasattr(delegate, "uses_combo"):
            return False
        if not delegate.uses_combo(index):
            return False
        # Drive selection + edit through the same code path the standard
        # double-click would have used, so selection sync to viewport
        # and the model's setData path are unchanged.
        sel = tree.selectionModel()
        sel.setCurrentIndex(
            index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        tree.edit(index)
        return True


class OpticalEditorBody(QWidget):
    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

        self.model = OpticalTreeModel(project, self)

        self.tree = QTreeView(self)
        self.tree.setModel(self.model)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.tree.setIconSize(QSize(18, 18))
        # Drag-and-drop reordering of element rows. The model declares which
        # rows are draggable (ElementNode only) and where drops are allowed
        # (root level only — between top-level rows). Overwrite mode off so
        # the drop indicator renders as a horizontal bar between rows, not a
        # rectangle around a row.
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDragDropOverwriteMode(False)
        self.tree.setDefaultDropAction(Qt.MoveAction)

        header = self.tree.header()
        header.setStretchLastSection(False)
        self._apply_header_layout()

        # One delegate for the whole tree — dispatches per-cell via the
        # ``Slot`` returned by the model under ``SlotRole``. NAME-column
        # cells have no slot and fall through to the default paint/no-editor
        # behavior. Pickers registered here are looked up by the tag in
        # ``slot.options["picker"]``; Material's Name slot uses
        # ``"material_glass"``.
        self._slot_delegate = SlotDelegate(
            self,
            pickers={
                "material_glass": _open_material_glass_picker,
                "coating_tint": open_tint_picker,
                "coating_data": open_coating_data_dialog,
                "coating_preset": open_coating_preset_picker,
            },
            project=project,
        )
        self.tree.setItemDelegate(self._slot_delegate)

        self.toolbar = OpticalEditorToolbar(project, self)
        self.toolbar.addElementRequested.connect(self._on_add_element_requested)
        self.toolbar.removeElementRequested.connect(self._on_remove_element_requested)
        self.toolbar.expandAllRequested.connect(self.tree.expandAll)
        self.toolbar.collapseAllRequested.connect(self.tree.collapseAll)
        self.toolbar.unsoloAllRequested.connect(self._on_unsolo_all_requested)
        self.toolbar.unmuteAllRequested.connect(self._on_unmute_all_requested)
        self.toolbar.unflagAllVariablesRequested.connect(
            self._on_unflag_all_variables_requested
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.tree)

        # Scrubbability is declared on the slot, not inferred from the cell
        # type — so a float slot that's editable but shouldn't scrub
        # (e.g. a sensor exposure that wants explicit-typed input only) can
        # opt out via ``options={"scrubbable": False}`` without changing the
        # generic scrubber.
        self._scrub_trigger = attach_value_scrubber(
            self.tree, self._project, is_scrubbable=_slot_is_scrubbable
        )
        # Single-click into combo-edited cells: skips the standard
        # "click-to-select then double-click-to-edit" dance, and pops the
        # dropdown immediately via the delegate's QTimer.singleShot hook in
        # setEditorData. Other cell types keep the standard double-click.
        self._combo_click_filter = _ComboClickToEdit(self.tree)

        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        # Guard re-entry when the editor itself is the source of the
        # selection change: tree → project → selectionChanged → tree.
        self._applying_project_selection: bool = False

        # Expansion state preserved across model resets. ``None`` means no
        # snapshot was captured (first show, or a New/Open swap) — fall back
        # to expanding everything. Otherwise, restore exactly these keys so
        # an in-place edit (add/remove element, swap form, scrub a cell)
        # doesn't blow away whatever the user had expanded or collapsed.
        # Keys are stringly-typed with a namespace prefix so element and
        # surface IDs share one set: ``"e:<element_id>"`` / ``"s:<surface_uuid>"``.
        self._pre_reset_expanded_keys: Optional[Set[str]] = None
        # Identity of the system instance we last saw a reset for. New() /
        # load() create a fresh OpticalSystem, so id() changes; undo / redo
        # reload INTO the same instance, so id() stays. We can't use a
        # separate systemReplaced slot to flag this — Qt fires the model's
        # rebuild slot before ours, so by the time we'd set a flag the
        # reset has already finished. Detecting it here means the gate
        # closes before the snapshot is collected.
        self._last_seen_system_id: int = id(project.system)

        self.model.modelAboutToBeReset.connect(self._on_model_about_to_reset)
        self.model.modelReset.connect(self._on_model_reset)
        self.tree.selectionModel().currentRowChanged.connect(
            self._on_current_row_changed
        )
        project.selectionChanged.connect(self._on_project_selection_changed)
        project.surfaceSelectionChanged.connect(
            self._on_project_surface_selection_changed
        )
        # Repaint on variable-flag changes so the amber stripe appears /
        # disappears immediately when the user toggles a cell via the
        # right-click menu. Cheap — just a viewport update; no model
        # reset is needed because the flag isn't part of the tree
        # structure, only of the cell's paint.
        #
        # Connect to bound methods (not lambdas): Qt tracks the receiver
        # QObject and auto-disconnects when this body is destroyed. A
        # lambda capturing ``self`` has no registered receiver, so the
        # connection outlives the widget and the next flag emission
        # crashes on a dead ``QTreeView`` C++ object.
        project.variableFlagChanged.connect(self._on_variable_flag_changed)
        project.variableFlagsReplaced.connect(self._on_variable_flags_replaced)
        project.offAxisRevealChanged.connect(self._on_off_axis_reveal_changed)
        self._on_model_reset()

    # ------------------------------------------------------------------
    # Header layout + the collapsible off-axis column block
    # ------------------------------------------------------------------

    def _apply_header_layout(self) -> None:
        """Size every column and hide the collapsed off-axis block.

        Re-applied on every model reset, not just at construction: the column
        count grows with the system's asphere terms, and a column that was
        hidden and then revealed would otherwise come back at Qt's default
        width rather than the one ``row_schemas.COLUMN_WIDTHS`` declares.
        """
        try:
            header = self.tree.header()
            count = self.model.columnCount()
        except RuntimeError:
            return
        for c in range(count):
            header.setSectionResizeMode(c, QHeaderView.Interactive)
            width = schemas.COLUMN_WIDTHS.get(c)
            if width is not None:
                self.tree.setColumnWidth(c, width)
        self._apply_off_axis_visibility()

    def _apply_off_axis_visibility(self) -> None:
        """Show the Pos X..Pivot Z block iff some element is revealed.

        Columns belong to the view, not to a row, so one revealed element
        shows them for the whole tree; every other row just renders blank
        cells there (no slot -> no value, no editor).
        """
        try:
            show = self.model.any_off_axis_revealed()
        except (AttributeError, RuntimeError):
            return
        for key in schemas.OFF_AXIS_COLUMN_KEYS:
            column = schemas.canonical_column_for(key)
            if column is None:
                continue
            try:
                self.tree.setColumnHidden(column, not show)
            except RuntimeError:
                return
            if show:
                width = schemas.COLUMN_WIDTHS.get(column)
                if width is not None:
                    self.tree.setColumnWidth(column, width)

    def _on_off_axis_reveal_changed(self) -> None:
        """Slot for :sig:`Project.offAxisRevealChanged`."""
        try:
            self._apply_off_axis_visibility()
            self.tree.viewport().update()
        except RuntimeError:
            pass

    def _on_variable_flag_changed(self, _uuid: str, _attr: str) -> None:
        """Slot for :sig:`Project.variableFlagChanged`.

        The signal fires on every toggle / bounds edit; we just need to
        repaint the tree so the amber stripe reflects the new state.
        Guarded because a queued emission could race a widget teardown
        that Qt hasn't finished tearing down at the Python layer yet.
        """
        try:
            self.tree.viewport().update()
        except RuntimeError:
            # Underlying C++ object gone — connection is about to be
            # dropped by Qt; nothing to do.
            pass

    def _on_variable_flags_replaced(self) -> None:
        try:
            self.tree.viewport().update()
        except RuntimeError:
            pass

    def _on_model_about_to_reset(self) -> None:
        current_id = id(self._project.system)
        if current_id != self._last_seen_system_id:
            # System was swapped (New / Open). The old snapshot would be
            # for a different document; drop it so _on_model_reset falls
            # through to expandAll() for the fresh load.
            self._pre_reset_expanded_keys = None
            self._last_seen_system_id = current_id
        else:
            self._pre_reset_expanded_keys = self._collect_expanded_keys()

    def _on_model_reset(self) -> None:
        if self._pre_reset_expanded_keys is None:
            self.tree.expandAll()
        else:
            self._apply_expanded_keys(self._pre_reset_expanded_keys)
        self._pre_reset_expanded_keys = None
        # Column count tracks the system's asphere terms, so a reset can add
        # or drop trailing columns; re-run the whole header layout rather than
        # leaving new sections at Qt's defaults.
        self._apply_header_layout()
        # The view's selection model is reset alongside the data — re-apply
        # whatever the project still considers selected (may be ``None`` if
        # the system swap cleared it).  Resolve surface FIRST so its target
        # row wins over the element row (a surface node is more specific).
        if self._project.selected_surface_index is not None:
            self._on_project_surface_selection_changed(
                self._project.selected_surface_index
            )
        else:
            self._on_project_selection_changed(self._project.selected_element)

    def _on_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if self._applying_project_selection:
            return
        element = self._element_for_index(current)
        surface_index = self._surface_index_for_index(current)
        # Push element first so the project canonicalises the parent; then
        # push surface (which validates against the just-set element).
        self._project.set_selected_element(element)
        self._project.set_selected_surface_index(surface_index)

    def _on_project_selection_changed(self, element) -> None:
        # If a surface is also selected in the project, the surface handler
        # owns the row choice — let it run instead so we don't fight it.
        if self._project.selected_surface_index is not None:
            return
        self._apply_target_index(self._index_for_element(element))

    def _on_project_surface_selection_changed(self, surface_index) -> None:
        if surface_index is None:
            # Fall back to element-level row when surface clears.
            self._apply_target_index(
                self._index_for_element(self._project.selected_element)
            )
            return
        self._apply_target_index(self._index_for_surface(int(surface_index)))

    def _apply_target_index(self, target_index: QModelIndex) -> None:
        sel_model = self.tree.selectionModel()
        current = sel_model.currentIndex()
        if target_index == current:
            return
        # If the user is already on a descendant of the target (e.g. a
        # form-modifier child row of the surface we're about to select),
        # leave their row alone — otherwise project-selection round-trips
        # yank the cursor up to the ancestor mid-edit.
        if target_index.isValid() and current.isValid():
            probe = current.sibling(current.row(), 0)
            while probe.isValid():
                if probe == target_index:
                    return
                probe = probe.parent()
        self._applying_project_selection = True
        try:
            if not target_index.isValid():
                sel_model.clearSelection()
                sel_model.clearCurrentIndex()
            else:
                sel_model.setCurrentIndex(
                    target_index,
                    QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
                )
                # Scroll into view so a surface picked in the 3D viewport
                # actually becomes visible in a long tree.
                self.tree.scrollTo(target_index)
        finally:
            self._applying_project_selection = False

    def _on_tree_context_menu(self, pos: QPoint) -> None:
        """Right-click dispatch:
          * flag-able cell (Radius / Pos Z on a Surface or form row) →
            Toggle Variable / Edit Bounds at the top
          * ElementNode  → Flip / Remove
          * SurfaceNode  → Swap Form / Pos Z mode
          * empty area   → Add Element
          * material / form-modifier rows → still get variable-flag actions
            when the click landed on a flag-able cell (Asphere / Cylindrical
            form Radius), else no menu.
          * every menu ends with the "Variable Flags" bulk submenu.
        """
        index = self.tree.indexAt(pos)
        node = index.internalPointer() if index.isValid() else None

        menu = QMenu(self.tree)

        # Per-cell variable actions first — they're specific to the
        # exact cell clicked and should read as the primary action.
        added_variable_action = variable_flag_actions.populate_cell_menu(
            menu, self._project, index, dialog_parent=self,
        )
        if added_variable_action:
            menu.addSeparator()

        if isinstance(node, ElementNode):
            self._populate_element_menu(menu, node.element)
        elif isinstance(node, SurfaceNode):
            self._populate_surface_menu(menu, node)
        elif isinstance(node, MaterialNode):
            self._populate_material_menu(menu, node)
        elif node is None:
            # Click landed outside any row.
            self._populate_add_element_menu(menu)
        elif not added_variable_action:
            # Form-modifier row with no flag-able cell → no menu.
            return

        # Bulk operations at the bottom of every menu we chose to show.
        menu.addSeparator()
        variable_flag_actions.populate_bulk_menu(menu, self._project)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _populate_add_element_menu(self, menu: QMenu) -> None:
        """Mirrors the toolbar's ``+`` dropdown. Reuses the same dispatch
        and the body's add handler so newly-created elements still go to
        the front of the chain and become selected."""
        add_menu = menu.addMenu("Add Element")
        for label, kind in [
            ("Singlet",       ADD_SINGLET),
            ("Doublet",       ADD_DOUBLET),
            ("Triplet",       ADD_TRIPLET),
            ("Aperture Stop", ADD_APERTURE_STOP),
        ]:
            act = QAction(label, add_menu)
            act.triggered.connect(
                lambda _checked=False, k=kind: self._on_add_element_requested(k)
            )
            add_menu.addAction(act)
        add_menu.addSeparator()
        for label, kind in [
            ("Import Lens → Front (object side)…", ADD_IMPORT_LENS_FRONT),
            ("Import Lens → Back (sensor side)…",  ADD_IMPORT_LENS_BACK),
        ]:
            act = QAction(label, add_menu)
            act.triggered.connect(
                lambda _checked=False, k=kind: self._on_add_element_requested(k)
            )
            add_menu.addAction(act)

    def _populate_element_menu(self, menu: QMenu, element) -> None:
        import ghostlight

        flip_action = QAction("Flip", menu)
        # Single-surface elements (aperture stops) have nothing to flip.
        flip_action.setEnabled(len(element.surface_ids) >= 2)
        flip_action.triggered.connect(
            lambda _checked=False, el=element:
                element_actions.flip_element(self._project, el)
        )
        menu.addAction(flip_action)

        # Mute + Solo toggles only apply to glass — the aperture pupil
        # doesn't participate in mute-skip or the ghost-solo filter.
        if element.kind != ghostlight.ElementKind.STOP:
            currently_muted = bool(element.is_muted(self._project.system))
            mute_label = "Unmute Element" if currently_muted else "Mute Element"
            mute_action = QAction(mute_label, menu)
            mute_action.triggered.connect(
                lambda _checked=False, el=element, m=not currently_muted:
                    element_actions.set_element_muted(self._project, el, m)
            )
            menu.addAction(mute_action)

            currently_solo = element_actions.is_element_ghost_solo(
                self._project, element,
            )
            solo_label = (
                "Show All Ghosts (Unsolo)" if currently_solo
                else "Solo Ghost Reflections"
            )
            solo_action = QAction(solo_label, menu)
            solo_action.setCheckable(True)
            solo_action.setChecked(currently_solo)
            solo_action.triggered.connect(
                lambda _checked=False, el=element, s=not currently_solo:
                    element_actions.set_element_ghost_solo(self._project, el, s)
            )
            menu.addAction(solo_action)

        menu.addSeparator()
        remove_action = QAction("Remove", menu)
        remove_action.triggered.connect(
            lambda _checked=False, el=element: self._remove_element_with_confirm(el)
        )
        menu.addAction(remove_action)

    def _populate_material_menu(self, menu: QMenu, node: MaterialNode) -> None:
        """Right-click menu on a Material row.

        ``Convert to <vendor>`` searches the vendor's catalogue for a
        glass within (nd, Vd) tolerance of the current material and
        switches the row to it. ``Convert to Custom`` synthesizes a
        project-local Abbe entry from the current nd/Vd. The vendor that
        matches the row's *current* designer is grayed out, since
        converting to yourself is a no-op.

        Below the convert menu, a ``Substitute in Optimization``
        submenu exposes the catalogue-hammer material-flag flow — one
        radio-checked action per vendor, plus an ``Off`` entry.
        """
        ctx_designer = self._material_designer(node)
        convert = menu.addMenu("Convert to")
        from .row_schemas import CUSTOM_DESIGNER  # avoid top-level circular
        for vendor in material_designers():
            act = QAction(vendor, convert)
            act.setEnabled(vendor != ctx_designer)
            act.triggered.connect(
                lambda _checked=False, n=node, v=vendor:
                    self._convert_material_to_vendor_with_message(n, v)
            )
            convert.addAction(act)
        convert.addSeparator()
        custom_act = QAction("Custom", convert)
        custom_act.setEnabled(ctx_designer != CUSTOM_DESIGNER)
        custom_act.triggered.connect(
            lambda _checked=False, n=node: convert_material_to_custom(
                self._project, n
            )
        )
        convert.addAction(custom_act)

        self._populate_material_substitution_menu(menu, node)

    def _populate_material_substitution_menu(
        self, menu: QMenu, node: MaterialNode,
    ) -> None:
        """Add the "Substitute in Optimization" submenu to ``menu``.

        Radio-group behavior: the currently-flagged vendor (if any) shows
        its action checked; picking a different vendor rewrites the flag
        in place, and ``Off`` clears it. Custom is not offered as a
        substitution target — the hammer only iterates catalogue glasses,
        so pointing a flag at Custom would be a no-op.
        """
        eid = getattr(node.element, "element_id", "") or ""
        if not eid:
            return
        mi = int(node.material_index)
        spec = self._project.material_flag_spec(eid, mi)
        current_vendor = str(getattr(spec, "vendor", "") or "")

        sub = menu.addMenu("Substitute in Optimization")
        off_act = QAction("Off", sub)
        off_act.setCheckable(True)
        off_act.setChecked(spec is None)
        off_act.setEnabled(spec is not None)
        off_act.triggered.connect(
            lambda _checked=False, e=eid, i=mi:
                self._project.clear_material_flag(e, i)
        )
        sub.addAction(off_act)
        sub.addSeparator()

        from ..project import SubstitutionSpec
        for vendor in material_designers():
            from .row_schemas import CUSTOM_DESIGNER
            if vendor == CUSTOM_DESIGNER:
                continue
            act = QAction(vendor, sub)
            act.setCheckable(True)
            act.setChecked(vendor == current_vendor)
            act.triggered.connect(
                lambda _checked=False, e=eid, i=mi, v=vendor:
                    self._project.set_material_flag(
                        e, i, SubstitutionSpec(vendor=v)
                    )
            )
            sub.addAction(act)

    def _material_designer(self, node: MaterialNode) -> str:
        """Read the Designer slot value for ``node`` via the row schema —
        keeps the menu in sync with whatever ``_designer_get`` returns
        without duplicating the catalogue-lookup logic."""
        from .row_schemas import SlotContext, slot_at, canonical_column_for
        designer_col = canonical_column_for("material_designer")
        if designer_col is None:
            return ""
        slot = slot_at(node, designer_col)
        if slot is None:
            return ""
        ctx = SlotContext(
            node=node,
            system=self._project.system,
            project=self._project,
        )
        value = slot.get(ctx)
        return str(value) if value else ""

    def _convert_material_to_vendor_with_message(
        self, node: MaterialNode, vendor: str
    ) -> None:
        ok, message = convert_material_to_vendor(self._project, node, vendor)
        if ok:
            return
        QMessageBox.information(self, f"Convert to {vendor}", message)

    def _populate_surface_menu(self, menu: QMenu, node: SurfaceNode) -> None:
        uuid = node.surface_uuid
        surface_index = node.surface_index
        current = self.model.pos_z_mode(uuid)

        self._populate_form_submenu(menu, surface_index)
        self._populate_coating_submenu(menu, surface_index)
        menu.addSeparator()
        abs_action = QAction("Show Pos Z: Absolute", menu)
        abs_action.setCheckable(True)
        abs_action.setChecked(current == "absolute")
        abs_action.triggered.connect(
            lambda _checked=False, u=uuid: self.model.set_pos_z_mode(u, "absolute")
        )
        rel_action = QAction("Show Pos Z: Relative (thickness)", menu)
        rel_action.setCheckable(True)
        rel_action.setChecked(current == "relative")
        rel_action.triggered.connect(
            lambda _checked=False, u=uuid: self.model.set_pos_z_mode(u, "relative")
        )
        menu.addAction(abs_action)
        menu.addAction(rel_action)

        # Ghost-solo toggle — hidden on aperture-stop surfaces (they
        # bound the pupil, they don't produce ghost reflections in the
        # sense the feature isolates). Solo is a view filter, not a
        # lens mutation, so the action does NOT push undo.
        system = self._project.system
        is_stop_surf = (
            0 <= surface_index < len(system.surfaces)
            and bool(system.surfaces[surface_index].is_stop)
        )
        if not is_stop_surf:
            menu.addSeparator()
            currently_solo = self._project.is_surface_ghost_solo(uuid)
            solo_label = (
                "Show All Ghosts (Unsolo)" if currently_solo
                else "Solo Ghost Reflections"
            )
            solo_action = QAction(solo_label, menu)
            solo_action.setCheckable(True)
            solo_action.setChecked(currently_solo)
            solo_action.triggered.connect(
                lambda _checked=False, si=surface_index, s=not currently_solo:
                    surface_actions.set_surface_ghost_solo(self._project, si, s)
            )
            menu.addAction(solo_action)

    def _populate_coating_submenu(self, menu: QMenu, surface_index: int):
        """Add a 'Coating' submenu applying a catalogue preset to this surface.

        This is the entry point for adding a coating to a bare surface — a
        surface has no coating row in the tree until one is applied. Hidden on
        aperture stops (a stop bounds the pupil, it isn't a coatable optical
        interface). When the surface is already coated, a "Remove Coating"
        item is offered too.

        Returns the created submenu (or ``None`` when skipped) so callers /
        tests can hold a reference — the parent ``menu`` owns it in production,
        where it is ``exec()``'d immediately.
        """
        system = self._project.system
        if not (0 <= surface_index < len(system.surfaces)):
            return None
        surf = system.surfaces[surface_index]
        if bool(getattr(surf, "is_stop", False)):
            return None

        coating_menu = menu.addMenu("Coating")
        for preset in get_coating_catalogue().all():
            act = QAction(preset.display_name, coating_menu)
            if preset.description:
                act.setToolTip(preset.description)
            act.triggered.connect(
                lambda _checked=False, si=surface_index, p=preset:
                    coating_actions.apply_coating_preset(
                        self._project, si, dict(p.payload),
                        label=f"Apply Coating: {p.display_name}")
            )
            coating_menu.addAction(act)

        if coating_actions.surface_has_coating(surf):
            coating_menu.addSeparator()
            remove_act = QAction("Remove Coating", coating_menu)
            remove_act.triggered.connect(
                lambda _checked=False, si=surface_index:
                    coating_actions.remove_coating(self._project, si)
            )
            coating_menu.addAction(remove_act)
        return coating_menu

    def _populate_form_submenu(self, menu: QMenu, surface_index: int) -> None:
        """Add a 'Swap Form' submenu listing every SurfaceForm; current one ticked."""
        system = self._project.system
        if not (0 <= surface_index < len(system.surfaces)):
            return
        current_form = int(system.surfaces[surface_index].form)
        form_menu = menu.addMenu("Swap Form")
        for form_int, label in surface_actions.available_forms():
            act = QAction(label, form_menu)
            act.setCheckable(True)
            act.setChecked(form_int == current_form)
            act.triggered.connect(
                lambda _checked=False, si=surface_index, f=form_int:
                    surface_actions.set_surface_form(self._project, si, f)
            )
            form_menu.addAction(act)

    def _element_for_index(self, index: QModelIndex):
        """Walk up the tree from ``index`` until we hit an ElementNode."""
        if not index.isValid():
            return None
        node = index.internalPointer()
        while node is not None and not isinstance(node, ElementNode):
            node = node.parent
        return node.element if isinstance(node, ElementNode) else None

    def _surface_index_for_index(self, index: QModelIndex):
        """Return the global surface index for ``index`` if it points at a
        :class:`SurfaceNode` or one of its form-modifier child rows
        (Asphere / Cylindrical), else ``None``.

        Form-modifier rows resolve to their parent surface — they are
        part of that surface conceptually, so navigating into them
        should keep the surface highlighted (in the 3D viewport and in
        the tree's restored selection after a model reset) rather than
        bubbling up to the bare element row.
        """
        if not index.isValid():
            return None
        node = index.internalPointer()
        if isinstance(node, SurfaceNode):
            return node.surface_index if node.surface_index >= 0 else None
        parent = getattr(node, "parent", None) if node is not None else None
        if isinstance(parent, SurfaceNode):
            return parent.surface_index if parent.surface_index >= 0 else None
        return None

    def _index_for_element(self, element) -> QModelIndex:
        if element is None:
            return QModelIndex()
        root_count = self.model.rowCount(QModelIndex())
        for ei in range(root_count):
            idx = self.model.index(ei, 0, QModelIndex())
            node = idx.internalPointer()
            if isinstance(node, ElementNode) and node.element is element:
                return idx
        return QModelIndex()

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------

    def _on_unsolo_all_requested(self) -> None:
        # View filter, not a lens mutation — no undo push. The toolbar's
        # gating already prevents this firing on a clean state, so we
        # don't need a here-be-dragons no-op guard.
        self._project.clear_ghost_solo()

    def _on_unmute_all_requested(self) -> None:
        # Batched compound edit → single undo entry, matching the single
        # toolbar click. Toolbar gating prevents this firing when nothing
        # is muted.
        element_actions.unmute_all_elements(self._project)

    def _on_unflag_all_variables_requested(self) -> None:
        # View state, not a lens mutation — variable flags aren't on the
        # undo stack (same call as ghost_solo). Toolbar gating
        # prevents this firing when nothing is flagged.
        variable_flag_actions.clear_all_variable_flags(self._project)

    def _on_add_element_requested(self, kind: str) -> None:
        if kind == ADD_ANAMORPHIC_FRONT:
            self._launch_anamorphic_wizard()
            return
        if kind in (ADD_IMPORT_LENS_FRONT, ADD_IMPORT_LENS_BACK):
            self._import_lens_from_file(to_front=(kind == ADD_IMPORT_LENS_FRONT))
            return
        builder = _ADD_DISPATCH.get(kind)
        if builder is None:
            return
        # Default-position adds: the action inserts at the FRONT of the
        # chain (furthest from sensor) so existing surfaces don't shift in
        # absolute z. The user can drag-drop the new row to reorder.
        new_element = builder(self._project)
        # The model reset triggered by the add preserved the prior
        # expansion state — so the new element's row is collapsed. Expand
        # just that subtree to max depth so the user sees what they added.
        self._expand_element_subtree(new_element)
        # Surface selection clears on element change; selecting the new
        # element gives the user immediate feedback in the tree + viewport.
        self._project.set_selection(new_element, None)

    def _import_lens_from_file(self, *, to_front: bool) -> None:
        """Pick a ``.lens`` file and splice its elements onto the front
        (object side) or back (sensor side) of the current system.

        Delegates the merge to :func:`element_actions.append_lens_from_file`
        (one undoable edit); on success the appended elements are expanded
        and the first is selected, mirroring the plain add flow.
        """
        side = "Front" if to_front else "Back"
        start_dir = os.path.dirname(self._project.path) if self._project.path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Import Lens to {side}", start_dir, _LENS_FILTER,
        )
        if not path:
            return
        try:
            new_elements = element_actions.append_lens_from_file(
                self._project, path, to_front=to_front,
            )
        except Exception as exc:  # noqa: BLE001 — user-facing, surface the error
            QMessageBox.warning(
                self,
                "Import Lens",
                f"Could not import {os.path.basename(path)}:\n\n{exc}",
            )
            return
        if not new_elements:
            return
        for element in new_elements:
            self._expand_element_subtree(element)
        self._project.set_selection(new_elements[0], None)

    def _launch_anamorphic_wizard(self) -> None:
        """Open the anamorphic setup dialog and, on accept, insert the
        block + launch the optimization preview.

        Reject at either stage leaves the project as it was before the
        stage started:
          * Reject in the setup dialog → nothing is inserted.
          * Reject in the preview dialog → the initial-guess block stays
            in place (one ``project.edit(...)`` entry the user can Ctrl+Z
            to remove).
        Accept in the preview dialog stacks a second edit entry so the
        first Ctrl+Z reverses the optimizer's apply while the block
        itself stays; a second Ctrl+Z removes the block.
        """
        # Deferred imports keep the module-load cost small for users who
        # never open the wizard, and dodge any circular graph between
        # optical_editor and optimization_panel.
        from .anamorphic_dialog import AnamorphicSetupDialog
        from ..optimization_panel.optimizer import OptimizationRun
        from ..optimization_panel.preview_dialog import OptimizationPreviewDialog

        setup = AnamorphicSetupDialog(self)
        if setup.exec() != QDialog.Accepted:
            return
        spec = setup.spec()

        try:
            new_elements, mf = element_actions.add_anamorphic_front_block(
                self._project, spec=spec,
            )
        except Exception as exc:  # noqa: BLE001 — user-facing wizard, log + surface
            QMessageBox.warning(
                self,
                "Anamorphic Front Block",
                f"Failed to build front block:\n\n{exc}",
            )
            return
        if not new_elements:
            return

        for element in new_elements:
            self._expand_element_subtree(element)
        self._project.set_selection(new_elements[0], None)

        try:
            run = OptimizationRun(
                self._project, mf, self._project.system_setup, self,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Anamorphic Front Block",
                (
                    "Front block was inserted, but the optimizer could not "
                    f"start:\n\n{exc}\n\nYou can still open the Optimization "
                    "panel to run the pre-built merit function manually."
                ),
            )
            return

        dlg = OptimizationPreviewDialog(self._project, mf, run, self)
        dlg.show()
        dlg.start_run()

    # ------------------------------------------------------------------
    # Expansion-state snapshot / restore (across model resets)
    # ------------------------------------------------------------------

    def _collect_expanded_keys(self) -> Set[str]:
        """Snapshot every expanded element AND surface row.

        Surfaces must be captured too, not just elements: a model reset
        rebuilds the whole tree, and a value-scrub edit triggers one on
        every commit, so anything not snapshotted here collapses under the
        user. Captured regardless of ancestor
        visibility — Qt persists isExpanded across an ancestor collapse,
        and the restoration mirrors that.
        """
        keys: Set[str] = set()
        root_count = self.model.rowCount(QModelIndex())
        for ei in range(root_count):
            el_idx = self.model.index(ei, 0, QModelIndex())
            el_node = el_idx.internalPointer()
            if isinstance(el_node, ElementNode) and el_node.element is not None:
                el_id = getattr(el_node.element, "element_id", None)
                if el_id and self.tree.isExpanded(el_idx):
                    keys.add(f"e:{el_id}")
            child_count = self.model.rowCount(el_idx)
            for ci in range(child_count):
                child_idx = self.model.index(ci, 0, el_idx)
                child_node = child_idx.internalPointer()
                if (
                    isinstance(child_node, SurfaceNode)
                    and child_node.surface_uuid
                    and self.tree.isExpanded(child_idx)
                ):
                    keys.add(f"s:{child_node.surface_uuid}")
        return keys

    def _apply_expanded_keys(self, keys: Set[str]) -> None:
        root_count = self.model.rowCount(QModelIndex())
        for ei in range(root_count):
            el_idx = self.model.index(ei, 0, QModelIndex())
            el_node = el_idx.internalPointer()
            if isinstance(el_node, ElementNode) and el_node.element is not None:
                el_id = getattr(el_node.element, "element_id", None)
                self.tree.setExpanded(
                    el_idx, bool(el_id and f"e:{el_id}" in keys)
                )
            child_count = self.model.rowCount(el_idx)
            for ci in range(child_count):
                child_idx = self.model.index(ci, 0, el_idx)
                child_node = child_idx.internalPointer()
                if isinstance(child_node, SurfaceNode) and child_node.surface_uuid:
                    self.tree.setExpanded(
                        child_idx, f"s:{child_node.surface_uuid}" in keys
                    )

    def _expand_element_subtree(self, element) -> None:
        idx = self._index_for_element(element)
        if not idx.isValid():
            return
        self.tree.expandRecursively(idx, -1)

    def _on_remove_element_requested(self) -> None:
        self._remove_element_with_confirm(self._project.selected_element)

    def _remove_element_with_confirm(self, element) -> None:
        if element is None:
            return
        name = element.name or "this element"
        answer = QMessageBox.question(
            self,
            "Remove element",
            f"Remove {name}? This will delete its surfaces and cannot be "
            f"undone except via Edit › Undo.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        element_actions.remove_element(self._project, element)

    def _index_for_surface(self, surface_index: int) -> QModelIndex:
        if surface_index < 0:
            return QModelIndex()
        root_count = self.model.rowCount(QModelIndex())
        for ei in range(root_count):
            el_idx = self.model.index(ei, 0, QModelIndex())
            child_count = self.model.rowCount(el_idx)
            for ci in range(child_count):
                child_idx = self.model.index(ci, 0, el_idx)
                node = child_idx.internalPointer()
                if isinstance(node, SurfaceNode) and node.surface_index == surface_index:
                    return child_idx
        return QModelIndex()
