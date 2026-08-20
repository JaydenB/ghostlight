"""``QAbstractItemModel`` exposing the optical-system tree."""
from __future__ import annotations

from PySide6.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from PySide6.QtGui import QFont

from ..project import Project
from . import element_actions
from . import row_schemas as schemas
from .delegates import NodeKindRole, SlotRole
from .icons import icon_for
from .nodes import ElementNode, RootNode, SurfaceNode, TreeNode, build_tree


ELEMENT_ROW_MIME = "application/x-ghostlight-element-row"


# Default mode for surface Pos Z when the user hasn't picked one.  Loaded
# documents start out showing every surface's thickness, since that's what's
# actually on disk; absolute z is a computed view.
DEFAULT_POS_Z_MODE = "relative"

# Canonical column index used for the per-surface Pos-Z broadcast and for the
# per-surface mode-toggle refresh. Resolved once at import; ``CANONICAL_COLUMNS``
# is the single source of truth.
_POS_Z_COLUMN = schemas.canonical_column_for("pos_z") or 0
_NAME_COLUMN = 0
_OFF_AXIS_COLUMN = schemas.canonical_column_for("off_axis") or 0


class OpticalTreeModel(QAbstractItemModel):
    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._root: RootNode = RootNode()
        self._suppress_modified_rebuild: int = 0
        # Per-surface Pos-Z display-mode overrides, keyed by surface UUID.
        # Absent key means ``DEFAULT_POS_Z_MODE``; presence means the user
        # explicitly flipped this surface away from the default.
        # Display-only; never persisted to .lens.
        self._pos_z_modes: dict[str, str] = {}
        project.systemReplaced.connect(self._on_system_changed)
        project.systemModified.connect(self._on_modified)
        # Reveal state lives on the Project (so the viewport / other panels
        # could read it) but never touches the system, so it doesn't fire
        # systemModified — repaint the toggle column directly instead. No
        # rebuild: the tree structure is identical either way.
        project.offAxisRevealChanged.connect(self._on_off_axis_reveal_changed)
        # Ghost-solo state lives outside the system mutation path, so a
        # solo toggle doesn't fire systemModified — listen directly here
        # so the tree rebuilds and the solo icon flips immediately.
        project.ghostSoloChanged.connect(self._on_modified)
        self._rebuild()

    @property
    def project(self) -> Project:
        """The bound Project — used by cell pickers to reach the live system."""
        return self._project

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        self.beginResetModel()
        try:
            self._root = build_tree(
                self._project.system,
                solo_uuids=self._project.ghost_solo_surface_uuids,
            )
        finally:
            self.endResetModel()

    def _on_system_changed(self, *_args) -> None:
        # Modes are scoped to the current document — a new system means new
        # surfaces (and new UUIDs); drop any held state before rebuilding.
        self._pos_z_modes.clear()
        self._rebuild()

    def _on_modified(self) -> None:
        if self._suppress_modified_rebuild > 0:
            return
        self._rebuild()

    def _node(self, index: QModelIndex) -> TreeNode:
        if not index.isValid():
            return self._root
        ptr = index.internalPointer()
        return ptr if ptr is not None else self._root

    def _ctx(self, node: TreeNode) -> schemas.SlotContext:
        return schemas.SlotContext(
            node=node,
            system=self._project.system,
            project=self._project,
            pos_z_mode=self._mode_for_node(node),
        )

    # ------------------------------------------------------------------
    # QAbstractItemModel
    # ------------------------------------------------------------------

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return schemas.column_count(self._project.system)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._node(parent)
        return len(node.children)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = self._node(parent)
        if row < 0 or row >= len(parent_node.children):
            return QModelIndex()
        return self.createIndex(row, column, parent_node.children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        node = self._node(index)
        parent_node = node.parent
        if parent_node is None or parent_node is self._root:
            return QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return schemas.header_text(section)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            # Allow drops at the root (between top-level element rows) so the
            # tree's drop indicator can land in the gap above / below elements.
            return Qt.ItemIsEnabled | Qt.ItemIsDropEnabled
        node = self._node(index)
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() != _NAME_COLUMN:
            slot = schemas.slot_at(node, index.column())
            if slot is not None and slot.is_editable(self._ctx(node)):
                base |= Qt.ItemIsEditable
        if isinstance(node, ElementNode):
            # Only elements are draggable for reordering — surfaces / materials
            # belong to their parent element and don't move independently.
            base |= Qt.ItemIsDragEnabled
        return base

    # ------------------------------------------------------------------
    # Drag-and-drop reordering — element rows only
    # ------------------------------------------------------------------

    def supportedDropActions(self) -> Qt.DropActions:
        return Qt.MoveAction

    def mimeTypes(self) -> list[str]:
        return [ELEMENT_ROW_MIME]

    def mimeData(self, indexes) -> QMimeData:
        rows = sorted({
            i.row() for i in indexes
            if i.isValid() and not i.parent().isValid()
        })
        data = QMimeData()
        if not rows:
            # Returning a valid (empty-payload) QMimeData prevents Qt from
            # initiating a drag for non-element rows.
            return data
        # Single-row payload — multi-select drag is intentionally not
        # supported; users reorder one row at a time.
        data.setData(ELEMENT_ROW_MIME, str(rows[0]).encode("utf-8"))
        return data

    def canDropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent: QModelIndex,
    ) -> bool:
        if action != Qt.MoveAction:
            return False
        if not data.hasFormat(ELEMENT_ROW_MIME):
            return False
        # Only allow drops between top-level rows (parent is root).
        return not parent.isValid()

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent: QModelIndex,
    ) -> bool:
        if not self.canDropMimeData(data, action, row, column, parent):
            return False
        try:
            src = int(bytes(data.data(ELEMENT_ROW_MIME)).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False
        # row == -1 means "dropped onto the parent" → for the root that's
        # equivalent to "append at the end".
        target = row if row >= 0 else self.rowCount(QModelIndex())
        element_actions.move_element(self._project, src, target)
        # Always return False so Qt's drag-drop machinery doesn't try to call
        # removeRows() on the source row — ``move_element`` already rebuilt
        # the system atomically through the Project's edit / undo path, and a
        # second removal would corrupt the new ordering.
        return False

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        node = self._node(index)
        ctx = self._ctx(node)

        # Italicise muted-element rows across every column, not just Name.
        if role == Qt.FontRole and isinstance(node, ElementNode) and node.muted:
            font = QFont()
            font.setItalic(True)
            return font

        if index.column() == _NAME_COLUMN:
            if role == Qt.DisplayRole:
                return schemas.name_label(ctx)
            if role == Qt.DecorationRole:
                return icon_for(node.icon_name)
            if role == NodeKindRole:
                return int(node.kind)
            return None

        if role == NodeKindRole:
            return int(node.kind)

        slot = schemas.slot_at(node, index.column())
        if slot is None:
            # Preserve the old per-column convention: empty *string* for an
            # unpopulated DisplayRole cell so view code that string-compares
            # against "" keeps working. EditRole stays ``None`` so the cell
            # is read as "no value" by the scrubber and editor factories.
            if role == Qt.DisplayRole:
                return ""
            return None
        if role == SlotRole:
            return slot
        if role == Qt.DisplayRole:
            return slot.resolve_display(ctx)
        if role == Qt.EditRole:
            return slot.get(ctx)
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if role != Qt.EditRole or not index.isValid():
            return False
        node = self._node(index)
        slot = schemas.slot_at(node, index.column())
        if slot is None:
            return False
        ctx = self._ctx(node)
        label = slot.resolve_undo_label(ctx)
        self._suppress_modified_rebuild += 1
        try:
            with self._project.edit(label) as txn:
                result = slot.write(ctx, value)
                if not result.changed:
                    txn.abort()
                    return False
                if result.requires_reset:
                    self.beginResetModel()
                    self._root = build_tree(
                        self._project.system,
                        solo_uuids=self._project.ghost_solo_surface_uuids,
                    )
                    self.endResetModel()
                else:
                    self.dataChanged.emit(
                        index, index, [Qt.DisplayRole, Qt.EditRole]
                    )
                    if result.broadcast_column == "pos_z":
                        # Pos-Z relative-mode write called ``system.finalize()``,
                        # which can shift z on surfaces other than the one
                        # edited. Broadcast — but as per-parent dataChanged,
                        # not a model reset, so an active value-scrubber drag
                        # survives.
                        self._emit_pos_z_column_changed()
        finally:
            self._suppress_modified_rebuild -= 1
        return True

    def _emit_pos_z_column_changed(self) -> None:
        """Emit ``dataChanged`` for every surface Pos Z cell in the tree."""
        for el_node in self._root.children:
            if not el_node.children:
                continue
            top = self.createIndex(0, _POS_Z_COLUMN, el_node.children[0])
            bot = self.createIndex(
                len(el_node.children) - 1, _POS_Z_COLUMN, el_node.children[-1]
            )
            self.dataChanged.emit(top, bot, [Qt.DisplayRole, Qt.EditRole])

    # ------------------------------------------------------------------
    # Off-axis reveal — the ">>>" toggle and the column block it controls
    #
    # Effective state is  explicit-reveal OR holds-a-non-zero-value.  The
    # second half is the rule that any non-zero position parameter forces Off
    # Axis on and blocks hiding it, which is what makes it safe not to persist
    # the flag: a saved file's off-axis data always comes back visible.
    # ------------------------------------------------------------------

    def off_axis_state(self, node: TreeNode) -> tuple[bool, bool]:
        """``(revealed, locked)`` for an element node; (False, False) otherwise."""
        element = getattr(node, "element", None)
        if not isinstance(node, ElementNode) or element is None:
            return (False, False)
        try:
            locked = schemas.element_has_off_axis_value(element)
        except (AttributeError, TypeError, IndexError):
            locked = False
        if locked:
            return (True, True)
        element_id = getattr(element, "element_id", "") or ""
        try:
            explicit = bool(self._project.is_off_axis_revealed(element_id))
        except AttributeError:
            explicit = False
        return (explicit, False)

    def any_off_axis_revealed(self) -> bool:
        """True when at least one element wants the extra columns on screen.

        Columns are a property of the whole view, so one revealed row shows
        them for every row; rows that aren't revealed simply render blank
        there. The view calls this to decide what to hide.
        """
        for node in self._root.children:
            if self.off_axis_state(node)[0]:
                return True
        return False

    def button_state(self, index: QModelIndex) -> tuple[bool, bool]:
        """``(checked, locked)`` for a ``SlotEditor.BUTTON`` cell.

        Read by the delegate to pick the button's paint state.
        """
        if not index.isValid():
            return (False, False)
        return self.off_axis_state(self._node(index))

    def toggle_button(self, index: QModelIndex) -> None:
        """Handle a click on a ``SlotEditor.BUTTON`` cell.

        Deliberately NOT routed through ``setData``: reveal state isn't part
        of the document, so it must not open a ``project.edit`` transaction
        or land on the undo stack.
        """
        if not index.isValid():
            return
        node = self._node(index)
        element = getattr(node, "element", None)
        if not isinstance(node, ElementNode) or element is None:
            return
        revealed, locked = self.off_axis_state(node)
        if locked:
            # Non-zero values hold the row open; there is nothing to toggle.
            return
        element_id = getattr(element, "element_id", "") or ""
        try:
            self._project.set_off_axis_revealed(element_id, not revealed)
        except AttributeError:
            return

    def _on_off_axis_reveal_changed(self) -> None:
        """Repaint the toggle column — the tree itself hasn't changed."""
        if not self._root.children:
            return
        top = self.createIndex(0, _OFF_AXIS_COLUMN, self._root.children[0])
        bot = self.createIndex(
            len(self._root.children) - 1, _OFF_AXIS_COLUMN, self._root.children[-1]
        )
        self.dataChanged.emit(top, bot, [Qt.DisplayRole])

    # ------------------------------------------------------------------
    # Per-surface Pos-Z mode (display-only)
    # ------------------------------------------------------------------

    def _mode_for_node(self, node: TreeNode) -> str:
        if isinstance(node, SurfaceNode):
            return self._pos_z_modes.get(node.surface_uuid, DEFAULT_POS_Z_MODE)
        # Non-surface rows don't render Pos Z, so mode is irrelevant for them.
        return "absolute"

    def pos_z_mode(self, surface_uuid: str) -> str:
        """Return ``"absolute"`` or ``"relative"`` for ``surface_uuid``."""
        return self._pos_z_modes.get(surface_uuid, DEFAULT_POS_Z_MODE)

    def set_pos_z_mode(self, surface_uuid: str, mode: str) -> None:
        """Set the per-surface display mode and refresh that row's Pos Z cell.

        ``mode`` must be ``"absolute"`` or ``"relative"``.  Storing the
        default mode clears the entry.
        """
        if mode not in ("absolute", "relative"):
            return
        current = self._pos_z_modes.get(surface_uuid, DEFAULT_POS_Z_MODE)
        if current == mode:
            return
        if mode == DEFAULT_POS_Z_MODE:
            self._pos_z_modes.pop(surface_uuid, None)
        else:
            self._pos_z_modes[surface_uuid] = mode
        idx = self._index_for_surface_uuid(surface_uuid)
        if idx.isValid():
            pos_idx = self.index(idx.row(), _POS_Z_COLUMN, self.parent(idx))
            self.dataChanged.emit(pos_idx, pos_idx, [Qt.DisplayRole, Qt.EditRole])

    def _index_for_surface_uuid(self, surface_uuid: str) -> QModelIndex:
        for el_node in self._root.children:
            if not isinstance(el_node, ElementNode):
                continue
            for ci, child in enumerate(el_node.children):
                if isinstance(child, SurfaceNode) and child.surface_uuid == surface_uuid:
                    return self.createIndex(ci, 0, child)
        return QModelIndex()
