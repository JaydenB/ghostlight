"""Value-scrubber popup for numeric cells in any tree view backed by ``Project``.

Ctrl+MiddleClick on a scrubbable cell opens a sensitivity picker.
MiddleClick on a sensitivity row both arms it and begins a drag; the
cell value live-updates from horizontal mouse motion via the model's
``setData(EditRole)``. Release ends the drag and the popup stays open.
Click Close X / press Escape / left- or right-click outside the popup
to dismiss.

By default a cell is considered scrubbable when its ``Qt.EditRole`` value
is a ``float`` and the model marks it ``Qt.ItemIsEditable``. Consumers
can pass a custom ``is_scrubbable`` callable to ``attach_value_scrubber``
to be more restrictive. The default ``compound_label`` reads the column's
header text so undo entries read as e.g. ``"Scrub Radius"``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QRect,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .project import Project


IsScrubbable = Callable[[QModelIndex], bool]
CompoundLabel = Callable[[QModelIndex], str]


def default_is_scrubbable(index: QModelIndex) -> bool:
    """Scrubbable when the cell is editable and its ``EditRole`` is numeric.

    Both ``float`` and ``int`` cells qualify. Int cells scrub at sub-unit
    sensitivity internally (the popup keeps a private float accumulator)
    and only push back to the model when the rounded integer changes —
    the user never sees the float.
    """
    if not index.isValid():
        return False
    model = index.model()
    if model is None:
        return False
    if not (model.flags(index) & Qt.ItemIsEditable):
        return False
    val = model.data(index, Qt.EditRole)
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, float))


def default_compound_label(index: QModelIndex) -> str:
    """Undo label derived from the column's header text."""
    if not index.isValid():
        return "Scrub"
    model = index.model()
    if model is None:
        return "Scrub"
    header = model.headerData(index.column(), Qt.Horizontal, Qt.DisplayRole)
    if header:
        return f"Scrub {header}"
    return "Scrub"


# Per-pixel rate: value_delta = delta_px * sensitivity / _PIXELS_PER_UNIT.
# Per locked design 1 px == sensitivity, so this stays at 1.0; raise it
# (e.g. 10.0) if the scrub ever feels too hot for the chosen multipliers.
_PIXELS_PER_UNIT = 1.0

_SENSITIVITIES: tuple[float, ...] = (100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001)

_BG = QColor("#2a2a2a")
_BG_ARMED = QColor("#3a6ea5")
_BG_HOVER = QColor("#383838")
_FG = QColor("#dddddd")
_FG_ARMED = QColor("#ffffff")
_SEPARATOR = QColor("#444444")
_BORDER = QColor("#555555")
_OVERLAY_FILL = QColor(80, 140, 255, 60)
_OVERLAY_BORDER = QColor(80, 140, 255, 220)


def _format_value(v: float) -> str:
    return f"{v:.4f}"


def _format_int_value(v: int) -> str:
    return f"{int(v)}"


def _format_sensitivity(s: float) -> str:
    if s >= 1.0:
        return f"{int(s)}"
    return f"{s:g}"


class _Row(QFrame):
    """Base for popup rows. Paints its own background so the parent's
    stylesheet does not bleed in."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hover = False
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)

    def _bg_color(self) -> QColor:
        return _BG_HOVER if self._hover else _BG

    def enterEvent(self, ev) -> None:
        self._hover = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color())


class _SensitivityRow(_Row):
    def __init__(self, sensitivity: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sensitivity = sensitivity
        self._armed = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 4, 24, 4)
        self._label = QLabel(_format_sensitivity(sensitivity), self)
        self._label.setStyleSheet(f"color: {_FG.name()};")
        layout.addWidget(self._label)
        layout.addStretch(1)

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def is_armed(self) -> bool:
        return self._armed

    def set_armed(self, armed: bool) -> None:
        if armed == self._armed:
            return
        self._armed = armed
        fg = _FG_ARMED if armed else _FG
        self._label.setStyleSheet(f"color: {fg.name()};")
        self.update()

    def _bg_color(self) -> QColor:
        if self._armed:
            return _BG_ARMED
        return super()._bg_color()


class _CloseRow(_Row):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 4, 24, 4)
        left = QLabel("Close", self)
        left.setStyleSheet(f"color: {_FG.name()};")
        right = QLabel("X", self)
        right.setStyleSheet(f"color: {_FG.name()};")
        layout.addWidget(left)
        layout.addStretch(1)
        layout.addWidget(right)


class _Separator(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _SEPARATOR)


class _RowHighlightOverlay(QWidget):
    """Translucent rectangle painted over the dragged tree row. Reparented
    to the tree viewport so it scrolls/repaints in sync with the rows.
    Mouse-transparent: never intercepts the drag itself."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _OVERLAY_FILL)
        painter.setPen(_OVERLAY_BORDER)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class ScrubPopup(QFrame):
    """Frameless popup that picks the per-pixel sensitivity and runs the
    middle-button drag for one tree cell."""

    def __init__(
        self,
        tree: QTreeView,
        index: QModelIndex,
        anchor_global: QPoint,
        project: "Project",
        compound_label: Optional[CompoundLabel] = None,
    ) -> None:
        super().__init__(
            tree,
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)

        self._tree = tree
        self._model = tree.model()
        self._pindex = QPersistentModelIndex(index)
        self._project = project
        self._column = index.column()
        self._compound_label: CompoundLabel = compound_label or default_compound_label
        self._compound_open = False
        # Int-mode is decided up-front by the initial EditRole type. The
        # cell-side type doesn't change during a drag, so we can lock the
        # decision in at popup-open time and trust it across writes.
        initial_raw = self._model.data(self._index(), Qt.EditRole)
        self._is_int = (
            isinstance(initial_raw, int) and not isinstance(initial_raw, bool)
        )
        # Sub-pixel scrub accumulator. The model only ever sees rounded ints
        # in int-mode; this float is internal-only so the per-pixel
        # sensitivity feels the same as a float cell.
        self._scrub_float: float = float(self._read_value())
        self._last_written_int: Optional[int] = (
            int(round(self._scrub_float)) if self._is_int else None
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        self._value_label = QLabel(self._format_current(), self)
        self._value_label.setAlignment(Qt.AlignCenter)
        font = QFont(self._value_label.font())
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        self._value_label.setFont(font)
        self._value_label.setContentsMargins(16, 6, 16, 6)
        self._value_label.setStyleSheet(
            f"color: {_FG.name()}; background-color: {_BG.name()};"
        )
        outer.addWidget(self._value_label)
        outer.addWidget(_Separator(self))

        self._rows: list[_SensitivityRow] = []
        for s in _SENSITIVITIES:
            row = _SensitivityRow(s, self)
            outer.addWidget(row)
            self._rows.append(row)

        outer.addWidget(_Separator(self))
        self._close_row = _CloseRow(self)
        outer.addWidget(self._close_row)

        # Drag state.
        self._dragging = False
        self._armed_row: Optional[_SensitivityRow] = None
        self._press_global = QPoint()
        self._press_value = 0.0
        self._total_dx = 0
        self._cursor_override = False
        self._overlay: Optional[_RowHighlightOverlay] = None

        # Live label updates from external edits + bail if model resets.
        self._model.dataChanged.connect(self._on_data_changed)
        self._model.modelReset.connect(self.close)

        self.adjustSize()
        self.move(self._clamp_to_screen(anchor_global))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BG)
        # A 1-px pen on drawRect's right/bottom edges has half its width
        # falling outside the widget bounds, where HiDPI scaling can clip it.
        # Use a 2-px pen inset by 1 px so the full stroke is always inside
        # the paintable area on every edge.
        pen = QPen(_BORDER, 2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

    # ------------------------------------------------------------------
    # Model glue
    # ------------------------------------------------------------------

    def _index(self) -> QModelIndex:
        if not self._pindex.isValid():
            return QModelIndex()
        return QModelIndex(self._pindex)

    def _read_value(self) -> float:
        idx = self._index()
        if not idx.isValid():
            return 0.0
        val = self._model.data(idx, Qt.EditRole)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _write_value(self, value: float) -> None:
        idx = self._index()
        if not idx.isValid():
            return
        if self._is_int:
            new_int = int(round(value))
            if new_int == self._last_written_int:
                # The accumulator advanced sub-pixel-of-a-unit; don't push
                # an identical int through the model or it allocates an
                # undo entry per pixel even when nothing changes.
                return
            self._last_written_int = new_int
            self._model.setData(idx, new_int, Qt.EditRole)
        else:
            self._model.setData(idx, float(value), Qt.EditRole)

    def _format_current(self) -> str:
        """Stringify the value label from the live model state.

        In int-mode we show the int (no decimals) so the popup matches the
        cell. The internal float accumulator is never surfaced.
        """
        if self._is_int:
            try:
                return _format_int_value(int(self._read_value()))
            except (TypeError, ValueError):
                return "0"
        return _format_value(self._read_value())

    def _on_data_changed(
        self,
        top_left: QModelIndex,
        bottom_right: QModelIndex,
        _roles=(),
    ) -> None:
        idx = self._index()
        if not idx.isValid():
            self.close()
            return
        if (top_left.parent() == idx.parent()
                and top_left.row() <= idx.row() <= bottom_right.row()
                and top_left.column() <= idx.column() <= bottom_right.column()):
            self._value_label.setText(self._format_current())

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _clamp_to_screen(self, pos: QPoint) -> QPoint:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return pos
        avail = screen.availableGeometry()
        size = self.sizeHint()
        x = max(avail.left(), min(pos.x(), avail.right() - size.width()))
        y = max(avail.top(), min(pos.y(), avail.bottom() - size.height()))
        return QPoint(x, y)

    def _row_under(self, local: QPoint) -> Optional[QWidget]:
        widget = self.childAt(local)
        while widget is not None and widget is not self:
            if isinstance(widget, (_SensitivityRow, _CloseRow)):
                return widget
            widget = widget.parentWidget()
        return None

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        local = ev.position().toPoint()
        inside = self.rect().contains(local)

        if not inside:
            # Outside-click policy: LMB/RMB close, MMB is ignored.
            if ev.button() == Qt.MiddleButton:
                ev.accept()
                return
            self.close()
            ev.accept()
            return

        target = self._row_under(local)
        if ev.button() == Qt.MiddleButton:
            if isinstance(target, _SensitivityRow):
                self._begin_drag(target, ev.globalPosition().toPoint())
            ev.accept()
            return
        if ev.button() == Qt.LeftButton:
            if isinstance(target, _CloseRow):
                self.close()
            # LMB on sensitivity rows / empty space: no-op (no arming, no close).
            ev.accept()
            return
        ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if not self._dragging:
            super().mouseMoveEvent(ev)
            return
        current = ev.globalPosition().toPoint()
        dx = current.x() - self._press_global.x()
        if dx == 0:
            return
        self._total_dx += dx
        new_value = (
            self._press_value
            + self._total_dx * self._armed_sensitivity() / _PIXELS_PER_UNIT
        )
        # Keep the internal float so int-mode can resolve sub-unit motion
        # across many small pixel deltas. _write_value rounds when needed.
        self._scrub_float = new_value
        self._write_value(new_value)
        QCursor.setPos(self._press_global)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._dragging and ev.button() == Qt.MiddleButton:
            self._end_drag()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------
    # Drag lifecycle
    # ------------------------------------------------------------------

    def _armed_sensitivity(self) -> float:
        return self._armed_row.sensitivity if self._armed_row is not None else 1.0

    def _begin_drag(self, row: _SensitivityRow, global_pos: QPoint) -> None:
        if self._armed_row is not None and self._armed_row is not row:
            self._armed_row.set_armed(False)
        self._armed_row = row
        row.set_armed(True)
        # Re-anchor the press point and zero the accumulator on every MMB
        # press so a mid-drag sensitivity switch rebases cleanly.
        self._press_global = QPoint(global_pos)
        # Int-mode uses the internal float accumulator as the press value so
        # sub-unit precision survives mid-drag sensitivity switches; float
        # cells rebase off the live model value as before.
        self._press_value = (
            self._scrub_float if self._is_int else self._read_value()
        )
        self._total_dx = 0
        if self._dragging:
            # Mid-drag MMB repress (e.g. switching sensitivity row): keep the
            # single cursor override + single undo compound we already
            # opened. Pushing again would leak the override stack — release
            # only pops once — and stuck blank-cursor on let-go is the
            # symptom.
            return
        self._dragging = True
        # Coalesce every per-pixel write during this drag into one undo
        # entry. Multiple drags within the same open popup produce separate
        # entries (compound opens/closes per drag, not per popup).
        self._project.begin_compound(self._compound_label(self._index()))
        self._compound_open = True
        QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
        self._cursor_override = True
        self._show_overlay()

    def _end_drag(self) -> None:
        self._dragging = False
        if self._cursor_override:
            QApplication.restoreOverrideCursor()
            self._cursor_override = False
        QCursor.setPos(self._press_global)
        self._hide_overlay()
        if self._compound_open:
            self._project.end_compound()
            self._compound_open = False

    # ------------------------------------------------------------------
    # Row-highlight overlay on the tree
    # ------------------------------------------------------------------

    def _show_overlay(self) -> None:
        idx = self._index()
        if not idx.isValid():
            return
        viewport = self._tree.viewport()
        rect = self._tree.visualRect(idx)
        if rect.isEmpty():
            return
        full = QRect(0, rect.top(), viewport.width(), rect.height())
        if self._overlay is None:
            self._overlay = _RowHighlightOverlay(viewport)
        self._overlay.setGeometry(full)
        self._overlay.show()
        self._overlay.raise_()

    def _hide_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.hide()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, ev) -> None:
        # Defensive: a popup dismissed mid-drag (Escape, click-away, model
        # reset) must still close the compound or later edits leak into it.
        if self._compound_open:
            self._project.end_compound()
            self._compound_open = False
        if self._cursor_override:
            QApplication.restoreOverrideCursor()
            self._cursor_override = False
        if self._overlay is not None:
            self._overlay.deleteLater()
            self._overlay = None
        try:
            self._model.dataChanged.disconnect(self._on_data_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            self._model.modelReset.disconnect(self.close)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(ev)


# ----------------------------------------------------------------------
# Attachment helper
# ----------------------------------------------------------------------


class _ScrubTrigger(QWidget):
    """Hidden helper: filters mouse events on the tree's viewport, opens a
    ``ScrubPopup`` on Ctrl+MMB over a scrubbable cell."""

    def __init__(
        self,
        tree: QTreeView,
        project: "Project",
        is_scrubbable: IsScrubbable,
        compound_label: CompoundLabel,
    ) -> None:
        super().__init__(tree)
        self.hide()
        self._tree: Optional[QTreeView] = tree
        self._project = project
        self._is_scrubbable = is_scrubbable
        self._compound_label = compound_label
        tree.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:
        # During panel-type switches and app/test teardown, Qt may dispatch
        # a final event to this filter while the tree (or our own Python
        # wrapper) is being torn down. Guard against both: a partially
        # stripped __dict__ (getattr default) AND a still-present Python
        # wrapper whose C++ is gone (RuntimeError on .viewport()).
        tree = getattr(self, "_tree", None)
        if tree is None:
            return False
        try:
            viewport = tree.viewport()
        except RuntimeError:
            self._tree = None
            return False
        if obj is not viewport:
            return False
        if ev.type() != QEvent.MouseButtonPress:
            return False
        if ev.button() != Qt.MiddleButton:
            return False
        if not (ev.modifiers() & Qt.ControlModifier):
            return False
        index = tree.indexAt(ev.position().toPoint())
        if not index.isValid():
            return False
        if not self._is_scrubbable(index):
            return False
        self._open_popup(index, ev.globalPosition().toPoint())
        return True

    def _open_popup(self, index: QModelIndex, anchor_global: QPoint) -> Optional["ScrubPopup"]:
        """Open a scrub popup at ``index``. Also surfaces the row in the
        tree's selection so the post-drag rebuild lands the user on the
        row they were scrubbing.

        Without the selection sync, the tree's current row stayed on
        whatever was selected before the Ctrl+MMB (typically the parent
        Element). After ``end_compound`` fires ``systemModified`` and
        the model rebuilds, ``_on_model_reset`` restores selection from
        the project — which still pointed at the Element. Net effect:
        finish a surface drag, end up with the *element* highlighted.
        Setting the current index here pushes the scrubbed row down to
        the project via the body's ``currentRowChanged`` slot, so the
        post-reset restoration finds the surface.
        """
        tree = self._tree
        if tree is None:
            return None
        tree.closePersistentEditor(index)
        sel_model = tree.selectionModel()
        if sel_model is not None:
            sel_model.setCurrentIndex(
                index,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )
        popup = ScrubPopup(
            tree, index, anchor_global, self._project, self._compound_label
        )
        popup.show()
        return popup


def attach_value_scrubber(
    tree: QTreeView,
    project: "Project",
    *,
    is_scrubbable: Optional[IsScrubbable] = None,
    compound_label: Optional[CompoundLabel] = None,
) -> _ScrubTrigger:
    """Install Ctrl+MMB value-scrubbing on ``tree`` for the given project.

    ``is_scrubbable(index) -> bool`` gates which cells respond to Ctrl+MMB.
    Defaults to :func:`default_is_scrubbable` (editable float cells).

    ``compound_label(index) -> str`` produces the undo-entry label for one
    drag session. Defaults to :func:`default_compound_label` which uses
    the column's header text.

    Returns the trigger object (parented to the tree) so the caller can
    keep a reference if desired; in practice the tree's parenthood keeps
    it alive.
    """
    return _ScrubTrigger(
        tree,
        project,
        is_scrubbable or default_is_scrubbable,
        compound_label or default_compound_label,
    )
