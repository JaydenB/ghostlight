"""Item delegate for the optical-editor tree view.

One ``SlotDelegate`` handles every cell. Editor creation, value sync, and
commit are dispatched off the ``Slot`` returned by the model's
``SlotRole`` — adding a new editor type is one entry in
``_EDITOR_OPS`` plus an enum member in :mod:`row_schemas`.

``NodeKindRole`` and ``SlotRole`` are kept as module-level constants so
the model + the body's click-to-edit filter agree on the role numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from PySide6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from . import row_schemas as schemas
from .nodes import surface_uuid_for
from .row_schemas import Slot, SlotEditor


# Picker callable signature: (parent_widget, index) -> new_value | None.
# Returning ``None`` means "user cancelled, leave the cell unchanged".
PickerFn = Callable[[QWidget, QModelIndex], Optional[str]]


# Paint constants for the two-strip cell layout: a small dim label up top
# (when the in-cell label adds info over the strip header), then the value
# centered in the rest of the cell.
_ROW_HEIGHT_MULT = 2.4        # cell height = roughly 2.4x a single text line
_LABEL_POINT_SCALE = 0.82     # label rendered at 82% of the value point size
_LABEL_MIN_POINT = 7.0
_CELL_PAD_X = 6
_LABEL_PAD_Y = 2

# Variable-flag left-edge stripe: a thin vertical bar hugging the left
# edge of a cell whose slot's ``variable_attr`` is currently flagged on
# the project. Amber is distinct from the selection highlight (blue) and
# stays legible against both alt-row banding shades.
_VAR_STRIPE_WIDTH = 3
_VAR_STRIPE_COLOR = QColor(0xF5, 0xA6, 0x23)  # amber


NodeKindRole = Qt.UserRole + 1
SlotRole = Qt.UserRole + 2


# ---------------------------------------------------------------------------
# Editor-type factories
# ---------------------------------------------------------------------------

def _make_text_editor(parent: QWidget, _slot: Slot) -> QWidget:
    editor = QLineEdit(parent)
    editor.setAutoFillBackground(True)
    return editor


def _set_text(editor: QLineEdit, value: Any) -> None:
    editor.setText("" if value is None else str(value))


def _commit_text(
    editor: QLineEdit, _slot: Slot, model: QAbstractItemModel, index: QModelIndex
) -> None:
    model.setData(index, editor.text(), Qt.EditRole)


def _make_float_editor(parent: QWidget, slot: Slot) -> QWidget:
    # MathDoubleSpinBox (not QDoubleSpinBox) so the cell accepts a typed
    # calculation, e.g. "12*2"; see ghostlight_designer.math_spinbox.
    editor = MathDoubleSpinBox(parent)
    editor.setDecimals(int(slot.options.get("decimals", 4)))
    editor.setRange(-1.0e9, 1.0e9)
    editor.setSingleStep(0.1)
    editor.setAutoFillBackground(True)
    return editor


def _set_float(editor: QDoubleSpinBox, value: Any) -> None:
    try:
        editor.setValue(float(value) if value is not None else 0.0)
    except (TypeError, ValueError):
        editor.setValue(0.0)


def _commit_float(
    editor: QDoubleSpinBox,
    _slot: Slot,
    model: QAbstractItemModel,
    index: QModelIndex,
) -> None:
    editor.interpretText()
    model.setData(index, float(editor.value()), Qt.EditRole)


def _make_int_spinbox_editor(parent: QWidget, slot: Slot) -> QWidget:
    editor = MathSpinBox(parent)
    editor.setRange(
        int(slot.options.get("min", -1_000_000)),
        int(slot.options.get("max", 1_000_000)),
    )
    editor.setSingleStep(int(slot.options.get("step", 1)))
    editor.setAutoFillBackground(True)
    return editor


def _set_int_spinbox(editor: QSpinBox, value: Any) -> None:
    try:
        editor.setValue(int(value) if value is not None else 0)
    except (TypeError, ValueError):
        editor.setValue(0)


def _commit_int_spinbox(
    editor: QSpinBox,
    _slot: Slot,
    model: QAbstractItemModel,
    index: QModelIndex,
) -> None:
    editor.interpretText()
    model.setData(index, int(editor.value()), Qt.EditRole)


def _make_enum_combo_editor(parent: QWidget, slot: Slot) -> QWidget:
    enum_cls = slot.options.get("enum")
    if enum_cls is None:
        raise KeyError(
            f"slot {slot.key!r}: SlotEditor.ENUM_COMBO needs options['enum']"
        )
    # `exclude` hides specific enum integers from the dropdown while
    # keeping the schema's `fmt` callback able to render them — useful
    # when the binding's enum carries a value the UI doesn't expose yet
    # (e.g. ApertureShape.IMAGE), but legacy loaded data might still
    # carry that value and we don't want it to format as "Unknown".
    exclude = {int(v) for v in slot.options.get("exclude", ())}
    editor = QComboBox(parent)
    for member in enum_cls.__members__.values():
        if int(member) in exclude:
            continue
        editor.addItem(member.name, int(member))
    # Opaque background so the cell's display string doesn't bleed through
    # the combo's transparent areas (the "duplicate text" ghost users were
    # seeing on the cyl-axis dropdown before this was set).
    editor.setAutoFillBackground(True)
    return editor


def _set_enum_combo(editor: QComboBox, value: Any) -> None:
    try:
        ival = int(value) if value is not None else 0
    except (TypeError, ValueError):
        ival = 0
    idx = editor.findData(ival)
    if idx >= 0:
        editor.setCurrentIndex(idx)


def _commit_enum_combo(
    editor: QComboBox,
    _slot: Slot,
    model: QAbstractItemModel,
    index: QModelIndex,
) -> None:
    value = editor.currentData()
    if value is None:
        value = int(editor.currentIndex())
    model.setData(index, int(value), Qt.EditRole)


@dataclass(frozen=True)
class _EditorOps:
    create: Callable[[QWidget, Slot], QWidget]
    set_data: Callable[[QWidget, Any], None]
    commit: Callable[[QWidget, Slot, QAbstractItemModel, QModelIndex], None]


def _make_string_combo_editor(parent: QWidget, slot: Slot) -> QWidget:
    choices_opt = slot.options.get("choices")
    if callable(choices_opt):
        choices = list(choices_opt())
    else:
        choices = list(choices_opt or [])
    editor = QComboBox(parent)
    for choice in choices:
        s = str(choice)
        editor.addItem(s, s)
    editor.setAutoFillBackground(True)
    return editor


def _set_string_combo(editor: QComboBox, value: Any) -> None:
    s = "" if value is None else str(value)
    idx = editor.findData(s)
    if idx < 0:
        idx = editor.findText(s)
    if idx >= 0:
        editor.setCurrentIndex(idx)


def _commit_string_combo(
    editor: QComboBox,
    _slot: Slot,
    model: QAbstractItemModel,
    index: QModelIndex,
) -> None:
    value = editor.currentData()
    if value is None:
        value = editor.currentText()
    model.setData(index, str(value), Qt.EditRole)


_EDITOR_OPS: dict[SlotEditor, _EditorOps] = {
    SlotEditor.TEXT: _EditorOps(_make_text_editor, _set_text, _commit_text),
    SlotEditor.FLOAT: _EditorOps(_make_float_editor, _set_float, _commit_float),
    SlotEditor.INT_SPINBOX: _EditorOps(
        _make_int_spinbox_editor, _set_int_spinbox, _commit_int_spinbox
    ),
    SlotEditor.ENUM_COMBO: _EditorOps(
        _make_enum_combo_editor, _set_enum_combo, _commit_enum_combo
    ),
    SlotEditor.STRING_COMBO: _EditorOps(
        _make_string_combo_editor, _set_string_combo, _commit_string_combo
    ),
    # TEXT_PICKER reuses the plain-text editor for inline typing; the
    # persistent picker button lives outside the editor (painted by the
    # delegate, hit-tested in ``editorEvent``).
    SlotEditor.TEXT_PICKER: _EditorOps(_make_text_editor, _set_text, _commit_text),
}


# ---------------------------------------------------------------------------
# Combo commit-on-pick
# ---------------------------------------------------------------------------

def _wire_combo_commit(
    delegate: QStyledItemDelegate, editor: QComboBox
) -> None:
    """Wire ``editor`` to commit + close on the first option click.

    Connects to *both* the popup view's ``clicked`` signal (fires on a real
    mouse press+release on an option, *before* ``activated`` and before the
    popup hides) and ``activated`` (covers keyboard Up/Down + Enter). A
    once-flag stops the duplicate from firing twice.

    ``editor.hide()`` after closeEditor forces the cell to repaint with
    the new value immediately — without it the closed editor lingers for
    a couple of frames and users perceive a lag between picking and the
    cell updating.
    """
    state = {"done": False}

    def commit() -> None:
        if state["done"]:
            return
        state["done"] = True
        delegate.commitData.emit(editor)
        delegate.closeEditor.emit(editor)
        editor.hide()

    editor.activated.connect(lambda _i: commit())
    view = editor.view()
    if view is not None:
        view.clicked.connect(lambda _idx: commit())


# ---------------------------------------------------------------------------
# Delegate
# ---------------------------------------------------------------------------

_PICKER_BUTTON_GLYPH = "…"  # "…"
_PICKER_BUTTON_WIDTH = 22
_PICKER_BUTTON_MARGIN = 2

# SlotEditor.BUTTON cells. Unlike the picker button (a small square tucked into
# the right of a cell that also shows a value), these own their whole cell, so
# they get a wide pill sized to the glyph rather than a fixed square.
_CELL_BUTTON_MARGIN = 3
_CELL_BUTTON_MIN_WIDTH = 34


class SlotDelegate(QStyledItemDelegate):
    """Single delegate for every cell — dispatches off the ``Slot`` returned
    by the model under ``SlotRole``. NAME column cells (column 0) have no
    slot and fall back to ``QStyledItemDelegate``'s default behavior, which
    paints icon + display text without offering an editor.
    """

    def __init__(
        self,
        parent=None,
        *,
        pickers: Optional[Mapping[str, PickerFn]] = None,
        project: Any = None,
    ) -> None:
        super().__init__(parent)
        # Slot.options["picker"] is a string tag; pickers maps tag → fn.
        # Kept on the delegate so body.py owns the dialog UI and the
        # delegate stays free of project / catalogue dependencies.
        self._pickers: Mapping[str, PickerFn] = pickers or {}
        # ``project`` is optional so headless tests that construct a
        # delegate without one still work. When present, the delegate
        # paints the variable-flag stripe on any cell whose slot carries
        # ``options["variable_attr"]`` and whose surface is flagged.
        self._project = project

    def _slot(self, index: QModelIndex) -> Optional[Slot]:
        data = index.data(SlotRole)
        return data if isinstance(data, Slot) else None

    def _variable_attr(self, slot: Slot) -> str:
        """Slot's ``variable_attr`` option, or "" if not flag-able."""
        return str(slot.options.get("variable_attr", "") or "")

    def _cell_is_variable_flagged(
        self, slot: Slot, index: QModelIndex,
    ) -> bool:
        """True iff this cell's slot is flag-able AND the project's
        variable-flag map has an entry for this surface + attr."""
        if self._project is None:
            return False
        attr = self._variable_attr(slot)
        if not attr:
            return False
        node = index.internalPointer()
        if node is None:
            return False
        uuid = surface_uuid_for(node)
        if not uuid:
            return False
        try:
            return bool(self._project.is_variable_flagged(uuid, attr))
        except AttributeError:
            # Project without the variable-flag API (test stub). Degrade
            # silently — the stripe simply won't paint.
            return False

    def _is_editable(self, index: QModelIndex) -> bool:
        model = index.model()
        if model is None:
            return False
        return bool(model.flags(index) & Qt.ItemIsEditable)

    def _picker_button_rect(self, cell_rect: QRect) -> QRect:
        """Right-aligned button inside the cell, centered vertically."""
        side = min(_PICKER_BUTTON_WIDTH, cell_rect.height() - 2 * _PICKER_BUTTON_MARGIN)
        side = max(side, 14)
        top = cell_rect.top() + (cell_rect.height() - side) // 2
        left = cell_rect.right() - side - _PICKER_BUTTON_MARGIN
        return QRect(left, top, side, side)

    def _cell_button_rect(self, cell_rect: QRect) -> QRect:
        """Centred pill filling most of a ``SlotEditor.BUTTON`` cell."""
        height = max(16, min(22, cell_rect.height() - 2 * _CELL_BUTTON_MARGIN))
        width = max(
            _CELL_BUTTON_MIN_WIDTH,
            cell_rect.width() - 2 * _CELL_BUTTON_MARGIN,
        )
        width = min(width, cell_rect.width() - 2 * _CELL_BUTTON_MARGIN)
        left = cell_rect.left() + (cell_rect.width() - width) // 2
        top = cell_rect.top() + (cell_rect.height() - height) // 2
        return QRect(left, top, max(width, 0), height)

    def _button_state(self, index: QModelIndex) -> tuple[bool, bool]:
        """``(checked, locked)`` for a BUTTON cell, from the model.

        ``locked`` means the toggle is forced on and can't be turned off —
        the model's rule for a row that carries real off-axis values, so
        non-zero placement can never be hidden behind a collapsed column.
        Models without the API (test stubs) read as (False, False).
        """
        model = index.model()
        getter = getattr(model, "button_state", None)
        if getter is None:
            return (False, False)
        try:
            state = getter(index)
        except Exception:  # pragma: no cover - defensive
            return (False, False)
        if not state:
            return (False, False)
        return (bool(state[0]), bool(state[1]))

    def uses_combo(self, index: QModelIndex) -> bool:
        """True when this cell would open a ``QComboBox`` editor.

        Read by the body's click-to-edit filter so a single left-click on a
        combo cell opens the editor and pops the dropdown — instead of
        making the user double-click to enter edit mode and then click the
        combo arrow as a third step.
        """
        slot = self._slot(index)
        return slot is not None and slot.editor in (
            SlotEditor.ENUM_COMBO, SlotEditor.STRING_COMBO,
        )

    def createEditor(self, parent: QWidget, option, index: QModelIndex):
        slot = self._slot(index)
        if slot is None or slot.editor == SlotEditor.READONLY:
            return None
        ops = _EDITOR_OPS.get(slot.editor)
        if ops is None:
            return None
        try:
            editor = ops.create(parent, slot)
        except Exception:
            # A misconfigured slot (wrong options callable signature,
            # missing required option) should NOT take the app down. Log
            # and return None so the cell just stays in display mode.
            import logging
            logging.getLogger(__name__).exception(
                "createEditor failed for slot %r", slot.key
            )
            return None
        if isinstance(editor, QComboBox):
            _wire_combo_commit(self, editor)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        slot = self._slot(index)
        if slot is None or editor is None:
            return
        ops = _EDITOR_OPS.get(slot.editor)
        if ops is None:
            return
        ops.set_data(editor, index.data(Qt.EditRole))
        if isinstance(editor, QComboBox):
            # Pop the dropdown right after the editor is fully laid out.
            # Direct showPopup() in setEditorData fires before the editor's
            # geometry is final, so the popup either no-ops or positions
            # under the cell incorrectly; a 0-delay timer waits one
            # event-loop tick, by which point the editor is in place.
            QTimer.singleShot(0, editor.showPopup)

    def setModelData(
        self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        slot = self._slot(index)
        if slot is None or editor is None:
            return
        ops = _EDITOR_OPS.get(slot.editor)
        if ops is None:
            return
        ops.commit(editor, slot, model, index)

    # ------------------------------------------------------------------
    # Painting & geometry — two-strip cell layout
    # ------------------------------------------------------------------

    def _in_cell_label(self, slot: Slot, index: QModelIndex) -> str:
        """In-cell label, or '' when it duplicates the column strip header.

        Surface rows show only values (their slot labels match the canonical
        column strip — Radius / Pos Z / Aperture Rad). Material rows whose
        slot labels diverge from the strip header (Type / Catalog / …) get
        the label rendered.
        """
        strip = schemas.header_text(index.column())
        return "" if slot.label.strip() == strip.strip() else slot.label

    def _label_font(self, base: QFont) -> QFont:
        f = QFont(base)
        f.setPointSizeF(max(_LABEL_MIN_POINT, f.pointSizeF() * _LABEL_POINT_SCALE))
        return f

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        base = super().sizeHint(option, index)
        target = int(option.fontMetrics.height() * _ROW_HEIGHT_MULT)
        return QSize(base.width(), max(base.height(), target))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        slot = self._slot(index)
        if slot is None:
            # NAME column + trailing empty cells: standard render handles
            # icon + type label, alt rows, selection, hover — and vertically
            # centers everything in the taller row.
            super().paint(painter, option, index)
            return

        # Cell chrome (selection / alt-row / hover) with text suppressed,
        # then we overlay label + value ourselves.
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        # Variable-flag stripe. Painted AFTER chrome so it draws on top of
        # the alt-row banding, BEFORE label/value text so the amber sits
        # to the left of the digits. Keeps the value column geometry
        # unchanged — the stripe eats a few pixels off the alt-row band's
        # left edge, not off the cell padding.
        if self._cell_is_variable_flagged(slot, index):
            self._paint_variable_stripe(painter, option)

        # BUTTON cells own their whole cell — no label / value strips.
        if slot.editor == SlotEditor.BUTTON:
            self._paint_cell_button(painter, option, index, slot)
            return

        inner = option.rect.adjusted(_CELL_PAD_X, 0, -_CELL_PAD_X, 0)
        display = index.data(Qt.DisplayRole) or ""
        # READONLY cells are informational labels (e.g. the element row's
        # EFL). Render the value dimmer + italic so it reads as a
        # computed annotation rather than an editable field, without
        # changing the cell geometry.
        is_readonly_label = slot.editor == SlotEditor.READONLY
        if option.state & QStyle.State_Selected:
            text_color = option.palette.color(QPalette.HighlightedText)
        elif is_readonly_label:
            text_color = option.palette.color(QPalette.PlaceholderText)
        else:
            text_color = option.palette.color(QPalette.Text)

        # Compress empty non-editable cells — chrome is already painted,
        # so a cell with nothing to display and no editor is fully blank.
        # This is what drops the "nd" / "Vd" labels in catalog mode, and
        # the "Name" label in Custom mode, without per-row column hiding.
        if not display and not self._is_editable(index):
            return

        label_text = self._in_cell_label(slot, index)
        if not label_text:
            painter.save()
            painter.setPen(text_color)
            if is_readonly_label:
                value_font = QFont(option.font)
                value_font.setItalic(True)
                painter.setFont(value_font)
            painter.drawText(
                inner, Qt.AlignLeft | Qt.AlignVCenter, str(display)
            )
            painter.restore()
            return

        label_font = self._label_font(option.font)
        label_h = QFontMetrics(label_font).height()
        label_rect = QRect(
            inner.left(), inner.top() + _LABEL_PAD_Y,
            inner.width(), label_h,
        )
        value_rect = QRect(
            inner.left(),
            label_rect.bottom() + _LABEL_PAD_Y,
            inner.width(),
            inner.bottom() - (label_rect.bottom() + _LABEL_PAD_Y),
        )

        painter.save()
        painter.setFont(label_font)
        painter.setPen(option.palette.color(QPalette.PlaceholderText))
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, label_text)
        painter.restore()

        painter.save()
        painter.setPen(text_color)
        if is_readonly_label:
            value_font = QFont(option.font)
            value_font.setItalic(True)
            painter.setFont(value_font)
        painter.drawText(value_rect, Qt.AlignLeft | Qt.AlignVCenter, str(display))
        painter.restore()

        if (
            slot.editor == SlotEditor.TEXT_PICKER
            and slot.options.get("picker") in self._pickers
            and self._is_editable(index)
        ):
            self._paint_picker_button(painter, option)
            if slot.options.get("color_swatch"):
                self._paint_color_swatch(painter, option, str(display))

    def _paint_variable_stripe(
        self, painter: QPainter, option: QStyleOptionViewItem
    ) -> None:
        """Amber left-edge stripe marking a variable-flagged cell.

        A thin, filled rectangle hugging the left edge of the cell rect.
        Uses ``fillRect`` (not drawRect) so the exact ``_VAR_STRIPE_WIDTH``
        pixels are painted regardless of pen scaling.
        """
        rect = QRect(
            option.rect.left(),
            option.rect.top(),
            _VAR_STRIPE_WIDTH,
            option.rect.height(),
        )
        painter.save()
        painter.fillRect(rect, _VAR_STRIPE_COLOR)
        painter.restore()

    def _paint_color_swatch(
        self, painter: QPainter, option: QStyleOptionViewItem, hexstr: str
    ) -> None:
        """Small filled colour chip left of the picker button for a
        ``#RRGGBB`` value cell (the artist tint). Silent no-op when the
        display isn't a parseable hex triplet."""
        text = hexstr.strip().lstrip("#")
        if len(text) != 6:
            return
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
        except ValueError:
            return
        button = self._picker_button_rect(option.rect)
        side = button.height()
        chip = QRect(button.left() - side - _PICKER_BUTTON_MARGIN,
                     button.top(), side, side)
        painter.save()
        painter.setPen(option.palette.color(QPalette.Mid))
        painter.setBrush(QColor(r, g, b))
        painter.drawRoundedRect(chip, 3, 3)
        painter.restore()

    def _paint_picker_button(
        self, painter: QPainter, option: QStyleOptionViewItem
    ) -> None:
        rect = self._picker_button_rect(option.rect)
        painter.save()
        bg = option.palette.color(QPalette.Button)
        border = option.palette.color(QPalette.Mid)
        painter.setBrush(bg)
        painter.setPen(border)
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(option.palette.color(QPalette.ButtonText))
        painter.drawText(rect, Qt.AlignCenter, _PICKER_BUTTON_GLYPH)
        painter.restore()

    def _paint_cell_button(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
        slot: Slot,
    ) -> None:
        """Paint a ``SlotEditor.BUTTON`` cell in one of three states.

        Off — plain button chrome. On — highlight fill, so an expanded row is
        obvious at a glance. Locked — highlight fill plus dimmed text, saying
        "this is on because the row holds values, and you can't turn it off".
        """
        checked, locked = self._button_state(index)
        rect = self._cell_button_rect(option.rect)
        if rect.width() <= 0:
            return
        glyph = str(slot.options.get("glyph", "") or _PICKER_BUTTON_GLYPH)

        painter.save()
        if checked:
            bg = option.palette.color(QPalette.Highlight)
            fg = option.palette.color(
                QPalette.PlaceholderText if locked else QPalette.HighlightedText
            )
        else:
            bg = option.palette.color(QPalette.Button)
            fg = option.palette.color(QPalette.ButtonText)
        painter.setBrush(bg)
        painter.setPen(option.palette.color(QPalette.Mid))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(fg)
        painter.drawText(rect, Qt.AlignCenter, glyph)
        painter.restore()

    def editorEvent(
        self,
        event: QEvent,
        model: QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        slot = self._slot(index)
        if (
            slot is not None
            and slot.editor == SlotEditor.BUTTON
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
        ):
            if self._cell_button_rect(option.rect).contains(
                event.position().toPoint()
            ):
                # Straight to the model, NOT through setData. This is view
                # state (which columns are revealed), not a document edit —
                # routing it through setData would open a project.edit
                # transaction and push a bogus "Set Off Axis" undo entry.
                handler = getattr(model, "toggle_button", None)
                if handler is not None:
                    try:
                        handler(index)
                    except Exception:  # pragma: no cover - defensive
                        import logging
                        logging.getLogger(__name__).exception(
                            "button toggle raised for slot %r", slot.key
                        )
                return True
        if (
            slot is not None
            and slot.editor == SlotEditor.TEXT_PICKER
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
            and self._is_editable(index)
        ):
            tag = slot.options.get("picker")
            picker = self._pickers.get(tag) if tag else None
            if picker is not None:
                button_rect = self._picker_button_rect(option.rect)
                if button_rect.contains(event.position().toPoint()):
                    # Fall back to the delegate's parent (body widget) when
                    # ``option.widget`` is None — Qt allows ``option.widget``
                    # to be unset, and dialog parents must be a real QWidget
                    # to position + raise reliably on Windows.
                    parent = option.widget or self.parent()
                    try:
                        new_value = picker(parent, index)
                    except Exception:
                        # A picker bug crashing the whole app on a single
                        # click is a bad user experience; log and swallow
                        # so the cell stays editable for the next click.
                        import logging
                        logging.getLogger(__name__).exception(
                            "material picker raised"
                        )
                        return True
                    if new_value is not None:
                        model.setData(index, str(new_value), Qt.EditRole)
                    return True
        return super().editorEvent(event, model, option, index)

    def updateEditorGeometry(
        self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        slot = self._slot(index)
        if slot is None:
            super().updateEditorGeometry(editor, option, index)
            return
        label_text = self._in_cell_label(slot, index)
        if not label_text:
            editor.setGeometry(option.rect)
            return
        # Editor occupies the value strip so the label stays visible above
        # while the user types — matches the view-state layout.
        label_h = QFontMetrics(self._label_font(option.font)).height()
        editor_top = option.rect.top() + _LABEL_PAD_Y + label_h + _LABEL_PAD_Y
        editor.setGeometry(QRect(
            option.rect.left(),
            editor_top,
            option.rect.width(),
            max(0, option.rect.bottom() - editor_top),
        ))
