"""Ctrl+MMB value-scrubber for QAbstractSpinBox (QSpinBox / QDoubleSpinBox).

The existing :class:`ghostlight_designer.value_scrubber.ScrubPopup` is
heavily tied to a ``QTreeView`` + ``QAbstractItemModel`` cell — which
makes sense for the editor trees, less so for one-off spinboxes in
dialogs. This module adapts that popup to a single spinbox by:

* building a hidden one-row :class:`QStandardItemModel` mirroring the
  spinbox value (bidirectionally),
* parenting an invisible :class:`QTreeView` over that model purely so
  the popup has something to anchor + the model has the standard
  ``dataChanged`` plumbing,
* feeding ``ScrubPopup`` a duck-typed no-op project (spinbox edits
  don't go through undo).

Usage::

    from .spinbox_scrub import attach_spinbox_scrubber
    attach_spinbox_scrubber(my_spinbox, label="Exposure")

Ctrl+MiddleClick on the spinbox now opens the same sensitivity-picker
popup the editor trees use; the drag updates the spinbox value via
``setValue`` (which fires its ``valueChanged`` signal exactly once per
step).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QTreeView

from ..value_scrubber import ScrubPopup


class _NoopProject:
    """Duck-typed ``Project`` for ``ScrubPopup`` — dialog scrubs don't
    participate in the project undo stack, so begin/end_compound are
    no-ops."""

    def begin_compound(self, _label: str) -> None:
        return None

    def end_compound(self) -> None:
        return None


class _SpinBoxAdapterModel(QStandardItemModel):
    """One-row, one-column model that mirrors a ``QAbstractSpinBox``.

    Bidirectional: spinbox ``valueChanged`` updates the model (so the
    popup's value label live-updates if the user types into the
    spinbox while the popup is open), and ``setData(EditRole)``
    (the path the popup uses to write) routes back to ``spinbox.setValue``.

    The model's ``EditRole`` type matches the spinbox's value type
    (``int`` for ``QSpinBox``, ``float`` for ``QDoubleSpinBox``) —
    that's the cue ``ScrubPopup`` uses to enable its int-only mode.
    """

    def __init__(self, spinbox: QAbstractSpinBox, header: str, parent: Optional[QObject] = None) -> None:
        super().__init__(1, 1, parent)
        self.setHorizontalHeaderLabels([header])
        item = QStandardItem()
        item.setEditable(True)
        self.setItem(0, 0, item)
        self._spinbox = spinbox
        self._is_int = isinstance(spinbox, QSpinBox)
        self._suppress = False
        self._push_value_from_spinbox()
        spinbox.valueChanged.connect(self._push_value_from_spinbox)

    def _push_value_from_spinbox(self, *_args) -> None:
        if self._suppress:
            return
        self._suppress = True
        try:
            val = self._spinbox.value()
            typed = int(val) if self._is_int else float(val)
            # super().setData so we don't recurse into our override.
            super().setData(self.index(0, 0), typed, Qt.EditRole)
        finally:
            self._suppress = False

    def setData(self, index, value, role: int = Qt.EditRole) -> bool:  # type: ignore[override]
        if role != Qt.EditRole or index.row() != 0 or index.column() != 0:
            return super().setData(index, value, role)
        if self._suppress:
            return super().setData(index, value, role)
        # Coerce + clamp via the spinbox itself so range / step rules
        # are enforced consistently.
        try:
            if self._is_int:
                coerced = int(round(float(value)))
            else:
                coerced = float(value)
        except (TypeError, ValueError):
            return False
        self._suppress = True
        try:
            self._spinbox.setValue(coerced)
            # Re-read the post-clamp value so the model EditRole stays
            # in sync with whatever the spinbox actually accepted.
            settled = self._spinbox.value()
            if self._is_int:
                settled = int(settled)
            else:
                settled = float(settled)
            ok = super().setData(index, settled, role)
        finally:
            self._suppress = False
        return ok


class _SpinBoxScrubTrigger(QObject):
    """Event filter — on Ctrl+MMB over the spinbox, open ``ScrubPopup``
    anchored at the cursor.

    QAbstractSpinBox routes mouse events to its internal QLineEdit
    over the text area, so a filter on the spinbox alone would miss
    clicks inside the editable region. We install on both targets.
    """

    def __init__(
        self,
        spinbox: QAbstractSpinBox,
        tree: QTreeView,
        model: _SpinBoxAdapterModel,
        project: _NoopProject,
    ) -> None:
        super().__init__(spinbox)
        self._spinbox = spinbox
        self._tree = tree
        self._model = model
        self._project = project
        spinbox.installEventFilter(self)
        line_edit = spinbox.lineEdit() if hasattr(spinbox, "lineEdit") else None
        if line_edit is not None:
            line_edit.installEventFilter(self)
        self._line_edit = line_edit

    def eventFilter(self, obj, ev) -> bool:
        if obj is not self._spinbox and obj is not self._line_edit:
            return False
        if ev.type() != QEvent.MouseButtonPress:
            return False
        if ev.button() != Qt.MiddleButton:
            return False
        if not (ev.modifiers() & Qt.ControlModifier):
            return False
        index = self._model.index(0, 0)
        anchor_global = ev.globalPosition().toPoint()
        popup = ScrubPopup(self._tree, index, anchor_global, self._project)
        popup.show()
        return True


def attach_spinbox_scrubber(
    spinbox: QAbstractSpinBox,
    *,
    label: str = "Value",
) -> _SpinBoxScrubTrigger:
    """Install Ctrl+MMB value-scrubbing on ``spinbox``.

    Works on both ``QSpinBox`` (int-mode scrubbing — the popup picks
    rounded ints via its internal float accumulator) and
    ``QDoubleSpinBox`` (float-mode).

    The returned trigger is parented to the spinbox so its lifetime
    matches; callers don't normally need to hold a reference.
    """
    model = _SpinBoxAdapterModel(spinbox, label, parent=spinbox)
    # Hidden tree just so ScrubPopup has the QTreeView + QAbstractItemModel
    # combo it expects. visualRect() returns an empty rect on a hidden
    # tree, which makes the popup's row-highlight overlay silently skip
    # — that's fine, the overlay is over the cell, not the spinbox.
    tree = QTreeView(spinbox)
    tree.setModel(model)
    tree.hide()
    trigger = _SpinBoxScrubTrigger(spinbox, tree, model, _NoopProject())
    return trigger
