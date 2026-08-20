"""Per-cell editor factory + paint customisation for the optimization tree.

This delegate solves three problems the bare ``QStyledItemDelegate``
ignored:

1. **Row height** — bare delegate cells were one text-line tall and read
   as a flat list; the ODE uses ``_ROW_HEIGHT_MULT * font_height`` so
   each row breathes. We do the same here, with a slightly taller
   multiplier for merit-function rows so the parent/child hierarchy
   reads at a glance.

2. **Combo cells need single-click + commit-on-pick.** Without the
   ``_wire_combo_commit`` pattern from the ODE, the user has to
   double-click to enter edit mode, click the combo arrow, click an
   option — four clicks. With it, one click opens the dropdown, one
   click commits.

3. **Visual hierarchy** — merit-function rows render bold so they stand
   apart from the goal rows that live underneath them.

The ``uses_combo`` method is read by the body's click-to-edit event
filter (same shape as in :mod:`ghostlight_designer.optical_editor.body`).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QModelIndex, QSize, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from ..project import Project
from .columns import base_column_count
from .goals.base import param_schema_for
from .nodes import GoalNode, MeritFunctionNode


# Rows are this multiple of a text line tall. Matches the ODE convention
# so the two trees look like they belong to the same app.
_ROW_HEIGHT_MULT = 2.0
_MF_ROW_HEIGHT_MULT = 2.2


class OptimizationCellDelegate(QStyledItemDelegate):
    """Tree-cell delegate for the optimization panel.

    See module docstring for what it solves beyond
    ``QStyledItemDelegate``. The combo logic mirrors
    :class:`ghostlight_designer.optical_editor.delegates.SlotDelegate` so any
    fix to one should be applied to the other.
    """

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self._project = project

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        base = super().sizeHint(option, index)
        mult = _MF_ROW_HEIGHT_MULT if isinstance(
            index.internalPointer(), MeritFunctionNode,
        ) else _ROW_HEIGHT_MULT
        target = int(option.fontMetrics.height() * mult)
        return QSize(base.width(), max(base.height(), target))

    # ------------------------------------------------------------------
    # Paint — bold the MF row so the parent/child distinction is obvious
    # ------------------------------------------------------------------

    def initStyleOption(
        self, option: QStyleOptionViewItem, index: QModelIndex,
    ) -> None:
        super().initStyleOption(option, index)
        node = index.internalPointer()
        if isinstance(node, MeritFunctionNode):
            # Bold so MF rows visibly group their goal children
            # underneath them. The default font sizing keeps fontMetrics
            # consistent with sizeHint's measurement.
            f = QFont(option.font)
            f.setBold(True)
            option.font = f

    # ------------------------------------------------------------------
    # Combo wiring — single-click open + commit on first pick
    # ------------------------------------------------------------------

    def uses_combo(self, index: QModelIndex) -> bool:
        """True iff this cell will open a ``QComboBox`` editor.

        Read by :class:`ghostlight_designer.optimization_panel.body._ComboClickToEdit`
        so a single left-click on a combo cell opens the editor + pops
        the dropdown, rather than needing a double-click first.
        """
        return self._param_kind_at(index) in (
            "wavelength_pick", "field_pick", "surface_pick", "axis",
        )

    def _param_kind_at(self, index: QModelIndex) -> Optional[str]:
        col = index.column()
        if col < base_column_count():
            return None
        node = index.internalPointer()
        if not isinstance(node, GoalNode):
            return None
        mfs = self._project.merit_functions
        if not (0 <= node.mf_index < len(mfs)):
            return None
        mf = mfs[node.mf_index]
        if not (0 <= node.goal_index < len(mf.goals)):
            return None
        goal = mf.goals[node.goal_index]
        schema = param_schema_for(goal.kind)
        param_idx = col - base_column_count()
        if not (0 <= param_idx < len(schema)):
            return None
        return schema[param_idx].kind

    def createEditor(
        self,
        parent: QWidget,
        option,
        index: QModelIndex,
    ) -> Optional[QWidget]:
        kind = self._param_kind_at(index)
        if kind == "wavelength_pick":
            editor = self._make_wavelength_combo(parent)
        elif kind == "field_pick":
            editor = self._make_field_combo(parent)
        elif kind == "surface_pick":
            editor = self._make_surface_combo(parent)
        elif kind == "axis":
            editor = self._make_axis_combo(parent)
        else:
            return self._make_numeric_editor(parent, option, index)

        # Commit + close on the first option click / keyboard select.
        # Without this the dropdown closes but the cell sits in edit
        # mode until the user clicks elsewhere — the "double rendering"
        # they reported.
        self._wire_combo_commit(editor)
        return editor

    def _make_numeric_editor(
        self, parent: QWidget, option, index: QModelIndex
    ) -> Optional[QWidget]:
        """Math-capable spinbox for the numeric cells (Target / Weight).

        Qt's default item-editor factory would hand back a plain spinbox
        here, which cannot take a typed calculation. ``setEditorData`` /
        ``setModelData`` need no matching branch: their ``super()`` paths
        go through the editor's USER property, and ``QStyledItemDelegate``
        already calls ``interpretText()`` on spinbox editors before
        reading it.
        """
        value = index.data(Qt.EditRole)
        if isinstance(value, bool):
            return super().createEditor(parent, option, index)
        if isinstance(value, float):
            editor = MathDoubleSpinBox(parent)
            editor.setDecimals(6)
            editor.setRange(-1.0e12, 1.0e12)
            editor.setAutoFillBackground(True)
            return editor
        if isinstance(value, int):
            editor = MathSpinBox(parent)
            editor.setRange(-1_000_000, 1_000_000)
            editor.setAutoFillBackground(True)
            return editor
        return super().createEditor(parent, option, index)

    def _wire_combo_commit(self, editor: QComboBox) -> None:
        """Commit + close on first option click. Mirrors
        ``ghostlight_designer.optical_editor.delegates._wire_combo_commit``."""
        state = {"done": False}

        def commit() -> None:
            if state["done"]:
                return
            state["done"] = True
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)
            editor.hide()

        # ``activated`` covers keyboard Up/Down+Enter. The popup view's
        # ``clicked`` fires on the first mouse press+release on an item,
        # before ``activated`` and before the popup hides.
        editor.activated.connect(lambda _i: commit())
        view = editor.view()
        if view is not None:
            view.clicked.connect(lambda _idx: commit())

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if isinstance(editor, QComboBox):
            current = index.data(Qt.EditRole)
            ix = editor.findData(current)
            if ix < 0:
                ix = editor.findText(str(current))
            if ix >= 0:
                editor.setCurrentIndex(ix)
            # Pop the dropdown after geometry settles (one event-loop
            # tick). Direct showPopup in setEditorData fires before the
            # editor is positioned in the cell and either no-ops or pops
            # under the wrong cell.
            QTimer.singleShot(0, editor.showPopup)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        if isinstance(editor, QComboBox):
            value = editor.currentData()
            if value is None:
                value = editor.currentText()
            model.setData(index, value, Qt.EditRole)
            return
        super().setModelData(editor, model, index)

    # ------------------------------------------------------------------
    # Combo builders
    # ------------------------------------------------------------------

    def _make_wavelength_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        # Opaque background — without setAutoFillBackground the popup's
        # text bleeds through the cell underneath while the editor is
        # open. See [[feedback-qt-cell-editor-quirks]].
        combo.setAutoFillBackground(True)
        combo.addItem("Primary", "primary")
        combo.addItem("All", "all")
        sequences = self._project.system_setup.sequences
        if sequences:
            for wi, w in enumerate(sequences[0].source.wavelengths.wavelengths):
                combo.addItem(f"λ{wi + 1} — {w.value_nm:.2f} nm", wi)
        return combo

    def _make_field_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.setAutoFillBackground(True)
        combo.addItem("All", "all")
        sequences = self._project.system_setup.sequences
        if sequences:
            for fi, f in enumerate(sequences[0].source.fields):
                label = f"F{fi + 1} — {f.name} ({f.tilt_x_deg:.1f}, {f.tilt_y_deg:.1f})°"
                combo.addItem(label, fi)
        return combo

    def _make_surface_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.setAutoFillBackground(True)
        try:
            n = self._project.system.num_surfaces()
        except Exception:
            n = 0
        for si in range(n):
            combo.addItem(f"Surface {si}", si)
        return combo

    def _make_axis_combo(self, parent: QWidget) -> QComboBox:
        combo = QComboBox(parent)
        combo.setAutoFillBackground(True)
        for axis in ("Radial", "X", "Y"):
            combo.addItem(axis, axis)
        return combo
