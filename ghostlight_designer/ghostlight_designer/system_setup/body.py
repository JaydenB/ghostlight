"""Body widget for the ``system_setup`` panel — tree view of Sequences/Sensor."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..value_scrubber import attach_value_scrubber

from ..project import Project
from . import columns as col_mod
from .delegates import (
    ApertureTypeDelegate,
    DistributionTypeDelegate,
    FieldTypeDelegate,
    FloatSpinDelegate,
    IntSpinDelegate,
    LineEditDelegate,
    SensorPresetDelegate,
    SourceTypeDelegate,
    StopSurfaceDelegate,
    WavelengthIndexDelegate,
    _BaseEnumComboDelegate,
)
from .model import SystemSetupTreeModel
from .nodes import (
    DistributionFieldNode,
    DistributionProp,
    FieldFieldNode,
    FieldNode,
    FieldProp,
    SensorProp,
    SensorPropNode,
    SequenceFieldNode,
    SequenceNode,
    SequenceProp,
    SourceFieldNode,
    SourceProp,
    WavelengthNode,
    WavelengthsFieldNode,
    WavelengthsProp,
)


class _ValueColumnRouter(QStyledItemDelegate):
    """Routes value-column editing to the appropriate delegate by node kind."""

    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

        # Build all child delegates once. They are parented to this router
        # so Qt cleans them up with the tree.
        self._aperture_type = ApertureTypeDelegate(self)
        self._field_type = FieldTypeDelegate(self)
        self._source_type = SourceTypeDelegate(self)
        self._distribution_type = DistributionTypeDelegate(self)
        self._sensor_preset = SensorPresetDelegate(self)
        self._stop_surface = StopSurfaceDelegate(project, self)
        self._wavelength_primary = WavelengthIndexDelegate(
            project, include_primary=False, parent=self
        )
        self._wavelength_reference = WavelengthIndexDelegate(
            project, include_primary=True, parent=self
        )

        self._line_edit = LineEditDelegate(self)
        self._aperture_radius_spin = FloatSpinDelegate(
            decimals=3, minimum=0.0, maximum=1_000_000.0, step=0.5, parent=self
        )
        self._angle_spin = FloatSpinDelegate(
            decimals=3,
            minimum=-360.0,
            maximum=360.0,
            step=0.5,
            suffix="°",
            parent=self,
        )
        self._wavelength_value_spin = FloatSpinDelegate(
            decimals=2,
            minimum=1.0,
            maximum=100_000.0,
            step=1.0,
            suffix=" nm",
            parent=self,
        )
        self._sensor_mm_spin = FloatSpinDelegate(
            decimals=2, minimum=0.01, maximum=1000.0, step=0.1, parent=self
        )
        self._int_spin = IntSpinDelegate(minimum=1, maximum=10_000_000, parent=self)

    def _pick(self, index: QModelIndex) -> Optional[QStyledItemDelegate]:
        if not index.isValid():
            return None
        node = index.internalPointer()
        if node is None:
            return None

        if isinstance(node, SequenceNode):
            return self._line_edit
        if isinstance(node, SequenceFieldNode):
            if node.prop == SequenceProp.APERTURE_TYPE:
                return self._aperture_type
            if node.prop == SequenceProp.FIELD_TYPE:
                return self._field_type
            if node.prop == SequenceProp.STOP_SURFACE:
                return self._stop_surface

        if isinstance(node, SourceFieldNode):
            if node.prop == SourceProp.SOURCE_TYPE:
                return self._source_type
            if node.prop == SourceProp.APERTURE_RADIUS:
                return self._aperture_radius_spin

        if isinstance(node, DistributionFieldNode):
            if node.prop == DistributionProp.TYPE:
                return self._distribution_type
            if node.prop == DistributionProp.RAY_COUNT:
                return self._int_spin

        if isinstance(node, WavelengthsFieldNode):
            if node.prop == WavelengthsProp.PRIMARY:
                return self._wavelength_primary
            if node.prop == WavelengthsProp.REFERENCE:
                return self._wavelength_reference

        if isinstance(node, WavelengthNode):
            return self._wavelength_value_spin

        if isinstance(node, FieldNode):
            return self._line_edit
        if isinstance(node, FieldFieldNode):
            if node.prop in (FieldProp.TILT_X, FieldProp.TILT_Y):
                return self._angle_spin

        if isinstance(node, SensorPropNode):
            if node.prop == SensorProp.PRESET:
                return self._sensor_preset
            if node.prop in (SensorProp.WIDTH, SensorProp.HEIGHT):
                return self._sensor_mm_spin

        return None

    def uses_combo(self, index: QModelIndex) -> bool:
        """True when this cell would open a QComboBox editor.

        Lets the body's click-to-edit filter open the editor (and the
        dropdown) on the user's first click into the cell.
        """
        d = self._pick(index)
        return isinstance(
            d,
            (
                _BaseEnumComboDelegate,
                StopSurfaceDelegate,
                WavelengthIndexDelegate,
            ),
        )

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> Optional[QWidget]:
        d = self._pick(index)
        if d is None:
            return None
        editor = d.createEditor(parent, option, index)
        if isinstance(editor, QComboBox):
            _wire_combo_commit(self, editor)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        d = self._pick(index)
        if d is not None:
            d.setEditorData(editor, index)
        if isinstance(editor, QComboBox):
            # Pop the dropdown after the editor's geometry is finalised
            # (calling it during setEditorData is too early). One event-
            # loop tick later the editor is in place and the popup lands
            # under the cell.
            QTimer.singleShot(0, editor.showPopup)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        d = self._pick(index)
        if d is not None:
            d.setModelData(editor, model, index)

    def updateEditorGeometry(
        self,
        editor: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        if editor is not None:
            editor.setGeometry(option.rect)


def _wire_combo_commit(
    delegate: QStyledItemDelegate, editor: QComboBox
) -> None:
    """Wire ``editor`` so the first click on an option commits and closes.

    See ``optical_editor.delegates._wire_combo_commit`` for the full
    rationale. Briefly: ``view().clicked`` fires before ``activated`` and
    before the popup hides — that's the snappiest commit signal for mouse
    use. ``activated`` covers Up/Down + Enter keyboard activation. The
    once-flag dedupes when both fire for the same selection.
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


class _ComboClickToEdit(QObject):
    """Mouse filter that opens combo-edited cells on a single left-click.

    Mirror of the optical-editor's filter — see ``optical_editor.body``
    for the rationale. Only combo cells are intercepted; everything else
    falls through to the standard double-click flow.
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
        sel = tree.selectionModel()
        sel.setCurrentIndex(
            index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        tree.edit(index)
        return True


class SystemSetupBody(QWidget):
    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

        self.model = SystemSetupTreeModel(project, self)

        self.tree = QTreeView(self)
        self.tree.setModel(self.model)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        header = self.tree.header()
        # Values displayed here are short (numbers, dropdown labels) so the
        # Value column gets a fixed reasonable width and Name stretches to
        # fill the rest. As the panel narrows, Name is the one that crops —
        # Value stays fully readable. User can still drag the boundary to
        # widen Value when a long label needs it.
        header.setStretchLastSection(False)
        header.setSectionResizeMode(int(col_mod.Column.NAME), QHeaderView.Stretch)
        header.setSectionResizeMode(int(col_mod.Column.VALUE), QHeaderView.Interactive)
        header.setMinimumSectionSize(40)
        self.tree.setColumnWidth(int(col_mod.Column.VALUE), 130)

        # Name column is purely display — no editing.
        # Value column routes per node kind.
        self._value_router = _ValueColumnRouter(project, self)
        self.tree.setItemDelegateForColumn(
            int(col_mod.Column.VALUE), self._value_router
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

        # Ctrl+MMB value scrubber over any editable numeric cell. The
        # default ``is_scrubbable`` accepts both float and int EditRole
        # values — int cells scrub at sub-unit sensitivity internally and
        # only push rounded ints back to the model.
        self._scrub_trigger = attach_value_scrubber(self.tree, self._project)
        # Single-click into combo cells opens the editor + dropdown.
        self._combo_click_filter = _ComboClickToEdit(self.tree)

        self.model.modelReset.connect(self._on_model_reset)
        self._on_model_reset()

    def _on_model_reset(self) -> None:
        # Expand everything down to Source level by default; leave deep
        # collapsible groups (Distribution / Wavelengths / Fields) closed
        # so the panel isn't visually overwhelming on first open.
        self.tree.expandAll()
