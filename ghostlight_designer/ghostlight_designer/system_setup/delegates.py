"""Item delegates for the System Setup tree."""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLineEdit,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from ..project import Project
from ..system_setup_data import (
    ApertureType,
    CUSTOM_PRESET,
    DistributionType,
    FieldType,
    SENSOR_PRESETS,
    SourceType,
)


class _BaseEnumComboDelegate(QStyledItemDelegate):
    """Combobox delegate where items are populated from a string list."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def _items(self) -> Iterable[str]:
        raise NotImplementedError

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        combo = QComboBox(parent)
        for label in self._items():
            combo.addItem(label)
        combo.setAutoFillBackground(True)
        return combo

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        current = str(index.data(Qt.EditRole) or "")
        i = editor.findText(current)
        if i >= 0:
            editor.setCurrentIndex(i)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        model.setData(index, editor.currentText(), Qt.EditRole)


class FloatSpinDelegate(QStyledItemDelegate):
    def __init__(
        self,
        decimals: int = 3,
        minimum: float = -1e9,
        maximum: float = 1e9,
        step: float = 0.1,
        suffix: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._min = minimum
        self._max = maximum
        self._step = step
        self._suffix = suffix

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        # MathDoubleSpinBox (not QDoubleSpinBox) so the cell accepts a
        # typed calculation, e.g. "12*2"; see ghostlight_designer.math_spinbox.
        spin = MathDoubleSpinBox(parent)
        spin.setDecimals(self._decimals)
        spin.setMinimum(self._min)
        spin.setMaximum(self._max)
        spin.setSingleStep(self._step)
        if self._suffix:
            spin.setSuffix(self._suffix)
        spin.setFrame(False)
        # Without an opaque background the cell's display text bleeds
        # through the borderless spinbox; the user sees a "double" string.
        spin.setAutoFillBackground(True)
        return spin

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        try:
            editor.setValue(float(index.data(Qt.EditRole)))
        except (TypeError, ValueError):
            editor.setValue(0.0)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        editor.interpretText()
        model.setData(index, float(editor.value()), Qt.EditRole)


class IntSpinDelegate(QStyledItemDelegate):
    def __init__(
        self,
        minimum: int = 1,
        maximum: int = 10_000_000,
        step: int = 1,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._step = step

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        spin = MathSpinBox(parent)
        spin.setMinimum(self._min)
        spin.setMaximum(self._max)
        spin.setSingleStep(self._step)
        spin.setFrame(False)
        spin.setAutoFillBackground(True)
        return spin

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        try:
            editor.setValue(int(index.data(Qt.EditRole)))
        except (TypeError, ValueError):
            editor.setValue(self._min)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        editor.interpretText()
        model.setData(index, int(editor.value()), Qt.EditRole)


class LineEditDelegate(QStyledItemDelegate):
    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        editor = QLineEdit(parent)
        editor.setAutoFillBackground(True)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        editor.setText(str(index.data(Qt.EditRole) or ""))

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        model.setData(index, editor.text(), Qt.EditRole)


# ---------------------------------------------------------------------------
# Concrete enum dropdowns
# ---------------------------------------------------------------------------


class ApertureTypeDelegate(_BaseEnumComboDelegate):
    def _items(self):
        return [m.value for m in ApertureType]


class FieldTypeDelegate(_BaseEnumComboDelegate):
    def _items(self):
        return [m.value for m in FieldType]


class SourceTypeDelegate(_BaseEnumComboDelegate):
    def _items(self):
        return [m.value for m in SourceType]


class DistributionTypeDelegate(_BaseEnumComboDelegate):
    def _items(self):
        return [m.value for m in DistributionType]


class SensorPresetDelegate(_BaseEnumComboDelegate):
    def _items(self):
        return [p.name for p in SENSOR_PRESETS] + [CUSTOM_PRESET]


# ---------------------------------------------------------------------------
# Stop-surface dropdown (depends on the current OpticalSystem)
# ---------------------------------------------------------------------------


class StopSurfaceDelegate(QStyledItemDelegate):
    """Dropdown listing 'Auto' plus an entry per surface in the active system."""

    AUTO_LABEL = "Auto"

    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        combo = QComboBox(parent)
        combo.addItem(self.AUTO_LABEL, None)
        n = 0
        try:
            n = len(self._project.system.surfaces)
        except Exception:
            n = 0
        for i in range(n):
            combo.addItem(f"Surface {i}", i)
        combo.setAutoFillBackground(True)
        return combo

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        current = str(index.data(Qt.EditRole) or "")
        i = editor.findText(current)
        if i < 0:
            i = 0
        editor.setCurrentIndex(i)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        data = editor.currentData()
        if data is None:
            model.setData(index, self.AUTO_LABEL, Qt.EditRole)
        else:
            model.setData(index, int(data), Qt.EditRole)


# ---------------------------------------------------------------------------
# Wavelength reference dropdowns (depend on the current wavelength count)
# ---------------------------------------------------------------------------


class WavelengthIndexDelegate(QStyledItemDelegate):
    """Combobox of `Wavelength 1..N`. ``include_primary`` adds a leading
    ``Primary`` choice (used for the Reference row)."""

    def __init__(
        self,
        project: Project,
        include_primary: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._include_primary = include_primary

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        combo = QComboBox(parent)
        if self._include_primary:
            combo.addItem("Primary")
        for i in range(self._wavelength_count(index)):
            combo.addItem(f"Wavelength {i + 1}")
        combo.setAutoFillBackground(True)
        return combo

    def _wavelength_count(self, index: QModelIndex) -> int:
        node = index.internalPointer()
        si = getattr(node, "sequence_index", 0)
        try:
            return len(
                self._project.system_setup.sequences[si].source.wavelengths.wavelengths
            )
        except (IndexError, AttributeError):
            return 0

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        current = str(index.data(Qt.EditRole) or "")
        i = editor.findText(current)
        if i < 0:
            i = 0
        editor.setCurrentIndex(i)

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        model.setData(index, editor.currentText(), Qt.EditRole)
