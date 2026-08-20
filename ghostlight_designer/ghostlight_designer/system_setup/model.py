"""``QAbstractItemModel`` exposing ``Project.system_setup``."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt

from ..project import Project
from ..system_setup_data import (
    ApertureType,
    CUSTOM_PRESET,
    DistributionType,
    FieldType,
    SourceType,
    find_preset,
    match_preset,
)
from . import columns as col_mod
from .nodes import (
    CategoryNode,
    DistributionFieldNode,
    DistributionProp,
    FieldFieldNode,
    FieldNode,
    FieldProp,
    RootNode,
    SensorProp,
    SensorPropNode,
    SequenceFieldNode,
    SequenceNode,
    SequenceProp,
    SourceFieldNode,
    SourceNode,
    SourceProp,
    TreeNode,
    WavelengthNode,
    WavelengthsFieldNode,
    WavelengthsNode,
    WavelengthsProp,
    build_tree,
)


def _structure_signature(setup) -> tuple:
    """Cheap fingerprint used to detect when a full rebuild is required.

    Captures only structural counts (#sequences, per-sequence #fields and
    #wavelengths). Value changes do NOT bump the signature, so editing a
    name or a number triggers a per-cell ``dataChanged`` rather than a
    full reset.
    """
    return tuple(
        (
            len(seq.source.wavelengths.wavelengths),
            len(seq.source.fields),
        )
        for seq in setup.sequences
    )


class SystemSetupTreeModel(QAbstractItemModel):
    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._root: RootNode = RootNode()
        self._sig: tuple = ()
        project.systemReplaced.connect(self._on_replaced)
        project.systemSetupChanged.connect(self._on_setup_changed)
        self._rebuild()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        self.beginResetModel()
        try:
            self._root = build_tree(self._project.system_setup)
            self._sig = _structure_signature(self._project.system_setup)
        finally:
            self.endResetModel()

    def _on_replaced(self, *_args) -> None:
        self._rebuild()

    def _on_setup_changed(self) -> None:
        new_sig = _structure_signature(self._project.system_setup)
        if new_sig != self._sig:
            self._rebuild()
        # Value-only changes are emitted by ``setData`` per-cell, so no
        # explicit refresh is required here.

    def _node(self, index: QModelIndex) -> TreeNode:
        if not index.isValid():
            return self._root
        ptr = index.internalPointer()
        return ptr if ptr is not None else self._root

    # ------------------------------------------------------------------
    # QAbstractItemModel
    # ------------------------------------------------------------------

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return col_mod.column_count()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._node(parent)
        return len(node.children)

    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
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

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return col_mod.header_text(section)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        node = self._node(index)
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        col = index.column()
        if col == col_mod.Column.VALUE and self._row_is_editable(node):
            base |= Qt.ItemIsEditable
        return base

    @staticmethod
    def _row_is_editable(node: TreeNode) -> bool:
        if isinstance(node, (CategoryNode, RootNode)):
            return False
        if isinstance(node, (SourceNode,)):
            return False
        # All other concrete leaf/entity nodes carry an editable value in
        # the Value column.
        return isinstance(
            node,
            (
                SequenceNode,
                SequenceFieldNode,
                SourceFieldNode,
                DistributionFieldNode,
                WavelengthsFieldNode,
                WavelengthNode,
                FieldNode,
                FieldFieldNode,
                SensorPropNode,
            ),
        )

    # ------------------------------------------------------------------
    # data / setData
    # ------------------------------------------------------------------

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        node = self._node(index)
        col = index.column()
        setup = self._project.system_setup

        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == col_mod.Column.NAME:
                return node.label
            if col == col_mod.Column.VALUE:
                return self._value(node, setup, edit=(role == Qt.EditRole))
            return ""

        if role == Qt.TextAlignmentRole and col == col_mod.Column.VALUE:
            if self._row_is_editable(node):
                return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def _value(self, node: TreeNode, setup, edit: bool):
        if isinstance(node, SequenceNode):
            seq = setup.sequences[node.sequence_index]
            return seq.name

        if isinstance(node, SequenceFieldNode):
            seq = setup.sequences[node.sequence_index]
            if node.prop == SequenceProp.APERTURE_TYPE:
                return seq.aperture_type.value
            if node.prop == SequenceProp.FIELD_TYPE:
                return seq.field_type.value
            if node.prop == SequenceProp.STOP_SURFACE:
                return self._stop_surface_label(seq.stop_surface)

        if isinstance(node, SourceFieldNode):
            src = setup.sequences[node.sequence_index].source
            if node.prop == SourceProp.SOURCE_TYPE:
                return src.type.value
            if node.prop == SourceProp.APERTURE_RADIUS:
                return float(src.aperture_radius) if edit else f"{src.aperture_radius:.3f}"

        if isinstance(node, DistributionFieldNode):
            d = setup.sequences[node.sequence_index].source.distribution
            if node.prop == DistributionProp.TYPE:
                return d.type.value
            if node.prop == DistributionProp.RAY_COUNT:
                return int(d.ray_count)

        if isinstance(node, WavelengthsFieldNode):
            wc = setup.sequences[node.sequence_index].source.wavelengths
            if node.prop == WavelengthsProp.PRIMARY:
                idx = wc.primary_index
                if 0 <= idx < len(wc.wavelengths):
                    return f"Wavelength {idx + 1}"
                return ""
            if node.prop == WavelengthsProp.REFERENCE:
                if wc.reference_index is None:
                    return "Primary"
                if 0 <= wc.reference_index < len(wc.wavelengths):
                    return f"Wavelength {wc.reference_index + 1}"
                return "Primary"

        if isinstance(node, WavelengthNode):
            wc = setup.sequences[node.sequence_index].source.wavelengths
            w = wc.wavelengths[node.wavelength_index]
            return float(w.value_nm) if edit else f"{w.value_nm:.2f} nm"

        if isinstance(node, FieldNode):
            f = setup.sequences[node.sequence_index].source.fields[node.field_index]
            return f.name

        if isinstance(node, FieldFieldNode):
            f = setup.sequences[node.sequence_index].source.fields[node.field_index]
            if node.prop == FieldProp.TILT_X:
                return float(f.tilt_x_deg) if edit else f"{f.tilt_x_deg:.3f}"
            if node.prop == FieldProp.TILT_Y:
                return float(f.tilt_y_deg) if edit else f"{f.tilt_y_deg:.3f}"

        if isinstance(node, SensorPropNode):
            s = setup.sensor
            if node.prop == SensorProp.PRESET:
                return s.preset_name
            if node.prop == SensorProp.WIDTH:
                return float(s.width_mm) if edit else f"{s.width_mm:.2f}"
            if node.prop == SensorProp.HEIGHT:
                return float(s.height_mm) if edit else f"{s.height_mm:.2f}"

        return ""

    # ------------------------------------------------------------------
    # Stop surface label rendering
    # ------------------------------------------------------------------

    def _stop_surface_label(self, idx: Optional[int]) -> str:
        if idx is None:
            return "Auto"
        n = self._surface_count()
        if 0 <= idx < n:
            return f"Surface {idx}"
        return f"Surface {idx} (missing)"

    def _surface_count(self) -> int:
        sys = self._project.system
        try:
            return len(sys.surfaces)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # setData dispatch
    # ------------------------------------------------------------------

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.EditRole:
            return False
        node = self._node(index)
        if not self._row_is_editable(node):
            return False
        if index.column() != col_mod.Column.VALUE:
            return False

        setup = self._project.system_setup
        ok, extra_indices = self._apply(node, value, setup)
        if not ok:
            return False
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        for idx in extra_indices:
            self.dataChanged.emit(idx, idx, [Qt.DisplayRole, Qt.EditRole])
        self._project.mark_system_setup_modified()
        return True

    def _apply(self, node: TreeNode, value, setup) -> tuple[bool, list]:
        """Mutate ``setup`` for ``node`` to ``value``.

        Returns (ok, [extra indices to emit dataChanged for]).
        """
        if isinstance(node, SequenceNode):
            seq = setup.sequences[node.sequence_index]
            new_name = str(value).strip()
            if not new_name or new_name == seq.name:
                return False, []
            seq.name = new_name
            node.label = new_name
            return True, []

        if isinstance(node, SequenceFieldNode):
            seq = setup.sequences[node.sequence_index]
            if node.prop == SequenceProp.APERTURE_TYPE:
                try:
                    new = ApertureType(value)
                except ValueError:
                    return False, []
                if new == seq.aperture_type:
                    return False, []
                seq.aperture_type = new
                return True, []
            if node.prop == SequenceProp.FIELD_TYPE:
                try:
                    new = FieldType(value)
                except ValueError:
                    return False, []
                if new == seq.field_type:
                    return False, []
                seq.field_type = new
                return True, []
            if node.prop == SequenceProp.STOP_SURFACE:
                if value is None or value == "Auto":
                    if seq.stop_surface is None:
                        return False, []
                    seq.stop_surface = None
                    return True, []
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    return False, []
                if idx < 0:
                    return False, []
                if seq.stop_surface == idx:
                    return False, []
                seq.stop_surface = idx
                return True, []

        if isinstance(node, SourceFieldNode):
            src = setup.sequences[node.sequence_index].source
            if node.prop == SourceProp.SOURCE_TYPE:
                try:
                    new = SourceType(value)
                except ValueError:
                    return False, []
                if new == src.type:
                    return False, []
                src.type = new
                return True, []
            if node.prop == SourceProp.APERTURE_RADIUS:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return False, []
                if v < 0:
                    return False, []
                if v == src.aperture_radius:
                    return False, []
                src.aperture_radius = v
                return True, []

        if isinstance(node, DistributionFieldNode):
            d = setup.sequences[node.sequence_index].source.distribution
            if node.prop == DistributionProp.TYPE:
                try:
                    new = DistributionType(value)
                except ValueError:
                    return False, []
                if new == d.type:
                    return False, []
                d.type = new
                return True, []
            if node.prop == DistributionProp.RAY_COUNT:
                try:
                    v = int(value)
                except (TypeError, ValueError):
                    return False, []
                if v < 1:
                    return False, []
                if v == d.ray_count:
                    return False, []
                d.ray_count = v
                return True, []

        if isinstance(node, WavelengthsFieldNode):
            wc = setup.sequences[node.sequence_index].source.wavelengths
            if node.prop == WavelengthsProp.PRIMARY:
                idx = _parse_wavelength_index(value, len(wc.wavelengths))
                if idx is None or idx == wc.primary_index:
                    return False, []
                wc.primary_index = idx
                return True, []
            if node.prop == WavelengthsProp.REFERENCE:
                if value == "Primary" or value is None:
                    if wc.reference_index is None:
                        return False, []
                    wc.reference_index = None
                    return True, []
                idx = _parse_wavelength_index(value, len(wc.wavelengths))
                if idx is None or idx == wc.reference_index:
                    return False, []
                wc.reference_index = idx
                return True, []

        if isinstance(node, WavelengthNode):
            wc = setup.sequences[node.sequence_index].source.wavelengths
            w = wc.wavelengths[node.wavelength_index]
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False, []
            if v <= 0:
                return False, []
            if v == w.value_nm:
                return False, []
            w.value_nm = v
            return True, []

        if isinstance(node, FieldNode):
            f = setup.sequences[node.sequence_index].source.fields[node.field_index]
            new_name = str(value).strip()
            if not new_name or new_name == f.name:
                return False, []
            f.name = new_name
            node.label = new_name
            return True, []

        if isinstance(node, FieldFieldNode):
            f = setup.sequences[node.sequence_index].source.fields[node.field_index]
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False, []
            if node.prop == FieldProp.TILT_X:
                if v == f.tilt_x_deg:
                    return False, []
                f.tilt_x_deg = v
                return True, []
            if node.prop == FieldProp.TILT_Y:
                if v == f.tilt_y_deg:
                    return False, []
                f.tilt_y_deg = v
                return True, []

        if isinstance(node, SensorPropNode):
            return self._apply_sensor(node, value, setup)

        return False, []

    # ------------------------------------------------------------------
    # Sensor (preset reconciliation)
    # ------------------------------------------------------------------

    def _apply_sensor(self, node: SensorPropNode, value, setup) -> tuple[bool, list]:
        s = setup.sensor
        if node.prop == SensorProp.PRESET:
            name = str(value)
            preset = find_preset(name)
            if preset is not None:
                if (
                    s.preset_name == name
                    and s.width_mm == preset.width_mm
                    and s.height_mm == preset.height_mm
                ):
                    return False, []
                s.preset_name = name
                s.width_mm = preset.width_mm
                s.height_mm = preset.height_mm
                return True, self._sensor_value_indices(exclude=node.prop)
            if s.preset_name == CUSTOM_PRESET:
                return False, []
            s.preset_name = CUSTOM_PRESET
            return True, []

        if node.prop in (SensorProp.WIDTH, SensorProp.HEIGHT):
            try:
                mm = float(value)
            except (TypeError, ValueError):
                return False, []
            if mm <= 0:
                return False, []
            if node.prop == SensorProp.WIDTH:
                if mm == s.width_mm:
                    return False, []
                s.width_mm = mm
            else:
                if mm == s.height_mm:
                    return False, []
                s.height_mm = mm
            new_preset_name = match_preset(s.width_mm, s.height_mm)
            extras = []
            if new_preset_name != s.preset_name:
                s.preset_name = new_preset_name
                preset_idx = self._sensor_prop_index(SensorProp.PRESET)
                if preset_idx is not None:
                    extras.append(preset_idx)
            return True, extras

        return False, []

    def _sensor_category(self) -> Optional[CategoryNode]:
        for child in self._root.children:
            if isinstance(child, CategoryNode) and child.label == "Image Sensor":
                return child
        return None

    def _sensor_prop_index(self, prop: SensorProp) -> Optional[QModelIndex]:
        cat = self._sensor_category()
        if cat is None:
            return None
        for child in cat.children:
            if isinstance(child, SensorPropNode) and child.prop == prop:
                return self.createIndex(child.row(), col_mod.Column.VALUE, child)
        return None

    def _sensor_value_indices(self, exclude: Optional[SensorProp] = None) -> list:
        cat = self._sensor_category()
        out: list = []
        if cat is None:
            return out
        for child in cat.children:
            if isinstance(child, SensorPropNode) and child.prop != exclude:
                out.append(
                    self.createIndex(child.row(), col_mod.Column.VALUE, child)
                )
        return out


def _parse_wavelength_index(value, count: int) -> Optional[int]:
    """Parse user-entered wavelength refs like "Wavelength 1" or "1" or 0."""
    if isinstance(value, int):
        return value if 0 <= value < count else None
    s = str(value).strip()
    if not s:
        return None
    if s.lower().startswith("wavelength"):
        s = s[len("wavelength"):].strip()
    try:
        n = int(s)
    except ValueError:
        return None
    idx = n - 1  # user-facing wavelength numbers are 1-based
    return idx if 0 <= idx < count else None
