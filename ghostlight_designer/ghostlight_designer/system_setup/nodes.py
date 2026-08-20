"""Tree node hierarchy for the System Setup panel.

Tree shape::

    System Setup (root)
    ├── Image Sensor                    (CategoryNode)
    │   ├── Preset                      (SensorPropNode, PRESET)
    │   ├── Width  (mm)                 (SensorPropNode, WIDTH)
    │   └── Height (mm)                 (SensorPropNode, HEIGHT)
    └── Sequences                       (CategoryNode, top-level sibling)
        └── <sequence name>             (SequenceNode, value = sequence.name)
            ├── Aperture Type           (SequenceFieldNode, APERTURE_TYPE)
            ├── Field Type              (SequenceFieldNode, FIELD_TYPE)
            ├── Stop Surface            (SequenceFieldNode, STOP_SURFACE)
            └── Source                  (SourceNode)
                ├── Source Type         (SourceFieldNode, SOURCE_TYPE)
                ├── Aperture Radius     (SourceFieldNode, APERTURE_RADIUS)
                ├── Distribution        (DistributionNode)
                │   ├── Type            (DistributionFieldNode, TYPE)
                │   └── Ray Count       (DistributionFieldNode, RAY_COUNT)
                ├── Wavelengths         (WavelengthsNode)
                │   ├── Primary         (WavelengthsFieldNode, PRIMARY)
                │   ├── Reference       (WavelengthsFieldNode, REFERENCE)
                │   └── Wavelength N    (WavelengthNode, value = nm)
                └── Fields              (FieldsNode)
                    └── <field name>    (FieldNode, value = field.name)
                        ├── Tilt X      (FieldFieldNode, TILT_X)
                        └── Tilt Y      (FieldFieldNode, TILT_Y)
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Property IDs (which field of the underlying dataclass a row maps to)
# ---------------------------------------------------------------------------


class SequenceProp(Enum):
    APERTURE_TYPE = "aperture_type"
    FIELD_TYPE = "field_type"
    STOP_SURFACE = "stop_surface"


class SourceProp(Enum):
    SOURCE_TYPE = "source_type"
    APERTURE_RADIUS = "aperture_radius"


class DistributionProp(Enum):
    TYPE = "type"
    RAY_COUNT = "ray_count"


class WavelengthsProp(Enum):
    PRIMARY = "primary"
    REFERENCE = "reference"


class FieldProp(Enum):
    TILT_X = "tilt_x"
    TILT_Y = "tilt_y"


class SensorProp(Enum):
    PRESET = "preset"
    WIDTH = "width"
    HEIGHT = "height"


# ---------------------------------------------------------------------------
# Node classes
# ---------------------------------------------------------------------------


class TreeNode:
    __slots__ = ("parent", "children", "label")

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.parent: Optional["TreeNode"] = None
        self.children: List["TreeNode"] = []

    def add(self, child: "TreeNode") -> "TreeNode":
        child.parent = self
        self.children.append(child)
        return child

    def row(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.children.index(self)


class RootNode(TreeNode):
    pass


class CategoryNode(TreeNode):
    """Pure label, no editable value (e.g. 'Sequences', 'Image Sensor')."""


class SequenceNode(TreeNode):
    """Represents a Sequence; its editable value is ``sequence.name``."""

    __slots__ = ("sequence_index",)

    def __init__(self, sequence_index: int, label: str = "") -> None:
        super().__init__(label)
        self.sequence_index = sequence_index


class SequenceFieldNode(TreeNode):
    __slots__ = ("sequence_index", "prop")

    def __init__(self, sequence_index: int, prop: SequenceProp, label: str) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.prop = prop


class SourceNode(TreeNode):
    __slots__ = ("sequence_index",)

    def __init__(self, sequence_index: int) -> None:
        super().__init__("Source")
        self.sequence_index = sequence_index


class SourceFieldNode(TreeNode):
    __slots__ = ("sequence_index", "prop")

    def __init__(self, sequence_index: int, prop: SourceProp, label: str) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.prop = prop


class DistributionNode(TreeNode):
    __slots__ = ("sequence_index",)

    def __init__(self, sequence_index: int) -> None:
        super().__init__("Distribution")
        self.sequence_index = sequence_index


class DistributionFieldNode(TreeNode):
    __slots__ = ("sequence_index", "prop")

    def __init__(
        self, sequence_index: int, prop: DistributionProp, label: str
    ) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.prop = prop


class WavelengthsNode(TreeNode):
    __slots__ = ("sequence_index",)

    def __init__(self, sequence_index: int) -> None:
        super().__init__("Wavelengths")
        self.sequence_index = sequence_index


class WavelengthsFieldNode(TreeNode):
    __slots__ = ("sequence_index", "prop")

    def __init__(
        self, sequence_index: int, prop: WavelengthsProp, label: str
    ) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.prop = prop


class WavelengthNode(TreeNode):
    __slots__ = ("sequence_index", "wavelength_index")

    def __init__(
        self, sequence_index: int, wavelength_index: int, label: str
    ) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.wavelength_index = wavelength_index


class FieldsNode(TreeNode):
    __slots__ = ("sequence_index",)

    def __init__(self, sequence_index: int) -> None:
        super().__init__("Fields")
        self.sequence_index = sequence_index


class FieldNode(TreeNode):
    """Represents a Field; its editable value is ``field.name``."""

    __slots__ = ("sequence_index", "field_index")

    def __init__(
        self, sequence_index: int, field_index: int, label: str = ""
    ) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.field_index = field_index


class FieldFieldNode(TreeNode):
    __slots__ = ("sequence_index", "field_index", "prop")

    def __init__(
        self,
        sequence_index: int,
        field_index: int,
        prop: FieldProp,
        label: str,
    ) -> None:
        super().__init__(label)
        self.sequence_index = sequence_index
        self.field_index = field_index
        self.prop = prop


class SensorPropNode(TreeNode):
    __slots__ = ("prop",)

    def __init__(self, prop: SensorProp, label: str) -> None:
        super().__init__(label)
        self.prop = prop


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def build_tree(setup) -> RootNode:
    """Build the static tree given a ``SystemSetup`` snapshot."""
    root = RootNode()

    sensor_cat = CategoryNode("Image Sensor")
    root.add(sensor_cat)
    sensor_cat.add(SensorPropNode(SensorProp.PRESET, "Preset"))
    sensor_cat.add(SensorPropNode(SensorProp.WIDTH, "Width (mm)"))
    sensor_cat.add(SensorPropNode(SensorProp.HEIGHT, "Height (mm)"))

    sequences_cat = CategoryNode("Sequences")
    root.add(sequences_cat)

    for si, seq in enumerate(setup.sequences):
        seq_node = SequenceNode(si, seq.name)
        sequences_cat.add(seq_node)

        seq_node.add(SequenceFieldNode(si, SequenceProp.APERTURE_TYPE, "Aperture Type"))
        seq_node.add(SequenceFieldNode(si, SequenceProp.FIELD_TYPE, "Field Type"))
        seq_node.add(SequenceFieldNode(si, SequenceProp.STOP_SURFACE, "Stop Surface"))

        src_node = SourceNode(si)
        seq_node.add(src_node)
        src_node.add(SourceFieldNode(si, SourceProp.SOURCE_TYPE, "Source Type"))
        src_node.add(SourceFieldNode(si, SourceProp.APERTURE_RADIUS, "Aperture Radius"))

        dist_node = DistributionNode(si)
        src_node.add(dist_node)
        dist_node.add(DistributionFieldNode(si, DistributionProp.TYPE, "Type"))
        dist_node.add(DistributionFieldNode(si, DistributionProp.RAY_COUNT, "Ray Count"))

        waves_node = WavelengthsNode(si)
        src_node.add(waves_node)
        waves_node.add(WavelengthsFieldNode(si, WavelengthsProp.PRIMARY, "Primary"))
        waves_node.add(WavelengthsFieldNode(si, WavelengthsProp.REFERENCE, "Reference"))
        for wi, _w in enumerate(seq.source.wavelengths.wavelengths):
            waves_node.add(WavelengthNode(si, wi, f"Wavelength {wi + 1}"))

        fields_node = FieldsNode(si)
        src_node.add(fields_node)
        for fi, fld in enumerate(seq.source.fields):
            fnode = FieldNode(si, fi, fld.name)
            fields_node.add(fnode)
            fnode.add(FieldFieldNode(si, fi, FieldProp.TILT_X, "Tilt X (°)"))
            fnode.add(FieldFieldNode(si, fi, FieldProp.TILT_Y, "Tilt Y (°)"))

    return root
