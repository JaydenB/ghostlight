"""In-memory System Setup data attached to ``Project``.

A trimmed Sequence / Source / Distribution / Wavelength / Field
hierarchy, held in memory only: none of it is written to ``.lens``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ApertureType(str, Enum):
    FROM_STOP = "From Stop"
    NONE = "None"


class FieldType(str, Enum):
    ANGLE = "Angle"
    FREE = "Free"
    NONE = "None"


class SourceType(str, Enum):
    POINT_SOURCE = "Point Source"
    PLANE_WF = "Plane WF"


class DistributionType(str, Enum):
    SINGLE_RAY = "Single Ray"
    Y_FAN = "Y-Fan"
    X_FAN = "X-Fan"
    XY_FAN = "XY-Fan"
    RING = "Ring"
    RANDOM = "Random"


# ---------------------------------------------------------------------------
# Sensor presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorPreset:
    name: str
    width_mm: float
    height_mm: float


SENSOR_PRESETS: List[SensorPreset] = [
    SensorPreset("Super 35",         24.89, 18.66),
    SensorPreset("Alexa LF",         36.70, 25.54),
    SensorPreset("Alexa 65",         54.12, 25.58),
    SensorPreset("IMAX 15/70",       70.00, 48.50),
    SensorPreset("Full Frame",       35.60, 23.80),
    SensorPreset("VistaVision",      37.72, 24.92),
    SensorPreset("APS-C",            23.60, 15.60),
    SensorPreset("Micro 4/3",        17.30, 13.00),
    SensorPreset("Canon 5D Mark IV", 36.00, 24.00),
]

CUSTOM_PRESET = "Custom"


def find_preset(name: str) -> Optional[SensorPreset]:
    for p in SENSOR_PRESETS:
        if p.name == name:
            return p
    return None


def match_preset(width_mm: float, height_mm: float, tol: float = 1e-3) -> str:
    for p in SENSOR_PRESETS:
        if abs(p.width_mm - width_mm) < tol and abs(p.height_mm - height_mm) < tol:
            return p.name
    return CUSTOM_PRESET


@dataclass
class SensorSettings:
    width_mm: float = 24.89
    height_mm: float = 18.66
    preset_name: str = "Super 35"


# ---------------------------------------------------------------------------
# Sequence / Source / Distribution / Wavelengths / Field
# ---------------------------------------------------------------------------


@dataclass
class Field:
    """A single field point — name + tilt in degrees on each axis.

    For Field Type = "Angle", ``tilt_x_deg`` / ``tilt_y_deg`` define the
    incoming wavefront tilt. ``FREE`` / ``NONE`` reinterpret the same
    values per consumer.
    """

    name: str = "Field"
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0


@dataclass
class Distribution:
    type: DistributionType = DistributionType.Y_FAN
    ray_count: int = 8


@dataclass
class Wavelength:
    value_nm: float = 587.56


def _default_wavelengths() -> List[Wavelength]:
    return [
        Wavelength(486.13),  # F-line
        Wavelength(587.56),  # d-line (primary by default)
        Wavelength(656.27),  # C-line
    ]


@dataclass
class WavelengthContainer:
    wavelengths: List[Wavelength] = field(default_factory=_default_wavelengths)
    primary_index: int = 1                 # default to d-line
    # None means "use Primary"; otherwise an index into ``wavelengths``.
    reference_index: Optional[int] = None


def _default_fields() -> List[Field]:
    return [Field("Axial", 0.0, 0.0)]


@dataclass
class Source:
    type: SourceType = SourceType.PLANE_WF
    aperture_radius: float = 20.0
    distribution: Distribution = field(default_factory=Distribution)
    wavelengths: WavelengthContainer = field(default_factory=WavelengthContainer)
    fields: List[Field] = field(default_factory=_default_fields)


@dataclass
class Sequence:
    name: str = "Auto Sequence 1"
    aperture_type: ApertureType = ApertureType.FROM_STOP
    field_type: FieldType = FieldType.ANGLE
    # None = Auto (use the OpticalSystem's ``is_stop`` surface); otherwise an
    # index into the flat ``OpticalSystem.surfaces`` list.
    stop_surface: Optional[int] = None
    source: Source = field(default_factory=Source)


@dataclass
class SystemSetup:
    sequences: List[Sequence] = field(default_factory=lambda: [Sequence()])
    sensor: SensorSettings = field(default_factory=SensorSettings)
