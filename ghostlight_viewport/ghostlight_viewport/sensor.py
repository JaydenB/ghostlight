"""SensorSpec dataclass + sensor quad mesh helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ghostlight import LensCalibration


_CALIBRATED_SEGMENTS = 128


@dataclass
class SensorSpec:
    """Image plane description for the viewport.

    Fields use millimetre units to match Ghostlight's lens-space convention.
    The viewport never imports :class:`ghostlight.LensCalibration` directly —
    use :meth:`from_calibration` to translate.

    Convention: the sensor lives at world z=0; lens elements sit at z ≤ 0.
    """

    half_w: float
    half_h: float
    pixel_w: int = 0
    pixel_h: int = 0
    label: str = "sensor"

    @classmethod
    def from_calibration(
        cls,
        calib: "LensCalibration",
        *,
        pixel_w: int = 0,
        pixel_h: int = 0,
        label: str = "sensor",
    ) -> "SensorSpec":
        """Build a SensorSpec from a LensCalibration. Sensor is anchored at z=0."""
        return cls(
            half_w=float(calib.sensor_half_w),
            half_h=float(calib.sensor_half_h),
            pixel_w=int(pixel_w),
            pixel_h=int(pixel_h),
            label=str(label),
        )

    def bbox(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            (-self.half_w, -self.half_h, 0.0),
            ( self.half_w,  self.half_h, 0.0),
        )

    def build_quad(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (vertices [4, 3] float32, indices [6] uint32) for the quad."""
        hw, hh = float(self.half_w), float(self.half_h)
        verts = np.array(
            [[-hw, -hh, 0.0],
             [ hw, -hh, 0.0],
             [ hw,  hh, 0.0],
             [-hw,  hh, 0.0]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        return verts, indices

    def build_border(self) -> np.ndarray:
        """Return (8, 3) float32 line-strip vertices for the rectangle outline."""
        hw, hh = float(self.half_w), float(self.half_h)
        return np.array(
            [[-hw, -hh, 0.0],
             [ hw, -hh, 0.0],
             [ hw, -hh, 0.0],
             [ hw,  hh, 0.0],
             [ hw,  hh, 0.0],
             [-hw,  hh, 0.0],
             [-hw,  hh, 0.0],
             [-hw, -hh, 0.0]],
            dtype=np.float32,
        )


@dataclass
class CalibratedSensorSpec:
    """Circular 'calibrated sensor' showing the area the lens illuminates.

    The radius is the image-circle radius — the smallest circle containing the
    rectangle defined by ``LensCalibration.image_circle_semi_w`` /
    ``image_circle_semi_h`` (i.e. ``sqrt(half_w**2 + half_h**2)``).  Rendered at
    world z=0, same plane as :class:`SensorSpec`.

    Read off the image circle, not ``sensor_half_*``: the latter is where
    vignetting *begins* (90% of axial throughput), which sits well inside the
    illuminated circle on a lens that shades off gradually.
    """

    radius: float
    label: str = "calibrated sensor"

    @classmethod
    def from_calibration(
        cls,
        calib: "LensCalibration",
        *,
        label: str = "calibrated sensor",
    ) -> "CalibratedSensorSpec":
        hw = float(getattr(calib, "image_circle_semi_w", 0.0) or calib.sensor_half_w)
        hh = float(getattr(calib, "image_circle_semi_h", 0.0) or calib.sensor_half_h)
        return cls(radius=math.sqrt(hw * hw + hh * hh), label=str(label))

    def bbox(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        r = float(self.radius)
        return ((-r, -r, 0.0), (r, r, 0.0))

    def build_disk(self, segments: int = _CALIBRATED_SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
        """Triangle-fan disk: returns (vertices [N+2, 3] float32, indices [3*N] uint32)."""
        n = max(8, int(segments))
        r = float(self.radius)
        verts = np.empty((n + 2, 3), dtype=np.float32)
        verts[0] = (0.0, 0.0, 0.0)
        for i in range(n + 1):
            t = (i / n) * 2.0 * math.pi
            verts[i + 1] = (r * math.cos(t), r * math.sin(t), 0.0)
        indices = np.empty(n * 3, dtype=np.uint32)
        for i in range(n):
            indices[i * 3 + 0] = 0
            indices[i * 3 + 1] = i + 1
            indices[i * 3 + 2] = i + 2
        return verts, indices

    def build_circle(self, segments: int = _CALIBRATED_SEGMENTS) -> np.ndarray:
        """Return (2*N, 3) float32 GL_LINES vertices tracing the circle outline."""
        n = max(8, int(segments))
        r = float(self.radius)
        ring = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            t = (i / n) * 2.0 * math.pi
            ring[i] = (r * math.cos(t), r * math.sin(t), 0.0)
        verts = np.empty((2 * n, 3), dtype=np.float32)
        for i in range(n):
            verts[2 * i + 0] = ring[i]
            verts[2 * i + 1] = ring[(i + 1) % n]
        return verts
