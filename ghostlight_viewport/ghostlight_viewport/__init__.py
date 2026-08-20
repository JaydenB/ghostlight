"""PySide6 3D viewport for Ghostlight lens systems.

The viewport is a reusable :class:`QOpenGLWidget` subclass that visualises a
:class:`ghostlight.OpticalSystem` plus its element groupings, ray traces, and
sensor.  It never traces, never mutates the lens data — the host pushes
state in and receives selection signals out.

Usage::

    import ghostlight
    from ghostlight_viewport import LensViewport, SensorSpec, RayBundle

    lens = ghostlight.OpticalSystem.load("doublet.lens")
    elements = ghostlight.Element.from_lens_file("doublet.lens")
    sensor = SensorSpec.from_calibration(lens.calibration())

    viewport = LensViewport()
    viewport.set_lens(lens, elements)
    viewport.set_sensor(sensor)
"""

from __future__ import annotations

from .colors import wavelength_to_rgb
from .sensor import CalibratedSensorSpec, SensorSpec
from .rays import RayBundle
from .widget import LensViewport

__all__ = [
    "LensViewport",
    "SensorSpec",
    "CalibratedSensorSpec",
    "RayBundle",
    "wavelength_to_rgb",
]
