"""Spot diagram panel — image-plane spot pattern for a defined set of
fields, wavelengths, and pupil samples.

See the comments in :mod:`ghostlight_designer.evaluation_panels` for the
architectural pattern; this subpackage exposes the registration entry
point and type id.
"""
from __future__ import annotations

from .body import SpotDiagramBody
from .type import SPOT_DIAGRAM_TYPE_ID, register_spot_diagram_panel_type

__all__ = [
    "SpotDiagramBody",
    "SPOT_DIAGRAM_TYPE_ID",
    "register_spot_diagram_panel_type",
]
