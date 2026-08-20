"""Textures panel — load and visualise the lens's raster image inputs.

The single ``APERTURE_IMAGE`` bitmap mechanism (aperture matte + front-glass
"dirt map" folded into the diffraction pupil) is authored per-surface and
previewed four ways (raw texture / composited pupil / starburst sprite / PSF),
all off one ``_render_starburst_debug`` call.
"""
from __future__ import annotations

from .body import TexturesPanelBody
from .type import TEXTURES_TYPE_ID, register_textures_panel_type

__all__ = [
    "TexturesPanelBody",
    "TEXTURES_TYPE_ID",
    "register_textures_panel_type",
]
