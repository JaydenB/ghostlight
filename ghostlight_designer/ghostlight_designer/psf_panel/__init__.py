"""PSF panel type — renders a grid of geometric point-spread functions.

Mirrors the structure of :mod:`ghostlight_designer.sourceflare_panel` (auto-
render, threading, edit-settle debounce, settings dialog).  The visible
output is an N×N tiled composite where each tile shows the PSF at one
field point on the sensor; chief-ray centering means each tile shows
*aberration shape*, not geometric distortion.

Display knobs (per-tile normalisation, log-gain tone mapping) operate
on the cached float buffer without firing a GPU re-render.
"""
from __future__ import annotations

from .body import PSFPanelBody
from .type import PSF_TYPE_ID, register_psf_panel_type

__all__ = [
    "PSFPanelBody",
    "PSF_TYPE_ID",
    "register_psf_panel_type",
]
