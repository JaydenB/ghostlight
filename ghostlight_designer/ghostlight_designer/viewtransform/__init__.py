"""Designer-wide display colour management for the render panels.

The ghostlight renderers emit **scene-linear ACEScg** (AP1 primaries, D60
white, unbounded HDR float), matching the compositing output contract.
This package turns that buffer into display pixels through a real ACES 2.0
output transform (OpenColorIO ``DisplayViewTransform``) plus a viewer
exposure in stops, so the designer previews exactly what a Nuke session on an
ACES 2.0 OCIO config will show.

Colourspace contract (also the contract for the Nuke plugin):
    * The renderer output plane is scene-linear in ``RenderConfig.output_cs``.
      Panels pin ``output_cs = ACESCG`` so it is declared, not assumed.
    * The view transform's source space is therefore ``ACEScg`` (resolved by
      name/alias/role against whichever config is active).
    * Exposure is a linear pre-multiply (``* 2**stops``) applied *before* the
      view transform — identical to Nuke's Viewer gain.

The view/display selection is designer-wide (persisted in ``AppSettings``);
exposure is per-panel (like Nuke's per-Viewer gain).
"""
from __future__ import annotations

from .pipeline import (
    ViewTransformError,
    ViewTransformSpec,
    apply_view,
    available_views,
    compute_exposure_scale,
    default_config_name,
    get_processor,
    meter_auto_stops,
    resolve_default_display_view,
    spec_from_settings,
)
from .qimage import to_qimage

__all__ = [
    "ViewTransformError",
    "ViewTransformSpec",
    "apply_view",
    "available_views",
    "compute_exposure_scale",
    "default_config_name",
    "get_processor",
    "meter_auto_stops",
    "resolve_default_display_view",
    "spec_from_settings",
    "to_qimage",
]
