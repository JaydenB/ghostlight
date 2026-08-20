"""Source Flare Renderer panel — extended (area) light-source flares.

The light source is a geometric shape (point / circle / rectangle /
square) with an angular size in degrees, rendered by averaging point-flare
passes over samples of the shape via
``OpticalSystem.render_source_flare``.
"""
from .body import SourceFlarePanelBody
from .type import SOURCEFLARE_TYPE_ID, register_sourceflare_panel_type

__all__ = [
    "SourceFlarePanelBody",
    "SOURCEFLARE_TYPE_ID",
    "register_sourceflare_panel_type",
]
