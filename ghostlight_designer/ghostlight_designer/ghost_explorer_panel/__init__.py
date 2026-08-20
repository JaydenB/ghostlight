"""Ghost Explorer panel — scrub through a lens's ghosts one at a time.

Renders the source-flare stack from a fixed point source, with the aperture
starburst and veiling glare switched off and the render's ``GhostFilter``
pinned to a single pair, so each frame shows exactly one ghost reflection. A
slider along the bottom steps through the lens's renderable pairs, with a
readout naming the two surfaces the ghost bounces off and its ghost number.

One coarse whole-flare pass sets the viewer exposure — so every ghost is shown
at its true brightness relative to the others — and, while the cull is on
(the default), scores each pair so the scrubber can skip the ones too dim to
see.
"""
from .body import GhostExplorerPanelBody
from .ghost_survey import GhostEntry
from .type import GHOST_EXPLORER_TYPE_ID, register_ghost_explorer_panel_type

__all__ = [
    "GhostExplorerPanelBody",
    "GhostEntry",
    "GHOST_EXPLORER_TYPE_ID",
    "register_ghost_explorer_panel_type",
]
