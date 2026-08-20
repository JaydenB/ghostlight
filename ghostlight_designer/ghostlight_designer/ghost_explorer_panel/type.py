"""Registration for the ``ghost_explorer`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from ..settings import AppSettings
from .body import GhostExplorerPanelBody
from .menus import build_menus

GHOST_EXPLORER_TYPE_ID = "ghost_explorer"


def register_ghost_explorer_panel_type(settings: AppSettings) -> None:
    """Register the ghost-explorer panel type with the given ``settings``.

    See :func:`ghostlight_designer.sourceflare_panel.type.register_sourceflare_panel_type`
    for the closure-capture rationale.
    """
    registry.unregister(GHOST_EXPLORER_TYPE_ID)
    registry.register(
        PanelType(
            id=GHOST_EXPLORER_TYPE_ID,
            display_name="Ghost Explorer",
            build_body=lambda project, parent: GhostExplorerPanelBody(
                project, settings, parent
            ),
            build_menus=build_menus,
        )
    )
