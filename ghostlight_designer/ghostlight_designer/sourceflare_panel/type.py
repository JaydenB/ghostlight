"""Registration for the ``sourceflare`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from ..settings import AppSettings
from .body import SourceFlarePanelBody
from .menus import build_menus

SOURCEFLARE_TYPE_ID = "sourceflare"


def register_sourceflare_panel_type(settings: AppSettings) -> None:
    """Register the sourceflare panel type with the given ``settings``.

    ``settings`` is captured by the panel factory closure so each panel
    instance can reach the shared :class:`AppSettings` without a global.
    """
    registry.unregister(SOURCEFLARE_TYPE_ID)
    registry.register(
        PanelType(
            id=SOURCEFLARE_TYPE_ID,
            display_name="Source Flare Renderer",
            build_body=lambda project, parent: SourceFlarePanelBody(
                project, settings, parent
            ),
            build_menus=build_menus,
        )
    )
