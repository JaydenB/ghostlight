"""Registration for the ``textures`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from ..settings import AppSettings
from .body import TexturesPanelBody

TEXTURES_TYPE_ID = "textures"


def register_textures_panel_type(settings: AppSettings) -> None:
    """Register the Textures panel type with the given ``settings``.

    See :func:`ghostlight_designer.sourceflare_panel.type.register_sourceflare_panel_type`
    for the closure-capture rationale.
    """
    registry.unregister(TEXTURES_TYPE_ID)
    registry.register(
        PanelType(
            id=TEXTURES_TYPE_ID,
            display_name="Textures",
            build_body=lambda project, parent: TexturesPanelBody(
                project, settings, parent
            ),
        )
    )
