"""Registration for the ``psf`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from ..settings import AppSettings
from .body import PSFPanelBody
from .menus import build_menus

PSF_TYPE_ID = "psf"


def register_psf_panel_type(settings: AppSettings) -> None:
    """Register the PSF panel type with the given ``settings``.

    See :func:`ghostlight_designer.sourceflare_panel.type.register_sourceflare_panel_type`
    for the closure-capture rationale.
    """
    registry.unregister(PSF_TYPE_ID)
    registry.register(
        PanelType(
            id=PSF_TYPE_ID,
            display_name="PSF Grid Renderer",
            build_body=lambda project, parent: PSFPanelBody(
                project, settings, parent
            ),
            build_menus=build_menus,
        )
    )
