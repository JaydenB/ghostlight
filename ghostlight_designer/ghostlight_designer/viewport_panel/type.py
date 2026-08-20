"""Registration for the ``viewport`` panel type."""
from __future__ import annotations

from typing import Optional

from ..panel_system import PanelType, registry
from ..settings import AppSettings
from .body import ViewportPanelBody
from .menus import build_menus

VIEWPORT_TYPE_ID = "viewport"


def register_viewport_panel_type(settings: Optional[AppSettings] = None) -> None:
    """Register the viewport panel type.

    ``settings`` is captured in the build_body closure so every viewport
    panel can persist the context popup's Focus-row unit toggle. Re-register
    each call so the captured ``settings`` is fresh (tests spin up a new
    AppSettings + MainWindow per case)."""
    registry.unregister(VIEWPORT_TYPE_ID)
    registry.register(
        PanelType(
            id=VIEWPORT_TYPE_ID,
            display_name="Viewport",
            build_body=lambda project, parent: ViewportPanelBody(
                project, parent, settings=settings
            ),
            build_menus=build_menus,
        )
    )
