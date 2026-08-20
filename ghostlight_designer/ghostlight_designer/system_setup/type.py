"""Registration for the ``system_setup`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from .body import SystemSetupBody
from .menus import build_menus

SYSTEM_SETUP_TYPE_ID = "system_setup"


def register_system_setup_panel_type() -> None:
    if registry.get(SYSTEM_SETUP_TYPE_ID) is not None:
        return
    registry.register(
        PanelType(
            id=SYSTEM_SETUP_TYPE_ID,
            display_name="System Setup",
            build_body=lambda project, parent: SystemSetupBody(project, parent),
            build_menus=build_menus,
        )
    )
