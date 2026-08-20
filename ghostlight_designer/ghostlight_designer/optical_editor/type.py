"""Registration for the ``optical_editor`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from .body import OpticalEditorBody
from .menus import build_menus

OPTICAL_EDITOR_TYPE_ID = "optical_editor"


def register_optical_editor_panel_type() -> None:
    if registry.get(OPTICAL_EDITOR_TYPE_ID) is not None:
        return
    registry.register(
        PanelType(
            id=OPTICAL_EDITOR_TYPE_ID,
            display_name="Optical Design Editor",
            build_body=lambda project, parent: OpticalEditorBody(project, parent),
            build_menus=build_menus,
        )
    )
