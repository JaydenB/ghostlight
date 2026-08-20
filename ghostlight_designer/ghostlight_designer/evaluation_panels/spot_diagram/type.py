"""Registration for the ``spot_diagram`` evaluation panel type."""
from __future__ import annotations

from ...panel_system import PanelType, registry
from ...settings import AppSettings
from .body import SpotDiagramBody
from .menus import build_menus

SPOT_DIAGRAM_TYPE_ID = "evaluation_spot_diagram"


def register_spot_diagram_panel_type(settings: AppSettings) -> None:
    """Register the spot diagram panel type.

    ``settings`` is captured in the build_body closure so every panel
    instance — including any created by a split / layout-restore
    — gets the same ``AppSettings`` and respects the View →
    Auto-Update master switch.
    """
    registry.unregister(SPOT_DIAGRAM_TYPE_ID)
    registry.register(
        PanelType(
            id=SPOT_DIAGRAM_TYPE_ID,
            display_name="Spot Diagram",
            build_body=lambda project, parent: SpotDiagramBody(
                project, settings, parent
            ),
            build_menus=build_menus,
            # Groups every evaluation panel under an
            # "Evaluations" submenu in the per-panel "Panel" menu,
            # alongside the existing "Renderers" submenu.
            category="Evaluations",
        )
    )
