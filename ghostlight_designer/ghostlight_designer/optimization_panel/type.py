"""Registration for the ``optimization`` panel type."""
from __future__ import annotations

from ..panel_system import PanelType, registry
from .body import OptimizationPanelBody
from .menus import build_menus

OPTIMIZATION_TYPE_ID = "optimization"


def register_optimization_panel_type() -> None:
    if registry.get(OPTIMIZATION_TYPE_ID) is not None:
        return
    registry.register(
        PanelType(
            id=OPTIMIZATION_TYPE_ID,
            display_name="Optimization",
            build_body=lambda project, parent: OptimizationPanelBody(project, parent),
            build_menus=build_menus,
        )
    )
