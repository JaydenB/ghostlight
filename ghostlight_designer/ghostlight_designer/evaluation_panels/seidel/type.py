"""Registration for the Seidel evaluation panel type."""
from __future__ import annotations

from ...panel_system import PanelType, registry
from ...settings import AppSettings
from .body import SeidelBody
from .menus import build_menus

SEIDEL_TYPE_ID = "evaluation_seidel"


def register_seidel_panel_type(settings: AppSettings) -> None:
    """Register the Seidel bar-chart panel type.

    ``settings`` is captured in the build_body closure so every panel
    instance — including ones created by a split / layout-restore
    — gets the same ``AppSettings`` and respects the global
    Auto-Update master switch. See
    :mod:`ghostlight_designer.evaluation_panels.spot_diagram.type` for the
    full rationale.
    """
    registry.unregister(SEIDEL_TYPE_ID)
    registry.register(
        PanelType(
            id=SEIDEL_TYPE_ID,
            display_name="Seidel Bar Chart",
            build_body=lambda project, parent: SeidelBody(
                project, settings, parent
            ),
            build_menus=build_menus,
            category="Evaluations",
        )
    )
