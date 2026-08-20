"""Registration for the ``field_diagrams`` evaluation panel type."""
from __future__ import annotations

from ...panel_system import PanelType, registry
from ...settings import AppSettings
from .body import FieldDiagramBody
from .menus import build_menus

FIELD_DIAGRAMS_TYPE_ID = "evaluation_field_diagrams"


def register_field_diagrams_panel_type(settings: AppSettings) -> None:
    """Register the field-diagrams panel type.

    ``settings`` is captured in the build_body closure so every panel
    instance — including ones created by a split / layout-restore
    — gets the same ``AppSettings`` and respects the global
    Auto-Update master switch. See
    :mod:`ghostlight_designer.evaluation_panels.spot_diagram.type` for the
    full rationale.
    """
    registry.unregister(FIELD_DIAGRAMS_TYPE_ID)
    registry.register(
        PanelType(
            id=FIELD_DIAGRAMS_TYPE_ID,
            display_name="Field Diagrams",
            build_body=lambda project, parent: FieldDiagramBody(
                project, settings, parent
            ),
            build_menus=build_menus,
            category="Evaluations",
        )
    )
