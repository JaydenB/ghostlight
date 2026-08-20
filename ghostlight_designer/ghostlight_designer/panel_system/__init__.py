"""Panel system — type-agnostic infrastructure for the redesignable workspace."""
from __future__ import annotations

from .types import PanelType, PanelTypeRegistry, registry
from .menu_order import HIDDEN_TYPE_IDS, MENU_LAYOUT, ordered_menu_entries
from .panel import Panel
from .container import PanelRoot
from .floating import FloatingPanelWindow
from .layouts import PanelLayout, PanelLayoutRegistry, layout_registry

__all__ = [
    "PanelType",
    "PanelTypeRegistry",
    "registry",
    "HIDDEN_TYPE_IDS",
    "MENU_LAYOUT",
    "ordered_menu_entries",
    "Panel",
    "PanelRoot",
    "FloatingPanelWindow",
    "PanelLayout",
    "PanelLayoutRegistry",
    "layout_registry",
]
