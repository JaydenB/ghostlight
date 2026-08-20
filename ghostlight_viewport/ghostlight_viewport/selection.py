"""Selection state holder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SelectionState:
    """Tracks the currently-selected and hovered scene items.

    ``element`` / ``surface`` are the clicked picks (``surface`` is a global
    index into ``system.surfaces`` or ``None``); ``hover`` / ``hover_surface``
    are their "about to select" counterparts.  Hover is purely visual and
    never emitted as a selection signal — they're independent UI states.

    Surface mode populates both ``element`` + ``surface`` (and likewise
    ``hover`` + ``hover_surface``) so the host gets element-level context
    while the renderer picks the more-specific surface state for highlight.
    """

    element: Optional[Any] = None
    surface: Optional[int] = None
    hover: Optional[Any] = None
    hover_surface: Optional[int] = None

    def set_element(self, element: Any) -> bool:
        if self.element is element:
            return False
        self.element = element
        return True

    def set_surface(self, surface_index: Optional[int]) -> bool:
        if self.surface == surface_index:
            return False
        self.surface = surface_index
        return True

    def clear(self) -> bool:
        changed = self.element is not None or self.surface is not None
        self.element = None
        self.surface = None
        return changed

    def clear_surface(self) -> bool:
        if self.surface is None:
            return False
        self.surface = None
        return True

    def set_hover(self, element: Any) -> bool:
        if self.hover is element:
            return False
        self.hover = element
        return True

    def set_hover_surface(self, surface_index: Optional[int]) -> bool:
        if self.hover_surface == surface_index:
            return False
        self.hover_surface = surface_index
        return True

    def clear_hover(self) -> bool:
        changed = self.hover is not None or self.hover_surface is not None
        self.hover = None
        self.hover_surface = None
        return changed
