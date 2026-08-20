"""Panel type registry.

A ``PanelType`` describes one kind of panel body — its stable id, a human label,
and factories for the body widget and any type-specific menus. Concrete types
live in their own sibling subpackages (``viewport_panel/``, ``optical_editor/``,
...) and self-register via a one-line ``registry.register(...)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMenu, QWidget

from ..project import Project


def _no_menus(body: QWidget, project: Project) -> List[QMenu]:
    return []


def _no_state(body: QWidget) -> dict:
    return {}


def _restore_noop(body: QWidget, state: dict) -> None:
    return None


@dataclass(frozen=True)
class PanelType:
    id: str
    display_name: str
    build_body: Callable[[Project, QWidget], QWidget]
    build_menus: Callable[[QWidget, Project], List[QMenu]] = field(default=_no_menus)
    serialize_state: Callable[[QWidget], dict] = field(default=_no_state)
    restore_state: Callable[[QWidget, dict], None] = field(default=_restore_noop)
    # Empty string = no category; the type appears at the top level of
    # the Panels menu. A non-empty value groups all types sharing that
    # string under a same-named submenu (e.g. "Renderers", "Evaluations").
    # The grouping is a display concern only — type ids and behaviour
    # are unaffected.
    category: str = ""


class PanelTypeRegistry(QObject):
    typesChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._types: dict[str, PanelType] = {}

    def register(self, t: PanelType) -> None:
        self._types[t.id] = t
        self.typesChanged.emit()

    def unregister(self, type_id: str) -> None:
        if type_id in self._types:
            del self._types[type_id]
            self.typesChanged.emit()

    def get(self, type_id: str) -> Optional[PanelType]:
        return self._types.get(type_id)

    def all(self) -> List[PanelType]:
        return list(self._types.values())


registry = PanelTypeRegistry()
