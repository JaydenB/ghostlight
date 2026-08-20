"""Predefined panel layouts.

A ``PanelLayout`` is a named, immutable workspace template — a nested
split/leaf ``spec`` dict in the same form ``PanelRoot.build_default_layout``
expects. Layouts are registered at app startup (see ``main_window``) and
exposed in each panel's "Layouts" menu so users can flip between predefined
arrangements (and reset to default) with one click.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class PanelLayout:
    id: str
    display_name: str
    spec: dict


class PanelLayoutRegistry(QObject):
    layoutsChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._layouts: Dict[str, PanelLayout] = {}
        self._order: List[str] = []

    def register(self, layout: PanelLayout) -> None:
        if layout.id not in self._layouts:
            self._order.append(layout.id)
        self._layouts[layout.id] = layout
        self.layoutsChanged.emit()

    def get(self, layout_id: str) -> Optional[PanelLayout]:
        return self._layouts.get(layout_id)

    def all(self) -> List[PanelLayout]:
        return [self._layouts[i] for i in self._order]

    def clear(self) -> None:
        self._layouts.clear()
        self._order.clear()
        self.layoutsChanged.emit()


layout_registry = PanelLayoutRegistry()
