"""Floating-window host for undocked Panels.

A FloatingPanelWindow owns its own :class:`PanelRoot` so the user can
split, swap, and close panels inside the floating window using the same
mechanism as the docked workspace. Closing the last leaf closes the
window.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMainWindow

from ..project import Project
from .container import PanelRoot
from .layouts import PanelLayoutRegistry
from .panel import Panel
from .types import PanelTypeRegistry


class FloatingPanelWindow(QMainWindow):
    panelAdded = Signal(object)
    panelRemoved = Signal(object)
    windowClosed = Signal()

    def __init__(
        self,
        project: Project,
        registry: Optional[PanelTypeRegistry] = None,
        layout_registry: Optional[PanelLayoutRegistry] = None,
    ) -> None:
        super().__init__(None)
        self.setWindowTitle("Ghostlight Designer — Panel")
        self._panel_root = PanelRoot(
            project,
            parent=self,
            registry=registry,
            layout_registry=layout_registry,
        )
        self.setCentralWidget(self._panel_root)
        self._panel_root.panelAdded.connect(self._on_panel_added)
        self._panel_root.panelRemoved.connect(self._on_panel_removed)
        self._closing = False

    @property
    def panel_root(self) -> PanelRoot:
        return self._panel_root

    @property
    def panel(self) -> Optional[Panel]:
        """First leaf of this floating window, or None if empty.

        Retained as a convenience for the single-panel case; callers that
        need the full tree should go through :attr:`panel_root` instead.
        """
        leaves = self._panel_root.leaves()
        return leaves[0] if leaves else None

    @property
    def panels(self) -> List[Panel]:
        return self._panel_root.leaves()

    def set_root_panel(self, panel: Panel) -> None:
        self._panel_root.set_root_panel(panel)

    def _on_panel_added(self, panel: Panel) -> None:
        panel.set_floating(True)
        panel.set_close_visible(True)
        self.panelAdded.emit(panel)

    def _on_panel_removed(self, panel: Panel) -> None:
        panel.set_floating(False)
        self.panelRemoved.emit(panel)
        if self._panel_root.root_widget is None and not self._closing:
            self.close()

    def closeEvent(self, event) -> None:
        self._closing = True
        for p in self._panel_root.leaves():
            p.set_floating(False)
        self.windowClosed.emit()
        event.accept()
