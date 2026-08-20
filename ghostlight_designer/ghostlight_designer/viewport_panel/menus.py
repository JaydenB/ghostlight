"""Menus contributed by the ``viewport`` panel type."""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ..project import Project
from .body import ViewportPanelBody


def build_menus(body: ViewportPanelBody, project: Project) -> List[QMenu]:
    menu = QMenu("&View")

    act_axes = QAction("&Axes", menu)
    act_axes.setCheckable(True)
    act_axes.setChecked(True)
    act_axes.toggled.connect(body.viewport.set_show_axes)
    menu.addAction(act_axes)

    act_cube = QAction("View &Cube", menu)
    act_cube.setCheckable(True)
    act_cube.setChecked(True)
    act_cube.toggled.connect(body.viewport.set_show_view_cube)
    menu.addAction(act_cube)

    act_rays = QAction("&Rays", menu)
    act_rays.setCheckable(True)
    act_rays.setChecked(body.show_rays)
    act_rays.toggled.connect(body.set_show_rays)
    menu.addAction(act_rays)

    # Element centre-of-rotation markers. On by default; they only draw for
    # elements that declare a non-zero pivot, so this is a no-op on an
    # on-axis lens rather than permanent clutter.
    act_pivots = QAction("Pi&vots", menu)
    act_pivots.setCheckable(True)
    act_pivots.setChecked(body.viewport.show_pivots())
    act_pivots.toggled.connect(body.viewport.set_show_pivots)
    menu.addAction(act_pivots)

    menu.addSeparator()

    act_reset = QAction("&Reset View", menu)
    act_reset.triggered.connect(body.viewport.reset_view)
    menu.addAction(act_reset)

    return [menu]
