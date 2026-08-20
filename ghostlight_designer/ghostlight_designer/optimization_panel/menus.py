"""Menu-bar contribution for the optimization panel.

Adds a single "Optimization" menu when this panel is in the layout. The
menu mirrors the toolbar's actions plus keyboard shortcuts; everything
routes through :class:`OptimizationPanelBody` so behaviour stays in one
place.
"""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu

from ..project import Project
from .body import OptimizationPanelBody
from .data import GoalKind
from .goals.base import GOAL_REGISTRY, display_name_for
from .presets import PRESETS


def build_menus(body: OptimizationPanelBody, _project: Project) -> List[QMenu]:
    menu = QMenu("&Optimization")

    add_mf_menu = menu.addMenu("Add &Merit Function")
    for label, _builder in PRESETS:
        act = QAction(label, add_mf_menu)
        act.triggered.connect(
            lambda _checked=False, l=label: body._on_add_mf(l)
        )
        add_mf_menu.addAction(act)

    add_goal_menu = menu.addMenu("Add &Goal to Selected MF")
    for kind in GoalKind:
        if kind not in GOAL_REGISTRY:
            continue
        act = QAction(display_name_for(kind), add_goal_menu)
        act.triggered.connect(
            lambda _checked=False, k=kind.value: body._on_add_goal(k)
        )
        add_goal_menu.addAction(act)

    menu.addSeparator()

    act_run = QAction("&Run Selected", menu)
    act_run.setShortcut(QKeySequence("Ctrl+Shift+R"))
    act_run.triggered.connect(body._on_run_selected)
    menu.addAction(act_run)

    act_run_all = QAction("Run &All", menu)
    act_run_all.setShortcut(QKeySequence("Ctrl+Alt+R"))
    act_run_all.triggered.connect(body._on_run_all)
    menu.addAction(act_run_all)

    menu.addSeparator()

    act_remove = QAction("Remove &Selected", menu)
    act_remove.setShortcut(QKeySequence("Del"))
    act_remove.triggered.connect(body._on_remove)
    menu.addAction(act_remove)

    return [menu]
