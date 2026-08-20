"""Top-of-panel toolbar for the optimization editor.

Layout: ``+ MF ▾ | + Goal ▾ | ─ Remove | ▶ Run | ▶▶ Run All``

Same conventions as the Optical Design Editor's toolbar: emits intent
signals; the body owns the wiring.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from .data import GoalKind
from .goals.base import GOAL_REGISTRY, display_name_for
from .presets import PRESETS


class OptimizationToolbar(QFrame):
    """Toolbar widget; emits intent signals consumed by the body."""

    # Carries the preset *label* — the body looks up the builder in PRESETS.
    addMeritFunctionRequested = Signal(str)
    removeRequested = Signal()
    runSelectedRequested = Signal()
    runAllRequested = Signal()
    # Carries the :class:`GoalKind` value (str). The body decides which MF
    # the goal goes into based on the current selection.
    addGoalRequested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("optimizationToolbar")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.btn_add_mf = self._build_add_mf_button()
        layout.addWidget(self.btn_add_mf)

        self.btn_add_goal = self._build_add_goal_button()
        layout.addWidget(self.btn_add_goal)

        self.btn_remove = QPushButton("Remove", self)
        self.btn_remove.setToolTip(
            "Remove the selected merit function or goal."
        )
        self.btn_remove.clicked.connect(
            lambda _checked=False: self.removeRequested.emit()
        )
        layout.addWidget(self.btn_remove)

        layout.addSpacing(8)

        self.btn_run = QPushButton("Run", self)
        self.btn_run.setToolTip("Run the selected merit function in a preview window.")
        self.btn_run.clicked.connect(
            lambda _checked=False: self.runSelectedRequested.emit()
        )
        layout.addWidget(self.btn_run)

        self.btn_run_all = QPushButton("Run All", self)
        self.btn_run_all.setToolTip(
            "Run every enabled merit function one after another. "
            "Each opens its own preview dialog in turn."
        )
        self.btn_run_all.clicked.connect(
            lambda _checked=False: self.runAllRequested.emit()
        )
        layout.addWidget(self.btn_run_all)

        layout.addStretch(1)

    def _build_add_mf_button(self) -> QToolButton:
        btn = QToolButton(self)
        btn.setText("+ Merit Function ▾")
        btn.setToolTip(
            "Add a new merit function. Presets seed targets from the "
            "current lens; pick Empty for a blank slate."
        )
        btn.setPopupMode(QToolButton.InstantPopup)
        btn.setFocusPolicy(Qt.NoFocus)
        self._add_mf_menu = QMenu(btn)
        for label, _builder in PRESETS:
            act = QAction(label, self._add_mf_menu)
            act.triggered.connect(
                lambda _checked=False, l=label: self.addMeritFunctionRequested.emit(l)
            )
            self._add_mf_menu.addAction(act)
        btn.setMenu(self._add_mf_menu)
        return btn

    def _build_add_goal_button(self) -> QToolButton:
        btn = QToolButton(self)
        btn.setText("+ Goal ▾")
        btn.setToolTip(
            "Add a goal to the selected merit function. Select an MF "
            "(or any of its rows) first."
        )
        btn.setPopupMode(QToolButton.InstantPopup)
        btn.setFocusPolicy(Qt.NoFocus)
        self._add_goal_menu = QMenu(btn)
        # Iterate the registry in enum declaration order so the menu has
        # a stable shape across runs.
        for kind in GoalKind:
            if kind not in GOAL_REGISTRY:
                continue
            act = QAction(display_name_for(kind), self._add_goal_menu)
            act.triggered.connect(
                lambda _checked=False, k=kind.value: self.addGoalRequested.emit(k)
            )
            self._add_goal_menu.addAction(act)
        btn.setMenu(self._add_goal_menu)
        return btn
