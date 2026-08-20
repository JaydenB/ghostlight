"""Top-of-panel toolbar for the optical editor.

Exposes:

* Add ▾        — drop-down to insert a Singlet / Doublet / Triplet /
                 Aperture Stop after the currently-selected element (or
                 at the end of the chain if nothing is selected).
* Remove       — delete the currently-selected element (asks for
                 confirmation).
* Expand All   — expand every element / surface row in the tree.
* Collapse All — collapse every element / surface row in the tree.
* Un-Solo All  — clear the ghost-solo highlight from every surface.
                 Disabled when nothing is solo'd.
* Un-Mute All  — unmute every muted element in one undo entry.
                 Disabled when nothing is muted.
* Un-Flag All  — clear the variable flag from every surface attribute.
                 Disabled when nothing is flagged.

The toolbar is a thin shell: it owns no element / surface mutation logic
or view state itself, just calls into :mod:`element_actions` and emits
intent signals the body wires to the tree. Surface form changes live on
the tree's right-click context menu (see :class:`OpticalEditorBody`) so
the same entry point can be re-bound by a viewport radial menu later.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ghostlight_viewport.icons import make_icon

from ..project import Project
from . import element_actions


# Kind constants used by ``addElementRequested`` to tell the body which
# builder to invoke. Keeping them as strings rather than an enum lets the
# signal cross module boundaries (a radial menu, say) without dragging
# the toolbar's import along.
ADD_SINGLET = "singlet"
ADD_DOUBLET = "doublet"
ADD_TRIPLET = "triplet"
ADD_APERTURE_STOP = "aperture_stop"
# Wizard action — opens the anamorphic setup dialog, then inserts an
# optimizer-driven cyl block. Doesn't share the plain "add element at
# front" flow because it needs its own popup and post-insert dialog.
ADD_ANAMORPHIC_FRONT = "anamorphic_front"

# Import a whole lens file's element chain onto the front (object side) or
# back (sensor side, closest to sensor) of the current system. Like the
# anamorphic wizard, these need a file-picker flow rather than a fixed
# builder, so the body handles them specially outside _ADD_DISPATCH.
ADD_IMPORT_LENS_FRONT = "import_lens_front"
ADD_IMPORT_LENS_BACK = "import_lens_back"


_ICON_PX = 18


class OpticalEditorToolbar(QFrame):
    """Toolbar widget. Emits intent signals; the body wires them to actions."""

    addElementRequested = Signal(str)
    removeElementRequested = Signal()
    expandAllRequested = Signal()
    collapseAllRequested = Signal()
    unsoloAllRequested = Signal()
    unmuteAllRequested = Signal()
    unflagAllVariablesRequested = Signal()

    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

        self.setObjectName("opticalEditorToolbar")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.btn_add = self._build_add_button()
        layout.addWidget(self.btn_add)

        self.btn_remove = QToolButton(self)
        self.btn_remove.setIcon(make_icon("remove", size=_ICON_PX))
        self.btn_remove.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self.btn_remove.setToolTip("Remove the selected element")
        self.btn_remove.setAutoRaise(True)
        self.btn_remove.setFocusPolicy(Qt.NoFocus)
        self.btn_remove.setCursor(Qt.PointingHandCursor)
        self.btn_remove.clicked.connect(
            lambda _checked=False: self.removeElementRequested.emit()
        )
        layout.addWidget(self.btn_remove)

        layout.addStretch(1)

        # Tree view controls — expand / collapse all rows. Kept visually
        # separated from the +/- mutation group by the stretch so it's
        # clear these don't modify the lens. Same autoRaise / flat glyph
        # style as the mutation buttons, per the "not obviously a
        # pushbutton" brief.
        self.btn_expand_all = self._build_flat_button(
            "expand-all", "Expand all rows", self.expandAllRequested
        )
        layout.addWidget(self.btn_expand_all)

        self.btn_collapse_all = self._build_flat_button(
            "collapse-all", "Collapse all rows", self.collapseAllRequested
        )
        layout.addWidget(self.btn_collapse_all)

        # State-clear controls. These grey out when there's nothing to
        # clear so the user knows at a glance whether solo/mute is active
        # anywhere in the system.
        self.btn_unsolo_all = self._build_flat_button(
            "unsolo-all",
            "Clear ghost-solo highlight from every surface",
            self.unsoloAllRequested,
        )
        layout.addWidget(self.btn_unsolo_all)

        self.btn_unmute_all = self._build_flat_button(
            "unmute-all",
            "Unmute every muted element",
            self.unmuteAllRequested,
        )
        layout.addWidget(self.btn_unmute_all)

        self.btn_unflag_all_variables = self._build_flat_button(
            "unvariable-all",
            "Clear the variable flag from every surface attribute",
            self.unflagAllVariablesRequested,
        )
        layout.addWidget(self.btn_unflag_all_variables)

        project.selectionChanged.connect(self._on_selection_changed)
        project.surfaceSelectionChanged.connect(self._on_surface_selection_changed)
        # Solo state has its own signal (view-only, no undo). Mute state
        # rides on systemModified via project.edit(); systemReplaced
        # covers New / Open where both states are reset.
        project.ghostSoloChanged.connect(self._refresh_enabled_state)
        project.systemModified.connect(self._refresh_enabled_state)
        project.systemReplaced.connect(
            lambda _system: self._refresh_enabled_state()
        )
        # Variable flags live on the project too; refresh the button's
        # enabled state whenever the flag map changes. Bound method
        # (not lambda) so Qt auto-disconnects on toolbar destruction —
        # see feedback-pyside-lambda-signals.
        project.variableFlagChanged.connect(self._on_variable_flag_changed)
        project.variableFlagsReplaced.connect(self._refresh_enabled_state)

        self._refresh_enabled_state()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_flat_button(
        self, icon_kind: str, tooltip: str, signal: Signal
    ) -> QToolButton:
        btn = QToolButton(self)
        btn.setIcon(make_icon(icon_kind, size=_ICON_PX))
        btn.setIconSize(QSize(_ICON_PX, _ICON_PX))
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _checked=False: signal.emit())
        return btn

    def _build_add_button(self) -> QToolButton:
        btn = QToolButton(self)
        btn.setIcon(make_icon("add", size=_ICON_PX))
        btn.setIconSize(QSize(_ICON_PX, _ICON_PX))
        btn.setToolTip("Add a new element after the selected one")
        btn.setAutoRaise(True)
        btn.setPopupMode(QToolButton.InstantPopup)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setCursor(Qt.PointingHandCursor)
        # InstantPopup needs a real QMenu — store on self so it isn't GC'd.
        self.add_menu = QMenu(btn)
        for label, kind in [
            ("Singlet",       ADD_SINGLET),
            ("Doublet",       ADD_DOUBLET),
            ("Triplet",       ADD_TRIPLET),
            ("Aperture Stop", ADD_APERTURE_STOP),
        ]:
            act = QAction(label, self.add_menu)
            act.triggered.connect(
                lambda _checked=False, k=kind: self.addElementRequested.emit(k)
            )
            self.add_menu.addAction(act)
        # Separator + wizard action — visually distinguishes the plain
        # "insert a fixed element" items from the multi-step build.
        self.add_menu.addSeparator()
        anam_act = QAction("Anamorphic Front Block…", self.add_menu)
        anam_act.triggered.connect(
            lambda _checked=False:
                self.addElementRequested.emit(ADD_ANAMORPHIC_FRONT)
        )
        self.add_menu.addAction(anam_act)
        # Separator + import-from-file items — splice an existing lens onto
        # the object side (front) or sensor side (back) of the system.
        self.add_menu.addSeparator()
        for label, kind in [
            ("Import Lens → Front (object side)…", ADD_IMPORT_LENS_FRONT),
            ("Import Lens → Back (sensor side)…",  ADD_IMPORT_LENS_BACK),
        ]:
            act = QAction(label, self.add_menu)
            act.triggered.connect(
                lambda _checked=False, k=kind: self.addElementRequested.emit(k)
            )
            self.add_menu.addAction(act)
        btn.setMenu(self.add_menu)
        return btn

    # ------------------------------------------------------------------
    # Selection-driven enable / disable
    # ------------------------------------------------------------------

    def _on_selection_changed(self, _element) -> None:
        self._refresh_enabled_state()

    def _on_surface_selection_changed(self, _surface_index) -> None:
        self._refresh_enabled_state()

    def _on_variable_flag_changed(self, _uuid: str, _attr: str) -> None:
        self._refresh_enabled_state()

    def _refresh_enabled_state(self) -> None:
        self.btn_remove.setEnabled(self._project.selected_element is not None)
        self.btn_unsolo_all.setEnabled(
            bool(self._project.ghost_solo_surface_uuids)
        )
        self.btn_unmute_all.setEnabled(
            element_actions.any_element_muted(self._project)
        )
        self.btn_unflag_all_variables.setEnabled(
            bool(self._project.all_variable_flags())
        )
