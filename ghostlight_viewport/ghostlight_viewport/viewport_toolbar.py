"""Floating overlay toolbar for :class:`LensViewport`.

A row of square icon-enum buttons anchored to the top-right of the viewport,
sitting to the left of the existing view-cube gizmo.  Each button shows the
icon of its currently-selected option and pops a menu of icon+text choices.

Icon glyphs live in :mod:`ghostlight_viewport.icons` so host apps (e.g. the
designer's optical-editor tree) can reuse the same painted artwork.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QToolButton,
    QWidget,
)

from .icons import make_icon


_BUTTON_PX = 28
_ICON_PX = 20


# Each option is (key, label, icon_kind).
Option = tuple[str, str, str]


class IconEnumButton(QToolButton):
    """Square button whose icon reflects its current enum value.

    Click pops an :class:`QMenu` of (icon, label) actions; selecting one
    updates the button icon, stores the key, and emits :sig:`valueChanged`.
    """

    valueChanged = Signal(str)

    def __init__(
        self,
        options: Iterable[Option],
        default_key: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._options: list[Option] = list(options)
        if not self._options:
            raise ValueError("IconEnumButton needs at least one option")
        self._value: str = default_key
        self._actions: dict[str, QAction] = {}

        self.setFixedSize(_BUTTON_PX, _BUTTON_PX)
        self.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self.setAutoRaise(True)
        self.setPopupMode(QToolButton.InstantPopup)
        # Hide the popup-arrow indicator so the icon stays centred.
        self.setStyleSheet("QToolButton::menu-indicator { image: none; }")

        menu = QMenu(self)
        for key, label, kind in self._options:
            act = QAction(make_icon(kind), label, menu)
            act.triggered.connect(lambda _checked=False, k=key: self.set_value(k))
            menu.addAction(act)
            self._actions[key] = act
        self.setMenu(menu)

        # Initialise icon + tooltip.
        self.set_value(default_key, emit=False)

    def value(self) -> str:
        return self._value

    def set_value(self, key: str, emit: bool = True) -> None:
        if key not in self._actions:
            raise ValueError(f"unknown option key: {key!r}")
        option = next(o for o in self._options if o[0] == key)
        _, label, kind = option
        changed = self._value != key
        self._value = key
        self.setIcon(make_icon(kind))
        self.setToolTip(label)
        if emit and changed:
            self.valueChanged.emit(key)


class ViewportToolbar(QWidget):
    """Row of icon-enum buttons overlaid on :class:`LensViewport`.

    Hosts the Cutaway and Selection Mode buttons.  Pass-through signals let
    the parent viewport react without depending on the button widgets.
    """

    cutawayChanged = Signal(str)
    selectionModeChanged = Signal(str)

    _CUTAWAY_OPTIONS: list[Option] = [
        ("none", "None",     "none"),
        ("x",    "X Plane",  "x"),
        ("y",    "Y Plane",  "y"),
        ("xy",   "XY Plane", "xy"),
    ]
    _SELECTION_OPTIONS: list[Option] = [
        ("element", "Element", "elem"),
        ("surface", "Surface", "surf"),
        ("none",    "None",    "sel_none"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Non-opaque so child buttons render over the GL surface cleanly.
        self.setAutoFillBackground(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.cutaway = IconEnumButton(self._CUTAWAY_OPTIONS, "none", self)
        self.selection = IconEnumButton(self._SELECTION_OPTIONS, "element", self)
        layout.addWidget(self.cutaway)
        layout.addWidget(self.selection)

        self.cutaway.valueChanged.connect(self.cutawayChanged)
        self.selection.valueChanged.connect(self.selectionModeChanged)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def cutaway_value(self) -> str:
        return self.cutaway.value()

    def set_cutaway(self, key: str, emit: bool = True) -> None:
        self.cutaway.set_value(key, emit=emit)

    def selection_mode(self) -> str:
        return self.selection.value()

    def set_selection_mode(self, key: str, emit: bool = True) -> None:
        self.selection.set_value(key, emit=emit)
