"""Leaf panel widget — top bar (menu buttons + corner undock/close) + body."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..project import Project
from .layouts import PanelLayoutRegistry, layout_registry as default_layout_registry
from .menu_order import ordered_menu_entries
from .types import registry as default_registry, PanelTypeRegistry


MENUBAR_HEIGHT = 28

_MENUBAR_STYLE = """
QFrame#panelMenuBar {
    background: palette(window);
    border-bottom: 1px solid palette(mid);
}
QFrame#panelMenuBar QPushButton,
QFrame#panelMenuBar QToolButton {
    background: transparent;
    border: 1px solid transparent;
    padding: 2px 8px;
}
QFrame#panelMenuBar QToolButton {
    padding: 2px 4px;
}
QFrame#panelMenuBar QPushButton:hover,
QFrame#panelMenuBar QToolButton:hover {
    background: palette(midlight);
}
QFrame#panelMenuBar QPushButton:pressed,
QFrame#panelMenuBar QToolButton:pressed {
    background: palette(highlight);
    color: palette(highlighted-text);
}
QFrame#panelMenuBar QPushButton:disabled,
QFrame#panelMenuBar QToolButton:disabled {
    color: palette(mid);
}
"""


class Panel(QWidget):
    splitRequested = Signal(int, object)
    undockRequested = Signal(object)
    closeRequested = Signal(object)
    typeChanged = Signal(str)
    layoutRequested = Signal(str)

    def __init__(
        self,
        type_id: str,
        project: Project,
        parent: Optional[QWidget] = None,
        registry: Optional[PanelTypeRegistry] = None,
        layout_registry: Optional[PanelLayoutRegistry] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._registry = registry if registry is not None else default_registry
        self._layout_registry = (
            layout_registry if layout_registry is not None else default_layout_registry
        )
        self._type_id = type_id
        self._is_floating = False
        self._type_menus: List[QMenu] = []
        self._type_menu_buttons: List[QPushButton] = []
        self._type_actions: List[QAction] = []
        self._layout_actions: List[QAction] = []
        # Cache of category submenu widgets, keyed by category name
        # ("Renderers", "Evaluations", …). Reused across rebuilds so a
        # repeated menu open doesn't churn QMenu objects — each rebuild
        # clears the contents instead of reconstructing the menu.
        self._category_menus: Dict[str, QMenu] = {}

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._menubar = QFrame(self)
        self._menubar.setFrameShape(QFrame.NoFrame)
        self._menubar.setFixedHeight(MENUBAR_HEIGHT)
        self._menubar.setObjectName("panelMenuBar")
        self._menubar.setStyleSheet(_MENUBAR_STYLE)
        self._menubar_layout = QHBoxLayout(self._menubar)
        self._menubar_layout.setContentsMargins(2, 0, 2, 0)
        self._menubar_layout.setSpacing(0)
        layout.addWidget(self._menubar)

        self._body_container = QWidget(self)
        self._body_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body_layout = QVBoxLayout(self._body_container)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        layout.addWidget(self._body_container, 1)

        self._build_panel_menu()
        self._menubar_layout.addStretch(1)
        self._stretch_index = self._menubar_layout.count() - 1
        self._build_corner_buttons()
        self._build_body_and_type_menus()
        self._menubar_layout.activate()

    @property
    def type_id(self) -> str:
        return self._type_id

    @property
    def body(self) -> Optional[QWidget]:
        items = self._body_container.layout()
        if items.count() == 0:
            return None
        return items.itemAt(0).widget()

    @property
    def project(self) -> Project:
        return self._project

    def set_floating(self, floating: bool) -> None:
        self._is_floating = floating
        self.btn_undock.setEnabled(not floating)

    def is_floating(self) -> bool:
        return self._is_floating

    def set_close_visible(self, visible: bool) -> None:
        self.btn_close.setVisible(visible)

    def change_type(self, new_type_id: str) -> None:
        if new_type_id == self._type_id:
            return
        self._tear_down_body_and_type_menus()
        self._type_id = new_type_id
        self._build_body_and_type_menus()
        self.typeChanged.emit(new_type_id)

    def _build_panel_menu(self) -> None:
        # NB: QMenu must NOT be created with a parent widget — setParent on a
        # QMenu strips its Qt.Window flag (part of Qt.Popup), demoting it to a
        # regular child that Qt then renders inline. Keep menus parentless and
        # hold the reference via self.menu_panel / self._type_menus to prevent
        # garbage collection.
        self.menu_panel = QMenu()
        self.menu_panel.setTitle("Panel")

        self.action_split_h = QAction("Split &Horizontally", self)
        self.action_split_h.triggered.connect(
            lambda _checked=False: self.splitRequested.emit(Qt.Vertical.value, self)
        )
        self.menu_panel.addAction(self.action_split_h)

        self.action_split_v = QAction("Split &Vertically", self)
        self.action_split_v.triggered.connect(
            lambda _checked=False: self.splitRequested.emit(Qt.Horizontal.value, self)
        )
        self.menu_panel.addAction(self.action_split_v)

        self.menu_layouts = QMenu()
        self.menu_layouts.setTitle("&Layouts")
        self.action_layouts = self.menu_panel.addMenu(self.menu_layouts)
        self.menu_layouts.aboutToShow.connect(self._rebuild_layout_actions)

        self.menu_panel.addSeparator()
        self.menu_panel.aboutToShow.connect(self._rebuild_type_actions)

        self.btn_panel_menu = self._make_menu_button("&Panel", self.menu_panel)
        self._menubar_layout.addWidget(self.btn_panel_menu)

    def _rebuild_layout_actions(self) -> None:
        for a in self._layout_actions:
            self.menu_layouts.removeAction(a)
        self._layout_actions.clear()
        layouts = self._layout_registry.all()
        if not layouts:
            placeholder = QAction("(no layouts)", self.menu_layouts)
            placeholder.setEnabled(False)
            self.menu_layouts.addAction(placeholder)
            self._layout_actions.append(placeholder)
            return
        for lay in layouts:
            act = QAction(lay.display_name, self.menu_layouts)
            act.triggered.connect(
                lambda _checked=False, lid=lay.id: self.layoutRequested.emit(lid)
            )
            self.menu_layouts.addAction(act)
            self._layout_actions.append(act)

    def _make_menu_button(self, text: str, menu: QMenu) -> QPushButton:
        btn = QPushButton(text, self._menubar)
        btn.setFlat(True)
        btn.setFocusPolicy(Qt.NoFocus)
        btn.setCursor(Qt.PointingHandCursor)
        # WA_Hover makes :hover position-based (Qt polls the cursor each
        # paint) instead of relying on Enter/Leave events, which Qt is known
        # to skip when the cursor jumps directly between adjacent widgets —
        # the cause of the "two menu buttons highlighted at once" and
        # "highlight sticks after cursor leaves" symptoms.
        btn.setAttribute(Qt.WA_Hover, True)
        # Explicit popup on click — no setMenu(), no InstantPopup. Avoids the
        # PySide6 quirk where setMenu()+InstantPopup can spontaneously show the
        # menu before the first user interaction.
        btn.clicked.connect(
            lambda _checked=False, b=btn, m=menu: self._show_menu_popup(b, m)
        )
        return btn

    def _show_menu_popup(self, btn: QPushButton, menu: QMenu) -> None:
        menu.exec_(btn.mapToGlobal(btn.rect().bottomLeft()))
        # After the popup grabs and releases the mouse, Qt's cached hover
        # state for the button is often stale — force it to resync with
        # the actual cursor position.
        local = btn.mapFromGlobal(QCursor.pos())
        if not btn.rect().contains(local):
            btn.setAttribute(Qt.WA_UnderMouse, False)
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.update()

    def _rebuild_type_actions(self) -> None:
        for a in self._type_actions:
            self.menu_panel.removeAction(a)
        self._type_actions.clear()
        for menu in self._category_menus.values():
            menu.clear()

        # Curated order shared with the Windows menu — see
        # panel_system.menu_order. Hidden types are dropped and unlisted
        # ones fall through alphabetically there, so nothing here needs to
        # know per-type category rules.
        for entry in ordered_menu_entries(self._registry):
            if entry[0] == "group":
                _, cat_name, types = entry
                menu = self._category_menus.get(cat_name)
                if menu is None:
                    # Parentless, like menu_panel — Qt owns the popup.
                    menu = QMenu(cat_name)
                    self._category_menus[cat_name] = menu
                else:
                    menu.setTitle(cat_name)
                sub_action = self.menu_panel.addMenu(menu)
                self._type_actions.append(sub_action)
                for t in types:
                    menu.addAction(self._make_type_action(t, menu))
            else:
                act = self._make_type_action(entry[1], self.menu_panel)
                self.menu_panel.addAction(act)
                self._type_actions.append(act)

    def _make_type_action(self, t, parent: QMenu) -> QAction:
        act = QAction(t.display_name, parent)
        act.setCheckable(True)
        act.setChecked(t.id == self._type_id)
        act.setEnabled(t.id != self._type_id)
        act.triggered.connect(lambda _checked=False, tid=t.id: self.change_type(tid))
        return act

    def _build_corner_buttons(self) -> None:
        style = self.style()

        self.btn_undock = QToolButton(self._menubar)
        self.btn_undock.setIcon(style.standardIcon(QStyle.SP_TitleBarNormalButton))
        self.btn_undock.setAutoRaise(True)
        self.btn_undock.setToolTip("Undock panel")
        self.btn_undock.setFocusPolicy(Qt.NoFocus)
        self.btn_undock.setCursor(Qt.PointingHandCursor)
        self.btn_undock.setAttribute(Qt.WA_Hover, True)
        self.btn_undock.clicked.connect(lambda: self.undockRequested.emit(self))
        self._menubar_layout.addWidget(self.btn_undock)

        self.btn_close = QToolButton(self._menubar)
        self.btn_close.setIcon(style.standardIcon(QStyle.SP_TitleBarCloseButton))
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close panel")
        self.btn_close.setFocusPolicy(Qt.NoFocus)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setAttribute(Qt.WA_Hover, True)
        self.btn_close.clicked.connect(lambda: self.closeRequested.emit(self))
        self._menubar_layout.addWidget(self.btn_close)

    def _build_body_and_type_menus(self) -> None:
        t = self._registry.get(self._type_id)
        if t is None:
            placeholder = QWidget(self._body_container)
            self._body_container.layout().addWidget(placeholder)
            return
        body = t.build_body(self._project, self)
        self._body_container.layout().addWidget(body)
        type_menus = t.build_menus(body, self._project) or []
        insert_at = self._stretch_index  # insert just before the stretch
        for m in type_menus:
            title = m.title()
            btn = self._make_menu_button(title, m)
            # Do NOT setParent on the QMenu — see _build_panel_menu comment.
            # self._type_menus holds the reference.
            self._menubar_layout.insertWidget(insert_at, btn)
            insert_at += 1
            self._stretch_index += 1
            self._type_menus.append(m)
            self._type_menu_buttons.append(btn)
        self._menubar_layout.activate()

    def _tear_down_body_and_type_menus(self) -> None:
        for btn in self._type_menu_buttons:
            self._menubar_layout.removeWidget(btn)
            self._stretch_index -= 1
            btn.setParent(None)
            btn.deleteLater()
        self._type_menu_buttons.clear()
        for m in self._type_menus:
            m.setParent(None)
            m.deleteLater()
        self._type_menus.clear()
        body = self.body
        if body is not None:
            self._body_container.layout().removeWidget(body)
            body.setParent(None)
            body.deleteLater()
