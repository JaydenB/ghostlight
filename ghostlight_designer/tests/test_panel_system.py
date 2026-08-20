from __future__ import annotations

import logging
from typing import List

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMenu, QSplitter

from ghostlight_designer.panel_system import (
    FloatingPanelWindow,
    Panel,
    PanelLayout,
    PanelLayoutRegistry,
    PanelRoot,
    PanelType,
    PanelTypeRegistry,
)
from ghostlight_designer.project import Project


def _stub_body(label: str):
    def _build(project, parent):
        return QLabel(label, parent)
    return _build


def _stub_menus(name: str):
    def _build(body, project) -> List[QMenu]:
        return [QMenu(name)]
    return _build


def _stub_type(id: str, name: str | None = None, menu_label: str | None = None) -> PanelType:
    return PanelType(
        id=id,
        display_name=name or id.title(),
        build_body=_stub_body(menu_label or id),
        build_menus=_stub_menus(menu_label or f"{id}-menu"),
    )


@pytest.fixture
def reg():
    r = PanelTypeRegistry()
    r.register(_stub_type("alpha", "Alpha", "Alpha"))
    r.register(_stub_type("beta", "Beta", "Beta"))
    r.register(_stub_type("optical_editor", "Optical Editor", "Editor"))
    return r


def _normalize(d: dict) -> dict:
    """Strip non-structural fields (sizes that are zero pre-show) for compare."""
    if d.get("kind") == "split":
        return {
            "kind": "split",
            "orient": d["orient"],
            "children": [_normalize(c) for c in d["children"]],
        }
    if d.get("kind") == "leaf":
        return {"kind": "leaf", "type_id": d["type_id"]}
    return d


def test_registry_register_and_lookup():
    r = PanelTypeRegistry()
    t = _stub_type("alpha")
    r.register(t)
    assert r.get("alpha") is t
    assert t in r.all()
    assert r.get("missing") is None


def test_panel_menu_has_layouts_split_and_type_list(qapp, reg):
    project = Project()
    panel = Panel("alpha", project, registry=reg)
    try:
        # Trigger the dynamic rebuild that normally happens on aboutToShow.
        panel._rebuild_type_actions()
        labels = [
            a.text().replace("&", "").rstrip("…")
            for a in panel.menu_panel.actions()
            if not a.isSeparator()
        ]
        # Split actions stay at the top, Layouts submenu directly below them
        # (no separator between), then a single rule before the type list.
        assert labels[:3] == ["Split Horizontally", "Split Vertically", "Layouts"]
        # First separator must come AFTER Layouts, not between Splits and Layouts.
        raw_actions = list(panel.menu_panel.actions())
        first_sep_idx = next(
            i for i, a in enumerate(raw_actions) if a.isSeparator()
        )
        # actions 0..2: split_h, split_v, layouts_submenu_action
        assert first_sep_idx == 3
        assert "Alpha" in labels
        assert "Beta" in labels
        assert "Optical Editor" in labels
        # Undock/Close are corner-bar buttons now, not in the menu.
        assert "Undock" not in labels
        assert "Close" not in labels
    finally:
        panel.deleteLater()


def test_panel_menus_are_not_visible_at_startup(qapp, reg):
    """Regression: setParent on a QMenu strips its Qt.Window flag, demoting
    it to a regular child widget that Qt renders inline. Menus must remain
    parentless popups so they only appear on click."""
    project = Project()
    panel = Panel("alpha", project, registry=reg)
    try:
        panel.show()
        qapp.processEvents()
        assert panel.menu_panel.isVisible() is False
        for m in panel._type_menus:
            assert m.isVisible() is False, f"menu {m.title()!r} is visible at startup"
    finally:
        panel.deleteLater()


def test_panel_corner_buttons_emit_signals(qapp, reg):
    project = Project()
    panel = Panel("alpha", project, registry=reg)
    try:
        undocked: list = []
        closed: list = []
        panel.undockRequested.connect(lambda p: undocked.append(p))
        panel.closeRequested.connect(lambda p: closed.append(p))
        panel.btn_undock.click()
        panel.btn_close.click()
        assert undocked == [panel]
        assert closed == [panel]
    finally:
        panel.deleteLater()


def test_panel_undock_button_disabled_when_floating(qapp, reg):
    project = Project()
    panel = Panel("alpha", project, registry=reg)
    try:
        assert panel.btn_undock.isEnabled() is True
        panel.set_floating(True)
        assert panel.btn_undock.isEnabled() is False
        panel.set_floating(False)
        assert panel.btn_undock.isEnabled() is True
    finally:
        panel.deleteLater()


def test_change_type_swaps_body_and_menus(qapp, reg):
    project = Project()
    panel = Panel("alpha", project, registry=reg)
    try:
        assert panel.type_id == "alpha"
        first_body = panel.body
        assert isinstance(first_body, QLabel) and first_body.text() == "Alpha"

        panel.change_type("beta")
        assert panel.type_id == "beta"
        new_body = panel.body
        assert isinstance(new_body, QLabel) and new_body.text() == "Beta"

        type_menu_titles = [b.text() for b in panel._type_menu_buttons]
        assert "Beta" in type_menu_titles
        assert "Alpha" not in type_menu_titles
        assert panel.btn_panel_menu.text() == "&Panel"
    finally:
        panel.deleteLater()


def test_split_horizontal_inserts_panel(qapp, reg):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        leaf = root.make_panel("alpha")
        root.set_root_panel(leaf)
        root.split_panel(leaf, Qt.Horizontal, "beta")
        assert isinstance(root.root_widget, QSplitter)
        assert root.root_widget.orientation() == Qt.Horizontal
        leaves = root.leaves()
        assert len(leaves) == 2
        assert {l.type_id for l in leaves} == {"alpha", "beta"}
    finally:
        root.deleteLater()


def test_split_then_split_creates_nested_tree(qapp, reg):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        leaf = root.make_panel("alpha")
        root.set_root_panel(leaf)
        right = root.split_panel(leaf, Qt.Horizontal, "beta")
        root.split_panel(right, Qt.Vertical, "alpha")
        leaves = root.leaves()
        assert len(leaves) == 3
        assert len({id(l) for l in leaves}) == 3
    finally:
        root.deleteLater()


def test_remove_panel_keeps_single_child_splitter(qapp, reg):
    """Single-child splitters are intentionally NOT collapsed — reparenting
    a QOpenGLWidget out of one crashes PySide6 when the GL context isn't
    fully initialized yet (most reproducible right after launch)."""
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        leaf = root.make_panel("alpha")
        root.set_root_panel(leaf)
        right = root.split_panel(leaf, Qt.Horizontal, "beta")
        assert isinstance(root.root_widget, QSplitter)
        root.remove_panel(right)
        # Splitter stays, with just the surviving leaf inside.
        assert isinstance(root.root_widget, QSplitter)
        assert root.root_widget.count() == 1
        assert root.root_widget.widget(0).type_id == "alpha"
        assert root.leaves() == [root.root_widget.widget(0)]
    finally:
        root.deleteLater()


def test_remove_last_panel_prunes_empty_splitter(qapp, reg):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        leaf = root.make_panel("alpha")
        root.set_root_panel(leaf)
        right = root.split_panel(leaf, Qt.Horizontal, "beta")
        root.remove_panel(right)   # splitter still has [alpha]
        root.remove_panel(leaf)    # now splitter is empty
        assert root.root_widget is None
    finally:
        root.deleteLater()


def test_layout_round_trip(qapp, reg):
    project = Project()
    root1 = PanelRoot(project, registry=reg)
    try:
        leaf = root1.make_panel("alpha")
        root1.set_root_panel(leaf)
        right = root1.split_panel(leaf, Qt.Horizontal, "beta")
        root1.split_panel(right, Qt.Vertical, "alpha")
        d = root1.to_dict()

        root2 = PanelRoot(project, registry=reg)
        try:
            root2.from_dict(d)
            assert _normalize(root2.to_dict()["docked"]) == _normalize(d["docked"])
        finally:
            root2.deleteLater()
    finally:
        root1.deleteLater()


def test_unknown_panel_type_falls_back(qapp, reg, caplog):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        d = {"version": 1, "docked": {"kind": "leaf", "type_id": "nope", "state": {}}}
        with caplog.at_level(logging.WARNING, logger="ghostlight_designer.panel_system"):
            root.from_dict(d)
        leaves = root.leaves()
        assert len(leaves) == 1
        assert leaves[0].type_id == "optical_editor"
        assert any("nope" in r.message for r in caplog.records)
    finally:
        root.deleteLater()


def test_undock_creates_floating_window(qapp, reg):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        leaf = root.make_panel("alpha")
        root.set_root_panel(leaf)
        right = root.split_panel(leaf, Qt.Horizontal, "beta")

        # Simulate the undock handler: remove the docked panel, then build a
        # fresh same-type panel inside a floating window.
        root.remove_panel(right)
        right.deleteLater()
        win = FloatingPanelWindow(project, registry=reg)
        floating_panel = win.panel_root.make_panel("beta")
        win.set_root_panel(floating_panel)
        try:
            assert floating_panel.window() is win
            assert floating_panel.is_floating() is True
            # Splitter stays (with one child) — we no longer collapse singletons.
            assert isinstance(root.root_widget, QSplitter)
            assert root.root_widget.count() == 1
            assert root.root_widget.widget(0).type_id == "alpha"
        finally:
            win.deleteLater()
    finally:
        root.deleteLater()


def test_floating_close_emits_window_closed(qapp, reg):
    project = Project()
    win = FloatingPanelWindow(project, registry=reg)
    panel = win.panel_root.make_panel("alpha")
    win.set_root_panel(panel)
    win.show()

    closed: list = []
    win.windowClosed.connect(lambda: closed.append(True))
    win.close()
    assert closed == [True]
    assert panel.is_floating() is False


def test_floating_split_creates_sibling_panel(qapp, reg):
    project = Project()
    win = FloatingPanelWindow(project, registry=reg)
    panel = win.panel_root.make_panel("alpha")
    win.set_root_panel(panel)
    win.show()
    try:
        new_panel = win.panel_root.split_panel(panel, Qt.Horizontal, "beta")
        leaves = win.panel_root.leaves()
        assert len(leaves) == 2
        assert {p.type_id for p in leaves} == {"alpha", "beta"}
        assert new_panel.is_floating() is True
        assert panel.is_floating() is True
    finally:
        win.close()


def test_floating_close_last_leaf_closes_window(qapp, reg):
    project = Project()
    win = FloatingPanelWindow(project, registry=reg)
    panel = win.panel_root.make_panel("alpha")
    win.set_root_panel(panel)
    win.show()

    closed: list = []
    win.windowClosed.connect(lambda: closed.append(True))
    # Removing the sole leaf empties the tree → window auto-closes.
    win.panel_root.remove_panel(panel)
    qapp.processEvents()
    assert closed == [True]


def test_panel_added_fires_exactly_once_per_insertion(qapp, reg):
    project = Project()
    root = PanelRoot(project, registry=reg)
    try:
        seen: list = []
        root.panelAdded.connect(lambda p: seen.append(p))
        root.build_default_layout({
            "kind": "split",
            "orient": "v",
            "children": [
                {"kind": "leaf", "type_id": "alpha"},
                {"kind": "leaf", "type_id": "beta"},
            ],
        })
        assert len(seen) == 2
        new_panel = root.split_panel(seen[0], Qt.Horizontal, "alpha")
        assert seen.count(new_panel) == 1
        assert len(seen) == 3
    finally:
        root.deleteLater()


def test_main_window_close_panel_does_not_double_fire(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    try:
        leaves = w._panel_root.leaves()
        assert len(leaves) == 4
        editor = next(p for p in leaves if p.type_id == "optical_editor")
        editor.closeRequested.emit(editor)
        qapp.processEvents()
        # Survives without raising. The other panels stay; the inner
        # horizontal splitter becomes a single-child splitter (kept per
        # the no-collapse rationale).
        surviving = {p.type_id for p in w._panel_root.leaves()}
        assert surviving == {"system_setup", "viewport", "sourceflare"}
    finally:
        w.deleteLater()


def test_main_window_split_via_menu_action_end_to_end(qapp, isolated_settings):
    """Regression: action_split_h/v triggered the panel's signal lambda which
    used int(Qt.Horizontal) — PySide6's Qt.Orientation enum doesn't coerce
    through int(), so the lambda raised TypeError on every click."""
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        editor = next(p for p in w._panel_root.leaves() if p.type_id == "optical_editor")
        editor.action_split_h.trigger()
        qapp.processEvents()
        assert len(w._panel_root.leaves()) == 5
        # Trigger the vertical split path too.
        editor.action_split_v.trigger()
        qapp.processEvents()
        assert len(w._panel_root.leaves()) == 6
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_close_buttons_visible_with_default_panels(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    try:
        leaves = w._panel_root.leaves()
        assert len(leaves) > 1
        for p in leaves:
            assert p.btn_close.isHidden() is False
    finally:
        w.deleteLater()


def test_close_button_hides_when_only_one_panel_left(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    try:
        # Close panels until only the viewport remains.
        for type_id in ("optical_editor", "system_setup", "sourceflare"):
            p = next(p for p in w._panel_root.leaves() if p.type_id == type_id)
            p.closeRequested.emit(p)
            qapp.processEvents()
        leaves = w._panel_root.leaves()
        assert len(leaves) == 1
        assert leaves[0].btn_close.isHidden() is True
    finally:
        w.deleteLater()


def test_split_after_lone_restores_close_button(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    from PySide6.QtCore import Qt
    w = MainWindow(settings=isolated_settings)
    try:
        for type_id in ("optical_editor", "system_setup", "sourceflare"):
            p = next(p for p in w._panel_root.leaves() if p.type_id == type_id)
            p.closeRequested.emit(p)
            qapp.processEvents()
        lone = w._panel_root.leaves()[0]
        assert lone.btn_close.isHidden() is True
        w._panel_root.split_panel(lone, Qt.Horizontal, "optical_editor")
        qapp.processEvents()
        for p in w._panel_root.leaves():
            assert p.btn_close.isHidden() is False
    finally:
        w.deleteLater()


def test_undock_lone_docked_panel_duplicates(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        for type_id in ("optical_editor", "system_setup", "sourceflare"):
            p = next(p for p in w._panel_root.leaves() if p.type_id == type_id)
            p.closeRequested.emit(p)
            qapp.processEvents()
        lone = w._panel_root.leaves()[0]
        assert lone.type_id == "viewport"
        # Undock the lone panel — should DUPLICATE, not move.
        lone.undockRequested.emit(lone)
        qapp.processEvents()
        # Original still docked, same Python object.
        leaves_after = w._panel_root.leaves()
        assert len(leaves_after) == 1
        assert leaves_after[0] is lone
        # Fresh copy in the floating window — same type, different object.
        assert len(w._floating) == 1
        floating_panel = w._floating[0].panel
        assert floating_panel.type_id == "viewport"
        assert floating_panel is not lone
        # Floating panel always has its close button visible.
        assert floating_panel.btn_close.isHidden() is False
    finally:
        for win in list(w._floating):
            win.close()
        w.deleteLater()
        qapp.processEvents()


def test_panel_layouts_menu_lists_registered_layouts(qapp, reg):
    layouts = PanelLayoutRegistry()
    layouts.register(PanelLayout(id="one", display_name="Layout One",
                                 spec={"kind": "leaf", "type_id": "alpha"}))
    layouts.register(PanelLayout(id="two", display_name="Layout Two",
                                 spec={"kind": "leaf", "type_id": "beta"}))
    project = Project()
    panel = Panel("alpha", project, registry=reg, layout_registry=layouts)
    try:
        panel._rebuild_layout_actions()
        labels = [a.text() for a in panel.menu_layouts.actions()]
        assert labels == ["Layout One", "Layout Two"]
    finally:
        panel.deleteLater()


def test_panel_layouts_menu_emits_layout_requested(qapp, reg):
    layouts = PanelLayoutRegistry()
    layouts.register(PanelLayout(id="one", display_name="Layout One",
                                 spec={"kind": "leaf", "type_id": "alpha"}))
    project = Project()
    panel = Panel("alpha", project, registry=reg, layout_registry=layouts)
    try:
        emitted: list = []
        panel.layoutRequested.connect(lambda lid: emitted.append(lid))
        panel._rebuild_layout_actions()
        panel.menu_layouts.actions()[0].trigger()
        assert emitted == ["one"]
    finally:
        panel.deleteLater()


def test_panel_layouts_menu_handles_empty_registry(qapp, reg):
    project = Project()
    panel = Panel("alpha", project, registry=reg,
                  layout_registry=PanelLayoutRegistry())
    try:
        panel._rebuild_layout_actions()
        acts = panel.menu_layouts.actions()
        assert len(acts) == 1
        assert acts[0].isEnabled() is False
        assert "no layouts" in acts[0].text().lower()
    finally:
        panel.deleteLater()


def test_main_window_layout_request_replaces_tree(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    try:
        # Sanity: default has 4 panels.
        assert len(w._panel_root.leaves()) == 4
        any_leaf = w._panel_root.leaves()[0]
        any_leaf.layoutRequested.emit("viewport_focus")
        qapp.processEvents()
        leaves_after = w._panel_root.leaves()
        assert len(leaves_after) == 1
        assert leaves_after[0].type_id == "viewport"
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_main_window_layout_request_closes_floating(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        viewport_panel = next(p for p in w._panel_root.leaves() if p.type_id == "viewport")
        viewport_panel.undockRequested.emit(viewport_panel)
        qapp.processEvents()
        assert len(w._floating) == 1
        # Switching layouts must clear floating windows too.
        w._panel_root.leaves()[0].layoutRequested.emit("default")
        qapp.processEvents()
        assert len(w._floating) == 0
    finally:
        for win in list(w._floating):
            win.close()
        w.deleteLater()
        qapp.processEvents()


def test_main_window_layout_request_with_unknown_id_is_noop(qapp, isolated_settings, caplog):
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    try:
        before = len(w._panel_root.leaves())
        with caplog.at_level(logging.WARNING, logger="ghostlight_designer.main_window"):
            w._panel_root.leaves()[0].layoutRequested.emit("does_not_exist")
        qapp.processEvents()
        assert len(w._panel_root.leaves()) == before
        assert any("does_not_exist" in r.message for r in caplog.records)
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_main_window_close_persists_workspace(qapp, isolated_settings):
    """Closing the window stores the docked tree and floating windows so the
    next launch restores them."""
    from ghostlight_designer.main_window import MainWindow
    from PySide6.QtCore import Qt as _Qt
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        # Reshape: start from a single panel, then split and undock one.
        w._panel_root.leaves()[0].layoutRequested.emit("editor_focus")
        qapp.processEvents()
        assert len(w._panel_root.leaves()) == 1
        only = w._panel_root.leaves()[0]
        w._panel_root.split_panel(only, _Qt.Horizontal, "viewport")
        qapp.processEvents()
        viewport_leaf = next(p for p in w._panel_root.leaves() if p.type_id == "viewport")
        viewport_leaf.undockRequested.emit(viewport_leaf)
        qapp.processEvents()
        assert len(w._floating) == 1
        w.close()
        qapp.processEvents()
    finally:
        w.deleteLater()
        qapp.processEvents()
    saved = isolated_settings.workspace_layout()
    assert saved is not None
    assert "panel_root" in saved
    assert "floating" in saved
    assert len(saved["floating"]) == 1
    floating_tree = saved["floating"][0]["tree"]
    assert floating_tree["docked"]["type_id"] == "viewport"


def test_main_window_restores_workspace_on_launch(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    # Seed a tiny non-default workspace in settings.
    isolated_settings.set_workspace_layout({
        "version": 1,
        "panel_root": {
            "version": 1,
            "docked": {"kind": "leaf", "type_id": "viewport", "state": {}},
        },
        "floating": [],
    })
    w = MainWindow(settings=isolated_settings)
    try:
        leaves = w._panel_root.leaves()
        assert len(leaves) == 1
        assert leaves[0].type_id == "viewport"
        assert len(w._floating) == 0
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_main_window_restore_falls_back_on_corrupt_workspace(qapp, isolated_settings):
    from ghostlight_designer.main_window import MainWindow
    isolated_settings.set_workspace_layout({"junk": True})
    w = MainWindow(settings=isolated_settings)
    try:
        leaves = w._panel_root.leaves()
        # Falls back to the default 4-panel layout.
        assert {p.type_id for p in leaves} == {
            "optical_editor", "sourceflare", "system_setup", "viewport",
        }
    finally:
        w.deleteLater()
        qapp.processEvents()


def test_main_window_split_duplicates_panel_type(qapp, isolated_settings):
    """The Panel menu's Split actions should mint a sibling of the SAME type
    as the source panel, in both the docked tree and a floating window."""
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        # Docked: split a viewport → expect another viewport, not an ODE.
        viewport_panel = next(p for p in w._panel_root.leaves() if p.type_id == "viewport")
        viewport_panel.splitRequested.emit(Qt.Horizontal.value, viewport_panel)
        qapp.processEvents()
        viewports = [p for p in w._panel_root.leaves() if p.type_id == "viewport"]
        assert len(viewports) == 2

        # Floating: undock a viewport, then split it inside the floating window.
        viewports[0].undockRequested.emit(viewports[0])
        qapp.processEvents()
        floating_win = w._floating[0]
        floating_leaf = floating_win.panel
        floating_leaf.splitRequested.emit(Qt.Horizontal.value, floating_leaf)
        qapp.processEvents()
        floating_types = [p.type_id for p in floating_win.panel_root.leaves()]
        assert floating_types == ["viewport", "viewport"]
    finally:
        for win in list(w._floating):
            win.close()
        w.deleteLater()
        qapp.processEvents()


def test_main_window_undock_panel_end_to_end(qapp, isolated_settings):
    """Reproduces the user-reported undock crash path: click → emit → defer → undock."""
    from ghostlight_designer.main_window import MainWindow
    w = MainWindow(settings=isolated_settings)
    w.show()
    qapp.processEvents()
    try:
        viewport_panel = next(p for p in w._panel_root.leaves() if p.type_id == "viewport")
        viewport_panel.undockRequested.emit(viewport_panel)
        qapp.processEvents()
        assert len(w._floating) == 1
        new_panel = w._floating[0].panel
        assert new_panel.type_id == "viewport"
        assert new_panel.is_floating() is True
        assert new_panel.btn_undock.isEnabled() is False
        # The viewport is gone; what remains is the optical editor, the
        # system setup, and the source-flare panel.
        docked_ids = {p.type_id for p in w._panel_root.leaves()}
        assert docked_ids == {"optical_editor", "system_setup", "sourceflare"}
        # Close the floating window cleanly.
        w._floating[0].close()
        qapp.processEvents()
        assert len(w._floating) == 0
    finally:
        for win in list(w._floating):
            win.close()
        w.deleteLater()
        qapp.processEvents()


