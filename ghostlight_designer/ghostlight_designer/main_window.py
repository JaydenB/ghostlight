"""Main application window.

Holds a single ``Project`` and a single ``AppSettings``. The central widget
is a :class:`PanelRoot` that owns a tree of user-redesignable panels. Panel
bodies receive ``Project`` by constructor injection and connect to its
``systemReplaced`` / ``systemModified`` signals directly.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Dict, List, Optional

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
)

from .optical_editor import (
    OPTICAL_EDITOR_TYPE_ID,
    register_optical_editor_panel_type,
)
from .panel_system import (
    FloatingPanelWindow,
    Panel,
    PanelLayout,
    PanelRoot,
    layout_registry,
    ordered_menu_entries,
    registry as panel_type_registry,
)
from .evaluation_panels.field_diagrams import (
    FIELD_DIAGRAMS_TYPE_ID,
    register_field_diagrams_panel_type,
)
from .evaluation_panels.seidel import (
    SEIDEL_TYPE_ID,
    register_seidel_panel_type,
)
from .evaluation_panels.spot_diagram import (
    SPOT_DIAGRAM_TYPE_ID,
    register_spot_diagram_panel_type,
)
from .ghost_explorer_panel import register_ghost_explorer_panel_type
from .optimization_panel import (
    OPTIMIZATION_TYPE_ID,
    register_optimization_panel_type,
)
from .project import Project
from .psf_panel import PSF_TYPE_ID, register_psf_panel_type
from .settings import AppSettings
from .sourceflare_panel import SOURCEFLARE_TYPE_ID, register_sourceflare_panel_type
from .system_setup import SYSTEM_SETUP_TYPE_ID, register_system_setup_panel_type
from .textures_panel import TEXTURES_TYPE_ID, register_textures_panel_type
from .viewport_panel import VIEWPORT_TYPE_ID, register_viewport_panel_type

_log = logging.getLogger("ghostlight_designer.main_window")

APP_TITLE = "Ghostlight Designer"
LENS_FILTER = "Lens files (*.lens);;All files (*)"

DEFAULT_LAYOUT_ID = "default"


def _default_layout_spec() -> dict:
    return {
        "kind": "split",
        "orient": "v",
        "sizes": [4_000, 6_000],
        "children": [
            {
                "kind": "split",
                "orient": "h",
                "sizes": [5_000, 3_000, 2_000],
                "children": [
                    {"kind": "leaf", "type_id": OPTICAL_EDITOR_TYPE_ID},
                    {"kind": "leaf", "type_id": SOURCEFLARE_TYPE_ID},
                    {"kind": "leaf", "type_id": SYSTEM_SETUP_TYPE_ID},
                ],
            },
            {"kind": "leaf", "type_id": VIEWPORT_TYPE_ID},
        ],
    }


def _builtin_layouts() -> List[PanelLayout]:
    return [
        PanelLayout(
            id=DEFAULT_LAYOUT_ID,
            display_name="Default",
            spec=_default_layout_spec(),
        ),
        PanelLayout(
            id="editor_viewport",
            display_name="Editor + Viewport",
            spec={
                "kind": "split",
                "orient": "h",
                "sizes": [5_000, 5_000],
                "children": [
                    {"kind": "leaf", "type_id": OPTICAL_EDITOR_TYPE_ID},
                    {"kind": "leaf", "type_id": VIEWPORT_TYPE_ID},
                ],
            },
        ),
        PanelLayout(
            id="editor_focus",
            display_name="Optical Editor Only",
            spec={"kind": "leaf", "type_id": OPTICAL_EDITOR_TYPE_ID},
        ),
        PanelLayout(
            id="viewport_focus",
            display_name="Viewport Only",
            spec={"kind": "leaf", "type_id": VIEWPORT_TYPE_ID},
        ),
        PanelLayout(
            id="renderers",
            display_name="Renderers Comparison",
            spec={
                "kind": "split",
                "orient": "h",
                "sizes": [1, 1],
                "children": [
                    {"kind": "leaf", "type_id": SOURCEFLARE_TYPE_ID},
                    {"kind": "leaf", "type_id": PSF_TYPE_ID},
                ],
            },
        ),
        PanelLayout(
            id="full_workstation",
            display_name="Full Workstation",
            spec={
                "kind": "split",
                "orient": "v",
                "sizes": [6_000, 4_000],
                "children": [
                    {
                        "kind": "split",
                        "orient": "h",
                        "sizes": [4_000, 3_000, 3_000],
                        "children": [
                            {"kind": "leaf", "type_id": OPTICAL_EDITOR_TYPE_ID},
                            {"kind": "leaf", "type_id": SYSTEM_SETUP_TYPE_ID},
                            {"kind": "leaf", "type_id": VIEWPORT_TYPE_ID},
                        ],
                    },
                    {
                        "kind": "split",
                        "orient": "h",
                        "sizes": [1, 1],
                        "children": [
                            {"kind": "leaf", "type_id": SOURCEFLARE_TYPE_ID},
                            {"kind": "leaf", "type_id": PSF_TYPE_ID},
                        ],
                    },
                ],
            },
        ),
    ]


def _register_builtin_layouts() -> None:
    # Layouts are app-level data; register once. Re-registering the same id
    # is a no-op apart from re-emitting layoutsChanged.
    for lay in _builtin_layouts():
        if layout_registry.get(lay.id) is None:
            layout_registry.register(lay)


class MainWindow(QMainWindow):
    def __init__(
        self,
        project: Optional[Project] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        super().__init__()
        self.project = project if project is not None else Project(self)
        self.settings = settings if settings is not None else AppSettings(parent=self)

        self.resize(1280, 800)
        self._restore_window_state()

        self._floating: List[FloatingPanelWindow] = []
        self._build_panel_root()
        self._build_menus()

        self.project.dirtyChanged.connect(self._on_dirty_changed)
        self.project.pathChanged.connect(self._on_path_changed)
        self.project.systemReplaced.connect(self._on_system_replaced)
        self.project.canUndoChanged.connect(self._refresh_edit_actions)
        self.project.canRedoChanged.connect(self._refresh_edit_actions)
        self.settings.recentFilesChanged.connect(self._rebuild_recent_menu)

        self._rebuild_recent_menu(self.settings.recent_files())
        self._refresh_title()
        self._refresh_actions()
        self._refresh_edit_actions()

    def _build_panel_root(self) -> None:
        register_optical_editor_panel_type()
        register_viewport_panel_type(self.settings)
        register_system_setup_panel_type()
        register_sourceflare_panel_type(self.settings)
        register_ghost_explorer_panel_type(self.settings)
        register_psf_panel_type(self.settings)
        register_textures_panel_type(self.settings)
        register_spot_diagram_panel_type(self.settings)
        register_field_diagrams_panel_type(self.settings)
        register_seidel_panel_type(self.settings)
        register_optimization_panel_type()
        _register_builtin_layouts()
        self._panel_root = PanelRoot(self.project, self)
        self._panel_root.panelAdded.connect(self._wire_panel_signals)
        self._panel_root.panelAdded.connect(self._refresh_close_buttons)
        self._panel_root.panelRemoved.connect(self._refresh_close_buttons)
        self.setCentralWidget(self._panel_root)
        self._restore_workspace_or_default()

    def _refresh_close_buttons(self, *_args) -> None:
        leaves = self._panel_root.leaves()
        show = len(leaves) > 1
        for p in leaves:
            p.set_close_visible(show)

    def _wire_panel_signals(self, panel: Panel) -> None:
        panel.splitRequested.connect(self._on_panel_split_requested)
        panel.undockRequested.connect(self._on_panel_undock_requested)
        panel.closeRequested.connect(self._on_panel_close_requested)
        panel.layoutRequested.connect(self._on_panel_layout_requested)

    def _on_panel_layout_requested(self, layout_id: str) -> None:
        lay = layout_registry.get(layout_id)
        if lay is None:
            _log.warning("Unknown layout id %r requested", layout_id)
            return
        self.apply_layout(lay.spec)

    def apply_layout(self, spec: dict) -> None:
        """Replace the docked tree (and close all floating windows) with ``spec``.

        The user picked a fresh layout — any prior floating windows are part
        of the previous workspace, not this one, so close them too.
        """
        for win in list(self._floating):
            win.close()
        self._floating.clear()
        self._panel_root.build_default_layout(spec)

    def _on_panel_split_requested(self, orient_int: int, panel: Panel) -> None:
        orient = Qt.Horizontal if orient_int == Qt.Horizontal.value else Qt.Vertical
        root = self._panel_root_for(panel)
        if root is None:
            return
        root.split_panel(panel, orient, panel.type_id)

    def _on_panel_undock_requested(self, panel: Panel) -> None:
        if panel.is_floating():
            return
        QTimer.singleShot(0, lambda: self._do_undock(panel))

    def _do_undock(self, panel: Panel) -> None:
        if panel.is_floating():
            return
        if panel.parentWidget() is None and panel is not self._panel_root.root_widget:
            return
        size = panel.size()
        type_id = panel.type_id

        # If this is the only docked panel, undocking it would leave the main
        # window empty. In that case, DUPLICATE it into a floating window
        # instead — the original stays put.
        is_lone = len(self._panel_root.leaves()) == 1
        if not is_lone:
            # Re-parenting a QOpenGLWidget across top-level windows crashes in
            # PySide6. Destroy the old panel and build a fresh panel of the
            # same type inside the floating window.
            self._panel_root.remove_panel(panel)
            panel.deleteLater()
            if self._panel_root.root_widget is None:
                seed = self._panel_root.make_panel(OPTICAL_EDITOR_TYPE_ID)
                self._panel_root.set_root_panel(seed)

        win = self._make_floating_window()
        new_panel = win.panel_root.make_panel(type_id)
        win.set_root_panel(new_panel)
        target_w = size.width() if size.width() > 100 else int(self.width() * 0.4)
        target_h = size.height() if size.height() > 100 else int(self.height() * 0.6)
        win.resize(target_w, target_h)
        win.show()
        self._floating.append(win)

    def _on_panel_close_requested(self, panel: Panel) -> None:
        QTimer.singleShot(0, lambda: self._do_close(panel))

    def _do_close(self, panel: Panel) -> None:
        if panel.is_floating():
            win = self._floating_window_of(panel)
            if win is None:
                w = panel.window()
                if isinstance(w, FloatingPanelWindow):
                    w.close()
                return
            win.panel_root.remove_panel(panel)
            # FloatingPanelWindow closes itself when its tree empties.
            return
        if panel.parentWidget() is None and panel is not self._panel_root.root_widget:
            return
        self._panel_root.remove_panel(panel)
        if self._panel_root.root_widget is None:
            seed = self._panel_root.make_panel(OPTICAL_EDITOR_TYPE_ID)
            self._panel_root.set_root_panel(seed)

    def _on_floating_panel_destroyed(self) -> None:
        win = self.sender()
        if win is None:
            return
        self._floating = [w for w in self._floating if w is not win]
        win.deleteLater()

    def _floating_window_of(self, panel: Panel) -> Optional[FloatingPanelWindow]:
        w = panel.window()
        return w if isinstance(w, FloatingPanelWindow) else None

    def _panel_root_for(self, panel: Panel) -> Optional[PanelRoot]:
        win = self._floating_window_of(panel)
        if win is not None:
            return win.panel_root
        if panel.parentWidget() is None and panel is not self._panel_root.root_widget:
            return None
        return self._panel_root

    def _make_floating_window(self) -> FloatingPanelWindow:
        win = FloatingPanelWindow(self.project)
        win.panelAdded.connect(self._wire_panel_signals)
        win.windowClosed.connect(self._on_floating_panel_destroyed)
        return win

    def _build_menus(self) -> None:
        bar = self.menuBar()
        self.menu_file = bar.addMenu("&File")

        self.action_new = QAction("&New", self)
        self.action_new.setShortcut(QKeySequence.New)
        self.action_new.triggered.connect(self._on_new)

        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.triggered.connect(self._on_open)

        self.menu_recent = QMenu("Open &Recent", self)

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self._on_save)

        self.action_save_as = QAction("Save &As…", self)
        self.action_save_as.setShortcut(QKeySequence.SaveAs)
        self.action_save_as.triggered.connect(self._on_save_as)

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.Quit)
        self.action_exit.triggered.connect(self.close)

        self.menu_file.addAction(self.action_new)
        self.menu_file.addAction(self.action_open)
        self.menu_file.addMenu(self.menu_recent)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_save)
        self.menu_file.addAction(self.action_save_as)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)

        self.menu_edit = bar.addMenu("&Edit")

        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.triggered.connect(self.project.undo)

        self.action_redo = QAction("&Redo", self)
        self.action_redo.setShortcuts(
            [QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")]
        )
        self.action_redo.triggered.connect(self.project.redo)

        self.menu_edit.addAction(self.action_undo)
        self.menu_edit.addAction(self.action_redo)

        self.menu_view = bar.addMenu("&View")

        self.action_auto_update = QAction(
            "&Auto-Update Panels on Lens Changes", self
        )
        self.action_auto_update.setCheckable(True)
        self.action_auto_update.setChecked(self.settings.auto_update_enabled())
        self.action_auto_update.setToolTip(
            "Master switch. Uncheck to stop every auto-updating panel from "
            "re-rendering when the lens or system setup changes. Each panel's "
            "manual Refresh / Re-render still works."
        )
        self.action_auto_update.toggled.connect(self.settings.set_auto_update_enabled)
        # If another widget flips the setting (a preferences dialog, say),
        # keep the menu's check state in sync.
        self.settings.autoUpdateChanged.connect(self.action_auto_update.setChecked)
        self.menu_view.addAction(self.action_auto_update)

        # Designer-wide display (view) transform. Populated on show from the
        # active OCIO config so it always reflects the current config's
        # displays/views and the persisted selection.
        self.menu_view.addSeparator()
        self.menu_view_transform = QMenu("Display &Transform", self)
        self.menu_view_transform.setToolTip(
            "ACES 2.0 output transform applied to every render panel. Preview "
            "matches a Nuke session on an ACES 2.0 OCIO config. (The PSF panel "
            "keeps a separate diagnostic tone map.)"
        )
        self._view_transform_group: Optional[QActionGroup] = None
        self.menu_view_transform.aboutToShow.connect(
            self._rebuild_view_transform_menu
        )
        self.menu_view.addMenu(self.menu_view_transform)

        self.menu_windows = bar.addMenu("&Windows")
        self.menu_windows_layouts = QMenu("&Layouts", self.menu_windows)
        self.menu_windows.addMenu(self.menu_windows_layouts)
        self.menu_windows_separator = self.menu_windows.addSeparator()
        self._windows_layout_actions: List[QAction] = []
        self._windows_panel_actions: List[QAction] = []
        self._windows_category_menus: Dict[str, QMenu] = {}
        self.menu_windows_layouts.aboutToShow.connect(
            self._rebuild_windows_layout_actions
        )
        self.menu_windows.aboutToShow.connect(self._rebuild_windows_panel_actions)

    def _rebuild_view_transform_menu(self) -> None:
        """Rebuild the Display Transform submenu from the active OCIO config."""
        from . import viewtransform as vt

        menu = self.menu_view_transform
        menu.clear()

        config_key = self.settings.view_ocio_config()
        cur_display, cur_view = self.settings.view_display_view()
        group = QActionGroup(menu)
        group.setExclusive(True)
        group.triggered.connect(self._on_view_transform_selected)
        self._view_transform_group = group

        try:
            views_by_display = vt.available_views(config_key)
            if not cur_display or not cur_view:
                cur_display, cur_view = vt.resolve_default_display_view(config_key)
        except Exception as exc:  # bad $OCIO / unreadable file
            warn = menu.addAction(f"⚠ View transform unavailable: {exc}")
            warn.setEnabled(False)
            views_by_display = []

        for display, views in views_by_display:
            sub = menu.addMenu(display)
            for view in views:
                act = sub.addAction(view)
                act.setCheckable(True)
                act.setData([display, view])
                act.setActionGroup(group)
                act.setChecked(display == cur_display and view == cur_view)

        menu.addSeparator()
        builtin = menu.addAction("Use Built-in ACES 2.0 Config")
        builtin.setCheckable(True)
        builtin.setChecked(config_key == "")
        builtin.triggered.connect(self._use_builtin_ocio_config)
        env = menu.addAction("Use $OCIO Environment Config")
        env.setCheckable(True)
        env.setChecked(config_key == "$OCIO")
        env.triggered.connect(self._use_env_ocio_config)
        menu.addAction("Choose OCIO Config File…").triggered.connect(
            self._choose_ocio_config
        )

    def _on_view_transform_selected(self, action: QAction) -> None:
        data = action.data()
        if isinstance(data, (list, tuple)) and len(data) == 2:
            self.settings.set_view_display_view(str(data[0]), str(data[1]))

    def _use_builtin_ocio_config(self) -> None:
        self.settings.set_view_ocio_config("")

    def _use_env_ocio_config(self) -> None:
        self.settings.set_view_ocio_config("$OCIO")

    def _choose_ocio_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose OCIO Config",
            self.settings.last_open_dir() or "",
            "OCIO Config (*.ocio);;All Files (*)",
        )
        if path:
            self.settings.set_view_ocio_config(path)

    def _rebuild_windows_layout_actions(self) -> None:
        for a in self._windows_layout_actions:
            self.menu_windows_layouts.removeAction(a)
        self._windows_layout_actions.clear()
        layouts = layout_registry.all()
        if not layouts:
            placeholder = QAction("(no layouts)", self.menu_windows_layouts)
            placeholder.setEnabled(False)
            self.menu_windows_layouts.addAction(placeholder)
            self._windows_layout_actions.append(placeholder)
            return
        for lay in layouts:
            act = QAction(lay.display_name, self.menu_windows_layouts)
            act.triggered.connect(
                lambda _checked=False, lid=lay.id: self._on_panel_layout_requested(lid)
            )
            self.menu_windows_layouts.addAction(act)
            self._windows_layout_actions.append(act)

    def _rebuild_windows_panel_actions(self) -> None:
        # Tear down previous dynamic entries (everything past the Layouts +
        # separator pair stays put across rebuilds).
        for a in self._windows_panel_actions:
            self.menu_windows.removeAction(a)
        self._windows_panel_actions.clear()
        for menu in self._windows_category_menus.values():
            menu.clear()

        # Curated order shared with Panel._rebuild_type_actions — see
        # panel_system.menu_order.
        for entry in ordered_menu_entries(panel_type_registry):
            if entry[0] == "group":
                _, cat_name, types = entry
                menu = self._windows_category_menus.get(cat_name)
                if menu is None:
                    menu = QMenu(cat_name, self.menu_windows)
                    self._windows_category_menus[cat_name] = menu
                else:
                    menu.setTitle(cat_name)
                sub_action = self.menu_windows.addMenu(menu)
                self._windows_panel_actions.append(sub_action)
                for t in types:
                    act = QAction(t.display_name, menu)
                    act.triggered.connect(
                        lambda _checked=False, tid=t.id: self._spawn_floating_panel(tid)
                    )
                    menu.addAction(act)
            else:
                t = entry[1]
                act = QAction(t.display_name, self.menu_windows)
                act.triggered.connect(
                    lambda _checked=False, tid=t.id: self._spawn_floating_panel(tid)
                )
                self.menu_windows.addAction(act)
                self._windows_panel_actions.append(act)

    def _spawn_floating_panel(self, type_id: str) -> None:
        win = self._make_floating_window()
        new_panel = win.panel_root.make_panel(type_id)
        win.set_root_panel(new_panel)
        win.resize(int(self.width() * 0.4), int(self.height() * 0.6))
        win.show()
        self._floating.append(win)

    def _rebuild_recent_menu(self, files=None) -> None:
        if files is None:
            files = self.settings.recent_files()
        self.menu_recent.clear()
        if not files:
            self.menu_recent.setEnabled(False)
            return
        self.menu_recent.setEnabled(True)
        for path in files:
            label = os.path.basename(path) or path
            act = QAction(label, self.menu_recent)
            act.setToolTip(path)
            act.triggered.connect(lambda _checked=False, p=path: self._open_path(p))
            self.menu_recent.addAction(act)
        self.menu_recent.addSeparator()
        clear_act = QAction("Clear Recent", self.menu_recent)
        clear_act.triggered.connect(self.settings.clear_recent_files)
        self.menu_recent.addAction(clear_act)

    def _refresh_title(self) -> None:
        self.setWindowTitle(f"{self.project.display_name} - {APP_TITLE}")

    def _refresh_actions(self) -> None:
        has_path = self.project.path is not None
        self.action_save.setEnabled(has_path and self.project.is_dirty)
        self.action_save_as.setEnabled(True)

    def _refresh_edit_actions(self, *_args) -> None:
        can_u = self.project.can_undo
        can_r = self.project.can_redo
        self.action_undo.setEnabled(can_u)
        self.action_redo.setEnabled(can_r)
        self.action_undo.setText(
            f"&Undo {self.project.undo_label}" if can_u else "&Undo"
        )
        self.action_redo.setText(
            f"&Redo {self.project.redo_label}" if can_r else "&Redo"
        )

    def _on_dirty_changed(self, _dirty: bool) -> None:
        self._refresh_title()
        self._refresh_actions()

    def _on_path_changed(self, _path) -> None:
        self._refresh_title()
        self._refresh_actions()

    def _on_system_replaced(self, _system) -> None:
        self._refresh_title()
        self._refresh_actions()

    def _on_new(self) -> None:
        if not self._confirm_discard_changes():
            return
        self.project.new()

    def _on_open(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Lens", self.settings.last_open_dir(), LENS_FILTER,
        )
        if not path:
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        try:
            self.project.load(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Open failed", f"Could not open {path}:\n{exc}"
            )
            self.settings.remove_recent_file(path)
            return
        self.settings.add_recent_file(path)
        self.settings.set_last_open_dir(os.path.dirname(path) or "")

    def _on_save(self) -> None:
        if self.project.path is None:
            self._on_save_as()
            return
        try:
            self.project.save()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _on_save_as(self) -> None:
        suggested = self.project.path or os.path.join(
            self.settings.last_open_dir() or "", "Untitled.lens"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Lens As", suggested, LENS_FILTER,
        )
        if not path:
            return
        try:
            self.project.save_as(path)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.settings.add_recent_file(path)
        self.settings.set_last_open_dir(os.path.dirname(path) or "")

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.settings.set_window_geometry(self.saveGeometry())
        self.settings.set_window_state(self.saveState())
        # Serialize the workspace BEFORE tearing the floating windows down —
        # closing them clears self._floating and disconnects their panels.
        try:
            self.settings.set_workspace_layout(self._serialize_workspace())
        except Exception:
            _log.exception("Failed to serialize workspace layout")
        for win in list(self._floating):
            win.close()
        self._floating.clear()
        event.accept()

    def _confirm_discard_changes(self) -> bool:
        if not self.project.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            APP_TITLE,
            f"'{self.project.display_name.rstrip('*')}' has unsaved changes.\n"
            "Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Discard:
            return True
        self._on_save()
        return not self.project.is_dirty

    def _restore_window_state(self) -> None:
        geom = self.settings.window_geometry()
        if geom is not None:
            self.restoreGeometry(geom)
        state = self.settings.window_state()
        if state is not None:
            self.restoreState(state)

    def _restore_workspace_or_default(self) -> None:
        saved = self.settings.workspace_layout()
        if saved is None or not self._apply_workspace(saved):
            self._panel_root.build_default_layout(_default_layout_spec())

    def _apply_workspace(self, workspace: dict) -> bool:
        """Apply a previously serialized workspace dict.

        Returns ``False`` if the dict is unusable; the caller falls back to
        the default layout. Any failure to restore a single floating panel
        is logged but does not abort the rest of the restore.
        """
        if not isinstance(workspace, dict):
            return False
        panel_root_state = workspace.get("panel_root")
        if not isinstance(panel_root_state, dict):
            return False
        try:
            self._panel_root.from_dict(panel_root_state)
        except Exception:
            _log.exception("Failed to restore docked panel tree")
            return False
        if self._panel_root.root_widget is None:
            # Workspace stored an empty tree — that's unusable; reset.
            return False
        for fl in workspace.get("floating") or []:
            try:
                self._restore_floating(fl)
            except Exception:
                _log.exception("Failed to restore floating panel %r", fl)
        return True

    def _restore_floating(self, fl: dict) -> None:
        tree = fl.get("tree")
        if not isinstance(tree, dict):
            # Backward-compat: pre-multi-panel format stored a single
            # type_id + state. Wrap it as a one-leaf tree.
            type_id = fl.get("type_id")
            if not isinstance(type_id, str):
                return
            tree = {
                "version": 1,
                "docked": {
                    "kind": "leaf",
                    "type_id": type_id,
                    "state": fl.get("state") or {},
                },
            }
        win = self._make_floating_window()
        try:
            win.panel_root.from_dict(tree)
        except Exception:
            _log.exception("Failed to restore floating panel tree")
            win.deleteLater()
            return
        if win.panel_root.root_widget is None:
            win.deleteLater()
            return
        geom_b64 = fl.get("geometry")
        if isinstance(geom_b64, str) and geom_b64:
            try:
                win.restoreGeometry(QByteArray(base64.b64decode(geom_b64)))
            except Exception:
                _log.exception("Failed to restore floating window geometry")
        win.show()
        self._floating.append(win)

    def _serialize_workspace(self) -> dict:
        return {
            "version": 1,
            "panel_root": self._panel_root.to_dict(),
            "floating": [self._serialize_floating(w) for w in self._floating],
        }

    def _serialize_floating(self, win: FloatingPanelWindow) -> dict:
        geom = bytes(win.saveGeometry())
        return {
            "tree": win.panel_root.to_dict(),
            "geometry": base64.b64encode(geom).decode("ascii"),
        }
