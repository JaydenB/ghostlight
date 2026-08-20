from __future__ import annotations

from PySide6.QtCore import QByteArray, QModelIndex, Qt
from PySide6.QtGui import QKeySequence

import ghostlight

from ghostlight_designer.main_window import MainWindow, APP_TITLE
from ghostlight_designer.optical_editor.columns import Column
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.panel_system import PanelRoot
from ghostlight_designer.project import Project


def test_main_window_constructs(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        assert w.windowTitle() == f"Untitled - {APP_TITLE}"
        labels = [
            a.text().replace("&", "").rstrip("…")
            for a in w.menu_file.actions()
            if not a.isSeparator()
        ]
        assert labels == ["New", "Open", "Open Recent", "Save", "Save As", "Exit"]
    finally:
        w.deleteLater()


def test_save_disabled_initially(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        assert w.action_save.isEnabled() is False
        assert w.action_save_as.isEnabled() is True
    finally:
        w.deleteLater()


def test_save_enables_after_modify_with_path(qapp, isolated_settings, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    w = MainWindow(project=project, settings=isolated_settings)
    try:
        assert w.action_save.isEnabled() is False
        project.mark_modified()
        assert w.action_save.isEnabled() is True
        assert w.windowTitle() == f"{sample_lens_path.name}* - {APP_TITLE}"
    finally:
        w.deleteLater()


def test_title_updates_on_load_and_dirty(qapp, isolated_settings, sample_lens_path):
    project = Project()
    w = MainWindow(project=project, settings=isolated_settings)
    try:
        assert w.windowTitle() == f"Untitled - {APP_TITLE}"
        project.load(str(sample_lens_path))
        assert w.windowTitle() == f"{sample_lens_path.name} - {APP_TITLE}"
        project.mark_modified()
        assert w.windowTitle() == f"{sample_lens_path.name}* - {APP_TITLE}"
    finally:
        w.deleteLater()


def test_main_window_uses_panel_root(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        assert isinstance(w.centralWidget(), PanelRoot)
        type_ids = {p.type_id for p in w._panel_root.leaves()}
        assert type_ids == {"optical_editor", "system_setup", "viewport", "sourceflare"}
    finally:
        w.deleteLater()


def test_recent_menu_disabled_when_empty(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        assert w.menu_recent.isEnabled() is False
        assert w.menu_recent.actions() == []
    finally:
        w.deleteLater()


def test_recent_menu_populates_on_change(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        isolated_settings.add_recent_file("foo.lens")
        assert w.menu_recent.isEnabled() is True
        labels = [a.text() for a in w.menu_recent.actions() if not a.isSeparator()]
        assert "foo.lens" in labels
        assert "Clear Recent" in labels
    finally:
        w.deleteLater()


def test_open_path_adds_to_recents(qapp, isolated_settings, sample_lens_path):
    w = MainWindow(settings=isolated_settings)
    try:
        w._open_path(str(sample_lens_path))
        assert isolated_settings.recent_files() == [str(sample_lens_path)]
        assert w.project.path == str(sample_lens_path)
    finally:
        w.deleteLater()


def test_open_path_failure_removes_from_recents(qapp, isolated_settings, monkeypatch):
    """A stale recent-files entry that no longer loads should disappear."""
    bogus = "Z:/does/not/exist.lens"
    isolated_settings.add_recent_file(bogus)
    assert isolated_settings.recent_files() == [bogus]

    from ghostlight_designer import main_window as mw
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: None)

    w = MainWindow(settings=isolated_settings)
    try:
        w._open_path(bogus)
        assert isolated_settings.recent_files() == []
    finally:
        w.deleteLater()


def test_geometry_restored_from_settings(qapp, isolated_settings, monkeypatch):
    seed = MainWindow(settings=isolated_settings)
    seed.resize(1024, 600)
    isolated_settings.set_window_geometry(seed.saveGeometry())
    expected = bytes(isolated_settings.window_geometry())
    seed.deleteLater()

    # The offscreen platform doesn't update width()/height() until the window
    # is shown, and a saveGeometry round-trip doesn't reproduce the exact bytes
    # (window-handle metadata shifts). Stub restoreGeometry to capture the
    # bytes passed by _restore_window_state — that's the wiring we care about.
    captured: list[bytes] = []

    def fake_restore(self, data):
        captured.append(bytes(data))
        return True

    monkeypatch.setattr(MainWindow, "restoreGeometry", fake_restore)

    w = MainWindow(settings=isolated_settings)
    try:
        assert captured and captured[0] == expected
    finally:
        w.deleteLater()


def test_edit_menu_has_undo_redo_with_shortcuts(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        labels = [
            a.text().replace("&", "")
            for a in w.menu_edit.actions()
            if not a.isSeparator()
        ]
        assert labels == ["Undo", "Redo"]
        assert w.action_undo.shortcut() == QKeySequence.Undo
        assert QKeySequence.Redo in w.action_redo.shortcuts()
        assert QKeySequence("Ctrl+Shift+Z") in w.action_redo.shortcuts()
        # Both disabled with empty history.
        assert w.action_undo.isEnabled() is False
        assert w.action_redo.isEnabled() is False
    finally:
        w.deleteLater()


def test_edit_menu_reflects_can_undo_redo(qapp, isolated_settings, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    w = MainWindow(project=project, settings=isolated_settings)
    try:
        # Drive an edit through the optical-editor model.
        model = OpticalTreeModel(project)
        ei = 0
        identifier_idx = model.index(ei, int(Column.IDENTIFIER), QModelIndex())
        model.setData(identifier_idx, "RenamedViaTest", Qt.EditRole)

        assert w.action_undo.isEnabled() is True
        assert w.action_undo.text().startswith("&Undo ")
        assert "Rename" in w.action_undo.text()
        assert w.action_redo.isEnabled() is False

        project.undo()
        assert w.action_redo.isEnabled() is True
        assert w.action_redo.text().startswith("&Redo ")
    finally:
        w.deleteLater()


def test_windows_menu_layouts_submenu_at_top(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        actions = w.menu_windows.actions()
        assert actions, "Windows menu should have entries"
        # Layouts submenu sits first, followed immediately by a separator.
        assert actions[0].text().replace("&", "") == "Layouts"
        assert actions[1].isSeparator()

        # The Layouts submenu populates from layout_registry on aboutToShow.
        w.menu_windows_layouts.aboutToShow.emit()
        layout_labels = [a.text() for a in w.menu_windows_layouts.actions()]
        assert "Default" in layout_labels
    finally:
        w.deleteLater()


def test_windows_menu_spawns_floating_panel(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        before = len(w._floating)
        w._spawn_floating_panel("viewport")
        assert len(w._floating) == before + 1
        assert w._floating[-1].panel.type_id == "viewport"
        # Tear down the floating window so the test cleanup doesn't leak it.
        w._floating[-1].close()
    finally:
        w.deleteLater()


def test_windows_menu_lists_panel_types(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    try:
        w.menu_windows.aboutToShow.emit()
        # Either as top-level entries or as category submenu entries — flatten.
        labels: set[str] = set()
        for a in w.menu_windows.actions():
            sub = a.menu()
            if sub is not None and a is not w.menu_windows_layouts.menuAction():
                for child in sub.actions():
                    labels.add(child.text())
            elif not a.isSeparator():
                labels.add(a.text())
        # Sanity: a couple of the always-registered panel types are reachable.
        assert "Optical Design Editor" in labels
        assert "Viewport" in labels
    finally:
        w.deleteLater()


def test_close_event_persists_geometry(qapp, isolated_settings):
    w = MainWindow(settings=isolated_settings)
    w.resize(900, 700)
    assert isolated_settings.window_geometry() is None
    w.close()
    assert isolated_settings.window_geometry() is not None
    assert isinstance(isolated_settings.window_geometry(), QByteArray)
