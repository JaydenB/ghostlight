from __future__ import annotations

from PySide6.QtCore import QByteArray


def test_empty_store_has_no_recents(isolated_settings):
    assert isolated_settings.recent_files() == []


def test_add_recent_orders_most_recent_first(isolated_settings):
    received: list[list[str]] = []
    isolated_settings.recentFilesChanged.connect(lambda f: received.append(list(f)))

    isolated_settings.add_recent_file("a.lens")
    isolated_settings.add_recent_file("b.lens")

    assert isolated_settings.recent_files() == ["b.lens", "a.lens"]
    assert received == [["a.lens"], ["b.lens", "a.lens"]]


def test_add_recent_dedupes_and_moves_to_front(isolated_settings):
    isolated_settings.add_recent_file("a.lens")
    isolated_settings.add_recent_file("b.lens")
    isolated_settings.add_recent_file("a.lens")
    assert isolated_settings.recent_files() == ["a.lens", "b.lens"]


def test_add_recent_caps_at_max(isolated_settings):
    for i in range(isolated_settings.MAX_RECENT + 5):
        isolated_settings.add_recent_file(f"file_{i}.lens")
    files = isolated_settings.recent_files()
    assert len(files) == isolated_settings.MAX_RECENT
    # most recently added wins the front
    assert files[0] == f"file_{isolated_settings.MAX_RECENT + 4}.lens"


def test_remove_recent_emits_and_drops(isolated_settings):
    isolated_settings.add_recent_file("a.lens")
    isolated_settings.add_recent_file("b.lens")

    received: list[list[str]] = []
    isolated_settings.recentFilesChanged.connect(lambda f: received.append(list(f)))

    isolated_settings.remove_recent_file("a.lens")
    assert isolated_settings.recent_files() == ["b.lens"]
    assert received == [["b.lens"]]


def test_remove_recent_missing_is_noop(isolated_settings):
    isolated_settings.add_recent_file("a.lens")
    received: list[list[str]] = []
    isolated_settings.recentFilesChanged.connect(lambda f: received.append(list(f)))

    isolated_settings.remove_recent_file("missing.lens")
    assert received == []
    assert isolated_settings.recent_files() == ["a.lens"]


def test_clear_recent_emits_empty(isolated_settings):
    isolated_settings.add_recent_file("a.lens")
    received: list[list[str]] = []
    isolated_settings.recentFilesChanged.connect(lambda f: received.append(list(f)))

    isolated_settings.clear_recent_files()
    assert isolated_settings.recent_files() == []
    assert received == [[]]


def test_window_geometry_round_trip(isolated_settings):
    assert isolated_settings.window_geometry() is None
    geom = QByteArray(b"\x01\x02\x03\xff")
    isolated_settings.set_window_geometry(geom)
    assert bytes(isolated_settings.window_geometry()) == bytes(geom)


def test_last_open_dir_round_trip(isolated_settings):
    assert isolated_settings.last_open_dir() == ""
    isolated_settings.set_last_open_dir(r"C:\some\dir")
    assert isolated_settings.last_open_dir() == r"C:\some\dir"


def test_workspace_layout_empty_returns_none(isolated_settings):
    assert isolated_settings.workspace_layout() is None


def test_workspace_layout_round_trip(isolated_settings):
    workspace = {
        "version": 1,
        "panel_root": {"version": 1, "docked": {"kind": "leaf", "type_id": "viewport"}},
        "floating": [{"type_id": "psf", "geometry": "AAAA", "state": {}}],
    }
    isolated_settings.set_workspace_layout(workspace)
    assert isolated_settings.workspace_layout() == workspace


def test_workspace_layout_clear_with_none(isolated_settings):
    isolated_settings.set_workspace_layout({"version": 1, "panel_root": {}, "floating": []})
    isolated_settings.set_workspace_layout(None)
    assert isolated_settings.workspace_layout() is None


def test_workspace_layout_discards_corrupt_json(isolated_settings):
    # Force-write a non-JSON value through the underlying QSettings.
    isolated_settings._qs.setValue("workspace/layout", "{not json")
    assert isolated_settings.workspace_layout() is None
