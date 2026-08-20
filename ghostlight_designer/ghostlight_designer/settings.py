"""Persistent app preferences backed by ``QSettings``.

Storage location is whatever ``QSettings()`` picks for the org/app names set
on ``QApplication`` (e.g. ``HKCU\\Software\\Ghostlight\\Ghostlight Designer`` on
Windows). Tests inject a custom ``QSettings`` (an INI file in ``tmp_path``)
to avoid touching real user state.

The API is intentionally named, not generic key/value — new configs add
new methods so callsites stay greppable.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from PySide6.QtCore import QByteArray, QObject, QSettings, Signal

_log = logging.getLogger("ghostlight_designer.settings")


class AppSettings(QObject):
    MAX_RECENT = 10

    recentFilesChanged = Signal(list)
    autoUpdateChanged = Signal(bool)
    # Designer-wide display (view) transform selection changed. No payload —
    # panels re-display from their cached linear frame on receipt.
    viewTransformChanged = Signal()

    def __init__(
        self,
        qsettings: Optional[QSettings] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._qs = qsettings if qsettings is not None else QSettings()

    def recent_files(self) -> List[str]:
        raw = self._qs.value("recent_files", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [str(p) for p in raw]

    def add_recent_file(self, path: str) -> None:
        files = [f for f in self.recent_files() if f != path]
        files.insert(0, path)
        files = files[: self.MAX_RECENT]
        self._qs.setValue("recent_files", files)
        self.recentFilesChanged.emit(files)

    def remove_recent_file(self, path: str) -> None:
        files = self.recent_files()
        if path not in files:
            return
        files = [f for f in files if f != path]
        self._qs.setValue("recent_files", files)
        self.recentFilesChanged.emit(files)

    def clear_recent_files(self) -> None:
        self._qs.setValue("recent_files", [])
        self.recentFilesChanged.emit([])

    def window_geometry(self) -> Optional[QByteArray]:
        v = self._qs.value("window/geometry")
        if v is None or (isinstance(v, QByteArray) and v.isEmpty()):
            return None
        return v if isinstance(v, QByteArray) else QByteArray(bytes(v))

    def set_window_geometry(self, geom: QByteArray) -> None:
        self._qs.setValue("window/geometry", geom)

    def window_state(self) -> Optional[QByteArray]:
        v = self._qs.value("window/state")
        if v is None or (isinstance(v, QByteArray) and v.isEmpty()):
            return None
        return v if isinstance(v, QByteArray) else QByteArray(bytes(v))

    def set_window_state(self, state: QByteArray) -> None:
        self._qs.setValue("window/state", state)

    def last_open_dir(self) -> str:
        return str(self._qs.value("dirs/last_open", "") or "")

    def set_last_open_dir(self, path: str) -> None:
        self._qs.setValue("dirs/last_open", path)

    def last_export_dir(self) -> str:
        """Folder last used by the animation exporter's file picker."""
        return str(self._qs.value("dirs/last_export", "") or "")

    def set_last_export_dir(self, path: str) -> None:
        self._qs.setValue("dirs/last_export", path)

    def workspace_layout(self) -> Optional[dict]:
        """Return the persisted workspace dict, or ``None`` if missing/corrupt.

        Stored as a JSON string — QSettings round-trips nested dicts unreliably
        across backends (registry vs INI), so we encode once at the boundary.
        """
        raw = self._qs.value("workspace/layout")
        if not raw:
            return None
        try:
            data = json.loads(str(raw))
        except (ValueError, TypeError) as exc:
            _log.warning("Discarding corrupt workspace layout: %s", exc)
            return None
        if not isinstance(data, dict):
            return None
        return data

    def set_workspace_layout(self, layout: Optional[dict]) -> None:
        if layout is None:
            self._qs.remove("workspace/layout")
            return
        self._qs.setValue("workspace/layout", json.dumps(layout))

    def auto_update_enabled(self) -> bool:
        """Master switch for panels that auto-re-render on lens / setup edits.

        When False, View → Auto-Update Panels is unchecked: panels keep
        their per-panel auto-render flags but never actually dispatch
        work on a lens-changed signal. Manual "Refresh" / "Re-render"
        actions still fire. Defaults to True.
        """
        raw = self._qs.value("view/auto_update", True)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() not in ("false", "0", "no", "")
        try:
            return bool(int(raw))
        except (TypeError, ValueError):
            return True

    def set_auto_update_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.auto_update_enabled():
            return
        self._qs.setValue("view/auto_update", enabled)
        self.autoUpdateChanged.emit(enabled)

    def view_ocio_config(self) -> str:
        """OCIO config backing the designer-wide display transform.

        ``""`` (default) = the bundled builtin ACES 2.0 studio config;
        ``"$OCIO"`` = resolve the ``$OCIO`` environment config (to match a
        studio's own Nuke session); any other value = a path to an .ocio file.
        """
        return str(self._qs.value("view/ocio_config", "") or "")

    def set_view_ocio_config(self, config_key: str) -> None:
        config_key = str(config_key or "")
        if config_key == self.view_ocio_config():
            return
        self._qs.setValue("view/ocio_config", config_key)
        # Changing the config invalidates the persisted (display, view); clear
        # it so the next spec resolve falls back to the new config's default.
        self._qs.remove("view/display_view")
        self.viewTransformChanged.emit()

    def view_display_view(self) -> tuple:
        """The selected OCIO ``(display, view)`` pair.

        Returns ``("", "")`` when unset — the pipeline fills the active config's
        defaults. Stored as a JSON string (QSettings round-trips tuples/lists
        unreliably across registry vs INI backends, per ``workspace_layout``).
        """
        raw = self._qs.value("view/display_view")
        if not raw:
            return ("", "")
        try:
            data = json.loads(str(raw))
        except (ValueError, TypeError):
            return ("", "")
        if (
            isinstance(data, list)
            and len(data) == 2
            and all(isinstance(x, str) for x in data)
        ):
            return (data[0], data[1])
        return ("", "")

    def set_view_display_view(self, display: str, view: str) -> None:
        pair = (str(display), str(view))
        if pair == self.view_display_view():
            return
        self._qs.setValue("view/display_view", json.dumps([pair[0], pair[1]]))
        self.viewTransformChanged.emit()

    def viewport_focus_unit(self) -> str:
        """Unit the viewport context popup's Focus row scrubs in.

        ``"mm"`` scrubs effective focal length directly; ``"dpt"`` scrubs
        optical power in diopters. Defaults to ``"mm"``. No signal — only
        the popup reads it, at open / toggle time.
        """
        raw = str(self._qs.value("viewport/focus_unit", "mm") or "mm")
        return raw if raw in ("dpt", "mm") else "mm"

    def set_viewport_focus_unit(self, unit: str) -> None:
        if unit in ("dpt", "mm"):
            self._qs.setValue("viewport/focus_unit", unit)
