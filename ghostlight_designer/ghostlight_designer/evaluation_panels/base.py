"""Base class for evaluation panels (spot diagram, field diagrams, MTF, Seidel …).

Every evaluation panel shares the same orchestration shape:

* It listens for ``systemReplaced`` / ``systemModified`` / ``systemSetupChanged``
  on the project, plus the global ``AppSettings.autoUpdateChanged`` master
  switch from View → Auto-Update Panels.
* On a lens edit it schedules a refresh through a 350 ms edit-settle
  debounce — the same pattern the render panels use — so a scrub-drag
  doesn't pump the CPU.
* It runs the compute on a background thread so the UI stays responsive,
  then applies the result back on the GUI thread via a poll timer.
* It pauses work when not visible (collapsed splitter, hidden tab, etc.)
  and catches up when it becomes visible again.

Subclasses provide the actual work and visualization:

* :meth:`compute` (worker thread, no Qt access) returns an arbitrary
  result object.
* :meth:`apply_result` (GUI thread) updates the canvas with that result.
* :meth:`sync_from_system_setup` (GUI thread) is what the Sync button calls;
  default is a no-op so panels without setup-mappable parameters can ignore it.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QWidget

from ..project import Project
from ..settings import AppSettings

_log = logging.getLogger("ghostlight_designer.evaluation_panels")


DEBOUNCE_MS = 350
POLL_INTERVAL_MS = 50


class EvaluationPanelBody(QWidget):
    """Base class for evaluation panels.

    Subclasses must implement :meth:`compute` and :meth:`apply_result`.
    Subclasses own their own layout — they place the inherited
    :attr:`status_label` wherever fits (bottom-of-body, matching the
    other render panels) and build the rest of the UI as appropriate.

    The class deliberately ships no in-panel toolbar; user-facing
    actions (Refresh, Sync from System Setup, Auto-Update toggle, etc.)
    are exposed through the panel type's ``build_menus`` contribution
    on the host panel's menu bar — the convention every other panel
    type uses.
    """

    # Per-class override points. Subclass at the class-attribute level
    # if a panel wants different timings.
    debounce_ms: int = DEBOUNCE_MS
    poll_interval_ms: int = POLL_INTERVAL_MS

    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._app_settings = settings

        # Per-panel auto-update toggle. AND-ed with the global setting
        # from View → Auto-Update Panels — both must be on for a lens
        # edit to schedule a recompute.
        self._auto_update_local: bool = True

        # Visibility gate. Set in showEvent / hideEvent so that work
        # doesn't queue up behind a collapsed splitter or hidden tab.
        self._is_active: bool = False
        # Set whenever a system / setup change came in while the panel
        # was either hidden or globally gated. The next showEvent / Auto
        # ON transition picks this up.
        self._dirty_pending: bool = True

        # Threading state — one compute in flight, one slot pending.
        # The pending slot is just a bool: each new lens edit re-arms
        # it, so a burst collapses to a single recompute after the
        # current one returns.
        self._lock = threading.Lock()
        self._busy: bool = False
        self._pending: bool = False
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        # Bumped on every lens swap so the poller discards stale
        # results from a worker that captured the old lens.
        self._epoch: int = 0

        # Status line. A plain ``QLabel`` so subclasses can place it
        # wherever their layout calls for (typically at the bottom,
        # matching the existing render panels). Updated through
        # :meth:`set_status`.
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #aaa; padding: 2px 6px;")

        # Edit-settle debouncer — same pattern as the render panels.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(int(self.debounce_ms))
        self._debounce.timeout.connect(self._on_debounce_timeout)

        # Result poller. Idle cost is negligible; runs whenever the
        # widget is alive so results never get stuck in the queue.
        self._poller = QTimer(self)
        self._poller.setInterval(int(self.poll_interval_ms))
        self._poller.timeout.connect(self._poll_results)
        self._poller.start()

        # Subscribe to project + settings.
        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)
        project.systemSetupChanged.connect(self._on_system_setup_changed)
        settings.autoUpdateChanged.connect(self._on_global_auto_update_changed)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def compute(self) -> Any:
        """Run on a worker thread. Must not touch Qt widgets.

        Subclasses gather a self-contained snapshot of inputs at the
        start of this call (e.g. ``self._project.system``, the spec
        dataclass) and return an arbitrary result object that will be
        handed back to :meth:`apply_result` on the GUI thread.

        The default implementation is a no-op so a misconfigured panel
        doesn't crash — it just stays blank.
        """
        return None

    def apply_result(self, result: Any) -> None:
        """Run on the GUI thread after :meth:`compute` returns.

        Subclasses paint into their canvas / update labels from ``result``.
        ``result`` is whatever :meth:`compute` produced — never ``None``
        (those get routed to :meth:`apply_error` instead).
        """
        pass

    def apply_error(self, exc: BaseException) -> None:
        """Run on the GUI thread if :meth:`compute` raised.

        Default: log + set status to "Compute failed". Subclasses can
        override to e.g. dim the canvas.
        """
        _log.exception("EvaluationPanelBody compute failed", exc_info=exc)
        self.set_status(f"Compute failed: {exc}")

    def sync_from_system_setup(self) -> None:
        """Pull this panel's spec from ``project.system_setup``.

        Default is a no-op. Subclasses with a custom spec (wavelengths,
        fields, etc.) override to copy values across, then call
        :meth:`force_refresh_now` so the user sees the result.
        """
        pass

    # ------------------------------------------------------------------
    # Public API used by menus / toolbar
    # ------------------------------------------------------------------

    @property
    def auto_update_local(self) -> bool:
        return self._auto_update_local

    def set_auto_update_local(self, enabled: bool) -> None:
        """Per-panel auto-update toggle.

        Mirrors :meth:`SourceFlarePanelBody.set_auto_render` in naming —
        callable from the panel's View → Auto-Update menu action.
        """
        enabled = bool(enabled)
        if enabled == self._auto_update_local:
            return
        self._auto_update_local = enabled
        if enabled and self._dirty_pending and self._app_settings.auto_update_enabled():
            self.request_refresh()
        elif not enabled:
            self._debounce.stop()

    def request_refresh(self) -> None:
        """Debounced refresh request.

        Each call restarts the settle timer. Cheap to call in a tight
        loop during a scrub-drag — the actual recompute waits for
        :data:`debounce_ms` of quiet.
        """
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        self.set_status("Edits settling…")
        self._debounce.start()

    def force_refresh_now(self) -> None:
        """Bypass the debounce and the auto-update gates.

        This is the manual-action path — Refresh button, an explicit
        user click. Runs even when the global toggle is off because
        the user explicitly asked for it.
        """
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        self._debounce.stop()
        self._dispatch()

    def set_status(self, text: str) -> None:
        """Update the inherited status label. Cheap — guards against
        identical-text setText churn during a sustained drag."""
        if self.status_label.text() != text:
            self.status_label.setText(text)

    # ------------------------------------------------------------------
    # Internal: signal handlers
    # ------------------------------------------------------------------

    def _should_auto_update(self) -> bool:
        return self._auto_update_local and self._app_settings.auto_update_enabled()

    def _on_system_replaced(self, _system: Any) -> None:
        # A brand-new lens load — treat as an explicit user action so
        # the panel always reflects the new system, regardless of toggles.
        self._epoch += 1
        if self._is_active:
            self.force_refresh_now()
        else:
            self._dirty_pending = True

    def _on_system_modified(self) -> None:
        if not self._should_auto_update():
            self._dirty_pending = True
            return
        self.request_refresh()

    def _on_system_setup_changed(self) -> None:
        if not self._should_auto_update():
            self._dirty_pending = True
            return
        self.request_refresh()

    def _on_global_auto_update_changed(self, enabled: bool) -> None:
        """User flipped View → Auto-Update Panels.

        When OFF: pending lens edits will skip dispatch — nothing to do
        here besides letting the next request_refresh() short-circuit.
        When ON: catch up if we missed an edit while gated.
        """
        if enabled and self._dirty_pending and self._auto_update_local:
            self.request_refresh()

    def apply_sync_from_system_setup(self) -> None:
        """Public counterpart to :meth:`sync_from_system_setup`.

        Calls the subclass-provided sync and then forces a refresh on
        the spot — Sync is an explicit user action, so it bypasses the
        debounce. Wired up by the panel type's View menu.
        """
        try:
            self.sync_from_system_setup()
        except Exception:
            _log.exception("sync_from_system_setup raised")
            self.set_status("Sync failed (see log)")
            return
        self.force_refresh_now()

    # ------------------------------------------------------------------
    # Internal: dispatch
    # ------------------------------------------------------------------

    def _on_debounce_timeout(self) -> None:
        # Re-check state — visibility / toggles may have flipped between
        # the timer arming and firing.
        if not self._is_active or not self._should_auto_update():
            return
        self._dispatch()

    def _dispatch(self) -> None:
        with self._lock:
            self._pending = True
        self.set_status("Computing…")
        self._maybe_launch()

    def _maybe_launch(self) -> None:
        if not self._is_active:
            return
        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True

        epoch = self._epoch
        threading.Thread(
            target=self._worker,
            args=(epoch,),
            daemon=True,
        ).start()

    def _worker(self, epoch: int) -> None:
        try:
            result = self.compute()
            self._results.put((epoch, result, None))
        except BaseException as exc:  # noqa: BLE001
            # Capture every exception so a buggy subclass compute can't
            # leak a Python thread or strand the busy flag.
            self._results.put((epoch, None, exc))
        finally:
            with self._lock:
                self._busy = False

    def _poll_results(self) -> None:
        latest: Any = _NOT_SET
        latest_error: Optional[BaseException] = None
        while not self._results.empty():
            epoch, result, err = self._results.get()
            if epoch != self._epoch:
                # Stale frame from a worker that captured an older lens.
                continue
            if err is not None:
                latest_error = err
                latest = _NOT_SET  # error trumps a previous success
            else:
                latest = result
                latest_error = None

        if latest is not _NOT_SET:
            # The status line was "Computing…" while the worker was running.
            # Let the subclass overwrite it inside apply_result with something
            # useful ("3 fields × 3 λ", "RMS spot 7 µm", …) — that's the
            # whole point of the status. If the subclass doesn't write one,
            # fall back to "Idle." so the user knows the work is done.
            try:
                self.apply_result(latest)
            except Exception:
                _log.exception("apply_result raised")
                self.set_status("Display failed (see log)")
            else:
                if self.status_label.text() == "Computing…":
                    self.set_status("Idle.")
        elif latest_error is not None:
            self.apply_error(latest_error)

        # Drain any pending recompute that piled up while we were busy.
        self._maybe_launch()

    # ------------------------------------------------------------------
    # Visibility lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._is_active = True
        if self._dirty_pending:
            # Catch up on whatever happened while we were hidden — but
            # respect the auto-update gates. A panel that came back
            # visible while the global toggle is off should NOT silently
            # render; the user wants nothing to render until they
            # re-enable auto-update or click Refresh.
            if self._should_auto_update():
                self.force_refresh_now()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().hideEvent(event)
        self._is_active = False
        self._dirty_pending = True
        self._debounce.stop()


# Sentinel so ``None`` can be a legitimate "successful empty result" from
# ``compute``. Using ``object()`` rather than ``None`` keeps the polling
# loop unambiguous: ``latest is _NOT_SET`` means "no fresh result this
# tick", regardless of what the subclass returns.
_NOT_SET: Any = object()
