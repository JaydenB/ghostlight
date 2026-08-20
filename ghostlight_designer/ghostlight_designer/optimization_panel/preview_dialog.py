"""Modal dialog that drives one :class:`OptimizationRun`.

Layout::

    ┌────────────────────────────────────────────────────────────────┐
    │ Optimization: <merit function name>                            │
    ├────────────────────────────────────────────┬───────────────────┤
    │ Status:     Iteration 23 / 80              │                   │
    │ Total:      1.453 → 0.0314                 │   LensViewport    │
    │                                            │   (virtual lens)  │
    │ ┌── Goals ───────────────────────────────┐ │                   │
    │ │ Spot RMS    0.018   res 0.018          │ │                   │
    │ │ EFL         49.72   res 0.28           │ │                   │
    │ └────────────────────────────────────────┘ │                   │
    │                                            │                   │
    │ ┌── Variables ───────────────────────────┐ │                   │
    │ │ Surf 0 radius   30.12 → 28.94          │ │                   │
    │ │ ...                                    │ │                   │
    │ └────────────────────────────────────────┘ │                   │
    ├────────────────────────────────────────────┴───────────────────┤
    │ [Restart] [Cancel]            [Reject]    [Accept Solution]    │
    └────────────────────────────────────────────────────────────────┘

The dialog NEVER writes into ``project.system`` until the user clicks
Accept. Rejecting / closing / Esc / OS-close all leave the project
pristine; the worker is cancelled in the background.

A 100 ms throttle on the viewport refresh keeps the GPU cost manageable
on long runs — without it the viewport rebuilds GLB on every new-best
signal and the dialog stutters.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

from ghostlight.writer import build_optical_system_doc
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ghostlight_viewport import LensViewport, SensorSpec

from .. import ray_tracing as rt_mod
from ..project import Project
from .data import MeritFunction, RunResult
from .goals.base import display_name_for
from .optimizer import (
    GoalSnapshot,
    HammerProgress,
    OptimizationRun,
    VariableSnapshot,
)

_log = logging.getLogger("ghostlight_designer.optimization_panel.preview_dialog")


# How often the viewport may rebuild from the virtual system during a run.
# Each rebuild walks the lens, builds GLB, ships it to the GPU — cheap
# but not free. 100 ms keeps the visual feedback "live" without bottling
# the worker (which mutates the system between rebuilds).
_VIEWPORT_REFRESH_MS = 100


class OptimizationPreviewDialog(QDialog):
    """Modal dialog hosting one optimization run.

    Caller pattern::

        run = OptimizationRun(project.system, mf, project.system_setup)
        dlg = OptimizationPreviewDialog(project, mf, run, parent=main_window)
        if dlg.exec() == QDialog.Accepted:
            # the dialog has already applied the solution into project.system
            pass
    """

    def __init__(
        self,
        project: Project,
        mf: MeritFunction,
        run: OptimizationRun,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._mf = mf
        self._run = run

        self.setWindowTitle(f"Optimization: {mf.name}")
        self.setModal(True)
        self.resize(960, 640)

        # ---- State for throttled viewport repaints --------------------
        self._needs_viewport_refresh: bool = True
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(False)
        self._refresh_timer.setInterval(_VIEWPORT_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._maybe_refresh_viewport)

        # Final goal + variable snapshots delivered with the very last
        # iteration. Held here so Accept reads them for the final summary
        # log line, and also so the right-hand tables don't blank out at
        # the moment the run finishes.
        self._last_goals: list[GoalSnapshot] = []
        self._last_vars: list[VariableSnapshot] = []
        self._initial_total: Optional[float] = None
        self._current_total: Optional[float] = None
        self._current_iter: int = 0
        self._final_result: Optional[RunResult] = None

        self._build_ui()
        self._wire_run(run)
        # Seed the viewport with the cloned starting state so the user
        # can see "before" before they hit Start (the run starts itself
        # right after construction — see start_run()).
        self._push_viewport_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # ---- Left pane: status + goal + variable tables ---------------
        left = QFrame(splitter)
        left.setFrameShape(QFrame.NoFrame)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)

        self._status_label = QLabel("Initialising…", left)
        self._status_label.setStyleSheet("font-weight: bold;")
        ll.addWidget(self._status_label)

        self._total_label = QLabel("Total: —", left)
        ll.addWidget(self._total_label)

        # Hammer sub-status: shown only while a catalogue-hammer run is
        # active. Reads "Round 2/3 · Element 0 · material 0 — trying
        # Schott N-BK7 (42/120) → total 0.031" during the sweep.
        self._hammer_label = QLabel("", left)
        self._hammer_label.setStyleSheet("color: #888;")
        self._hammer_label.setVisible(False)
        ll.addWidget(self._hammer_label)

        ll.addWidget(self._section_header("Goals", left))
        self._goals_table = QTableWidget(0, 4, left)
        self._goals_table.setHorizontalHeaderLabels(
            ["Goal", "Target", "Value", "Residual"]
        )
        self._goals_table.verticalHeader().setVisible(False)
        self._goals_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._goals_table.setSelectionMode(QTableWidget.NoSelection)
        self._goals_table.horizontalHeader().setStretchLastSection(True)
        ll.addWidget(self._goals_table, 1)

        ll.addWidget(self._section_header("Variables", left))
        self._vars_table = QTableWidget(0, 4, left)
        self._vars_table.setHorizontalHeaderLabels(
            ["Surface", "Attribute", "Start", "Current"]
        )
        self._vars_table.verticalHeader().setVisible(False)
        self._vars_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._vars_table.setSelectionMode(QTableWidget.NoSelection)
        self._vars_table.horizontalHeader().setStretchLastSection(True)
        ll.addWidget(self._vars_table, 1)

        splitter.addWidget(left)

        # ---- Right pane: virtual-system viewport ----------------------
        self._viewport = LensViewport(splitter)
        self._viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Cross-section view by default — the preview lens is the only
        # thing the user is looking at, so we show the X-plane cutaway
        # so radii / thicknesses are immediately visible.
        try:
            self._viewport.set_cutaway_mode("x")
        except Exception:
            _log.exception("Preview viewport: set_cutaway_mode('x') failed")
        # Disable element / surface picking — this viewport is read-only.
        # Clicking inside it shouldn't change the project's selection or
        # the dialog's own state; the picking would also be confusing
        # because it'd select against the virtual lens, not the live one.
        try:
            self._viewport.set_selection_mode("none")
        except Exception:
            _log.exception("Preview viewport: set_selection_mode('none') failed")
        # Belt-and-braces: also disable the right-click context menu so this
        # read-only viewport never offers editing actions (mode "none"
        # already gates it, and no host wires contextMenuRequested here).
        try:
            self._viewport.set_context_menu_enabled(False)
        except Exception:
            _log.exception("Preview viewport: set_context_menu_enabled(False) failed")
        splitter.addWidget(self._viewport)
        splitter.setSizes([380, 580])

        # ---- Buttons --------------------------------------------------
        button_row = QHBoxLayout()
        button_row.setContentsMargins(8, 4, 8, 8)

        self._btn_cancel_run = QPushButton("Cancel Run", self)
        self._btn_cancel_run.setToolTip("Stop the optimiser at the next checkpoint.")
        self._btn_cancel_run.clicked.connect(self._on_cancel_run)
        button_row.addWidget(self._btn_cancel_run)

        button_row.addStretch(1)

        self._btn_reject = QPushButton("Reject", self)
        self._btn_reject.setToolTip(
            "Close without applying. The project lens is unchanged."
        )
        self._btn_reject.clicked.connect(self._on_reject_clicked)
        button_row.addWidget(self._btn_reject)

        self._btn_accept = QPushButton("Accept Solution", self)
        self._btn_accept.setDefault(True)
        self._btn_accept.setEnabled(False)
        self._btn_accept.setToolTip(
            "Apply the optimised lens to the project (one undo entry)."
        )
        self._btn_accept.clicked.connect(self._on_accept_clicked)
        button_row.addWidget(self._btn_accept)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(splitter, 1)
        outer.addLayout(button_row, 0)

    @staticmethod
    def _section_header(text: str, parent: QWidget) -> QLabel:
        lab = QLabel(text, parent)
        lab.setStyleSheet(
            "color: #888; font-size: 11px; text-transform: uppercase; "
            "padding-top: 4px;"
        )
        return lab

    # ------------------------------------------------------------------
    # Wiring & lifecycle
    # ------------------------------------------------------------------

    def _wire_run(self, run: OptimizationRun) -> None:
        run.iterationCompleted.connect(self._on_iteration)
        run.previewUpdated.connect(self._on_preview_dirty)
        run.goalsRecomputed.connect(self._on_goals)
        run.variablesRecomputed.connect(self._on_vars)
        run.hammerProgress.connect(self._on_hammer_progress)
        run.runFinished.connect(self._on_run_finished)

    def start_run(self) -> None:
        """Kick off the optimization. Call once after constructing the dialog."""
        self._status_label.setText("Running…")
        self._refresh_timer.start()
        self._run.start()

    # ------------------------------------------------------------------
    # Signal slots
    # ------------------------------------------------------------------

    def _on_iteration(self, iter_num: int, total: float) -> None:
        self._current_iter = int(iter_num)
        self._current_total = float(total)
        if self._initial_total is None:
            self._initial_total = float(total)
        self._refresh_status()

    def _on_preview_dirty(self) -> None:
        # Mark dirty — the throttled timer will rebuild on its next tick.
        self._needs_viewport_refresh = True

    def _on_goals(self, snaps) -> None:
        # ``snaps`` is a list[GoalSnapshot] from the worker. Update the
        # table in-place — preserves the row order the merit function
        # declared, and lets the user track per-goal progress visually.
        self._last_goals = list(snaps)
        self._populate_goals_table(self._last_goals)

    def _on_vars(self, snaps) -> None:
        self._last_vars = list(snaps)
        self._populate_vars_table(self._last_vars)

    def _on_hammer_progress(self, progress: HammerProgress) -> None:
        """Update the hammer status label with the current candidate.

        Emitted per candidate trial by the worker. Cheap to update — it's
        just a QLabel — so we don't throttle it; the sweep pace itself is
        bounded by the sub-optimization budget so the label doesn't churn
        faster than the user can read.
        """
        self._hammer_label.setVisible(True)
        if progress.total_candidates == 0:
            text = (
                f"Round {progress.round_index}/{progress.total_rounds} · "
                f"{progress.flag_label} — no vendor candidates"
            )
        else:
            cand = f"{progress.candidate_index}/{progress.total_candidates}"
            best = (
                "" if progress.best_total is None
                else f" · best {progress.best_total:.4g}"
            )
            trial = (
                "" if progress.candidate_total is None
                else f" → {progress.candidate_total:.4g}"
            )
            glass = progress.glass_key or "?"
            text = (
                f"Round {progress.round_index}/{progress.total_rounds} · "
                f"{progress.flag_label} — trying {glass} ({cand}){trial}{best}"
            )
        self._hammer_label.setText(text)

    def _on_run_finished(self, result: RunResult) -> None:
        self._final_result = result
        self._refresh_timer.stop()
        # One last forced repaint so the user sees the final lens state
        # even if they happened to be between throttled ticks.
        self._needs_viewport_refresh = True
        self._maybe_refresh_viewport()

        self._btn_cancel_run.setEnabled(False)
        if result.status == "ok":
            self._status_label.setText(
                f"Done in {result.iterations or 0} iterations."
            )
            self._btn_accept.setEnabled(True)
        elif result.status == "cancelled":
            self._status_label.setText("Cancelled.")
            # The lens at the cancel moment may still be an improvement
            # over the starting state — let the user choose to accept.
            self._btn_accept.setEnabled(True)
        elif result.status == "no_variables":
            self._status_label.setText("No variables flagged.")
        elif result.status == "no_goals":
            self._status_label.setText("No goals enabled.")
        else:
            self._status_label.setText(f"Failed: {result.message}")
        if result.total is not None:
            self._current_total = float(result.total)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Status / table population
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        max_iters = self._mf.max_iters
        self._status_label.setText(
            self._status_label.text()
            if self._final_result is not None
            else f"Iteration {self._current_iter} / {max_iters}"
        )
        if self._current_total is None:
            self._total_label.setText("Total: —")
            return
        if self._initial_total is None:
            self._total_label.setText(f"Total: {self._current_total:.6g}")
        else:
            self._total_label.setText(
                f"Total: {self._initial_total:.6g} → {self._current_total:.6g}"
            )

    def _populate_goals_table(self, snaps) -> None:
        # Map goal_id → snapshot so we can render in the merit function's
        # row order regardless of what order the worker emitted in.
        by_id = {s.goal_id: s for s in snaps}
        goals = [g for g in self._mf.goals if g.enabled]
        self._goals_table.setRowCount(len(goals))
        for row, g in enumerate(goals):
            kind_name = display_name_for(g.kind)
            label = f"{kind_name}" + (f" — {g.name}" if g.name else "")
            target = f"{g.target:.4g}"
            snap = by_id.get(g.goal_id)
            value = "—" if snap is None or snap.value is None else f"{snap.value:.4g}"
            res = "—" if snap is None or snap.residual is None else f"{snap.residual:.4g}"
            self._goals_table.setItem(row, 0, QTableWidgetItem(label))
            self._goals_table.setItem(row, 1, QTableWidgetItem(target))
            self._goals_table.setItem(row, 2, QTableWidgetItem(value))
            self._goals_table.setItem(row, 3, QTableWidgetItem(res))

    def _populate_vars_table(self, snaps) -> None:
        self._vars_table.setRowCount(len(snaps))
        for row, v in enumerate(snaps):
            self._vars_table.setItem(row, 0, QTableWidgetItem(str(v.surface_index)))
            self._vars_table.setItem(row, 1, QTableWidgetItem(str(v.attr)))
            self._vars_table.setItem(row, 2, QTableWidgetItem(f"{v.start:.6g}"))
            self._vars_table.setItem(row, 3, QTableWidgetItem(f"{v.current:.6g}"))

    # ------------------------------------------------------------------
    # Throttled viewport refresh
    # ------------------------------------------------------------------

    def _maybe_refresh_viewport(self) -> None:
        if not self._needs_viewport_refresh:
            return
        self._needs_viewport_refresh = False
        self._push_viewport_state()

    def _push_viewport_state(self) -> None:
        """Push the current virtual_system geometry into the viewport.

        Re-reads ``virtual_system`` every time — the worker may have
        mutated it since the last push, which is the whole point. The
        viewport copies into GLB internally, so subsequent worker writes
        don't corrupt what's on the GPU.
        """
        try:
            elements = list(self._run.virtual_system.elements)
        except Exception:
            elements = []
        try:
            self._viewport.set_lens(
                self._run.virtual_system, elements, fit_view=False,
            )
        except Exception:
            _log.exception("Preview viewport: set_lens failed")
            return
        # Mirror the System Setup's sensor onto the preview so the user
        # sees the same sensor rectangle they see in the main viewport.
        try:
            s = self._project.system_setup.sensor
            self._viewport.set_sensor(SensorSpec(
                half_w=float(s.width_mm) / 2.0,
                half_h=float(s.height_mm) / 2.0,
            ))
        except Exception:
            _log.exception("Preview viewport: set_sensor failed")

        # Trace and push the System Setup's field rays through the
        # virtual system so the user can see the current solve state
        # as *rays through the tuned lens*, not just as static geometry.
        # Rebuilt on every preview refresh (throttled to 100 ms) so the
        # rays update in step with the amber left-stripe repaint on the
        # geometry — the two views stay in sync.
        try:
            bundles = rt_mod.build_ray_bundles(
                self._run.virtual_system,
                self._project.system_setup,
            )
        except Exception:
            _log.exception("Preview viewport: build_ray_bundles failed")
            self._viewport.clear_trace_results()
            return
        if bundles:
            self._viewport.set_trace_results(bundles)
        else:
            self._viewport.clear_trace_results()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_cancel_run(self) -> None:
        self._run.request_cancel()
        self._status_label.setText("Cancelling…")
        self._btn_cancel_run.setEnabled(False)

    def _on_reject_clicked(self) -> None:
        # Always cancel before closing — if the run is still going, this
        # lets the worker exit cleanly before the QThread is torn down.
        self._run.request_cancel()
        self.reject()

    def _on_accept_clicked(self) -> None:
        if not self._run.is_finished:
            # Defensive: the button is gated until finished, but stay safe.
            return
        if not self._apply_solution_to_project():
            return
        self.accept()

    def _apply_solution_to_project(self) -> bool:
        """Serialize the virtual lens and reload it into the real project.

        Wrapped in ``project.edit(...)`` so the entire optimisation lands
        as a single undo entry whose label is "Optimize <MF name>".
        Returns True on success.
        """
        try:
            doc = build_optical_system_doc(
                system=self._run.virtual_system,
                metadata=getattr(self._run.virtual_system, "_raw_metadata", None) or {},
                glass_catalogue=getattr(
                    self._run.virtual_system, "_raw_glass_catalogue", None,
                ) or {},
            )
        except Exception:
            _log.exception("Failed to serialise virtual system for accept")
            self._status_label.setText("Failed to serialise virtual system.")
            return False

        fd, tmp_path = tempfile.mkstemp(suffix=".lens")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with self._project.edit(f"Optimize {self._mf.name}"):
                self._project.system.reload(tmp_path)
        except Exception as exc:
            _log.exception("Failed to apply optimization solution")
            self._status_label.setText(f"Apply failed: {exc}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Also stamp the merit function's last-run cache so the panel
        # tree shows the just-applied total. Use the worker's final
        # cached values — they correspond to virtual_system at the
        # exact state we just committed.
        if self._final_result is not None:
            self._mf.apply_run_result(self._final_result)
        self._refresh_goal_caches_on_mf()
        self._project.mark_merit_functions_modified()
        return True

    def _refresh_goal_caches_on_mf(self) -> None:
        """Copy the last goal snapshots back onto the merit function so
        the tree's Value column matches what the dialog just showed."""
        by_id = {s.goal_id: s for s in self._last_goals}
        for g in self._mf.goals:
            s = by_id.get(g.goal_id)
            if s is None:
                continue
            g.cached_value = s.value
            g.cached_residual = s.residual

    # ------------------------------------------------------------------
    # Close handling — ensure the worker always cancels
    # ------------------------------------------------------------------

    def reject(self) -> None:  # noqa: D401 (Qt API)
        self._run.request_cancel()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._run.request_cancel()
        super().closeEvent(event)
