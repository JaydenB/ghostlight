"""Optimization driver — runs a :class:`MeritFunction` against a *cloned*
:class:`ghostlight.OpticalSystem` so the live project is never mutated mid-run.

The flow:

1. :class:`OptimizationRun` is constructed with the source system. It
   immediately deep-clones it (writer-doc + temp-file reload, same path
   the Project's undo system uses, so we know it round-trips).
2. ``start()`` launches a worker :class:`QThread` that calls
   ``scipy.optimize.least_squares`` on a residuals function. Every
   evaluator runs against ``virtual_system`` only.
3. Signals fire on the main thread:
     - :sig:`iterationCompleted(iter_num, total)` — best-so-far tracked
       internally so we don't fire on every residuals call (least_squares
       calls residuals many times per iteration for finite-difference
       Jacobians; we'd flood the UI otherwise).
     - :sig:`previewUpdated()` — same throttling, signals the preview
       dialog to repaint the virtual lens in its embedded viewport.
     - :sig:`goalsRecomputed(list[GoalSnapshot])` — current per-goal
       values + residuals, for the dialog's live goal table.
     - :sig:`runFinished(RunResult)` — fires once at the end (ok,
       cancelled, no_variables, failed).
4. The dialog calls :meth:`request_cancel` to ask the worker to abort
   at the next residuals checkpoint.

The Project is **never** touched here. Applying the result is the
dialog's job (it serialises ``virtual_system`` and reloads it into
``project.system`` inside a ``project.edit(...)`` block).

scipy import is deferred to ``start()`` so importing this module in a
headless env without scipy doesn't fail — the panel's "Run" handler
catches the ImportError and surfaces a friendly message.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional

import ghostlight
from ghostlight.writer import build_optical_system_doc
from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..material_catalogue import get_catalogue
from ..material_substitution import candidates_within_spec
from .data import GoalEntry, MeritFunction, RunResult
from .geometry_penalties import evaluate_geometry_penalties, n_geometry_residuals
from .goals import GOAL_REGISTRY
from .goals.base import safe_evaluate
from .variables import (
    MaterialFlagRef,
    VariableRef,
    apply_variables,
    apply_variables_scipy,
    collect_material_flags,
    collect_variables,
    install_candidate_glass,
    pack_bounds_scipy,
)


_log = logging.getLogger("ghostlight_designer.optimization_panel.optimizer")


# ---------------------------------------------------------------------------
# Result records used by signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalSnapshot:
    """Per-goal value at a moment in time — drives the live dialog table."""
    goal_id: str
    value: Optional[float]
    residual: Optional[float]


@dataclass(frozen=True)
class VariableSnapshot:
    """Per-variable starting + current values for the dialog's variable table."""
    surface_index: int
    attr: str
    start: float
    current: float


@dataclass(frozen=True)
class HammerProgress:
    """One tick of the catalogue-hammer status feed.

    Fields:

    * ``round_index`` / ``total_rounds`` — outer sequential-greedy sweep
      counter (1-based for display; ``total_rounds`` is the configured
      max). Zero-value ``round_index`` means "outer loop is deciding
      whether to start another sweep" and is used for the summary line
      the dialog shows at start / end.
    * ``flag_label`` — human-readable "Element E · material M" identifier
      for the flag being worked on right now.
    * ``candidate_index`` / ``total_candidates`` — position within the
      current flag's vendor-candidate list (1-based).
    * ``glass_key`` — the catalogue key being trialled at this tick.
      Empty string on summary ticks.
    * ``best_total`` — best sum-of-squared-residuals seen so far across
      the whole run. Drives the "improvement" indicator.
    * ``candidate_total`` — the total achieved for *this* candidate's
      inner sub-optimization; ``None`` before it's computed.
    """
    round_index: int
    total_rounds: int
    flag_label: str
    candidate_index: int
    total_candidates: int
    glass_key: str
    best_total: Optional[float]
    candidate_total: Optional[float]


# ---------------------------------------------------------------------------
# System cloning — same writer + temp-file path the undo system uses
# ---------------------------------------------------------------------------


def clone_system(source: ghostlight.OpticalSystem) -> ghostlight.OpticalSystem:
    """Return a deep-clone of ``source`` via the writer+reload round-trip.

    Cost: one JSON dump + one C++ load. For a 4-element lens this is
    sub-millisecond; large systems could be ~10 ms. Same path the
    project uses for undo snapshots, so we inherit its correctness.
    """
    doc = build_optical_system_doc(
        system=source,
        metadata=getattr(source, "_raw_metadata", None) or {},
        glass_catalogue=getattr(source, "_raw_glass_catalogue", None) or {},
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".lens")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        clone = ghostlight.OpticalSystem.load(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return clone


# ---------------------------------------------------------------------------
# Worker — runs scipy.least_squares off the GUI thread
# ---------------------------------------------------------------------------


class _Worker(QObject):
    """QObject living on the worker thread. Emits cross-thread signals
    that the run translates to its own user-facing API.
    """

    iterationCompleted = Signal(int, float)         # iter_num, total
    previewUpdated = Signal()
    goalsRecomputed = Signal(list)                  # list[GoalSnapshot]
    variablesRecomputed = Signal(list)              # list[VariableSnapshot]
    hammerProgress = Signal(object)                 # HammerProgress
    finished = Signal(object)                       # RunResult

    def __init__(
        self,
        *,
        virtual_system: ghostlight.OpticalSystem,
        mf: MeritFunction,
        setup,
        variables: list[VariableRef],
        starting_values: list[float],
        bounds: tuple[list[float], list[float]],
        is_curvature: list[bool],
        material_flags: list[MaterialFlagRef],
        cancel_flag: threading.Event,
    ) -> None:
        super().__init__()
        self._sys = virtual_system
        self._mf = mf
        self._setup = setup
        self._vars = variables
        self._x0 = starting_values
        self._bounds = bounds
        # Parallel to ``self._vars``: True where scipy sees the variable
        # in curvature space and we must invert before writing back to
        # ``Surface.radius``. Every write path that consumes scipy's
        # ``x`` vector must go through :func:`apply_variables_scipy`
        # (not :func:`apply_variables`) or it will corrupt the tracer.
        self._is_curvature = list(is_curvature)
        # Display-space starts (R for radius vars, matches what the user
        # sees in the ODE). The dialog's Variables table renders these
        # instead of the raw scipy-space starts so a radius flagged at
        # R=50 doesn't show as start=0.02.
        self._display_starts: list[float] = [
            (1.0 / s if is_c and s != 0.0 else 0.0 if is_c else s)
            for s, is_c in zip(starting_values, self._is_curvature)
        ]
        self._material_flags = material_flags
        self._cancel = cancel_flag
        # Best-so-far tracking. The first residuals call seeds this.
        self._best_total: float = math.inf
        self._iter_count: int = 0

    def run(self) -> None:
        try:
            self._run_inner()
        except BaseException as exc:  # noqa: BLE001
            _log.exception("Optimization worker crashed")
            self.finished.emit(RunResult(
                status="failed",
                message=str(exc),
            ))

    def _run_inner(self) -> None:
        try:
            from scipy.optimize import least_squares  # noqa: F401
        except ImportError as exc:
            self.finished.emit(RunResult(
                status="failed",
                message=(
                    "scipy.optimize is unavailable — install scipy>=1.11 to "
                    f"run optimization. ({exc})"
                ),
            ))
            return

        enabled_goals = [g for g in self._mf.goals if g.enabled]
        if not enabled_goals:
            self.finished.emit(RunResult(
                status="no_goals",
                message="Merit function has no enabled goals.",
            ))
            return

        # Initial seed evaluation so the dialog has something to show on
        # frame 0 (before scipy has called residuals yet).
        self._publish_current_state(enabled_goals)

        # Dispatch: material flags trigger the hammer wrapper. When
        # neither material flags nor geometric variables exist the caller
        # already short-circuited to no_variables (see OptimizationRun.start),
        # so we can assume at least one is populated here.
        if self._material_flags:
            self._run_hammer(enabled_goals)
        else:
            self._run_standard(enabled_goals)

    # ------------------------------------------------------------------
    # Standard scipy path — the today-before-hammer flow, unchanged.
    # ------------------------------------------------------------------

    def _run_standard(self, enabled_goals: list[GoalEntry]) -> None:
        try:
            result = self._run_scipy(
                enabled_goals,
                max_nfev=int(self._mf.max_iters),
                emit_progress=True,
            )
        except _Cancelled:
            self.finished.emit(RunResult(
                status="cancelled",
                message="Cancelled by user.",
                total=self._best_total if math.isfinite(self._best_total) else None,
                iterations=self._iter_count,
            ))
            return
        except Exception as exc:  # noqa: BLE001
            _log.exception("least_squares raised")
            self.finished.emit(RunResult(
                status="failed",
                message=f"least_squares failed: {exc}",
            ))
            return

        # scipy may have stepped away from result.x during its last
        # trust-region trial; commit the winning x explicitly so the
        # virtual system + snapshots reflect the accepted solution.
        try:
            if self._vars and hasattr(result, "x") and len(result.x) > 0:
                apply_variables_scipy(
                    self._vars, self._sys, list(result.x), self._is_curvature,
                )
                self._sys.finalize()
        except Exception:
            _log.exception("Final apply after least_squares failed")

        self._publish_current_state(enabled_goals)
        self.previewUpdated.emit()
        self.finished.emit(RunResult(
            status="ok",
            message=str(result.message),
            total=self._compute_result_total(result),
            iterations=int(result.nfev),
        ))

    def _run_scipy(
        self,
        enabled_goals: list[GoalEntry],
        *,
        max_nfev: int,
        emit_progress: bool,
    ):
        """One scipy.optimize.least_squares invocation.

        Factored out of :meth:`_run_standard` so the hammer's inner loop
        can call the exact same code — same convergence knobs, same
        residuals contract, same cancel handshake. ``emit_progress`` is
        False on hammer sub-runs so a 200-candidate sweep doesn't emit
        200 × N iteration signals; the hammer emits its own coarse ticks.
        """
        from scipy.optimize import least_squares

        # If nothing is geometrically variable (material-only run inside
        # the hammer, for instance), least_squares refuses an empty x0.
        # Just evaluate residuals once and return a shim result.
        if not self._vars:
            residuals_out = self._compute_residuals(enabled_goals, [])
            return _NoGeomResult(
                x=[],
                fun=residuals_out,
                nfev=1,
                message="No geometric variables — evaluated at current state.",
            )

        n_pen = n_geometry_residuals(self._sys)
        fallback_len = len(enabled_goals) + n_pen

        def residuals(x):
            if self._cancel.is_set():
                raise _Cancelled()
            apply_variables_scipy(
                self._vars, self._sys, list(x), self._is_curvature,
            )
            try:
                self._sys.finalize()
            except Exception:
                # Length must match the healthy path — scipy fixes the
                # residuals vector shape on the first call.
                return [1.0e6] * fallback_len
            out = self._compute_residuals(enabled_goals, list(x), applied=True)
            total = float(sum(r * r for r in out))
            if total < self._best_total:
                self._best_total = total
                self._iter_count += 1
                if emit_progress:
                    self.iterationCompleted.emit(self._iter_count, total)
                    self._publish_current_state(enabled_goals)
                    self.previewUpdated.emit()
            return out

        return least_squares(
            residuals,
            self._x0,
            bounds=self._bounds,
            method="trf",
            max_nfev=int(max_nfev),
            ftol=float(self._mf.f_tol),
            # Ghostlight's ray tracer runs at float32. scipy's default
            # finite-difference step (~1.5e-8, sized for float64) is
            # smaller than the tracer's numerical precision — see
            # feedback-scipy-diff-step. 1e-4 (relative) is well above the
            # float32 noise floor while staying deep enough in the linear
            # regime that Jacobian estimates match reality.
            diff_step=1e-4,
        )

    def _compute_residuals(
        self,
        enabled_goals: list[GoalEntry],
        x: list[float],
        *,
        applied: bool = False,
    ) -> list[float]:
        """Evaluate goal residuals + append geometry-validity penalties.

        ``applied`` skips the redundant apply+finalize when the caller
        has just done it (residuals inside least_squares) — saves one
        finalize per iteration in the hot loop.

        Geometry penalties come from
        :func:`.geometry_penalties.evaluate_geometry_penalties` and steer
        scipy out of invalid states (negative thickness, aperture >
        radius). They contribute 0 to a healthy lens and grow linearly
        with the violation size, so scipy's squared residuals sum turns
        the violation into a quadratic bowl.
        """
        if not applied:
            if self._vars and x:
                # ``x`` may be scipy-space (curvatures for radii). Route
                # through the aware writer.
                apply_variables_scipy(
                    self._vars, self._sys, list(x), self._is_curvature,
                )
            try:
                self._sys.finalize()
            except Exception:
                return [1.0e6] * (
                    len(enabled_goals) + n_geometry_residuals(self._sys)
                )
        out: list[float] = []
        for g in enabled_goals:
            ev = GOAL_REGISTRY.get(g.kind)
            if ev is None:
                out.append(0.0)
                continue
            v = safe_evaluate(ev, self._sys, self._setup, g.params)
            out.append(g.weight * (v - g.target))
        # Geometry validity barriers — zero when valid, positive when
        # scipy tries to push us into an unphysical state.
        out.extend(evaluate_geometry_penalties(self._sys))
        return out

    def _compute_result_total(self, result) -> float:
        try:
            return float(sum(float(r) * float(r) for r in result.fun))
        except TypeError:
            return (
                float(self._best_total)
                if math.isfinite(self._best_total) else 0.0
            )

    # ------------------------------------------------------------------
    # Hammer path — sequential-greedy discrete search over materials
    # ------------------------------------------------------------------

    def _run_hammer(self, enabled_goals: list[GoalEntry]) -> None:
        """Sequential-greedy catalogue-hammer.

        Per outer round, per flag: enumerate candidate glasses from that
        flag's vendor (restricted by its spec bounds), install each,
        run a short scipy sub-optimization over the geometric variables,
        and keep the best-scoring glass + geometry for that flag before
        moving on to the next flag. Stop early when a full sweep leaves
        every flag with the same choice it started with (no improvement).

        The inner scipy call is exactly the standard path with a small
        ``max_nfev`` budget; ranking candidates by best-attainable total
        doesn't need full convergence per candidate.
        """
        catalogue = get_catalogue()

        # Baseline: whatever the virtual system looks like right now.
        baseline_glasses = [f.current_key for f in self._material_flags]
        baseline_geom = self._snapshot_variables()
        baseline_total = self._score_current(enabled_goals)
        best_total = baseline_total
        best_glasses = list(baseline_glasses)
        best_geom = list(baseline_geom)
        # Ensure the virtual system reflects the current best so
        # dialog viewport + snapshots stay coherent between rounds.
        self._restore_variables(best_geom)
        try:
            self._sys.finalize()
        except Exception:
            pass
        self._best_total = min(self._best_total, best_total)

        total_rounds = max(1, int(self._mf.max_hammer_rounds))
        # The whole hammer loop is wrapped in a _Cancelled catch so a
        # cancel mid-sub-run finishes cleanly as "cancelled" instead of
        # crashing out through _Worker.run's generic BaseException catcher
        # (which would report it as "failed"). The catch restores the
        # virtual system to the last-known-best state before finishing so
        # the dialog viewport shows something coherent on cancel.
        try:
            for round_idx in range(1, total_rounds + 1):
                round_improved = False
                for flag_idx, flag in enumerate(self._material_flags):
                    if self._cancel.is_set():
                        raise _Cancelled()

                    candidates = candidates_within_spec(catalogue, flag.spec)
                    if not candidates:
                        self.hammerProgress.emit(HammerProgress(
                            round_index=round_idx,
                            total_rounds=total_rounds,
                            flag_label=self._flag_label(flag),
                            candidate_index=0,
                            total_candidates=0,
                            glass_key="",
                            best_total=best_total if math.isfinite(best_total) else None,
                            candidate_total=None,
                        ))
                        continue

                    # Reset every OTHER flag to whatever "best so far" holds
                    # so ranking measures this flag's marginal contribution
                    # against the best-known configuration.
                    self._install_glass_choices(best_glasses)
                    self._restore_variables(best_geom)

                    best_glass_for_flag = best_glasses[flag_idx]
                    best_geom_for_flag = list(best_geom)
                    best_total_for_flag = best_total

                    for cand_idx, glass in enumerate(candidates, start=1):
                        if self._cancel.is_set():
                            raise _Cancelled()
                        # Install this candidate + reset geometry to the
                        # current best before the sub-optimization runs;
                        # every candidate starts from the same launch pad.
                        install_candidate_glass(self._sys, flag, glass)
                        self._restore_variables(best_geom)
                        try:
                            self._sys.finalize()
                        except Exception:
                            continue

                        total, geom_after = self._run_hammer_sub(enabled_goals)
                        self.hammerProgress.emit(HammerProgress(
                            round_index=round_idx,
                            total_rounds=total_rounds,
                            flag_label=self._flag_label(flag),
                            candidate_index=cand_idx,
                            total_candidates=len(candidates),
                            glass_key=str(glass.key),
                            best_total=(
                                best_total if math.isfinite(best_total) else None
                            ),
                            candidate_total=total,
                        ))
                        # Publish per-candidate so the dialog viewport +
                        # goal/variable tables reflect the sweep. The
                        # dialog throttles its viewport rebuild to 100 ms
                        # so this is safe to fire on every candidate; the
                        # tables just re-populate cheap QTableWidgetItems.
                        self._publish_current_state(enabled_goals)
                        self.previewUpdated.emit()
                        if total is None or not math.isfinite(total):
                            continue
                        if total < best_total_for_flag - 1e-12:
                            best_total_for_flag = total
                            best_glass_for_flag = str(glass.key)
                            best_geom_for_flag = geom_after

                    # Commit this flag's best pick + geometry.
                    if best_glass_for_flag != best_glasses[flag_idx]:
                        best_glasses[flag_idx] = best_glass_for_flag
                        round_improved = True
                    if best_total_for_flag < best_total - 1e-12:
                        best_total = best_total_for_flag
                        best_geom = list(best_geom_for_flag)
                        round_improved = True
                        self._best_total = min(self._best_total, best_total)
                        self._iter_count += 1
                        self.iterationCompleted.emit(self._iter_count, best_total)
                    # Re-install the winning combo so the next flag's search
                    # starts from the confirmed best state.
                    self._install_glass_choices(best_glasses)
                    self._restore_variables(best_geom)
                    try:
                        self._sys.finalize()
                    except Exception:
                        pass
                    self._publish_current_state(enabled_goals)
                    self.previewUpdated.emit()

                if not round_improved:
                    # Sweep produced no change — further rounds would be
                    # deterministic no-ops. Stop early.
                    break
        except _Cancelled:
            # Restore to the last-known-best combo so the viewport shows
            # a coherent state instead of whatever half-tried candidate
            # was current when cancel hit.
            self._install_glass_choices(best_glasses)
            self._restore_variables(best_geom)
            try:
                self._sys.finalize()
            except Exception:
                pass
            self._publish_current_state(enabled_goals)
            self.previewUpdated.emit()
            self._finish_hammer(
                best_total, "cancelled", "Cancelled by user.",
            )
            return

        # Final commit: leave the virtual system at the winning combo.
        self._install_glass_choices(best_glasses)
        self._restore_variables(best_geom)
        try:
            self._sys.finalize()
        except Exception:
            pass
        self._publish_current_state(enabled_goals)
        self.previewUpdated.emit()

        summary_bits = [f"best total {best_total:.6g}"]
        summary_bits.append(
            "no improvement" if best_total >= baseline_total - 1e-12
            else f"improved from {baseline_total:.6g}"
        )
        self._finish_hammer(best_total, "ok", "; ".join(summary_bits))

    def _run_hammer_sub(
        self, enabled_goals: list[GoalEntry],
    ) -> tuple[Optional[float], list[float]]:
        """Inner sub-optimization for one candidate glass.

        Returns (total, geometry-after). ``total`` is None when the
        sub-run raised. Silently swallowed here (returned as None) so
        one flaky candidate doesn't kill the whole hammer.
        """
        try:
            result = self._run_scipy(
                enabled_goals,
                max_nfev=int(self._mf.hammer_sub_max_nfev),
                emit_progress=False,
            )
        except _Cancelled:
            raise
        except Exception:
            _log.exception("Hammer sub-run raised")
            return (None, self._snapshot_variables())
        total = self._compute_result_total(result)
        # Apply the winning x from the sub-run to the system so
        # snapshot_variables captures the post-optimization state.
        # ``result.x`` is scipy-space (curvatures for radius vars) —
        # must go through the aware writer.
        try:
            if self._vars and hasattr(result, "x") and len(result.x) > 0:
                apply_variables_scipy(
                    self._vars, self._sys, list(result.x), self._is_curvature,
                )
                self._sys.finalize()
        except Exception:
            pass
        return (total, self._snapshot_variables())

    def _finish_hammer(
        self, best_total: float, status: str, message: str,
    ) -> None:
        self.finished.emit(RunResult(
            status=status,
            message=message,
            total=best_total if math.isfinite(best_total) else None,
            iterations=self._iter_count,
        ))

    # ------------------------------------------------------------------
    # Hammer state helpers
    # ------------------------------------------------------------------

    def _snapshot_variables(self) -> list[float]:
        """Read the current values of every geometric variable."""
        out: list[float] = []
        for v in self._vars:
            try:
                out.append(v.read(self._sys))
            except Exception:
                out.append(0.0)
        return out

    def _restore_variables(self, values: list[float]) -> None:
        if not values or len(values) != len(self._vars):
            return
        try:
            apply_variables(self._vars, self._sys, values)
        except Exception:
            pass

    def _score_current(self, enabled_goals: list[GoalEntry]) -> float:
        try:
            self._sys.finalize()
        except Exception:
            return math.inf
        residuals = self._compute_residuals(enabled_goals, [], applied=True)
        return float(sum(r * r for r in residuals))

    def _install_glass_choices(self, glass_keys: list[str]) -> None:
        """Restore each material flag to a specific catalogue key.

        Used between candidate trials to reset "other" flags to the
        current best. Skips flags whose target key isn't in the bundled
        catalogue (shouldn't happen — every key we install here was
        already yielded by :func:`candidates_within_spec` — but be safe).
        """
        if len(glass_keys) != len(self._material_flags):
            return
        catalogue = get_catalogue()
        for flag, key in zip(self._material_flags, glass_keys):
            if not key:
                continue
            glass = catalogue.by_key(key)
            if glass is None or glass.nd is None or glass.vd is None:
                continue
            install_candidate_glass(self._sys, flag, glass)

    def _flag_label(self, flag: MaterialFlagRef) -> str:
        # Resolve to an element index if we can; falls back to element_id
        # when the element isn't reachable (which shouldn't happen).
        try:
            for i, el in enumerate(self._sys.elements):
                if getattr(el, "element_id", None) == flag.element_id:
                    return f"Element {i} · material {flag.material_index}"
        except Exception:
            pass
        return f"{flag.element_id[:8]} · material {flag.material_index}"

    def _publish_current_state(self, enabled_goals: list[GoalEntry]) -> None:
        """Compute every enabled goal once + emit snapshots for the dialog.

        Called sparingly (only on new-best, on entry, on exit) — each
        invocation re-runs every evaluator, so it's the same cost as one
        residuals iteration.
        """
        snaps: list[GoalSnapshot] = []
        for g in enabled_goals:
            ev = GOAL_REGISTRY.get(g.kind)
            if ev is None:
                snaps.append(GoalSnapshot(g.goal_id, None, None))
                continue
            try:
                v = safe_evaluate(ev, self._sys, self._setup, g.params)
            except Exception:
                snaps.append(GoalSnapshot(g.goal_id, None, None))
                continue
            res = g.weight * (v - g.target)
            snaps.append(GoalSnapshot(g.goal_id, float(v), float(res)))
        self.goalsRecomputed.emit(snaps)

        # Variables snapshot — always in display space (R for radius vars)
        # so the dialog table matches what the user sees in the ODE.
        var_snaps: list[VariableSnapshot] = []
        for v, start in zip(self._vars, self._display_starts):
            try:
                cur = v.read(self._sys)
            except Exception:
                cur = float("nan")
            var_snaps.append(VariableSnapshot(
                surface_index=v.surface_index,
                attr=v.attr,
                start=float(start),
                current=float(cur),
            ))
        self.variablesRecomputed.emit(var_snaps)


class _Cancelled(Exception):
    """Internal sentinel — distinguishes user-cancel from real failures."""
    pass


@dataclass
class _NoGeomResult:
    """Shim result for :meth:`_Worker._run_scipy` when no geometric
    variables exist. Mirrors just the attributes the callers read
    off a real ``scipy.optimize.OptimizeResult``.
    """
    x: list = field(default_factory=list)
    fun: list = field(default_factory=list)
    nfev: int = 1
    message: str = ""


# ---------------------------------------------------------------------------
# Public-facing run object
# ---------------------------------------------------------------------------


class OptimizationRun(QObject):
    """One optimization invocation.

    Owns:
      * ``virtual_system`` — the cloned lens being tuned (panel-visible).
      * ``variables`` — the design variables snapshotted at construction.
      * the worker thread.
    """

    iterationCompleted = Signal(int, float)
    previewUpdated = Signal()
    goalsRecomputed = Signal(list)
    variablesRecomputed = Signal(list)
    hammerProgress = Signal(object)               # HammerProgress
    runFinished = Signal(object)                  # RunResult

    def __init__(
        self,
        project,
        mf: MeritFunction,
        setup,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._mf = mf
        self._setup = setup
        # Variables live on the Project (keyed by surface UUID), so we
        # read them from the source project BEFORE cloning. The clone's
        # surface UUIDs are the same as the source's (writer preserves
        # them) but the flags themselves aren't attached to the clone;
        # keeping the read seam on the project side is cleaner.
        self.variables: list[VariableRef] = collect_variables(project)
        self.material_flags: list[MaterialFlagRef] = collect_material_flags(project)
        # Clone once at construction so we can hand a stable preview
        # to the dialog even before start() is called.
        self.virtual_system: ghostlight.OpticalSystem = clone_system(project.system)
        self._cancel = threading.Event()
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._finished: bool = False
        self._result: Optional[RunResult] = None

    @property
    def merit_function(self) -> MeritFunction:
        return self._mf

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def result(self) -> Optional[RunResult]:
        return self._result

    def start(self) -> None:
        """Launch the worker. Returns immediately.

        If there are no variables flagged we short-circuit to a
        :class:`RunResult` of ``no_variables`` without starting a thread.
        """
        if self._thread is not None:
            return  # idempotent
        if not self.variables and not self.material_flags:
            QTimer.singleShot(
                0, lambda: self._publish_finished(RunResult(
                    status="no_variables",
                    message=(
                        "No design variables are flagged on the lens. Open the "
                        "Optical Design Editor and mark at least one surface "
                        "attribute as variable — or flag a material for "
                        "catalogue substitution — before optimizing."
                    ),
                )),
            )
            return

        try:
            x0, lb, ub, is_curvature = pack_bounds_scipy(
                self.variables,
                self.virtual_system,
                preserve_radius_signs=bool(
                    getattr(self._mf, "preserve_radius_signs", False)
                ),
            )
        except Exception as exc:
            QTimer.singleShot(
                0, lambda: self._publish_finished(RunResult(
                    status="failed",
                    message=f"Failed to pack variable bounds: {exc}",
                )),
            )
            return

        self._thread = QThread(self)
        self._worker = _Worker(
            virtual_system=self.virtual_system,
            mf=self._mf,
            setup=self._setup,
            variables=self.variables,
            starting_values=x0,
            bounds=(lb, ub),
            is_curvature=is_curvature,
            material_flags=self.material_flags,
            cancel_flag=self._cancel,
        )
        self._worker.moveToThread(self._thread)
        # Forward worker signals through the run so callers don't have
        # to know about the inner QObject.
        self._worker.iterationCompleted.connect(self.iterationCompleted)
        self._worker.previewUpdated.connect(self.previewUpdated)
        self._worker.goalsRecomputed.connect(self.goalsRecomputed)
        self._worker.variablesRecomputed.connect(self.variablesRecomputed)
        self._worker.hammerProgress.connect(self.hammerProgress)
        self._worker.finished.connect(self._on_worker_finished)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def request_cancel(self) -> None:
        """Ask the worker to abort at the next residuals checkpoint.

        Safe to call multiple times. Has no effect after the run has
        already finished.
        """
        self._cancel.set()

    def _on_worker_finished(self, result: RunResult) -> None:
        self._publish_finished(result)

    def _publish_finished(self, result: RunResult) -> None:
        if self._finished:
            return
        self._finished = True
        self._result = result
        # Tear down the worker thread cleanly. quit() + wait() must run
        # on the GUI thread (where we are now); the worker has finished
        # its run() method by the time _on_worker_finished is invoked.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None
        self.runFinished.emit(result)
