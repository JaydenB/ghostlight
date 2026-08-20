"""Dataclasses for the optimization panel.

Lifecycle:

* :class:`MeritFunction` and its :class:`GoalEntry` rows live on
  ``Project.merit_functions`` (in-memory, not persisted to ``.lens``).
* ``cached_value`` / ``cached_residual`` are display-only refreshes the panel
  populates after each evaluation pass — never persisted, never feed back
  into the optimizer (the optimizer recomputes inside its residuals fn).
* Variables are NOT stored here — they're flagged on individual surfaces
  in the Optical Design Editor and read at run time through
  :mod:`ghostlight_designer.optimization_panel.variables`.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def _new_id() -> str:
    """Short opaque id for merit functions and goals.

    Just needs to be locally unique within a single Project — short to keep
    debug logs readable, not collision-proof across machines.
    """
    return secrets.token_hex(4)


# ---------------------------------------------------------------------------
# Goal kinds
# ---------------------------------------------------------------------------


class GoalKind(str, Enum):
    """Built-in goal evaluators: Spot RMS, EFL, F#, Field Curvature and
    Distortion."""
    SPOT_RMS = "spot_rms"
    EFL = "efl"            # scalar (= efl_y by lens_metrics convention)
    EFL_X = "efl_x"
    EFL_Y = "efl_y"
    F_NUMBER = "f_number"
    FIELD_CURVATURE = "field_curvature"
    DISTORTION = "distortion"
    # efl_y / efl_x — the anamorphic squeeze ratio driven by the "Add
    # Anamorphic Front Block" wizard, but also usable standalone.
    SQUEEZE_RATIO = "squeeze_ratio"


# ---------------------------------------------------------------------------
# Goal entry
# ---------------------------------------------------------------------------


@dataclass
class GoalEntry:
    """One row in a merit function.

    ``params`` is a per-kind dict — its keys are declared by each
    evaluator's ``param_schema`` (see :mod:`.goals.base`). Examples:

        Spot RMS:   {"wavelength": "primary" | "all" | int, "field": "all" | int}
        EFL:        {}  (always on-axis, primary wavelength)
        Distortion: {"wavelength": ..., "axis": "Radial" | "X" | "Y"}
    """

    goal_id: str
    kind: GoalKind
    name: str = ""
    target: float = 0.0
    weight: float = 1.0
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    # Display caches — never read by the optimizer.
    cached_value: Optional[float] = None
    cached_residual: Optional[float] = None

    @staticmethod
    def make(kind: GoalKind, **overrides) -> "GoalEntry":
        """Convenience builder: fresh id + the supplied overrides."""
        entry = GoalEntry(goal_id=_new_id(), kind=kind)
        for k, v in overrides.items():
            if not hasattr(entry, k):
                raise TypeError(f"GoalEntry has no attribute {k!r}")
            setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Merit function
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of one optimization run.

    Display-only — written into MeritFunction.last_* on completion and
    surfaced in the tree's status row.
    """
    status: str           # "ok" | "cancelled" | "no_variables" | "failed" | etc.
    message: str = ""
    total: Optional[float] = None
    iterations: Optional[int] = None


@dataclass
class MeritFunction:
    """A named bundle of goals targeted by one optimization pass.

    ``max_iters`` / ``f_tol`` are passed straight to scipy.optimize.least_squares.
    ``80`` / ``1e-8`` mirror the scipy defaults rounded to numbers a UI
    can show without exponents; tuneable per-MF in the body's settings later.

    Hammer knobs: ``max_hammer_rounds`` caps the outer sequential-greedy
    sweep count when at least one material-substitution flag is set on
    the project. ``hammer_sub_max_nfev`` caps the inner scipy budget
    spent per candidate glass trial — small numbers (20-40) are plenty
    because the hammer only needs to rank candidates by best-attainable
    total, not converge each one to full precision. Neither field
    matters when no material flag is set (standard scipy path).
    """

    mf_id: str
    name: str = "Merit Function"
    enabled: bool = True
    goals: list[GoalEntry] = field(default_factory=list)

    max_iters: int = 80
    f_tol: float = 1e-8

    max_hammer_rounds: int = 3
    hammer_sub_max_nfev: int = 30

    # When True, radius variables can flatten but never change sign
    # during optimization. The scipy step is bounded on the same
    # half-line as the starting curvature (excluding zero), so scipy can
    # push R toward ±∞ (flat) but not cross to the opposite sign. Useful
    # when the user wants "tune the shape, don't restructure my design".
    preserve_radius_signs: bool = False

    # Display caches from the last run.
    last_total: Optional[float] = None
    last_iters: Optional[int] = None
    last_status: Optional[str] = None
    last_message: str = ""

    @staticmethod
    def make(name: str = "Merit Function", **overrides) -> "MeritFunction":
        mf = MeritFunction(mf_id=_new_id(), name=name)
        for k, v in overrides.items():
            if not hasattr(mf, k):
                raise TypeError(f"MeritFunction has no attribute {k!r}")
            setattr(mf, k, v)
        return mf

    def apply_run_result(self, result: RunResult) -> None:
        """Copy the run outcome onto this MF's display caches."""
        self.last_total = result.total
        self.last_iters = result.iterations
        self.last_status = result.status
        self.last_message = result.message
