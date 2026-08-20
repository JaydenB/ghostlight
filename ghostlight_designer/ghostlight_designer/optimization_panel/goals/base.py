"""Evaluator protocol + central registry for optimization goals.

An :class:`Evaluator` is the per-:class:`GoalKind` plug-in that:

* Tells the tree which parameters this goal exposes (via :attr:`param_schema`).
* Computes a scalar value from a system + setup + a params dict.

The optimizer drives evaluators; the tree displays their schema. Adding a
new goal kind is: subclass :class:`Evaluator`, register it in
:func:`register_evaluator`, add an enum member in
:class:`ghostlight_designer.optimization_panel.data.GoalKind`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import ghostlight

from ..data import GoalKind


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamDef:
    """Declares one knob on a goal — drives a slot in the tree.

    ``kind`` is the editor type the row uses:

    * ``"wavelength_pick"`` — combo of "Primary" / "All" / one entry per
      wavelength in the project's first sequence. Stored as ``"primary"``,
      ``"all"``, or an int index.
    * ``"field_pick"`` — combo of "All" + each field. Stored as ``"all"``
      or int index.
    * ``"surface_pick"`` — combo of every surface in the lens. Stored as
      int.
    * ``"axis"`` — combo of "Radial" / "X" / "Y". Stored as the string.

    ``default`` is the value the tree seeds for newly-created goals.
    """
    name: str
    label: str
    kind: str
    default: Any = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator(Protocol):
    """One goal kind's compute + UI metadata.

    Implementations live in :mod:`.first_order`, :mod:`.spot`, and
    :mod:`.field_distortion`. Every Evaluator has the same shape — see
    those modules for examples.
    """

    kind: GoalKind
    display_name: str
    default_target: float
    # Per-knob schema. May be empty (e.g. on-axis EFL has no parameters).
    param_schema: tuple[ParamDef, ...]

    def evaluate(
        self,
        system: ghostlight.OpticalSystem,
        setup,
        params: dict[str, Any],
    ) -> float:
        """Return the scalar value for this goal.

        Pure CPU; runs in the optimizer's worker thread (no Qt). Should
        return a finite float; non-finite values are translated into a
        large penalty by the caller so least_squares doesn't crash.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


GOAL_REGISTRY: dict[GoalKind, Evaluator] = {}


def register_evaluator(evaluator: Evaluator) -> Evaluator:
    """Register ``evaluator`` under its ``kind``.

    Idempotent on identical re-registration (used as a decorator in the
    submodules); raises on a conflicting double-registration.
    """
    existing = GOAL_REGISTRY.get(evaluator.kind)
    if existing is not None and existing is not evaluator:
        raise RuntimeError(
            f"Goal kind {evaluator.kind!r} already registered by {type(existing).__name__}"
        )
    GOAL_REGISTRY[evaluator.kind] = evaluator
    return evaluator


def evaluator_for(kind: GoalKind) -> Optional[Evaluator]:
    return GOAL_REGISTRY.get(kind)


def param_schema_for(kind: GoalKind) -> tuple[ParamDef, ...]:
    ev = GOAL_REGISTRY.get(kind)
    return () if ev is None else tuple(ev.param_schema)


def default_target_for(kind: GoalKind) -> float:
    ev = GOAL_REGISTRY.get(kind)
    return 0.0 if ev is None else float(ev.default_target)


def display_name_for(kind: GoalKind) -> str:
    ev = GOAL_REGISTRY.get(kind)
    return kind.name if ev is None else ev.display_name


def default_params_for(kind: GoalKind) -> dict[str, Any]:
    """Build the params dict a freshly-created goal of ``kind`` should carry."""
    return {p.name: p.default for p in param_schema_for(kind)}


# ---------------------------------------------------------------------------
# Safe-evaluate wrapper used by the optimizer
# ---------------------------------------------------------------------------


# Penalty value returned when an evaluator raises or produces a non-finite
# number. Large enough that least_squares treats the step as bad without
# being so extreme it overwhelms the Jacobian's finite-difference scaling.
_EVAL_PENALTY = 1.0e6


def safe_evaluate(
    evaluator: Evaluator,
    system: ghostlight.OpticalSystem,
    setup,
    params: dict[str, Any],
) -> float:
    """Run ``evaluator.evaluate`` returning :data:`_EVAL_PENALTY` on failure.

    Used inside the optimizer's residuals function — a single
    pathological lens step (rays vignetted, divide-by-zero in a paraxial
    trace, etc.) shouldn't crash the whole run. The penalty is multiplied
    by the goal's weight upstream, so a normally-weighted goal contributes
    a residual of ~ weight × 1e6 on failure, which least_squares will
    interpret as "step away from here".
    """
    import math
    try:
        v = evaluator.evaluate(system, setup, params)
    except Exception:
        return _EVAL_PENALTY
    if v is None:
        return _EVAL_PENALTY
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _EVAL_PENALTY
    if not math.isfinite(v):
        return _EVAL_PENALTY
    return v
