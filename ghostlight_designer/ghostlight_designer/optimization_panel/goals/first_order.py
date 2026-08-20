"""First-order goal evaluators: Effective Focal Length and F-number.

Both lean on :mod:`ghostlight_designer.lens_metrics` for the actual EFL
calculation — that module already runs a paraxial-probe trace at d-line,
so a second implementation here would be drift bait.

F-number is the working effective F# for an object at infinity:
``F# = EFL / (2 · entrance_pupil_radius)``. We take the entrance pupil
radius from the system's stop surface when one is flagged, otherwise from
the front surface's semi-aperture — matches the convention the spot
diagram uses for its pupil-sampling fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import ghostlight

from ... import lens_metrics as lm
from ..data import GoalKind
from .base import Evaluator, ParamDef, register_evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entrance_pupil_radius(system: ghostlight.OpticalSystem) -> Optional[float]:
    """Best-effort entrance pupil radius (mm).

    Order of preference:
      1. ``is_stop`` surface's semi-aperture (matches what the tracer uses
         as the limiting aperture).
      2. Front surface's semi-aperture (good fallback for systems that
         don't explicitly tag a stop).

    Returns ``None`` when neither is positive (degenerate / blank lens).
    """
    try:
        for surf in system.surfaces:
            if bool(surf.is_stop) and float(surf.semi_aperture) > 0.0:
                return float(surf.semi_aperture)
        if len(system.surfaces) > 0:
            r = float(system.surfaces[0].semi_aperture)
            if r > 0.0:
                return r
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# EFL — scalar (Y-axis, lens_metrics convention)
# ---------------------------------------------------------------------------


@dataclass
class _EflEvaluator:
    kind: GoalKind = GoalKind.EFL
    display_name: str = "Effective Focal Length"
    default_target: float = 50.0
    param_schema: tuple[ParamDef, ...] = ()

    def evaluate(
        self,
        system: ghostlight.OpticalSystem,
        _setup,
        _params: dict[str, Any],
    ) -> float:
        efl = lm._effective_focal_length_on_axis(system, "y")
        if efl is None:
            raise ValueError("EFL probe failed")
        return float(efl)


@dataclass
class _EflXEvaluator:
    kind: GoalKind = GoalKind.EFL_X
    display_name: str = "EFL (X axis)"
    default_target: float = 50.0
    param_schema: tuple[ParamDef, ...] = ()

    def evaluate(self, system, _setup, _params):
        efl = lm._effective_focal_length_on_axis(system, "x")
        if efl is None:
            raise ValueError("EFL_X probe failed")
        return float(efl)


@dataclass
class _EflYEvaluator:
    kind: GoalKind = GoalKind.EFL_Y
    display_name: str = "EFL (Y axis)"
    default_target: float = 50.0
    param_schema: tuple[ParamDef, ...] = ()

    def evaluate(self, system, _setup, _params):
        efl = lm._effective_focal_length_on_axis(system, "y")
        if efl is None:
            raise ValueError("EFL_Y probe failed")
        return float(efl)


# ---------------------------------------------------------------------------
# Effective F#
# ---------------------------------------------------------------------------


@dataclass
class _SqueezeRatioEvaluator:
    kind: GoalKind = GoalKind.SQUEEZE_RATIO
    display_name: str = "Squeeze Ratio (EFL_y / EFL_x)"
    default_target: float = 2.0
    param_schema: tuple[ParamDef, ...] = ()

    def evaluate(self, system, _setup, _params):
        efl_x = lm._effective_focal_length_on_axis(system, "x")
        efl_y = lm._effective_focal_length_on_axis(system, "y")
        if efl_x is None or efl_y is None:
            raise ValueError("Squeeze ratio probe failed")
        if abs(efl_x) < 1e-9:
            raise ValueError("Squeeze ratio: EFL_x degenerate")
        return float(efl_y) / float(efl_x)


@dataclass
class _FNumberEvaluator:
    kind: GoalKind = GoalKind.F_NUMBER
    display_name: str = "F-number"
    default_target: float = 2.8
    param_schema: tuple[ParamDef, ...] = ()

    def evaluate(self, system, _setup, _params):
        efl = lm._effective_focal_length_on_axis(system, "y")
        if efl is None or efl <= 0.0:
            raise ValueError("F# requires positive EFL")
        r = _entrance_pupil_radius(system)
        if r is None or r <= 0.0:
            raise ValueError("F# requires a positive entrance pupil radius")
        diameter = 2.0 * r
        if diameter <= 0.0 or not math.isfinite(diameter):
            raise ValueError("F# entrance pupil diameter degenerate")
        return float(efl) / diameter


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


register_evaluator(_EflEvaluator())
register_evaluator(_EflXEvaluator())
register_evaluator(_EflYEvaluator())
register_evaluator(_SqueezeRatioEvaluator())
register_evaluator(_FNumberEvaluator())
