"""Spot RMS goal evaluator.

Mirrors :class:`ghostlight_designer.evaluation_panels.spot_diagram.compute`
but condensed to a single scalar: the RMS distance of every traced ray
landing from its field's centroid on the image plane.

Sampling defaults are deliberately modest (8 rings × 12 fans = 97 rays
per (field, λ) pair) so a typical residuals evaluation stays in the
~milliseconds range. The user can tune in a later iteration once we
have a sense of typical merit-function sizes in practice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import ghostlight

from ... import tracing_util as tu
from ..data import GoalKind
from .base import Evaluator, ParamDef, register_evaluator


# Pupil sampling for the spot RMS evaluator. 8 × 12 rings/fans is enough
# to give a stable RMS for the routine field/wavelength selections without
# breaking the bank on per-step trace cost.
_RINGS = 8
_FANS = 12


@dataclass
class _SpotRmsEvaluator:
    kind: GoalKind = GoalKind.SPOT_RMS
    display_name: str = "Spot RMS"
    default_target: float = 0.0
    param_schema: tuple[ParamDef, ...] = field(
        default_factory=lambda: (
            ParamDef("wavelength", "Wavelength", "wavelength_pick", "primary"),
            ParamDef("field", "Field", "field_pick", "all"),
        )
    )

    def evaluate(
        self,
        system: ghostlight.OpticalSystem,
        setup,
        params: dict[str, Any],
    ) -> float:
        wavelengths = tu.resolve_wavelengths(setup, params.get("wavelength", "primary"))
        fields = tu.resolve_fields(setup, params.get("field", "all"))
        if not wavelengths or not fields:
            raise ValueError("Spot RMS: empty wavelength or field set")

        pupil_r = tu.resolve_pupil_radius(system)
        samples = list(tu.ring_fan_samples(_RINGS, _FANS))
        if not samples:
            raise ValueError("Spot RMS: no pupil samples generated")

        # We accumulate sum-of-squares across every (field) — each field
        # has its own centroid, so the RMS is "average per-field RMS
        # squared", which is what the textbook Spot Radius RMS reports
        # when "field=all" is selected.
        total_sq = 0.0
        total_pts = 0

        for (tx, ty) in fields:
            direction = tu.direction_for_field(tx, ty)
            xs: List[float] = []
            ys: List[float] = []
            for lam in wavelengths:
                for (u, v) in samples:
                    path = tu.trace_collimated_ray(
                        system,
                        pupil_xy_mm=(u * pupil_r, v * pupil_r),
                        direction=direction,
                        wavelength_nm=lam,
                    )
                    xy = tu.landing_xy(path)
                    if xy is None:
                        continue
                    xs.append(xy[0])
                    ys.append(xy[1])
            if not xs:
                # Whole field vignetted — let safe_evaluate raise into
                # the penalty so the optimizer pushes away from here.
                raise ValueError("Spot RMS: every ray vignetted for a field")
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            for x, y in zip(xs, ys):
                dx = x - mx
                dy = y - my
                total_sq += dx * dx + dy * dy
            total_pts += len(xs)

        if total_pts == 0:
            raise ValueError("Spot RMS: no rays survived")
        return math.sqrt(total_sq / total_pts)


register_evaluator(_SpotRmsEvaluator())
