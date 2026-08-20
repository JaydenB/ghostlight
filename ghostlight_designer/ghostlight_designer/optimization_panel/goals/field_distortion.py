"""Field curvature + F-tan-θ distortion goal evaluators.

Both use chief-ray (on-axis pupil) traces for each non-axial field. Field
curvature reports the largest absolute z-deviation of the per-field best-
focus plane from the system's image plane (the last surface). Distortion
reports the maximum percent deviation of chief-ray image height from the
F-tan-θ ideal.

Both are cheaper than fitting a parabola to each field's marginal-ray
bundle, and accurate enough for general-purpose lens-design optimization.
They share the trace infrastructure with Spot RMS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import ghostlight

from ... import lens_metrics as lm
from ... import tracing_util as tu
from ..data import GoalKind
from .base import Evaluator, ParamDef, register_evaluator


# ---------------------------------------------------------------------------
# Field curvature
# ---------------------------------------------------------------------------


@dataclass
class _FieldCurvatureEvaluator:
    """Max |Δz| between each field's chief-ray crossing of the axis and
    the system's image plane.

    Interpretation: when a chief ray for a non-axial field crosses the
    optical axis (or the closest extrapolation to the optical axis) at a
    z that differs from the image plane, the off-axis field is focusing
    at a different depth. The maximum across fields tells us how curved
    the Petzval surface is in the worst case.

    Returns ``0.0`` when only the on-axis field is configured (nothing
    to curve).
    """
    kind: GoalKind = GoalKind.FIELD_CURVATURE
    display_name: str = "Field Curvature"
    default_target: float = 0.0
    param_schema: tuple[ParamDef, ...] = field(
        default_factory=lambda: (
            ParamDef("wavelength", "Wavelength", "wavelength_pick", "primary"),
        )
    )

    def evaluate(self, system: ghostlight.OpticalSystem, setup, params: dict[str, Any]) -> float:
        wavelengths = tu.resolve_wavelengths(
            setup, params.get("wavelength", "primary"),
        )
        if not wavelengths:
            raise ValueError("Field Curvature: no wavelength")
        lam = wavelengths[0]
        sensor_z = tu.sensor_z(system)

        # Pull every configured field so the user's choice in
        # System Setup drives the metric.  Skip pure on-axis (tx==ty==0)
        # entries — those contribute zero deviation by definition.
        fields = tu.resolve_fields(setup, "all")
        offax = [(tx, ty) for (tx, ty) in fields if (tx != 0.0 or ty != 0.0)]
        if not offax:
            return 0.0

        max_dz = 0.0
        any_valid = False
        for (tx, ty) in offax:
            chief = tu.trace_chief_ray(
                system, tilt_x_deg=tx, tilt_y_deg=ty, wavelength_nm=lam,
            )
            if chief is None:
                continue
            events = list(chief.events or [])
            if len(events) < 2:
                continue
            last = events[-1].hit_point
            prev = events[-2].hit_point
            # Linear extrapolation along the final segment to where this
            # chief ray crosses the optical axis. The image plane is at
            # ``sensor_z``; the depth at which the chief ray reaches
            # (x,y) = (0,0) defines this field's effective focal depth.
            lx, ly, lz = float(last.x), float(last.y), float(last.z)
            px, py, pz = float(prev.x), float(prev.y), float(prev.z)
            dz = lz - pz
            if abs(dz) < 1e-9:
                continue
            # Pick whichever axis the field is tilted along to extrapolate;
            # use the maximum-magnitude direction so we never divide by
            # a near-zero slope.
            dxdz = (lx - px) / dz
            dydz = (ly - py) / dz
            if abs(dxdz) > abs(dydz):
                if abs(dxdz) < 1e-12:
                    continue
                z_cross = lz - lx / dxdz
            else:
                if abs(dydz) < 1e-12:
                    continue
                z_cross = lz - ly / dydz
            any_valid = True
            d = abs(float(z_cross) - sensor_z)
            if d > max_dz:
                max_dz = d
        if not any_valid:
            raise ValueError("Field Curvature: no chief rays survived")
        return float(max_dz)


# ---------------------------------------------------------------------------
# Distortion (F-tan-θ)
# ---------------------------------------------------------------------------


@dataclass
class _DistortionEvaluator:
    """Max |actual - ideal| / ideal across fields, in %.

    ``ideal_height = EFL · tan(θ)`` (F-tan-θ); the evaluator measures
    the chief-ray landing height for each off-axis field and returns
    the worst-case percent deviation.

    Returns ``0.0`` when there are no off-axis fields.
    """
    kind: GoalKind = GoalKind.DISTORTION
    display_name: str = "Distortion (F·tan θ)"
    default_target: float = 0.0
    param_schema: tuple[ParamDef, ...] = field(
        default_factory=lambda: (
            ParamDef("wavelength", "Wavelength", "wavelength_pick", "primary"),
            ParamDef("axis", "Axis", "axis", "Radial"),
        )
    )

    def evaluate(self, system: ghostlight.OpticalSystem, setup, params: dict[str, Any]) -> float:
        wavelengths = tu.resolve_wavelengths(
            setup, params.get("wavelength", "primary"),
        )
        if not wavelengths:
            raise ValueError("Distortion: no wavelength")
        lam = wavelengths[0]
        axis = str(params.get("axis", "Radial"))

        efl = lm._effective_focal_length_on_axis(system, "y")
        if efl is None or efl <= 0.0:
            raise ValueError("Distortion: EFL probe failed")

        fields = tu.resolve_fields(setup, "all")
        offax = [(tx, ty) for (tx, ty) in fields if (tx != 0.0 or ty != 0.0)]
        if not offax:
            return 0.0

        max_pct = 0.0
        any_valid = False
        for (tx, ty) in offax:
            theta = math.radians(math.hypot(tx, ty))
            if axis == "X":
                theta = math.radians(abs(tx))
            elif axis == "Y":
                theta = math.radians(abs(ty))
            if theta <= 0.0:
                continue
            ideal = efl * math.tan(theta)
            if ideal <= 0.0 or not math.isfinite(ideal):
                continue
            chief = tu.trace_chief_ray(
                system, tilt_x_deg=tx, tilt_y_deg=ty, wavelength_nm=lam,
            )
            xy = tu.landing_xy(chief)
            if xy is None:
                continue
            x, y = xy
            if axis == "X":
                actual = abs(float(x))
            elif axis == "Y":
                actual = abs(float(y))
            else:
                actual = math.hypot(float(x), float(y))
            any_valid = True
            pct = 100.0 * abs(actual - ideal) / ideal
            if pct > max_pct:
                max_pct = pct
        if not any_valid:
            raise ValueError("Distortion: no chief rays survived")
        return float(max_pct)


register_evaluator(_FieldCurvatureEvaluator())
register_evaluator(_DistortionEvaluator())
