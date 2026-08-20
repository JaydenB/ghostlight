"""Field-dependent aberration computation.

For each field angle in the spec, we compute three quantities:

* **Sagittal / tangential focus z** (astigmatism). Trace a small fan of
  rays — sagittal = pupil offsets in X, tangential = pupil offsets in
  Y — at the field's chief direction. Each ray crosses the sensor (z=0)
  at some position with some slope. The best focus z is the value that
  minimises the bundle's spread along that axis; the closed-form solution
  is ``z* = -Cov(x, dx/dz) / Var(dx/dz)`` (and analogously for y).
* **Distortion percent**. Chief-ray sensor landing minus the paraxial
  expected landing, divided by the paraxial expected landing × 100.
  The paraxial slope is fit from the smallest non-zero field's chief
  ray — that's a linear extrapolation through the origin assumed to be
  free of distortion by construction (the field is small enough that
  Seidel ``D = V * y'_p`` is negligible).
* **Lateral chromatic deviation**. Chief-ray sensor landing per
  wavelength, plotted relative to the primary wavelength's landing.
  The Y-coordinate of the chief ray is the relevant axis for fields
  tilted in Y.

All trace work uses :mod:`ghostlight_designer.tracing_util` — sensor is
the virtual z=0 plane (see ``feedback_sensor_plane_z0``); never read
``events[-1].hit_point`` as the image landing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import ghostlight

from ... import tracing_util as tu
from .spec import FieldDiagramSpec

_log = logging.getLogger("ghostlight_designer.evaluation_panels.field_diagrams")


# Sensor lives at z=0 by convention; defocus values are reported
# *relative* to that plane (positive = behind the sensor).
SENSOR_Z_MM = 0.0


@dataclass(frozen=True)
class FieldDiagramResult:
    """Output of one field-diagrams compute.

    All arrays are aligned along the same field-angle axis stored in
    ``field_angles_deg`` — a 1-D float array of length ``N``. Missing
    samples (trace failed, fan didn't converge, etc.) are ``np.nan``.
    """

    spec: FieldDiagramSpec
    field_angles_deg: np.ndarray  # (N,)

    # Astigmatism / Petzval: defocus from the sensor at each field, in mm.
    sagittal_defocus_mm: np.ndarray   # (N,)
    tangential_defocus_mm: np.ndarray  # (N,)

    # Distortion: percentage at each field.
    distortion_pct: np.ndarray  # (N,)

    # Lateral chromatic: chief-ray y-landing at each (field, wavelength).
    # Plot the per-wavelength deviation from the primary column.
    chief_y_per_wavelength_mm: np.ndarray  # (N, n_wavelengths)
    wavelengths_nm: Tuple[float, ...]
    primary_wavelength_index: int

    # Per-system metrics (used for the title bar).
    pupil_radius_mm: float
    # Linear paraxial slope dy/dθ_rad (mm per radian) — used both to
    # build the paraxial-expected curve for the distortion plot and as
    # an EFL-ish sanity number for the title.
    paraxial_slope_mm_per_rad: float


# ---------------------------------------------------------------------------
# Focus finding (astigmatism)
# ---------------------------------------------------------------------------


def _fan_landings_and_slopes(
    system: ghostlight.OpticalSystem,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    wavelength_nm: float,
    pupil_axis: str,  # 'x' or 'y'
    pupil_r_mm: float,
    n_rays: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trace one sagittal- or tangential-style fan.

    Returns ``(sensor_axis, axis_slope)`` arrays — for ``pupil_axis='x'``
    each entry is ``(x_at_sensor, dx/dz_at_sensor)``; for 'y' it's the
    y-equivalents. Failed rays are skipped (not NaN-padded) since the
    consumer just feeds these into a 1-D best-focus fit.

    The slope is recovered from the segment between the last refracting
    surface and the sensor — both points are in image space (no further
    refraction), so the line is exact.
    """
    direction = tu.direction_for_field(tilt_x_deg, tilt_y_deg)

    # Sample the pupil along the chosen axis, including the centre.
    fan = np.linspace(-pupil_r_mm, pupil_r_mm, n_rays)

    axis_pos: List[float] = []
    axis_slope: List[float] = []

    for offset in fan:
        if pupil_axis == "x":
            pupil_xy = (float(offset), 0.0)
        else:
            pupil_xy = (0.0, float(offset))
        path = tu.trace_collimated_ray(
            system,
            pupil_xy_mm=pupil_xy,
            direction=direction,
            wavelength_nm=wavelength_nm,
        )
        if path is None:
            continue
        landing = tu.landing_xy(path)
        if landing is None:
            continue
        events = getattr(path, "events", None) or []
        if not events:
            continue
        last = events[-1]
        try:
            lx = float(last.hit_point.x)
            ly = float(last.hit_point.y)
            lz = float(last.hit_point.z)
        except (AttributeError, TypeError):
            continue
        # Sensor position from result; recover direction from the
        # segment (last lens surface → sensor).
        sx, sy = landing
        sz = SENSOR_Z_MM
        dz = sz - lz
        if abs(dz) < 1e-9:
            continue
        if pupil_axis == "x":
            axis_pos.append(sx)
            axis_slope.append((sx - lx) / dz)
        else:
            axis_pos.append(sy)
            axis_slope.append((sy - ly) / dz)

    return np.asarray(axis_pos, dtype=np.float64), np.asarray(axis_slope, dtype=np.float64)


def _best_focus_z(axis_pos: np.ndarray, axis_slope: np.ndarray) -> float:
    """Return ``z*`` that minimises ``Var(axis_pos + z * axis_slope)``.

    Closed form: ``z* = -Cov(p, s) / Var(s)``. Returns ``nan`` when the
    rays are nearly parallel (``Var(s)`` is below numerical noise),
    which is the case for a marginally lensed system or a degenerate
    fan.
    """
    if axis_pos.size < 2 or axis_slope.size != axis_pos.size:
        return float("nan")
    p = axis_pos - axis_pos.mean()
    s = axis_slope - axis_slope.mean()
    var_s = float(np.sum(s * s))
    if var_s < 1e-18:
        return float("nan")
    cov_ps = float(np.sum(p * s))
    return -cov_ps / var_s


def _sagittal_tangential_defocus(
    system: ghostlight.OpticalSystem,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    wavelength_nm: float,
    pupil_r_mm: float,
    n_rays: int,
) -> Tuple[float, float]:
    """Sagittal and tangential focus defocus (mm relative to sensor)."""
    sx, ss = _fan_landings_and_slopes(
        system,
        tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg,
        wavelength_nm=wavelength_nm,
        pupil_axis="x", pupil_r_mm=pupil_r_mm, n_rays=n_rays,
    )
    ty, ts = _fan_landings_and_slopes(
        system,
        tilt_x_deg=tilt_x_deg, tilt_y_deg=tilt_y_deg,
        wavelength_nm=wavelength_nm,
        pupil_axis="y", pupil_r_mm=pupil_r_mm, n_rays=n_rays,
    )
    return _best_focus_z(sx, ss), _best_focus_z(ty, ts)


# ---------------------------------------------------------------------------
# Distortion + lateral chromatic
# ---------------------------------------------------------------------------


def _chief_ray_y_landing(
    system: ghostlight.OpticalSystem,
    *,
    field_deg: float,
    wavelength_nm: float,
) -> float:
    """Y-coord of the chief ray's sensor landing for a Y-tilted field.

    The field axis is the Y axis (the textbook convention — tilt
    happens in Y). The X-component of the landing is symmetrically 0
    for a rotationally-symmetric lens, so we only need Y.
    """
    path = tu.trace_chief_ray(
        system,
        tilt_x_deg=0.0,
        tilt_y_deg=float(field_deg),
        wavelength_nm=float(wavelength_nm),
    )
    landing = tu.landing_xy(path)
    if landing is None:
        return float("nan")
    return float(landing[1])


def _paraxial_slope_from_smallest_field(
    field_angles_deg: np.ndarray,
    chief_y_mm: np.ndarray,
) -> float:
    """Linear paraxial fit through the origin using the smallest non-zero field.

    Distortion is by definition zero at axis (chief ray lands at 0).
    For a rotationally-symmetric lens, y(θ) = slope · tan(θ) + (higher
    order). At small θ the higher-order terms vanish, so the smallest
    non-zero field gives an unbiased ``slope`` estimate. The full
    paraxial curve is then ``slope · tan(θ)``.

    Returns ``nan`` if no usable point exists.
    """
    for i, theta_deg in enumerate(field_angles_deg):
        if theta_deg <= 0.0:
            continue
        y = float(chief_y_mm[i])
        if math.isnan(y):
            continue
        tan_theta = math.tan(math.radians(float(theta_deg)))
        if abs(tan_theta) < 1e-9:
            continue
        return y / tan_theta
    return float("nan")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_field_diagrams(
    system: ghostlight.OpticalSystem,
    spec: FieldDiagramSpec,
) -> FieldDiagramResult:
    """Run the field-angle sweep and assemble a :class:`FieldDiagramResult`.

    Pure CPU, no Qt. Safe to call from a worker thread. Reads the
    lens through its bound C++ API only.
    """
    spec = spec.clamp()
    pupil_r = tu.resolve_pupil_radius(system, spec.pupil_radius_mm)
    primary_lambda = spec.primary_wavelength_nm

    field_angles = np.linspace(
        0.0, float(spec.max_field_deg), int(spec.field_samples)
    )
    n = field_angles.size
    n_lambda = len(spec.wavelengths_nm)

    sag_def = np.full(n, np.nan, dtype=np.float64)
    tan_def = np.full(n, np.nan, dtype=np.float64)
    chief_y_lambda = np.full((n, n_lambda), np.nan, dtype=np.float64)

    for i, theta_deg in enumerate(field_angles):
        # Astigmatism — at the primary wavelength.
        sag, tan = _sagittal_tangential_defocus(
            system,
            tilt_x_deg=0.0, tilt_y_deg=float(theta_deg),
            wavelength_nm=primary_lambda,
            pupil_r_mm=pupil_r, n_rays=spec.rays_per_fan,
        )
        sag_def[i] = sag
        tan_def[i] = tan

        # Chief ray landing per wavelength (for distortion + lateral chromatic).
        for li, lam in enumerate(spec.wavelengths_nm):
            chief_y_lambda[i, li] = _chief_ray_y_landing(
                system, field_deg=float(theta_deg), wavelength_nm=float(lam),
            )

    # Distortion uses the primary-wavelength chief ray landings.
    primary_chief_y = chief_y_lambda[:, spec.primary_wavelength_index]
    paraxial_slope = _paraxial_slope_from_smallest_field(
        field_angles, primary_chief_y
    )
    distortion_pct = np.full(n, np.nan, dtype=np.float64)
    if not math.isnan(paraxial_slope) and abs(paraxial_slope) > 1e-12:
        for i, theta_deg in enumerate(field_angles):
            y_actual = float(primary_chief_y[i])
            if math.isnan(y_actual):
                continue
            tan_theta = math.tan(math.radians(float(theta_deg)))
            y_paraxial = paraxial_slope * tan_theta
            if abs(y_paraxial) < 1e-9:
                # Origin sample is by construction 0% — anchor the curve.
                distortion_pct[i] = 0.0
                continue
            distortion_pct[i] = (y_actual - y_paraxial) / y_paraxial * 100.0

    return FieldDiagramResult(
        spec=spec,
        field_angles_deg=field_angles,
        sagittal_defocus_mm=sag_def,
        tangential_defocus_mm=tan_def,
        distortion_pct=distortion_pct,
        chief_y_per_wavelength_mm=chief_y_lambda,
        wavelengths_nm=tuple(spec.wavelengths_nm),
        primary_wavelength_index=int(spec.primary_wavelength_index),
        pupil_radius_mm=float(pupil_r),
        paraxial_slope_mm_per_rad=float(paraxial_slope)
            if not math.isnan(paraxial_slope) else 0.0,
    )
