"""Smooth barrier residuals that steer the optimizer out of invalid states.

``ghostlight.OpticalSystem.finalize`` silently accepts negative thicknesses, ``R=0``,
apertures larger than the surface can support, etc. — the tracer will
happily produce garbage rays from any of those, but scipy sees the garbage
as a legitimate objective value and can happily drive further into it.

These penalties do the enforcement in the residuals: each returns 0 when
the geometry is valid, and grows linearly with how far into violation
we've gone. scipy squares residuals internally, so a linear penalty term
becomes a quadratic bowl in the merit total, giving least_squares a
gradient that steers back toward valid.

Design principles:

* **Independent of user weights.** These penalties add fixed-scale
  residuals, so a user with unusually weak goal weights still gets the
  same "don't drive thickness negative" barrier.
* **Zero outside violation.** Squared-zero contributes nothing, so a
  well-behaved lens carries no overhead in the objective total.
* **Cheap.** No tracer calls — read a few surface attributes and do
  scalar arithmetic. Called every residuals evaluation, so it must
  stay lightweight.
"""
from __future__ import annotations

import math
from typing import List

import ghostlight


# ---------------------------------------------------------------------------
# Tunables — module-level so tests can monkey-patch and the optimizer can
# read them if we ever want to expose them per-MF.
# ---------------------------------------------------------------------------

# Minimum surface-to-surface spacing (mm). Below this we start
# penalising. 0.05 mm is small enough not to interfere with realistic
# thin air gaps but large enough that we never confuse "valid but small"
# with "surface has crashed into the next one".
MIN_THICKNESS_MM: float = 0.05

# Sag geometry: for a spherical surface, sag is defined only when the
# aperture radius r < |R|. We stay a safety margin inside so a hemisphere
# doesn't sit on the exact boundary. 0.95 gives a 5% cushion.
APERTURE_RADIUS_MARGIN: float = 0.95

# Penalty weight — squared, this shows up as `PENALTY_WEIGHT**2` inside
# the least_squares sum. 100 gives a squared contribution of 10000 per
# violated millimetre, which dominates typical goal residuals (usually
# ≤ 1e2 squared) without being so large that scipy's Levenberg step
# gets numerically pinned to the boundary.
PENALTY_WEIGHT: float = 100.0


# ---------------------------------------------------------------------------
# Individual barrier evaluations
# ---------------------------------------------------------------------------


def _thickness_penalty(surf) -> float:
    """Residual for one surface's thickness.

    Zero when ``thickness >= MIN_THICKNESS_MM``; grows linearly with the
    shortfall otherwise. Muted (inactive) surfaces are excluded — they
    don't participate in the ray trace so their thickness doesn't affect
    the optical model.
    """
    if not bool(getattr(surf, "is_active", True)):
        return 0.0
    try:
        t = float(surf.thickness)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(t) or math.isinf(t):
        # Something has already gone catastrophically wrong; return a
        # large but finite residual so scipy still sees a gradient
        # toward valid space.
        return PENALTY_WEIGHT * 10.0
    shortfall = MIN_THICKNESS_MM - t
    if shortfall <= 0.0:
        return 0.0
    return PENALTY_WEIGHT * shortfall


def _aperture_penalty(surf) -> float:
    """Residual for the aperture-vs-radius sag validity.

    Only applies to spherical (or default) surfaces — asphere /
    cylindrical / stop rows have different sag rules and are left alone.
    ``R = 0`` (flat, in ghostlight's convention) has no sag
    constraint so we skip it too.

    Zero when ``r <= APERTURE_RADIUS_MARGIN * |R|``.
    """
    if not bool(getattr(surf, "is_active", True)):
        return 0.0
    if bool(getattr(surf, "is_stop", False)):
        # Stop surfaces are flat — no sag equation to violate.
        return 0.0
    try:
        form_int = int(getattr(surf, "form", 0))
    except (TypeError, ValueError):
        form_int = 0
    if form_int != int(ghostlight.SurfaceForm.SPHERE):
        # Asphere / cylindrical sag rules differ; not constrained here.
        return 0.0
    try:
        R = float(surf.radius)
        r = float(surf.semi_aperture)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(R) or math.isnan(r) or R == 0.0 or r <= 0.0:
        return 0.0
    max_r = APERTURE_RADIUS_MARGIN * abs(R)
    overshoot = r - max_r
    if overshoot <= 0.0:
        return 0.0
    return PENALTY_WEIGHT * overshoot


# ---------------------------------------------------------------------------
# Public entry — called by _Worker._compute_residuals
# ---------------------------------------------------------------------------


def evaluate_geometry_penalties(system: ghostlight.OpticalSystem) -> List[float]:
    """Return a list of validity-barrier residuals, one entry per check.

    Order is:

      * one thickness penalty per active surface
      * one aperture penalty per active spherical surface

    Appended to the goal residuals inside
    :meth:`_Worker._compute_residuals` so scipy sees them as objectives.
    Length is deterministic given ``system`` (doesn't shrink when a
    surface becomes valid) — least_squares would otherwise trip on a
    changing residuals-vector length.
    """
    out: List[float] = []
    try:
        surfaces = list(system.surfaces)
    except Exception:
        return out
    for surf in surfaces:
        out.append(_thickness_penalty(surf))
    for surf in surfaces:
        out.append(_aperture_penalty(surf))
    return out


def n_geometry_residuals(system: ghostlight.OpticalSystem) -> int:
    """How many barrier residuals :func:`evaluate_geometry_penalties`
    will emit for this system. Used by pre-allocation / test paths."""
    try:
        return 2 * len(list(system.surfaces))
    except Exception:
        return 0
