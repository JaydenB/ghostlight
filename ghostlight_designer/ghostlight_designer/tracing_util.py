"""Shared ray-trace helpers used by evaluation panels and optimization goals.

These were originally inline in :mod:`ghostlight_designer.evaluation_panels.spot_diagram.compute`.
Lifted here so the optimization goals don't have to import from another
panel package (which would couple optimization to spot-diagram lifecycle).

All functions are pure / thread-safe — they never touch Qt and never
cache anything on the system.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import ghostlight


# Distance (mm) in front of surface 0 where collimated launches start.
# Matches :mod:`ghostlight_designer.ray_tracing` and the spot diagram so
# every panel paints the same geometry.
LAUNCH_MARGIN_MM = 20.0


# ---------------------------------------------------------------------------
# Pupil sampling
# ---------------------------------------------------------------------------


def ring_fan_samples(rings: int, fans: int) -> Iterator[Tuple[float, float]]:
    """Generate normalised pupil samples ``(u, v) in [-1, 1]``.

    Always yields the on-axis ray first. Subsequent samples sweep rings
    outward (``i=1`` closest to centre) and fans counter-clockwise.
    ``rings=0`` or ``fans=0`` yields only the axial ray.
    """
    yield (0.0, 0.0)
    if rings <= 0 or fans <= 0:
        return
    for i in range(1, rings + 1):
        r = i / rings
        for j in range(fans):
            theta = 2.0 * math.pi * j / fans
            yield (r * math.cos(theta), r * math.sin(theta))


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------


def launch_z(system: ghostlight.OpticalSystem) -> float:
    """Z-coordinate of the launch plane (a fixed margin before surface 0)."""
    try:
        return float(system.surfaces[0].z) - LAUNCH_MARGIN_MM
    except Exception:
        return -LAUNCH_MARGIN_MM


def sensor_z(system: ghostlight.OpticalSystem) -> float:
    """Z of the image / sensor surface (last surface in the system)."""
    try:
        n = system.num_surfaces()
        return float(system.surfaces[n - 1].z)
    except Exception:
        return 0.0


def resolve_pupil_radius(
    system: ghostlight.OpticalSystem,
    explicit_mm: Optional[float] = None,
) -> float:
    """Pick a pupil radius (mm) for sampling.

    ``explicit_mm > 0`` wins. Otherwise we use the stop surface's
    semi-aperture if one is tagged, else the front surface's
    semi-aperture, else a 10 mm fallback so a half-built lens still
    traces something.
    """
    if explicit_mm is not None and explicit_mm > 0.0:
        return float(explicit_mm)
    try:
        for surf in system.surfaces:
            if bool(surf.is_stop) and float(surf.semi_aperture) > 0.0:
                return float(surf.semi_aperture)
        if len(system.surfaces) > 0:
            r = float(system.surfaces[0].semi_aperture)
            if r > 0.0:
                return max(0.5, r)
    except Exception:
        pass
    return 10.0


def direction_for_field(tilt_x_deg: float, tilt_y_deg: float) -> "ghostlight.Vec3f":
    """Unit-vector direction for a collimated field tilt.

    Matches the spot-diagram and viewport convention: tan-of-tilt on each
    axis, normalised. Axial field returns (0, 0, 1).
    """
    bx = math.tan(math.radians(float(tilt_x_deg)))
    by = math.tan(math.radians(float(tilt_y_deg)))
    mag = math.sqrt(bx * bx + by * by + 1.0)
    return ghostlight.Vec3f(bx / mag, by / mag, 1.0 / mag)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def status_ok(path) -> bool:
    """True iff the diagnostic path's status is ``OK``."""
    result = getattr(path, "result", None)
    if result is None:
        return False
    status = getattr(result, "status", None)
    name = getattr(status, "name", None)
    return name == "OK"


def trace_collimated_ray(
    system: ghostlight.OpticalSystem,
    *,
    pupil_xy_mm: Tuple[float, float],
    direction: "ghostlight.Vec3f",
    wavelength_nm: float,
):
    """Launch one collimated ray and return its diagnostic path or ``None``.

    Origin sits on the launch plane at ``(px, py, launch_z)``; direction
    is the supplied unit vector. ``None`` on any trace failure — callers
    discard / penalise it.
    """
    px, py = float(pupil_xy_mm[0]), float(pupil_xy_mm[1])
    origin = ghostlight.Vec3f(px, py, launch_z(system))
    try:
        path = ghostlight.trace_primary_ray_diagnostic(
            ghostlight.Ray(origin, direction, float(wavelength_nm)),
            system,
        )
    except Exception:
        return None
    if path is None or not status_ok(path):
        return None
    return path


def trace_chief_ray(
    system: ghostlight.OpticalSystem,
    *,
    tilt_x_deg: float,
    tilt_y_deg: float,
    wavelength_nm: float,
):
    """Launch the on-axis (pupil-centre) ray for a given field tilt."""
    return trace_collimated_ray(
        system,
        pupil_xy_mm=(0.0, 0.0),
        direction=direction_for_field(tilt_x_deg, tilt_y_deg),
        wavelength_nm=wavelength_nm,
    )


def landing_xy(path) -> Optional[Tuple[float, float]]:
    """Return ``(x, y)`` where ``path`` met the image plane, or ``None``.

    Prefers ``path.result.position`` (the bound C++ result already
    contains the intersection point); falls back to the last event's
    hit point for paths that don't surface a result.
    """
    if path is None:
        return None
    pos = getattr(getattr(path, "result", None), "position", None)
    if pos is not None:
        try:
            return float(pos.x), float(pos.y)
        except (AttributeError, TypeError):
            pass
    events = getattr(path, "events", None)
    if events:
        try:
            hp = events[-1].hit_point
            return float(hp.x), float(hp.y)
        except (AttributeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Wavelength / field resolution from system_setup
# ---------------------------------------------------------------------------


def resolve_wavelengths(
    setup,
    spec: object,
) -> List[float]:
    """Translate a ``params["wavelength"]`` spec into a list of nm values.

    ``spec`` accepts:
        ``"primary"`` — the primary wavelength of the first sequence.
        ``"all"``     — every wavelength of the first sequence.
        int           — index into that sequence's wavelength list.

    Falls back to d-line (587.56 nm) when the spec is unrecognised or
    the index is out of range — better than crashing the residuals fn.
    """
    fallback = [587.56]
    sequences = getattr(setup, "sequences", None) or []
    if not sequences:
        return fallback
    wc = sequences[0].source.wavelengths
    if not wc.wavelengths:
        return fallback
    values = [float(w.value_nm) for w in wc.wavelengths]
    if spec == "all":
        return values
    if spec == "primary" or spec is None:
        idx = wc.primary_index
        if 0 <= idx < len(values):
            return [values[idx]]
        return [values[0]]
    try:
        i = int(spec)
    except (TypeError, ValueError):
        return [values[0]]
    if 0 <= i < len(values):
        return [values[i]]
    return [values[0]]


def resolve_fields(
    setup,
    spec: object,
) -> List[Tuple[float, float]]:
    """Translate ``params["field"]`` into ``[(tilt_x_deg, tilt_y_deg), ...]``.

    ``spec`` accepts ``"all"``, an int index, or ``None`` (=> on-axis).
    Falls back to ``[(0.0, 0.0)]`` if the setup has no fields.
    """
    sequences = getattr(setup, "sequences", None) or []
    if not sequences:
        return [(0.0, 0.0)]
    fields = sequences[0].source.fields
    if not fields:
        return [(0.0, 0.0)]
    pairs = [(float(f.tilt_x_deg), float(f.tilt_y_deg)) for f in fields]
    if spec == "all":
        return pairs
    if spec is None:
        return [pairs[0]]
    try:
        i = int(spec)
    except (TypeError, ValueError):
        return [pairs[0]]
    if 0 <= i < len(pairs):
        return [pairs[i]]
    return [pairs[0]]
