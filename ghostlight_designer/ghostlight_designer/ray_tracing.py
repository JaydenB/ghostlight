"""Trace a ``SystemSetup``'s sequences/sources through an ``OpticalSystem``.

The viewport panel calls :func:`build_ray_bundles` whenever the lens or the
system setup changes; the returned list goes straight to
:meth:`ghostlight_viewport.LensViewport.set_trace_results`.

This module is responsible for:

* Deciding whether the current lens is ready to trace at all
  (see :func:`is_ready_to_trace`).
* Resolving each sequence's *effective stop* index — explicit override,
  the system's ``is_stop`` surface, or (fallback) the last surface.
* Turning each ``Distribution`` choice into a set of normalised pupil
  samples ``(u, v) ∈ [-1, 1]²``.
* Mapping the chosen ``SourceType`` / ``FieldType`` / ``Footprint`` to a
  concrete launch geometry (origin + direction) per ray.
* Tracing every (field × wavelength × pupil-sample) ray through
  :func:`ghostlight.trace_primary_ray_diagnostic` and grouping by field /
  wavelength into :class:`RayBundle` objects for display.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import ghostlight
from ghostlight_viewport import RayBundle

from .system_setup_data import (
    ApertureType,
    Distribution,
    DistributionType,
    Field,
    FieldType,
    Sequence,
    Source,
    SourceType,
    SystemSetup,
    Wavelength,
    WavelengthContainer,
)

_log = logging.getLogger("ghostlight_designer.ray_tracing")


# Distance (mm) in front of surface 0 where ray launches start.  Generous
# enough to keep the source plane clear of any reasonable front-element
# meniscus, small enough that the displayed launch leg doesn't dominate
# the viewport when the lens is short.
_LAUNCH_MARGIN_MM = 20.0

# Default object-side distance (mm) for ``POINT_SOURCE``-style trace.
# The system-setup data model does not (yet) carry a configurable object
# distance — we use a value far enough that the source acts close to
# collimated for normal lens scales.
_POINT_SOURCE_DISTANCE_MM = 1000.0


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------


def is_ready_to_trace(system: ghostlight.OpticalSystem) -> bool:
    """Return True iff the lens has at least one element with ≥2 surfaces.

    A 'singular lens element' here is any group reconstructed from the
    file's element list whose surface count is two or more (i.e. a real
    refracting element with at least front + back).  A bare stop, a single
    mirror, or an empty system is not enough.

    When the system was built programmatically (no ``elements`` reparse),
    we fall back to counting surfaces directly: two surfaces or more is
    treated as one implicit element.
    """
    try:
        elements = list(system.elements)
    except Exception:
        elements = []

    if elements:
        for el in elements:
            if len(el.surface_ids) >= 2:
                return True
        return False

    try:
        return system.num_surfaces() >= 2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stop resolution
# ---------------------------------------------------------------------------


def resolve_stop_index(
    system: ghostlight.OpticalSystem, seq: Sequence
) -> Optional[int]:
    """Pick the surface index that acts as the stop for ``seq``.

    * Explicit ``seq.stop_surface`` wins when it points at a valid surface.
    * Otherwise scan for a surface flagged ``is_stop=True``.
    * Otherwise (Auto with no flagged stop yet) fall back to the last
      surface — the spec calls for this so the trace stays alive while
      the user is still building the system.

    Returns ``None`` when the system has no surfaces at all.
    """
    try:
        n = system.num_surfaces()
    except Exception:
        n = 0
    if n <= 0:
        return None

    if seq.stop_surface is not None:
        idx = int(seq.stop_surface)
        if 0 <= idx < n:
            return idx
        return n - 1  # invalid override: fall back to last

    for i in range(n):
        try:
            if bool(system.surfaces[i].is_stop):
                return i
        except Exception:
            continue

    return n - 1


# ---------------------------------------------------------------------------
# Wavelength selection
# ---------------------------------------------------------------------------


def _used_wavelengths(container: WavelengthContainer) -> List[float]:
    """Return the wavelengths (nm) that should be traced.

    Every entry of ``container.wavelengths`` contributes — they're meant
    to be drawn as separate colours.  A container with no entries falls
    back to a single d-line ray so the viewport stays useful while the
    user is editing.
    """
    out: List[float] = []
    for w in container.wavelengths:
        if isinstance(w, Wavelength):
            v = float(w.value_nm)
        else:
            v = float(getattr(w, "value_nm", 587.56))
        if v > 0.0:
            out.append(v)
    if not out:
        out.append(587.56)
    return out


# ---------------------------------------------------------------------------
# Pupil sampling — one entry per Distribution type
# ---------------------------------------------------------------------------

PupilSamples = List[Tuple[float, float]]


def _samples_single_ray() -> PupilSamples:
    return [(0.0, 0.0)]


def _samples_y_fan(n: int) -> PupilSamples:
    if n <= 0:
        return [(0.0, 0.0)]
    if n == 1:
        return [(0.0, 0.0)]
    return [(0.0, (2.0 * i / (n - 1)) - 1.0) for i in range(n)]


def _samples_x_fan(n: int) -> PupilSamples:
    if n <= 0:
        return [(0.0, 0.0)]
    if n == 1:
        return [(0.0, 0.0)]
    return [((2.0 * i / (n - 1)) - 1.0, 0.0) for i in range(n)]


def _samples_xy_fan(n: int) -> PupilSamples:
    # An X-fan and a Y-fan share the (0, 0) ray; emit it once.
    out: PupilSamples = []
    seen: set = set()
    for s in _samples_x_fan(n) + _samples_y_fan(n):
        key = (round(s[0], 12), round(s[1], 12))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _samples_ring(n: int) -> PupilSamples:
    if n <= 0:
        return []
    return [
        (math.cos(2.0 * math.pi * k / n), math.sin(2.0 * math.pi * k / n))
        for k in range(n)
    ]


def _samples_random(n: int) -> PupilSamples:
    """Quasi-random samples on the unit disk (low-discrepancy, deterministic).

    Deterministic across calls so dragging a slider doesn't dance the
    sample set around — the ray pattern stays visually steady while the
    geometry beneath it moves.
    """
    if n <= 0:
        return []
    out: PupilSamples = []
    g = 1.32471795724474602596  # plastic constant; 2D Kronecker base
    a1 = 1.0 / g
    a2 = 1.0 / (g * g)
    for k in range(1, n + 1):
        u01 = (0.5 + a1 * k) % 1.0
        v01 = (0.5 + a2 * k) % 1.0
        # Map [0, 1)² to the unit disk with concentric-disk mapping.
        x = 2.0 * u01 - 1.0
        y = 2.0 * v01 - 1.0
        if x == 0.0 and y == 0.0:
            out.append((0.0, 0.0))
            continue
        if abs(x) > abs(y):
            r = abs(x)
            theta = (math.pi / 4.0) * (y / x)
        else:
            r = abs(y)
            theta = (math.pi / 2.0) - (math.pi / 4.0) * (x / y)
        out.append((r * math.cos(theta), r * math.sin(theta)))
    return out


def pupil_samples(dist: Distribution) -> PupilSamples:
    """Map a ``Distribution`` to its (u, v) unit-disk samples."""
    n = max(0, int(dist.ray_count))
    t = dist.type
    if t == DistributionType.SINGLE_RAY:
        return _samples_single_ray()
    if t == DistributionType.Y_FAN:
        return _samples_y_fan(n)
    if t == DistributionType.X_FAN:
        return _samples_x_fan(n)
    if t == DistributionType.XY_FAN:
        return _samples_xy_fan(n)
    if t == DistributionType.RING:
        return _samples_ring(n)
    if t == DistributionType.RANDOM:
        return _samples_random(n)
    return _samples_single_ray()


# ---------------------------------------------------------------------------
# Field → object-space angle helpers
# ---------------------------------------------------------------------------


def _angle_from_field(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
) -> Tuple[float, float]:
    """Map a Field's (tilt_x, tilt_y) interpretation to chief-ray angles
    in radians.  Returns ``(angle_x_rad, angle_y_rad)`` defining the
    direction the incoming rays should travel.

    All supported field types — ``ANGLE``, ``FREE``, ``NONE`` — pass the
    stored degrees through directly. ``FREE`` / ``NONE`` are placeholders
    today; they reuse the angle mapping so the viewport still shows
    something while a richer mapping is designed.
    """
    return math.radians(float(fld.tilt_x_deg)), math.radians(float(fld.tilt_y_deg))


# ---------------------------------------------------------------------------
# Launch geometry — origins + direction per (pupil sample, field, source)
# ---------------------------------------------------------------------------


# Marginal-ray search stays just inside the unvignetted limit so floating-
# point noise on the boundary doesn't bounce rays into the dim/dead region.
_PUPIL_SAFETY = 0.98

# Cap on binary-search iterations.  ~40 halvings get a 100 mm range down to
# sub-millimetre precision, which is more than enough for visualisation.
_MARGINAL_SEARCH_ITERS = 40
_MARGINAL_SEARCH_TOL_MM = 1e-3


def _launch_plane_z(system: ghostlight.OpticalSystem) -> float:
    try:
        return float(system.surfaces[0].z) - _LAUNCH_MARGIN_MM
    except Exception:
        return -_LAUNCH_MARGIN_MM


def _primary_wavelength(container: WavelengthContainer) -> float:
    """Wavelength used for the marginal-ray search.

    Vignetting is wavelength-dependent through dispersion, but only weakly
    across the visible band — searching once at the user's primary
    wavelength is enough to size a bundle that survives on the others too.
    """
    if not container.wavelengths:
        return 587.56
    idx = container.primary_index
    if 0 <= idx < len(container.wavelengths):
        w = container.wavelengths[idx]
        return float(getattr(w, "value_nm", 587.56))
    return float(getattr(container.wavelengths[0], "value_nm", 587.56))


def _make_launch_geometry(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    pupil_x_mm: float,
    pupil_y_mm: float,
) -> Tuple["ghostlight.Vec3f", "ghostlight.Vec3f"]:
    """Build the (origin, direction) pair for a single ray.

    Used by both the actual bundle assembly and the marginal-ray search
    so the search probes the *real* launch geometry — same source type,
    same direction math, same launch plane.
    """
    ang_x, ang_y = _angle_from_field(system, seq, fld)
    z0 = _launch_plane_z(system)

    px = float(pupil_x_mm)
    py = float(pupil_y_mm)
    pz = z0

    if source.type == SourceType.POINT_SOURCE:
        ps_z = z0 - _POINT_SOURCE_DISTANCE_MM
        ps_x = -math.tan(ang_x) * _POINT_SOURCE_DISTANCE_MM
        ps_y = -math.tan(ang_y) * _POINT_SOURCE_DISTANCE_MM
        origin = ghostlight.Vec3f(ps_x, ps_y, ps_z)
        dx = px - ps_x
        dy = py - ps_y
        dz = pz - ps_z
        dlen = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dlen <= 1e-12:
            direction = ghostlight.Vec3f(0.0, 0.0, 1.0)
        else:
            direction = ghostlight.Vec3f(dx / dlen, dy / dlen, dz / dlen)
    else:
        bx = math.tan(ang_x)
        by = math.tan(ang_y)
        mag = math.sqrt(bx * bx + by * by + 1.0)
        direction = ghostlight.Vec3f(bx / mag, by / mag, 1.0 / mag)
        origin = ghostlight.Vec3f(px, py, pz)

    return origin, direction


def _ray_reaches_sensor(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    pupil_x_mm: float,
    pupil_y_mm: float,
    lam: float,
) -> bool:
    """Return True iff a test ray launched at ``(pupil_x_mm, pupil_y_mm)``
    survives the full trace to the sensor (``path.result.status == OK``).

    Sized to ``OK`` at the sensor (not merely at the stop) so the bundle
    that comes out of the marginal-ray search shows *no* dead segments —
    every ray extends all the way through the system, matching the
    primary-trace look the demo viewport produces.  Rays that clear the
    stop but vignette downstream would still draw, but as visibly cut
    segments; sizing this way picks a pupil that fills the system instead.
    """
    try:
        origin, direction = _make_launch_geometry(
            system, seq, fld, source, pupil_x_mm, pupil_y_mm
        )
        path = ghostlight.trace_primary_ray_diagnostic(
            ghostlight.Ray(origin, direction, float(lam)), system
        )
    except Exception:
        return False
    if path is None:
        return False
    result = getattr(path, "result", None)
    if result is None:
        return False
    return _status_name(result.status) == "OK"


def _stop_hit_position(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    pupil_x_mm: float,
    pupil_y_mm: float,
    lam: float,
    stop_index: int,
) -> Optional[Tuple[float, float]]:
    """Return the ``(x, y)`` location where the ray hits the stop surface,
    or ``None`` if it never reaches that surface.

    Used by :func:`_find_chief_ray_center` to home in on the launch
    position whose ray lands at the stop's centre.  Status is ignored —
    we accept any event the tracer recorded at ``stop_index``, because
    even a vignetted intersection still tells us where the ray was
    heading, which is what the centred-fit needs.
    """
    try:
        origin, direction = _make_launch_geometry(
            system, seq, fld, source, pupil_x_mm, pupil_y_mm
        )
        path = ghostlight.trace_primary_ray_diagnostic(
            ghostlight.Ray(origin, direction, float(lam)), system
        )
    except Exception:
        return None
    if path is None:
        return None
    for ev in (path.events or []):
        if int(getattr(ev, "surface_index", -1)) == int(stop_index):
            hp = ev.hit_point
            return (float(hp.x), float(hp.y))
    return None


def _status_name(status) -> Optional[str]:
    if status is None:
        return None
    name = getattr(status, "name", None)
    if name is not None:
        return str(name)
    return None


def _ray_path_is_clean(path) -> bool:
    """True iff a traced path made it to the sensor without vignetting.

    Mirrors :func:`_ray_reaches_sensor` but operates on an already-traced
    path so we don't pay for a second trace per ray during bundle
    assembly.  Used to discard the diagonal samples that the cardinal-
    axis marginal search can't fully bound.
    """
    result = getattr(path, "result", None)
    if result is None:
        return False
    return _status_name(result.status) == "OK"


def _search_upper_bound(
    system: ghostlight.OpticalSystem, seq: Sequence, source: Source
) -> float:
    """Generous ceiling for the marginal-ray binary search.

    Set ~1.5× the largest plausible pupil radius for this lens:
    * the front surface's semi-aperture (any ray outside that misses it),
    * the stop's semi-aperture if larger,
    * the user's aperture_radius hint, when not pathological.
    """
    candidates: List[float] = []
    try:
        front_r = float(system.surfaces[0].semi_aperture)
        if front_r > 0.0:
            candidates.append(front_r)
    except Exception:
        pass
    stop_idx = resolve_stop_index(system, seq)
    if stop_idx is not None:
        try:
            stop_r = float(system.surfaces[stop_idx].semi_aperture)
            if stop_r > 0.0:
                candidates.append(stop_r)
        except Exception:
            pass
    ar = float(getattr(source, "aperture_radius", 0.0))
    if ar > 0.0:
        candidates.append(ar)
    if not candidates:
        return 10.0
    return max(candidates) * 1.5


def _find_chief_ray_center(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    lam: float,
    stop_index: Optional[int],
    upper: float,
) -> Tuple[float, float]:
    """Find the launch-plane point ``(cx, cy)`` whose ray lands at the stop
    centre — the chief-ray launch position for this field.

    For an off-axis field, the optical-axis launch ``(0, 0)`` is not the
    chief ray: a tilted bundle has to enter the pupil offset to pass
    through the (on-axis) stop.  Without this offset, the marginal-ray
    search probes a region where every ray misses the stop and collapses
    to zero — that's the bug that hid off-axis bundles entirely.

    Strategy: a coarse 7×7 grid finds the launch point closest to the
    stop centre, then a Newton step (linear-fit-and-invert on a small
    probe box) snaps it onto centre.  Newton is bounded so it can't
    overshoot off the unvignetted region; if it tries, we revert to the
    last good iterate.

    Always returns *some* centre: the grid best when the Newton step
    can't refine, ``(0, 0)`` when even the grid sees no stop hits.  This
    keeps the marginal search alive — without a stop or for systems that
    can't pass this field at all, the bundle is still attempted from the
    optical axis instead of vanishing entirely.
    """
    if stop_index is None:
        # No stop to centre on; the optical-axis launch is the fallback.
        return (0.0, 0.0)

    # Coarse grid: 7×7 across [-upper, +upper]² to seed the Newton step.
    # 7 (instead of 5) catches tighter unvignetted regions where the 5×5
    # spacing would skip over the only successful sample.
    N = 7
    best: Optional[Tuple[float, float]] = None
    best_dist = float("inf")
    for j in range(N):
        cy = upper * (2.0 * j / (N - 1) - 1.0)
        for i in range(N):
            cx = upper * (2.0 * i / (N - 1) - 1.0)
            hit = _stop_hit_position(
                system, seq, fld, source, cx, cy, lam, stop_index
            )
            if hit is None:
                continue
            d = hit[0] * hit[0] + hit[1] * hit[1]
            if d < best_dist:
                best_dist = d
                best = (cx, cy)

    if best is None:
        # Even the coarse grid sees no stop hits.  Fall back to optical
        # axis so the caller can still attempt a marginal search.
        return (0.0, 0.0)

    # Newton refinement: a couple of iterations are enough for paraxial
    # systems, more for strongly distorted ones.  Each step is clamped
    # to ``upper / 2`` per axis so we can't shoot off into unmapped
    # territory; if the refined point's residual is worse than the
    # previous iterate, revert.
    cx, cy = best
    best_cx, best_cy = cx, cy
    best_residual = best_dist
    h = upper / max(1, N - 1) * 0.5
    step_cap = upper * 0.5
    for _ in range(8):
        hit_c = _stop_hit_position(
            system, seq, fld, source, cx, cy, lam, stop_index
        )
        if hit_c is None:
            break
        residual = hit_c[0] * hit_c[0] + hit_c[1] * hit_c[1]
        if residual < best_residual:
            best_cx, best_cy = cx, cy
            best_residual = residual
        if residual < 1e-6:
            break
        hit_x = _stop_hit_position(
            system, seq, fld, source, cx + h, cy, lam, stop_index
        )
        hit_y = _stop_hit_position(
            system, seq, fld, source, cx, cy + h, lam, stop_index
        )
        if hit_x is None or hit_y is None:
            break
        # Jacobian: rows = d(sx, sy)/dcx, d(sx, sy)/dcy
        Jx_cx = (hit_x[0] - hit_c[0]) / h
        Jy_cx = (hit_x[1] - hit_c[1]) / h
        Jx_cy = (hit_y[0] - hit_c[0]) / h
        Jy_cy = (hit_y[1] - hit_c[1]) / h
        det = Jx_cx * Jy_cy - Jx_cy * Jy_cx
        if abs(det) < 1e-12:
            break
        dx = (-hit_c[0] * Jy_cy + hit_c[1] * Jx_cy) / det
        dy = (hit_c[0] * Jy_cx - hit_c[1] * Jx_cx) / det
        # Clamp the step so a degenerate Jacobian can't fling the centre
        # out of the unvignetted region — there's no recovery once we
        # land somewhere ``_stop_hit_position`` returns None.
        if dx > step_cap:
            dx = step_cap
        elif dx < -step_cap:
            dx = -step_cap
        if dy > step_cap:
            dy = step_cap
        elif dy < -step_cap:
            dy = -step_cap
        cx += dx
        cy += dy

    return (best_cx, best_cy)


def _search_max_offset_axis(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    lam: float,
    axis: str,
    upper: float,
    center: Tuple[float, float],
) -> float:
    """Largest pupil offset (mm) along ``axis`` from ``center`` that still
    traces all the way to the sensor.

    ``axis`` is one of ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``.  The search
    walks outward from the chief-ray centre, so an off-axis field with a
    non-(0,0) chief ray gets a non-zero pupil radius instead of collapsing.
    Returns ``0.0`` only when the chief ray itself fails to reach the sensor.
    """
    if upper <= 0.0:
        return 0.0
    cx, cy = center

    def offset(r: float) -> Tuple[float, float]:
        if axis == "+x":
            return (cx + r, cy)
        if axis == "-x":
            return (cx - r, cy)
        if axis == "+y":
            return (cx, cy + r)
        return (cx, cy - r)

    def ok(r: float) -> bool:
        x, y = offset(r)
        return _ray_reaches_sensor(system, seq, fld, source, x, y, lam)

    if not ok(0.0):
        return 0.0
    if ok(upper):
        return upper
    lo, hi = 0.0, upper
    for _ in range(_MARGINAL_SEARCH_ITERS):
        if hi - lo < _MARGINAL_SEARCH_TOL_MM:
            break
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _marginal_pupil_radii(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    lam: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """Return ``(rx, ry, (cx, cy))`` — marginal pupil radii in mm around the
    chief-ray launch point.

    The chief ray is the per-field offset that maps to the stop's centre;
    ``rx``/``ry`` are the tighter of ``±x``/``±y`` walks from that centre,
    so the final pupil ellipse sits entirely inside the unvignetted region
    even for asymmetric off-axis fields.  When the chief ray can't be
    found (system can't pass this field at all), returns zero radii and
    the optical-axis fallback centre.
    """
    stop_index = resolve_stop_index(system, seq)
    upper = _search_upper_bound(system, seq, source)
    center = _find_chief_ray_center(
        system, seq, fld, source, lam, stop_index, upper
    )
    rxp = _search_max_offset_axis(
        system, seq, fld, source, lam, "+x", upper, center
    )
    rxm = _search_max_offset_axis(
        system, seq, fld, source, lam, "-x", upper, center
    )
    ryp = _search_max_offset_axis(
        system, seq, fld, source, lam, "+y", upper, center
    )
    rym = _search_max_offset_axis(
        system, seq, fld, source, lam, "-y", upper, center
    )
    return min(rxp, rxm), min(ryp, rym), center


def _pupil_radii_for_field(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    source: Source,
    lam: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """The ``(rx, ry, (cx, cy))`` actually used to scale unit-pupil samples.

    ``FROM_STOP`` runs the marginal-ray search so the bundle fills the
    unvignetted pupil at the stop. ``NONE`` skips that optimisation
    entirely — rays trace independently of the stop, sized by the user's
    ``source.aperture_radius`` and launched from the optical axis.
    """
    if seq.aperture_type == ApertureType.NONE:
        r = max(0.0, float(getattr(source, "aperture_radius", 0.0)))
        return r, r, (0.0, 0.0)
    rx, ry, center = _marginal_pupil_radii(system, seq, fld, source, lam)
    return rx * _PUPIL_SAFETY, ry * _PUPIL_SAFETY, center


def _build_rays_for_field(
    system: ghostlight.OpticalSystem,
    seq: Sequence,
    fld: Field,
    samples: PupilSamples,
    wavelengths_nm: List[float],
) -> Tuple[List, List, List[float]]:
    """For one (sequence, field), build parallel ``(ray, origin, lambda)`` lists.

    Returns ``(rays, origins, wavelengths)`` where every list has the same
    length: outer loop over pupil samples, inner loop over wavelengths.
    """
    source = seq.source
    search_lam = _primary_wavelength(source.wavelengths)
    rx, ry, (cx, cy) = _pupil_radii_for_field(
        system, seq, fld, source, search_lam
    )
    if rx <= 0.0 or ry <= 0.0:
        return [], [], []

    rays: List = []
    origins: List = []
    wavelengths_out: List[float] = []

    # Pupil samples scale by (rx, ry) and shift to the chief-ray centre
    # (cx, cy).  Without the shift, off-axis fields launch from the
    # optical axis with their bundle direction tilted — every ray misses
    # the stop and the bundle vanishes.
    for (u, v) in samples:
        px = cx + u * rx
        py = cy + v * ry
        for lam in wavelengths_nm:
            origin, direction = _make_launch_geometry(
                system, seq, fld, source, px, py
            )
            rays.append(ghostlight.Ray(origin, direction, float(lam)))
            origins.append(origin)
            wavelengths_out.append(float(lam))

    return rays, origins, wavelengths_out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_ray_bundles(
    system: ghostlight.OpticalSystem,
    setup: SystemSetup,
) -> List[RayBundle]:
    """Trace every (sequence × field × wavelength × pupil sample) ray.

    Returns one :class:`RayBundle` per (sequence, field) pair.  A single
    bundle carries all wavelengths × all pupil samples for that field
    together so the viewport draws them with their wavelength tints.

    Returns ``[]`` when :func:`is_ready_to_trace` is False, or when the
    setup has no sequences or no fields.
    """
    if not is_ready_to_trace(system):
        return []
    if not setup.sequences:
        return []

    bundles: List[RayBundle] = []

    for seq_idx, seq in enumerate(setup.sequences):
        source = seq.source
        if seq.field_type == FieldType.NONE:
            continue
        fields = source.fields or []
        if not fields:
            continue

        samples = pupil_samples(source.distribution)
        if not samples:
            continue
        wavelengths_nm = _used_wavelengths(source.wavelengths)

        # ``NONE`` skips the stop optimisation entirely — see
        # :func:`_pupil_radii_for_field`. ``FROM_STOP`` discards rays that
        # don't reach the sensor cleanly so the bundle stays inside the
        # full-trace-OK pupil (matches the demo's primary-trace look).
        enforce_clean = seq.aperture_type == ApertureType.FROM_STOP

        for fld in fields:
            try:
                rays, origins, wls = _build_rays_for_field(
                    system, seq, fld, samples, wavelengths_nm
                )
            except Exception:
                _log.exception(
                    "ray_tracing: failed to build rays for seq %r / field %r",
                    seq.name, fld.name,
                )
                continue
            if not rays:
                continue

            paths: List = []
            kept_origins: List = []
            kept_wls: List[float] = []
            for ray, origin, lam in zip(rays, origins, wls):
                try:
                    path = ghostlight.trace_primary_ray_diagnostic(ray, system)
                except Exception:
                    # An ill-conditioned system (e.g. mid-edit zero radius)
                    # can throw — drop this ray but keep the rest going.
                    continue
                # In FROM_STOP mode, drop rays that don't make it cleanly
                # to the sensor.  The cardinal-axis marginal search bounds
                # ±x/±y, but a diagonal sample can still nick the
                # unvignetted boundary or pick up dispersion shift at a
                # non-primary wavelength; dropping those few keeps the
                # bundle's "full-system through-trace" look intact.
                if enforce_clean and not _ray_path_is_clean(path):
                    continue
                paths.append(path)
                kept_origins.append(origin)
                kept_wls.append(lam)

            if not paths:
                continue

            bundles.append(
                RayBundle(
                    paths=paths,
                    wavelengths_nm=kept_wls,
                    origins=kept_origins,
                    label=f"{seq.name} / {fld.name}",
                    base_alpha=0.85,
                    flat_alpha=True,
                )
            )

    return bundles
