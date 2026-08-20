"""Ray-trace one :class:`SpotDiagramSpec` against an ``OpticalSystem``.

Sampling pattern:

* For every (field, wavelength, pupil sample) tuple we launch a
  collimated ray (plane wavefront) from a launch plane a fixed margin
  in front of surface 0.
* Pupil samples follow the textbook "rings × fans" pattern — one ray on
  the optical axis plus ``rings × fans`` rays distributed on concentric
  circles inside the pupil.
* For each ray that survives the trace we read the last two events to
  recover its direction at the image plane, then propagate it linearly
  to each defocus slice in the spec. That avoids re-tracing per slice.

The result is a :class:`SpotResult` — purely numeric, no Qt — that the
panel body hands to its matplotlib canvas on the GUI thread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

import ghostlight

from .spec import SpotDiagramSpec


# Match :mod:`ghostlight_designer.ray_tracing` so users see consistent ray
# geometry between the viewport bundles and the evaluation spots.
_LAUNCH_MARGIN_MM = 20.0


@dataclass(frozen=True)
class SpotFieldResult:
    """One field's worth of spot data, sliced across defocus offsets.

    ``xs`` / ``ys`` are 3-D arrays shaped ``(n_defocus, n_wavelengths, n_samples)``,
    with ``np.nan`` for rays that didn't survive the trace. ``valid_mask``
    is the same shape and is True where the ray landed cleanly — that's
    what plotting filters on.
    """

    tilt_x_deg: float
    tilt_y_deg: float
    xs: np.ndarray  # (n_defocus, n_wavelengths, n_samples)
    ys: np.ndarray  # (n_defocus, n_wavelengths, n_samples)
    valid_mask: np.ndarray  # same shape, bool


@dataclass(frozen=True)
class SpotResult:
    """Top-level result returned to :meth:`SpotDiagramBody.apply_result`."""

    spec: SpotDiagramSpec
    fields: Tuple[SpotFieldResult, ...]
    # Echoed back so the canvas knows what to label colour-coded points.
    wavelengths_nm: Tuple[float, ...]
    defocus_offsets_mm: Tuple[float, ...]
    # Per-system metrics for the title bar.
    pupil_radius_mm: float
    sensor_z_mm: float


def _ring_fan_samples(rings: int, fans: int) -> Iterator[Tuple[float, float]]:
    """Generate normalised pupil samples on concentric rings + radial fans.

    Always yields the on-axis ray first; subsequent rays sweep rings
    outward (ring 1 = closest to centre) and fans counter-clockwise.
    Empty when ``rings == 0 and fans == 0`` apart from the axial ray.
    """
    yield (0.0, 0.0)
    if rings <= 0 or fans <= 0:
        return
    for i in range(1, rings + 1):
        r = i / rings
        for j in range(fans):
            theta = 2.0 * math.pi * j / fans
            yield (r * math.cos(theta), r * math.sin(theta))


def _resolve_pupil_radius(system: ghostlight.OpticalSystem, spec: SpotDiagramSpec) -> float:
    """Decide the entrance-pupil radius (mm) for this trace.

    Explicit ``pupil_radius_mm`` wins. Otherwise we fall back to the
    front surface's ``semi_aperture`` (the physical limit of the entrance
    aperture for a typical photographic lens) so the sampling fills the
    aperture without manual tweaking.
    """
    if spec.pupil_radius_mm > 0.0:
        return float(spec.pupil_radius_mm)
    try:
        return max(0.5, float(system.surfaces[0].semi_aperture))
    except Exception:
        # Mirrors ghostlight_designer.ray_tracing's "no idea, pick something
        # plausible" fallback so a half-built lens still produces output.
        return 10.0


def _launch_z(system: ghostlight.OpticalSystem) -> float:
    try:
        return float(system.surfaces[0].z) - _LAUNCH_MARGIN_MM
    except Exception:
        return -_LAUNCH_MARGIN_MM


# The sensor / image plane is the virtual z=0 plane — NOT
# ``system.surfaces[-1]``. ``OpticalSystem.finalize()`` rebases every
# lens surface so the last refracting interface ends up at some z < 0;
# rays propagate in +z and meet the sensor at z=0. Read the actual hit
# from ``path.result.position`` (see ``ghostlight_designer.tracing_util.landing_xy``).
SENSOR_Z_MM = 0.0


def _direction_for_field(tilt_x_deg: float, tilt_y_deg: float) -> "ghostlight.Vec3f":
    """Collimated direction vector for a given field tilt.

    Matches :func:`ghostlight_designer.ray_tracing._make_launch_geometry`'s
    PLANE_WF branch so the spot diagram traces the same rays the
    viewport draws.
    """
    bx = math.tan(math.radians(tilt_x_deg))
    by = math.tan(math.radians(tilt_y_deg))
    mag = math.sqrt(bx * bx + by * by + 1.0)
    return ghostlight.Vec3f(bx / mag, by / mag, 1.0 / mag)


def _status_ok(path) -> bool:
    """True iff the trace reached the sensor cleanly."""
    result = getattr(path, "result", None)
    if result is None:
        return False
    status = getattr(result, "status", None)
    name = getattr(status, "name", None)
    return name == "OK"


def _propagate_to_defocus(
    path,
    defocus_offsets_mm: Tuple[float, ...],
) -> Tuple[List[float], List[float]]:
    """Return (xs, ys) lists at z = SENSOR_Z_MM + d, one entry per offset.

    The Gaussian image plane is the virtual z=0 plane and the bound
    trace reports the actual sensor landing in ``path.result.position``
    — we use that as the d=0 anchor. Ray direction at the sensor is
    recovered from the last refracting surface's hit (``events[-1]``,
    at some z < 0) to the sensor landing: the segment between those
    two points is in image space (no further refraction), so a linear
    extrapolation in z is exact.

    NB: reading ``events[-1].hit_point`` as the image-plane landing is
    WRONG. That point sits on the last *lens* surface (z ≈ -30 mm for
    a typical photographic lens), not on the sensor — rays haven't
    converged yet at that depth, and the resulting "spot" was an
    artefact of the bundle's diameter at the lens back, not its focus.
    """
    n_def = len(defocus_offsets_mm)
    result = getattr(path, "result", None)
    pos = getattr(result, "position", None)
    if pos is None:
        return ([float("nan")] * n_def, [float("nan")] * n_def)

    try:
        sx = float(pos.x)
        sy = float(pos.y)
        sz = float(pos.z)
    except (AttributeError, TypeError):
        return ([float("nan")] * n_def, [float("nan")] * n_def)

    events = list(getattr(path, "events", None) or [])
    if not events:
        # No event history → no way to derive direction at the sensor.
        # Defocus offsets aren't meaningful here, so emit the sensor
        # landing for every slice.
        return ([sx] * n_def, [sy] * n_def)

    last = events[-1]
    try:
        lx = float(last.hit_point.x)
        ly = float(last.hit_point.y)
        lz = float(last.hit_point.z)
    except (AttributeError, TypeError):
        return ([sx] * n_def, [sy] * n_def)

    dz = sz - lz
    if abs(dz) < 1e-9:
        # Degenerate (sensor and last surface coincide, e.g. a one-
        # surface mirror). Defocus offsets don't have a well-defined
        # direction; return the sensor landing for every slice.
        return ([sx] * n_def, [sy] * n_def)
    dx_dz = (sx - lx) / dz
    dy_dz = (sy - ly) / dz

    xs: List[float] = []
    ys: List[float] = []
    for d in defocus_offsets_mm:
        xs.append(sx + d * dx_dz)
        ys.append(sy + d * dy_dz)
    return xs, ys


def compute_spot_diagram(
    system: ghostlight.OpticalSystem,
    spec: SpotDiagramSpec,
) -> SpotResult:
    """Run the trace and assemble a :class:`SpotResult`.

    Pure CPU. Safe to call from a worker thread — no Qt access. Reads
    the lens system through its bound C++ API only.
    """
    spec = spec.clamp()
    pupil_r = _resolve_pupil_radius(system, spec)
    launch_z = _launch_z(system)
    sensor_z = SENSOR_Z_MM

    samples = list(_ring_fan_samples(spec.rings, spec.fans))
    n_samples = len(samples)
    n_lambdas = len(spec.wavelengths_nm)
    n_defocus = len(spec.defocus_offsets_mm)

    field_results: List[SpotFieldResult] = []

    for (tx, ty) in spec.fields_deg:
        direction = _direction_for_field(tx, ty)
        # Pre-allocate NaN arrays — anything we don't fill in stays NaN
        # so the canvas can mask invalid samples without bookkeeping.
        xs = np.full((n_defocus, n_lambdas, n_samples), np.nan, dtype=np.float64)
        ys = np.full((n_defocus, n_lambdas, n_samples), np.nan, dtype=np.float64)
        valid = np.zeros((n_defocus, n_lambdas, n_samples), dtype=bool)

        for li, lam in enumerate(spec.wavelengths_nm):
            for si, (u, v) in enumerate(samples):
                px = u * pupil_r
                py = v * pupil_r
                origin = ghostlight.Vec3f(px, py, launch_z)
                try:
                    path = ghostlight.trace_primary_ray_diagnostic(
                        ghostlight.Ray(origin, direction, float(lam)),
                        system,
                    )
                except Exception:
                    continue
                if path is None or not _status_ok(path):
                    continue
                rx_list, ry_list = _propagate_to_defocus(
                    path, spec.defocus_offsets_mm
                )
                for di in range(n_defocus):
                    xs[di, li, si] = rx_list[di]
                    ys[di, li, si] = ry_list[di]
                    valid[di, li, si] = not (
                        math.isnan(rx_list[di]) or math.isnan(ry_list[di])
                    )

        field_results.append(SpotFieldResult(
            tilt_x_deg=float(tx),
            tilt_y_deg=float(ty),
            xs=xs,
            ys=ys,
            valid_mask=valid,
        ))

    return SpotResult(
        spec=spec,
        fields=tuple(field_results),
        wavelengths_nm=tuple(spec.wavelengths_nm),
        defocus_offsets_mm=tuple(spec.defocus_offsets_mm),
        pupil_radius_mm=pupil_r,
        sensor_z_mm=sensor_z,
    )
