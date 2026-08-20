"""First-order lens metrics for the viewport info bar.

Computes per-axis effective focal lengths, horizontal/vertical angle of
view, and anamorphic squeeze factor for a
:class:`ghostlight.OpticalSystem`.

EFFL is measured independently on the X and Y axes by tracing a small
paraxial ray parallel to the optical axis offset in each axis at the
d-line wavelength.  The output ray's slope (``dx/dz`` for the X probe,
``dy/dz`` for the Y probe) gives ``EFFL = -h / slope``.  For a
rotationally symmetric lens the two EFFLs are equal and the squeeze
ratio is 1.0; for an anamorphic / cylindrical system they diverge.

AFOV is reported separately for the two sensor axes:
``HFOV = 2·atan((W/2) / EFFL_x)``, likewise for V.

Returns ``None`` whenever either probe can't produce a meaningful
number (degenerate system, vignetting, TIR, output ray parallel to
axis, non-finite math).  The viewport hides the bar in that case so
the metrics for an unfinished or pathological lens stay out of the UI.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import ghostlight

from .ray_tracing import is_ready_to_trace

_log = logging.getLogger("ghostlight_designer.lens_metrics")


# Probe height (mm) for the paraxial EFFL trace.  Small enough to stay
# inside the paraxial regime even for fast lenses, large enough that the
# output slope clears single-precision trace noise for long focal lengths.
_PARAXIAL_PROBE_MM = 0.1

# Launch margin (mm) in front of surface 0.  Keeps the source plane clear
# of any front-element meniscus that might dip slightly into front of
# surface[0].z.
_LAUNCH_MARGIN_MM = 20.0

# d-line — wavelength dependence of EFFL is mild across the visible band;
# one trace at d-line is enough for a display value.
_PROBE_WAVELENGTH_NM = 587.56


@dataclass(frozen=True)
class LensMetrics:
    """Per-axis lens metrics for the info bar.

    ``efl_x_mm`` / ``efl_y_mm`` are the horizontal / vertical effective
    focal lengths (mm). ``hfov_deg`` / ``vfov_deg`` are the matching
    angles of view across the sensor's width / height (degrees).
    ``squeeze`` is the anamorphic squeeze factor (``efl_y / efl_x``):
    1.0 for spherical lenses, 2.0 for a "2× anamorphic", etc.
    """

    efl_x_mm: float
    efl_y_mm: float
    hfov_deg: float
    vfov_deg: float
    squeeze: float
    # Per-axis working f-number (EFFL / entrance-pupil diameter), read from
    # the C++ calibration — the SAME value that drives the starburst's
    # physical size.  ``None`` when the calibration can't produce it (partial
    # lens, no aperture stop); the info bar just omits the f-number then.
    f_number_x: Optional[float] = None
    f_number_y: Optional[float] = None

    @property
    def efl_mm(self) -> float:
        """Canonical scalar EFFL for display.

        Uses the vertical focal length — matches anamorphic naming
        convention (a "50 mm 2× anamorphic" has ``efl_y = 50``,
        ``efl_x = 25``).  For spherical lenses ``efl_x == efl_y`` so
        the choice doesn't matter.
        """
        return self.efl_y_mm

    @property
    def f_number(self) -> Optional[float]:
        """Canonical scalar f-number for display (vertical axis).

        Matches :attr:`efl_mm` — the vertical axis — so the pair reads
        consistently on anamorphics.
        """
        return self.f_number_y


def _effective_focal_length_on_axis(
    system: ghostlight.OpticalSystem, axis: str
) -> Optional[float]:
    """Estimate EFFL (mm) by probing along ``axis`` (``'x'`` or ``'y'``).

    Traces a small-height ray parallel to the optical axis through
    ``system``, offset in ``axis``, and reads the output slope along
    the same axis.  Returns ``None`` when the trace doesn't produce a
    usable result.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    if not is_ready_to_trace(system):
        return None

    try:
        front_z = float(system.surfaces[0].z)
    except Exception:
        return None

    z_launch = front_z - _LAUNCH_MARGIN_MM
    h = _PARAXIAL_PROBE_MM
    if axis == "x":
        origin = ghostlight.Vec3f(h, 0.0, z_launch)
    else:
        origin = ghostlight.Vec3f(0.0, h, z_launch)

    try:
        ray = ghostlight.Ray(
            origin,
            ghostlight.Vec3f(0.0, 0.0, 1.0),
            float(_PROBE_WAVELENGTH_NM),
        )
        path = ghostlight.trace_primary_ray_diagnostic(ray, system)
    except Exception:
        _log.exception(
            "lens_metrics: trace_primary_ray_diagnostic raised on axis=%s", axis
        )
        return None

    if path is None or not path.events:
        return None

    status_name = getattr(getattr(path.result, "status", None), "name", None)
    if status_name != "OK":
        # VIGNETTED / TIR / MISSED_SURFACE — the paraxial probe didn't
        # survive, so we can't read an output slope.  Treat as "no EFFL".
        return None

    last = path.events[-1]
    last_hp = last.hit_point
    sensor_pt = path.result.position

    if axis == "x":
        d_axis = float(sensor_pt.x) - float(last_hp.x)
    else:
        d_axis = float(sensor_pt.y) - float(last_hp.y)
    dz = float(sensor_pt.z) - float(last_hp.z)

    if abs(dz) < 1e-9:
        return None

    slope = d_axis / dz
    if abs(slope) < 1e-12:
        # Output is parallel to the axis — afocal / collimated.  No EFFL
        # to display.
        return None

    efl = -h / slope
    if not math.isfinite(efl):
        return None
    return float(efl)


def _fov_deg(efl_mm: float, span_mm: float) -> Optional[float]:
    """Half-angle-to-full-angle FOV (degrees) along one sensor axis.

    ``span_mm`` is the full sensor width or height; the chief-ray hits
    the corner at ``span_mm / 2``.  Returns ``None`` for invalid inputs.
    """
    if not (math.isfinite(efl_mm) and math.isfinite(span_mm)):
        return None
    if efl_mm <= 0.0 or span_mm <= 0.0:
        return None
    fov = 2.0 * math.degrees(math.atan((span_mm * 0.5) / efl_mm))
    if not math.isfinite(fov):
        return None
    return float(fov)


def _f_numbers(
    system: ghostlight.OpticalSystem,
) -> tuple[Optional[float], Optional[float]]:
    """Read the per-axis working f-number from the C++ calibration.

    This is the exact value the starburst pass consumes to set its
    physical size (``dx = lambda * f_number * pupil_fill``), so surfacing
    it in the info bar lets a wrong entrance-pupil / f-number solve be
    spotted at a glance.  Returns ``(None, None)`` — not an exception —
    for any lens the calibration can't fully solve.
    """
    try:
        cal = system.calibration()
    except Exception:
        _log.exception("lens_metrics: calibration() raised computing f-number")
        return (None, None)

    def _clean(value) -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if (math.isfinite(v) and v > 0.0) else None

    return (
        _clean(getattr(cal, "f_number_x", None)),
        _clean(getattr(cal, "f_number_y", None)),
    )


def compute_metrics(
    system: ghostlight.OpticalSystem,
    sensor_width_mm: float,
    sensor_height_mm: float,
) -> Optional[LensMetrics]:
    """Compute per-axis EFFL + AFOV + squeeze for a lens + sensor.

    Returns ``None`` if any required value can't be produced — callers
    are expected to hide / clear their UI in that case rather than
    display partial data.
    """
    efl_x = _effective_focal_length_on_axis(system, "x")
    efl_y = _effective_focal_length_on_axis(system, "y")
    if efl_x is None or efl_y is None:
        return None
    if efl_x <= 0.0 or efl_y <= 0.0:
        # Negative focal length means the probe ray diverged (system is
        # acting as a negative lens on that axis).  No useful AFOV in
        # that case — hide the bar.
        return None

    hfov = _fov_deg(efl_x, float(sensor_width_mm))
    vfov = _fov_deg(efl_y, float(sensor_height_mm))
    if hfov is None or vfov is None:
        return None

    squeeze = efl_y / efl_x
    if not math.isfinite(squeeze):
        return None

    fnum_x, fnum_y = _f_numbers(system)

    return LensMetrics(
        efl_x_mm=efl_x,
        efl_y_mm=efl_y,
        hfov_deg=hfov,
        vfov_deg=vfov,
        squeeze=float(squeeze),
        f_number_x=fnum_x,
        f_number_y=fnum_y,
    )
