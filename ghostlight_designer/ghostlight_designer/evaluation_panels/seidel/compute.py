"""Paraxial Seidel coefficient computation.

The five third-order monochromatic Seidel sums (S_I … S_V) plus the
two third-order chromatic sums (C_I, C_II), broken down per refracting
surface. All computed from a closed-form paraxial trace — no Monte
Carlo, no aspheric correction, no aperture vignetting.

Conventions (Welford / Smith):
* ``z`` increases toward the image. The sensor sits at ``z = 0``.
* ``y`` is the paraxial ray height at a surface; ``u`` is the slope
  (``dy/dz``). ``u`` is *after* refraction at the previous surface.
* The marginal ray launches from the object plane (here object-at-
  infinity, so ``u = 0``) at ``y = pupil_radius`` such that its height
  at the aperture stop equals the stop's semi-aperture.
* The chief ray launches at slope ``u = tan(field_deg)`` with a launch
  height chosen so that it crosses the aperture stop centre
  (``y = 0`` at the stop). For a system without a flagged stop, surface
  0 is treated as the stop.
* Lagrange invariant ``H = n·u·ȳ − n·ū·y`` is constant through the
  system (we verify this in the tests).

Seidel formulas (per surface, no conic term)::

    A   = n·i   = n·(c·y + u)         # refraction invariant, marginal
    Ā   = n·ī   = n·(c·ȳ + ū)         # refraction invariant, chief
    Δ(u/n) = u'/n' − u/n
    Δ(1/n) = 1/n' − 1/n

    S_I  = − A²  · y · Δ(u/n)         # spherical aberration
    S_II = − A · Ā · y · Δ(u/n)        # coma
    S_III = − Ā² · y · Δ(u/n)          # astigmatism
    S_IV = − H² · c · Δ(1/n)           # Petzval field curvature
    S_V  = (Ā/A) · (S_III + S_IV)       # distortion

    C_I  = − A · y · Δ(δn / n)         # axial colour, δn = n(λ_F) − n(λ_C)
    C_II = − Ā · y · Δ(δn / n)         # lateral colour

The stop surface (radius = 0 / curvature = 0) contributes nothing to
S_I…S_III (``c·y`` term zero) and nothing to S_IV (c = 0); its only
finite contribution is from the refraction itself which is also zero
(``Δ(u/n)`` collapses to ``Δu·(1/n)`` — and ``Δu = 0`` for c = 0).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import ghostlight

from .spec import SeidelSpec

_log = logging.getLogger("ghostlight_designer.evaluation_panels.seidel")


# Names of the seven aberrations in the order users see them on the chart.
SEIDEL_LABELS: Tuple[str, ...] = (
    "S_I (spherical)",
    "S_II (coma)",
    "S_III (astigmatism)",
    "S_IV (Petzval)",
    "S_V (distortion)",
    "C_I (axial colour)",
    "C_II (lateral colour)",
)
SEIDEL_KEYS: Tuple[str, ...] = (
    "spherical",
    "coma",
    "astigmatism",
    "petzval",
    "distortion",
    "axial_color",
    "lateral_color",
)


@dataclass(frozen=True)
class SeidelResult:
    """Output of one Seidel compute.

    All ``per_surface_*`` arrays are 1-D float arrays of length
    ``n_refracting`` — one entry per refracting surface (the sensor is
    excluded since there's no Δn there). ``surface_indices`` carries
    the original system-surface index for each entry so the bar chart
    can label them.
    """

    spec: SeidelSpec

    # System metadata for the title bar.
    field_deg: float
    pupil_radius_mm: float
    primary_wavelength_nm: float
    chromatic_wavelengths_nm: Tuple[float, float]  # (short, long); empty if N<2
    lagrange_invariant: float

    # Per refracting surface, length ``n_refracting``.
    surface_indices: np.ndarray  # (N,)
    surface_labels: Tuple[str, ...]  # readable surface names

    spherical_per_surface: np.ndarray
    coma_per_surface: np.ndarray
    astigmatism_per_surface: np.ndarray
    petzval_per_surface: np.ndarray
    distortion_per_surface: np.ndarray
    axial_color_per_surface: np.ndarray
    lateral_color_per_surface: np.ndarray

    # Convenience — sums for the title / status / sum bar.
    @property
    def sums(self) -> dict:
        return {
            "spherical": float(self.spherical_per_surface.sum()),
            "coma": float(self.coma_per_surface.sum()),
            "astigmatism": float(self.astigmatism_per_surface.sum()),
            "petzval": float(self.petzval_per_surface.sum()),
            "distortion": float(self.distortion_per_surface.sum()),
            "axial_color": float(self.axial_color_per_surface.sum()),
            "lateral_color": float(self.lateral_color_per_surface.sum()),
        }


# ---------------------------------------------------------------------------
# Surface walk — collect the data Seidel needs in a system-agnostic shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SurfaceData:
    """Geometry + indices for one refracting surface at one wavelength."""

    index: int            # system surface index
    name: str             # display label
    z: float              # vertex z coordinate
    radius: float         # 0 == flat
    n_before: float       # index of medium hit before this surface
    n_after: float        # index after this surface (== surfaces[k].ior at λ)
    is_stop: bool


def _collect_surfaces(
    system: ghostlight.OpticalSystem,
    wavelength_nm: float,
) -> List[_SurfaceData]:
    """Walk the system's surface chain at one wavelength.

    The last surface in :attr:`OpticalSystem.surfaces` is the image /
    flat termination (Ghostlight convention — sensor lives at ``z = 0``
    independent of that surface). We strip it: Seidel needs only the
    refracting surfaces. ``n_after`` for the last refracting surface
    is the medium between it and the sensor, which the C++ side stores
    on that surface's own ``ior``.
    """
    n_total = system.num_surfaces()
    if n_total < 2:
        return []

    # Determine refracting surface count. The last "surface" in the
    # system chain is the image plane terminator; drop it.
    refracting_count = n_total - 1
    out: List[_SurfaceData] = []
    # Object side starts in air.
    n_prev = 1.0
    for k in range(refracting_count):
        s = system.surfaces[k]
        try:
            n_after = float(s.ior_at(float(wavelength_nm)))
        except Exception:
            n_after = float(s.ior)
        if n_after <= 0.0 or not math.isfinite(n_after):
            n_after = 1.0
        out.append(_SurfaceData(
            index=k,
            name=f"S{k}",
            z=float(s.z),
            radius=float(s.radius),
            n_before=float(n_prev),
            n_after=float(n_after),
            is_stop=bool(s.is_stop),
        ))
        n_prev = n_after
    return out


# ---------------------------------------------------------------------------
# Paraxial trace
# ---------------------------------------------------------------------------


def _paraxial_trace(
    surfaces: List[_SurfaceData],
    *,
    y0: float,
    u0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paraxial trace through ``surfaces``.

    ``y0`` / ``u0`` are the ray's height and slope at the first
    refracting surface (i.e. immediately before its refraction —
    transfer from the object plane to surface 0 is the caller's job
    for an object-at-infinity launch we just need ``y0`` at surface 0).

    Returns three arrays of length ``len(surfaces)``::

        y[k] = ray height at surface k (before refraction)
        u[k] = ray slope before refraction at surface k
        u_out[k] = ray slope after refraction at surface k

    For an object at infinity ``u[0] == u0`` and ``y[0] == y0``.
    """
    n = len(surfaces)
    y = np.zeros(n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)
    u_out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return y, u, u_out

    y[0] = y0
    u[0] = u0
    for k in range(n):
        s = surfaces[k]
        c = 0.0 if abs(s.radius) < 1e-12 else 1.0 / s.radius
        # Snell-paraxial: n'·u' = n·u − (n' − n)·c·y
        # → u' = (n·u − (n' − n)·c·y) / n'
        u_out[k] = (s.n_before * u[k] - (s.n_after - s.n_before) * c * y[k]) / s.n_after
        if k + 1 < n:
            t = surfaces[k + 1].z - s.z
            y[k + 1] = y[k] + t * u_out[k]
            u[k + 1] = u_out[k]  # slope is preserved through the gap
    return y, u, u_out


def _find_stop_index(surfaces: List[_SurfaceData]) -> int:
    """Return the index of the surface flagged as stop, or 0 as fallback.

    Most lens files tag exactly one ``is_stop`` surface. When none is
    tagged we treat the front surface as the stop — same fallback the
    spot / field-diagram panels use for the pupil radius lookup.
    """
    for i, s in enumerate(surfaces):
        if s.is_stop:
            return i
    return 0


def _trace_marginal_and_chief(
    surfaces: List[_SurfaceData],
    *,
    pupil_radius_mm: float,
    field_deg: float,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
]:
    """Paraxial marginal + chief rays + Lagrange invariant.

    Marginal: object at infinity, height ``pupil_radius_mm`` at the
    aperture stop after refractions through the front group.

    Chief: object at infinity, slope ``tan(field_deg)``, launch height
    chosen so the chief ray crosses ``y = 0`` at the aperture stop
    (after the stop's refraction — but the stop is by definition flat /
    has c = 0 in practice, so before-refraction y = after-refraction y).

    Returns ``(marginal, chief, lagrange_invariant)``.
    """
    stop_k = _find_stop_index(surfaces)
    # Linearity trick: probe with (y0=1, u0=0) → record y at stop, call it
    # m1; probe with (y0=0, u0=1) → record y at stop, call it m2. Any
    # launch (y0, u0) gives y_at_stop = m1·y0 + m2·u0.
    y_m1, _, _ = _paraxial_trace(surfaces, y0=1.0, u0=0.0)
    y_m2, _, _ = _paraxial_trace(surfaces, y0=0.0, u0=1.0)
    m1 = float(y_m1[stop_k])
    m2 = float(y_m2[stop_k])

    # --- Marginal ---
    # Want y_at_stop = pupil_radius, u0 = 0  →  y0 = pupil_radius / m1.
    if abs(m1) < 1e-15:
        # Degenerate (parallel-plate stack with stop at index 0); fall
        # back to launching at pupil_radius itself.
        y0_marg = float(pupil_radius_mm)
    else:
        y0_marg = float(pupil_radius_mm) / m1
    y_marg, u_marg, u_out_marg = _paraxial_trace(
        surfaces, y0=y0_marg, u0=0.0,
    )

    # --- Chief ---
    u0_chief = math.tan(math.radians(float(field_deg)))
    # Want y_at_stop = 0  →  y0 = -u0·m2/m1.
    if abs(m1) < 1e-15:
        y0_chief = 0.0
    else:
        y0_chief = -u0_chief * m2 / m1
    y_chief, u_chief, u_out_chief = _paraxial_trace(
        surfaces, y0=y0_chief, u0=u0_chief,
    )

    # Lagrange invariant — evaluated at the first surface. For object
    # in air, ``n0 = 1``. Using ``H = n·u·ȳ − n·ū·y`` (Welford 4.21
    # without the sign-flipped form).
    if len(surfaces) > 0:
        n0 = surfaces[0].n_before
        H = n0 * (u_marg[0] * y_chief[0] - u_chief[0] * y_marg[0])
    else:
        H = 0.0

    return (
        (y_marg, u_marg, u_out_marg),
        (y_chief, u_chief, u_out_chief),
        float(H),
    )


# ---------------------------------------------------------------------------
# Seidel sums
# ---------------------------------------------------------------------------


def _seidel_per_surface(
    surfaces: List[_SurfaceData],
    marginal: Tuple[np.ndarray, np.ndarray, np.ndarray],
    chief: Tuple[np.ndarray, np.ndarray, np.ndarray],
    H: float,
) -> dict:
    """Compute S_I … S_V for every refracting surface."""
    y_m, u_m, u_out_m = marginal
    y_c, u_c, u_out_c = chief
    n = len(surfaces)
    out = {
        "spherical": np.zeros(n, dtype=np.float64),
        "coma": np.zeros(n, dtype=np.float64),
        "astigmatism": np.zeros(n, dtype=np.float64),
        "petzval": np.zeros(n, dtype=np.float64),
        "distortion": np.zeros(n, dtype=np.float64),
    }
    for k in range(n):
        s = surfaces[k]
        c = 0.0 if abs(s.radius) < 1e-12 else 1.0 / s.radius
        nb, na = s.n_before, s.n_after
        i_marg = c * y_m[k] + u_m[k]
        i_chief = c * y_c[k] + u_c[k]
        A = nb * i_marg
        A_bar = nb * i_chief
        du_over_n = u_out_m[k] / na - u_m[k] / nb
        d_one_over_n = 1.0 / na - 1.0 / nb

        s_i = -(A * A) * y_m[k] * du_over_n
        s_ii = -(A * A_bar) * y_m[k] * du_over_n
        s_iii = -(A_bar * A_bar) * y_m[k] * du_over_n
        s_iv = -(H * H) * c * d_one_over_n
        # S_V — the (Ā/A) factor is divergent on-axis (A ≈ 0); the
        # distortion contribution is also zero there, so we explicitly
        # set the result to 0 to avoid 0/0 NaN.
        if abs(A) < 1e-15:
            s_v = 0.0
        else:
            s_v = (A_bar / A) * (s_iii + s_iv)

        out["spherical"][k] = s_i
        out["coma"][k] = s_ii
        out["astigmatism"][k] = s_iii
        out["petzval"][k] = s_iv
        out["distortion"][k] = s_v
    return out


def _chromatic_per_surface(
    system: ghostlight.OpticalSystem,
    surfaces: List[_SurfaceData],
    marginal: Tuple[np.ndarray, np.ndarray, np.ndarray],
    chief: Tuple[np.ndarray, np.ndarray, np.ndarray],
    short_nm: float,
    long_nm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """C_I (axial colour) and C_II (lateral colour), per surface.

    Computed from ``δn = n(λ_short) − n(λ_long)`` at each surface,
    using the *primary*-wavelength paraxial trace (marginal + chief).
    The C sums share the same ``A / Ā / y`` factors as the
    monochromatic sums, but multiply by ``Δ(δn/n)`` instead of
    ``Δ(u/n)``.
    """
    y_m, u_m, _ = marginal
    y_c, u_c, _ = chief
    n = len(surfaces)
    c_i = np.zeros(n, dtype=np.float64)
    c_ii = np.zeros(n, dtype=np.float64)
    if short_nm <= 0.0 or long_nm <= 0.0 or short_nm == long_nm:
        return c_i, c_ii

    for k in range(n):
        s = surfaces[k]
        sys_surface = system.surfaces[s.index]
        try:
            n_after_short = float(sys_surface.ior_at(float(short_nm)))
            n_after_long = float(sys_surface.ior_at(float(long_nm)))
        except Exception:
            n_after_short = n_after_long = s.n_after
        delta_n_after = n_after_short - n_after_long

        if k == 0:
            # Object-side medium is air for every system we trace.
            delta_n_before = 0.0
        else:
            prev = system.surfaces[surfaces[k - 1].index]
            try:
                pn_short = float(prev.ior_at(float(short_nm)))
                pn_long = float(prev.ior_at(float(long_nm)))
            except Exception:
                pn_short = pn_long = surfaces[k - 1].n_after
            delta_n_before = pn_short - pn_long

        nb, na = s.n_before, s.n_after
        if abs(nb) < 1e-15 or abs(na) < 1e-15:
            continue
        c_curve = 0.0 if abs(s.radius) < 1e-12 else 1.0 / s.radius
        A = nb * (c_curve * y_m[k] + u_m[k])
        A_bar = nb * (c_curve * y_c[k] + u_c[k])
        d_dn_over_n = (delta_n_after / na) - (delta_n_before / nb)
        c_i[k] = -A * y_m[k] * d_dn_over_n
        c_ii[k] = -A_bar * y_m[k] * d_dn_over_n
    return c_i, c_ii


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_pupil_radius(
    system: ghostlight.OpticalSystem,
    explicit_mm: float,
) -> float:
    """Pupil radius for the marginal-ray launch.

    ``explicit_mm > 0`` wins. Otherwise prefer the flagged stop's
    semi-aperture, then the front surface's. Falls back to 10 mm so a
    half-built lens still produces a chart.
    """
    if explicit_mm > 0.0:
        return float(explicit_mm)
    try:
        for s in system.surfaces:
            if bool(s.is_stop) and float(s.semi_aperture) > 0.0:
                return float(s.semi_aperture)
        if system.num_surfaces() > 0:
            r = float(system.surfaces[0].semi_aperture)
            if r > 0.0:
                return r
    except Exception:
        pass
    return 10.0


def compute_seidel(
    system: ghostlight.OpticalSystem,
    spec: SeidelSpec,
) -> SeidelResult:
    """Compute per-surface Seidel + chromatic sums."""
    spec = spec.clamp()
    primary = spec.primary_wavelength_nm
    surfaces = _collect_surfaces(system, primary)
    n_refracting = len(surfaces)
    if n_refracting == 0:
        empty = np.zeros(0, dtype=np.float64)
        return SeidelResult(
            spec=spec,
            field_deg=float(spec.field_deg),
            pupil_radius_mm=0.0,
            primary_wavelength_nm=float(primary),
            chromatic_wavelengths_nm=(),
            lagrange_invariant=0.0,
            surface_indices=np.zeros(0, dtype=np.int64),
            surface_labels=tuple(),
            spherical_per_surface=empty,
            coma_per_surface=empty,
            astigmatism_per_surface=empty,
            petzval_per_surface=empty,
            distortion_per_surface=empty,
            axial_color_per_surface=empty,
            lateral_color_per_surface=empty,
        )

    pupil_r = _resolve_pupil_radius(system, spec.pupil_radius_mm)

    marginal, chief, H = _trace_marginal_and_chief(
        surfaces,
        pupil_radius_mm=pupil_r,
        field_deg=float(spec.field_deg),
    )

    mono = _seidel_per_surface(surfaces, marginal, chief, H)

    # Chromatic — bracket wavelengths. If only one wavelength configured,
    # leave the arrays at zero.
    wls = sorted(set(float(w) for w in spec.wavelengths_nm))
    if len(wls) >= 2:
        short_nm = wls[0]
        long_nm = wls[-1]
        c_i, c_ii = _chromatic_per_surface(
            system, surfaces, marginal, chief, short_nm, long_nm,
        )
        chrom_pair = (short_nm, long_nm)
    else:
        c_i = np.zeros(n_refracting, dtype=np.float64)
        c_ii = np.zeros(n_refracting, dtype=np.float64)
        chrom_pair = ()

    return SeidelResult(
        spec=spec,
        field_deg=float(spec.field_deg),
        pupil_radius_mm=float(pupil_r),
        primary_wavelength_nm=float(primary),
        chromatic_wavelengths_nm=chrom_pair,
        lagrange_invariant=float(H),
        surface_indices=np.array([s.index for s in surfaces], dtype=np.int64),
        surface_labels=tuple(s.name for s in surfaces),
        spherical_per_surface=mono["spherical"],
        coma_per_surface=mono["coma"],
        astigmatism_per_surface=mono["astigmatism"],
        petzval_per_surface=mono["petzval"],
        distortion_per_surface=mono["distortion"],
        axial_color_per_surface=c_i,
        lateral_color_per_surface=c_ii,
    )
