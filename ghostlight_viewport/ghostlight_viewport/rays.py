"""Ray bundle dataclass + ray-path -> line VBO builder.

A :class:`RayBundle` pairs a list of :class:`ghostlight.RayPath` objects with
their wavelengths (which RayPath itself doesn't carry).  The viewport calls
:func:`bundle_to_segments` to turn bundles into a single contiguous
``(xyz, rgba)`` line VBO, applying the active clip plane via Liang-Barsky
clipping where needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from .colors import wavelength_to_rgb

if TYPE_CHECKING:
    from ghostlight import RayPath, Vec3f


_OK_STATUS_NAMES = ("OK",)


@dataclass
class RayBundle:
    """One logical batch of traced rays the viewport should draw together.

    All lists are parallel; ``len(paths) == len(wavelengths_nm)``.  Optional
    ``origins`` / ``landings`` add the source-side and sensor-side legs that
    a bare ``RayPath`` (which only records per-surface events) doesn't carry.
    """

    paths: list
    wavelengths_nm: list[float]
    origins: Optional[list] = None        # list[Vec3f] | None
    landings: Optional[list] = None       # list[Vec3f] | None
    label: str = ""
    base_alpha: float = 1.0
    # When set, any ray that fails before reaching the sensor (vignetted,
    # TIR-trapped, etc.) is drawn in this RGB triple instead of its
    # wavelength colour.  Surviving rays in the same bundle keep the
    # wavelength colour.  Useful for highlighting vignetted ghost rays.
    dead_color_rgb: Optional[tuple[float, float, float]] = None
    # When True, every segment of every ray in this bundle draws at
    # ``base_alpha`` — no per-surface Fresnel attenuation, no dead-segment
    # dim.  Use when ray-energy attenuation would obscure geometry.
    flat_alpha: bool = False

    def __post_init__(self) -> None:
        if len(self.paths) != len(self.wavelengths_nm):
            raise ValueError(
                "RayBundle: paths and wavelengths_nm must be the same length "
                f"(got {len(self.paths)} vs {len(self.wavelengths_nm)})"
            )
        if self.origins is not None and len(self.origins) != len(self.paths):
            raise ValueError("RayBundle: origins must match paths length when provided")
        if self.landings is not None and len(self.landings) != len(self.paths):
            raise ValueError("RayBundle: landings must match paths length when provided")


def _vec3_to_tuple(v) -> tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


def _is_ok(event_or_result) -> bool:
    status = getattr(event_or_result, "status", None)
    if status is None:
        return False
    name = getattr(status, "name", None)
    if name is not None:
        return name in _OK_STATUS_NAMES
    return bool(status) is False  # fallback if status is an int with OK=0


def _liang_barsky_clip_segment(
    p0: np.ndarray, p1: np.ndarray, plane: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip segment p0->p1 to the half-space ``dot(plane.xyz, p) + plane.w <= 0``.

    Returns the (possibly shortened) segment, or None if both endpoints are on
    the discard side.
    """
    n = plane[:3]
    d = float(plane[3])
    f0 = float(np.dot(n, p0)) + d
    f1 = float(np.dot(n, p1)) + d
    if f0 <= 0.0 and f1 <= 0.0:
        return p0, p1
    if f0 > 0.0 and f1 > 0.0:
        return None
    # Straddle: solve for parameter t where f(t) = 0
    t = f0 / (f0 - f1)
    cut = p0 + t * (p1 - p0)
    if f0 <= 0.0:
        return p0, cut
    return cut, p1


def bundle_to_segments(
    bundles: Iterable[RayBundle],
    *,
    clip_plane: tuple[float, float, float, float] | None = None,
    clip_planes: Iterable[tuple[float, float, float, float]] | None = None,
    clip_mode: str = "segment",
    dead_alpha_scale: float = 0.3,
) -> np.ndarray:
    """Build an interleaved (xyz, rgba) line-segment VBO from RayBundles.

    Each *segment* contributes 2 vertices, 7 floats each (3 position + 4 colour).
    Returned shape: ``(N_VERTS, 7)`` float32, where ``N_VERTS = 2 * N_SEGMENTS``.
    Draw with ``GL_LINES``.

    Parameters
    ----------
    bundles:
        Iterable of :class:`RayBundle` objects.
    clip_plane:
        Single ``(a, b, c, d)`` with the half-space ``a*x+b*y+c*z+d > 0``
        discarded.  Pass ``None`` to skip clipping.  Kept for back-compat;
        prefer ``clip_planes`` for the two-plane viewport.
    clip_planes:
        Iterable of ``(a, b, c, d)`` tuples.  Segments are clipped against the
        intersection of all listed half-spaces.  Disabled (zero-normal) entries
        are silently ignored.
    clip_mode:
        How the planes act on rays.  ``"segment"`` (default) clips each line
        draw with Liang-Barsky so the visible portion of a ray ends at the
        plane.  ``"origin"`` instead drops the entire ray when its launch
        point lies in the discard half-space — surviving rays draw in full,
        even where they cross the plane.  The launch point is
        ``bundle.origins[i]`` when provided, otherwise the first event's hit.
    dead_alpha_scale:
        Multiplier applied to the segment leading into a failed event so dead
        rays remain visible but dim.
    """
    if clip_mode not in ("segment", "origin"):
        raise ValueError(
            f"clip_mode must be 'segment' or 'origin', got {clip_mode!r}"
        )
    raw_planes: list[tuple[float, float, float, float]] = []
    if clip_plane is not None:
        raw_planes.append(clip_plane)
    if clip_planes is not None:
        raw_planes.extend(clip_planes)

    planes: list[np.ndarray] = []
    for p in raw_planes:
        n = np.asarray(p[:3], dtype=np.float64)
        nl = float(np.linalg.norm(n))
        if nl > 1e-9:
            planes.append(np.array(
                [n[0] / nl, n[1] / nl, n[2] / nl, float(p[3]) / nl],
                dtype=np.float64,
            ))

    rows: list[np.ndarray] = []

    for bundle in bundles:
        dead_override = bundle.dead_color_rgb
        for i, (path, lam) in enumerate(zip(bundle.paths, bundle.wavelengths_nm)):
            events = list(path.events) if path.events is not None else []
            if not events:
                continue

            # Determine up front whether this ray was vignetted, so the whole
            # path can switch palette (instead of only the trailing segment).
            ray_died = any(not _is_ok(ev) for ev in events)
            if ray_died and dead_override is not None:
                r, g, b = dead_override
            else:
                r, g, b = wavelength_to_rgb(float(lam))
            base_rgb = np.array([r, g, b], dtype=np.float64) * float(bundle.base_alpha)

            points: list[tuple[float, float, float]] = []
            fresnels: list[float] = []
            dead_flags: list[bool] = []

            # Source-side leg, if origins provided
            if bundle.origins is not None:
                origin = bundle.origins[i]
                points.append(_vec3_to_tuple(origin))
                fresnels.append(1.0)
                dead_flags.append(False)

            # Walk events, stopping after the first non-OK one
            died_at: int | None = None
            for ev_idx, ev in enumerate(events):
                points.append(_vec3_to_tuple(ev.hit_point))
                if bundle.flat_alpha:
                    fresnels.append(1.0)
                else:
                    fresnels.append(float(max(0.05, min(1.0, ev.fresnel_weight))))
                dead_flags.append(False)
                if not _is_ok(ev):
                    died_at = ev_idx
                    if dead_flags:
                        dead_flags[-1] = True  # the segment leading into the death point dims
                    break

            # Sensor-side leg, if path completed OK
            if died_at is None:
                final_pos: tuple[float, float, float] | None = None
                if bundle.landings is not None and bundle.landings[i] is not None:
                    final_pos = _vec3_to_tuple(bundle.landings[i])
                elif path.result is not None and _is_ok(path.result):
                    fx = float(path.result.position.x)
                    fy = float(path.result.position.y)
                    final_pos = (fx, fy, 0.0)
                if final_pos is not None:
                    points.append(final_pos)
                    fresnels.append(fresnels[-1] if fresnels else 1.0)
                    dead_flags.append(False)

            if len(points) < 2:
                continue

            # Origin-cull mode tests the launch point once against every
            # active plane.  Whole ray in or whole ray out — per-segment
            # Liang-Barsky is skipped below so kept rays draw end to end,
            # even where they cross the plane.
            if clip_mode == "origin" and planes:
                origin_pt = np.asarray(points[0], dtype=np.float64)
                cull_ray = False
                for plane_n in planes:
                    if (
                        float(np.dot(plane_n[:3], origin_pt))
                        + float(plane_n[3])
                    ) > 0.0:
                        cull_ray = True
                        break
                if cull_ray:
                    continue

            for k in range(len(points) - 1):
                p0 = np.asarray(points[k], dtype=np.float64)
                p1 = np.asarray(points[k + 1], dtype=np.float64)
                a_seg = fresnels[k + 1] if (k + 1) < len(fresnels) else fresnels[k]
                if not bundle.flat_alpha and (
                    dead_flags[k + 1] if (k + 1) < len(dead_flags) else False
                ):
                    a_seg *= float(dead_alpha_scale)

                if clip_mode == "segment":
                    dropped = False
                    for plane_n in planes:
                        clipped = _liang_barsky_clip_segment(p0, p1, plane_n)
                        if clipped is None:
                            dropped = True
                            break
                        p0_new, p1_new = clipped
                        p0 = np.asarray(p0_new, dtype=np.float64)
                        p1 = np.asarray(p1_new, dtype=np.float64)
                    if dropped:
                        continue

                rgba = np.array([base_rgb[0], base_rgb[1], base_rgb[2], a_seg],
                                 dtype=np.float32)
                row0 = np.concatenate([p0.astype(np.float32), rgba])
                row1 = np.concatenate([p1.astype(np.float32), rgba])
                rows.append(row0)
                rows.append(row1)

    if not rows:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)
