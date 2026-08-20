"""In-memory element-transform to surface-pose bake.

Transform composition and sensor rebasing must match ``src/optical_system.cpp``.
Inter-element thicknesses are authored geometry and are not modified here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from ._ghostlight import _OpticalSystem


__all__ = [
    "make_rotation",
    "element_world_pivot",
    "bake_system_poses",
]


_IDENTITY = (1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Small 3×3 / vector helpers (row-major, matching optical_system.cpp)
# ---------------------------------------------------------------------------

def make_rotation(
    tilt_x_deg: float, tilt_y_deg: float, roll_deg: float
) -> tuple[float, ...]:
    """Row-major 3×3 from Euler angles in degrees.

    Convention: ``R = Ry(tilt_y) · Rx(tilt_x) · Rz(roll)``.
    """
    tx = math.radians(tilt_x_deg)
    ty = math.radians(tilt_y_deg)
    rz = math.radians(roll_deg)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        cy * cz + sy * sx * sz,   -cy * sz + sy * sx * cz,   sy * cx,
        cx * sz,                   cx * cz,                 -sx,
        -sy * cz + cy * sx * sz,   sy * sz + cy * sx * cz,   cy * cx,
    )


def _mat3_mul_vec(m: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def _mat3_mul_mat(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    out = []
    for i in range(3):
        for j in range(3):
            out.append(sum(a[i * 3 + k] * b[k * 3 + j] for k in range(3)))
    return tuple(out)


class _Transform:
    """Mirror of the loader's ``struct Transform``."""

    __slots__ = ("pos", "rot", "piv_corr")

    def __init__(self) -> None:
        self.pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.rot: tuple[float, ...] = _IDENTITY
        self.piv_corr: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _transform_for(element: Any) -> _Transform:
    """Build the pre-rig transform for one element.

    ``Element.position`` is already resolved-absolute (``element.py`` collapses
    ``relative_to_preceding`` at parse time), so there is no mode pass here —
    that is the one step of ``flatten_optical_system`` this mirror can skip.
    """
    t = _Transform()
    pos = tuple(float(v) for v in element.position)
    t.pos = (pos[0], pos[1], pos[2])
    rot_deg = tuple(float(v) for v in element.rotation_euler_deg)
    t.rot = make_rotation(rot_deg[0], rot_deg[1], rot_deg[2])
    pivot = tuple(float(v) for v in getattr(element, "pivot", (0.0, 0.0, 0.0)))
    if pivot[0] or pivot[1] or pivot[2]:
        rp = _mat3_mul_vec(t.rot, pivot)
        t.piv_corr = (pivot[0] - rp[0], pivot[1] - rp[1], pivot[2] - rp[2])
    return t


def _apply_rig_pivot(
    offset_position: Sequence[float],
    offset_rotation: Sequence[float],
    pivot_point: Sequence[float],
    t: _Transform,
) -> None:
    """Mirror of ``apply_pivot_to_transform`` — the group-level ``pivots[]`` rig."""
    r_off = make_rotation(
        float(offset_rotation[0]), float(offset_rotation[1]), float(offset_rotation[2])
    )
    rel = (
        t.pos[0] - pivot_point[0],
        t.pos[1] - pivot_point[1],
        t.pos[2] - pivot_point[2],
    )
    rotated = _mat3_mul_vec(r_off, rel)
    t.pos = (
        rotated[0] + pivot_point[0] + float(offset_position[0]),
        rotated[1] + pivot_point[1] + float(offset_position[1]),
        rotated[2] + pivot_point[2] + float(offset_position[2]),
    )
    t.rot = _mat3_mul_mat(r_off, t.rot)
    t.piv_corr = _mat3_mul_vec(r_off, t.piv_corr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def element_world_pivot(system: "_OpticalSystem", element: Any) -> Optional[tuple[float, float, float]]:
    """World position of ``element``'s centre of rotation, or ``None``.

    Returns ``None`` when the pivot is all-zero (it then coincides with the
    element's front vertex and is not worth drawing) or when the element's
    surfaces don't resolve.

    Derived from **baked surface data**, not from ``Element.position``. The
    loader rebases the whole chain so the last surface lands at the sensor
    (``z = 0``), which puts authored element z and baked surface z in different
    frames; the first surface's pose is already in the frame callers draw in.
    Since surface 0 sits at ``origin + piv_corr`` and the pivot sits at
    ``origin + P``, the difference is exactly ``rot · P``.
    """
    pivot = tuple(float(v) for v in getattr(element, "pivot", (0.0, 0.0, 0.0)))
    if not (pivot[0] or pivot[1] or pivot[2]):
        return None
    try:
        indices = element.resolve_surfaces(system)
    except (KeyError, AttributeError):
        return None
    if not indices:
        return None
    idx = indices[0]
    if not (0 <= idx < len(system.surfaces)):
        return None
    s0 = system.surfaces[idx]
    rot = tuple(float(v) for v in s0.rot)
    offs = _mat3_mul_vec(rot, pivot)
    return (
        float(s0.decenter_x) + offs[0],
        float(s0.decenter_y) + offs[1],
        float(s0.z) + offs[2],
    )


def bake_system_poses(system: "_OpticalSystem") -> bool:
    """Recompute every surface's ``decenter_x/y``, ``z`` and ``rot`` in place.

    Mirrors ``flatten_optical_system``'s pose passes: build each element's
    transform, compose the ``pivots[]`` rig on top, map each surface's local
    ``(0, 0, cum_z)`` through the result, then rebase so the chain ends at the
    sensor (``z = 0``).

    Returns True when the poses were written, False when the system is empty or
    structurally unresolvable (stale surface UUIDs mid-edit), in which case
    nothing is touched.
    """
    elements = list(getattr(system, "elements", None) or [])
    if not elements:
        return False

    surfaces = system.surfaces
    try:
        ids = list(system.surface_ids)
    except Exception:
        return False
    lookup = {uuid: i for i, uuid in enumerate(ids)}

    # Resolve every UUID before mutating any surface.
    per_element: list[list[int]] = []
    for el in elements:
        indices: list[int] = []
        for uuid in el.surface_ids:
            idx = lookup.get(uuid)
            if idx is None or not (0 <= idx < len(surfaces)):
                return False
            indices.append(idx)
        if not indices:
            return False
        per_element.append(indices)

    # ---- Pass 1: per-element transforms -------------------------------
    transforms = [_transform_for(el) for el in elements]
    pre_pivot_z = [t.pos[2] for t in transforms]

    # ---- Pass 2: the group-level pivots[] rig -------------------------
    _apply_rig(system, elements, transforms)

    # ---- Pass 3: map local vertices through the transforms ------------
    # Stage poses so the update is atomic.
    staged: list[tuple[int, float, float, float, tuple[float, ...]]] = []
    for indices, M in zip(per_element, transforms):
        cum_z = 0.0
        for idx in indices:
            staged.append((
                idx,
                M.pos[0] + M.piv_corr[0] + M.rot[2] * cum_z,
                M.pos[1] + M.piv_corr[1] + M.rot[5] * cum_z,
                M.pos[2] + M.piv_corr[2] + M.rot[8] * cum_z,
                M.rot,
            ))
            cum_z += float(surfaces[idx].thickness)

    # ---- Rebase to sensor = 0 -----------------------------------------
    # Uses **pre-pivot** z, matching the loader: that keeps the sensor anchor
    # stable so a rig offset visibly moves an element instead of being
    # cancelled out by a re-anchor.
    internal_total = sum(float(surfaces[i].thickness) for i in per_element[-1])
    total = pre_pivot_z[-1] + internal_total

    for idx, dx, dy, z, rot in staged:
        surf = surfaces[idx]
        surf.decenter_x = dx
        surf.decenter_y = dy
        surf.z = z - total
        surf.rot = list(rot)
    return True


def _apply_rig(system: "_OpticalSystem", elements: Iterable[Any], transforms: list[_Transform]) -> None:
    """Compose ``system.pivots`` onto ``transforms`` in array order, in place."""
    pivots = list(getattr(system, "pivots", None) or [])
    if not pivots:
        return
    id_to_idx = {
        el.element_id: i
        for i, el in enumerate(elements)
        if getattr(el, "element_id", "")
    }
    for p in pivots:
        targets = [id_to_idx[eid] for eid in p.element_ids if eid in id_to_idx]
        if not targets:
            continue
        if getattr(p, "pivot_point_mode", "centroid") == "manual":
            point = tuple(float(v) for v in p.pivot_point)
        else:
            # Centroid of the *current* origins, so successive rigs see the
            # accumulated motion of the ones before them.
            n = float(len(targets))
            point = tuple(
                sum(transforms[i].pos[axis] for i in targets) / n
                for axis in range(3)
            )
        for i in targets:
            _apply_rig_pivot(p.offset_position, p.offset_rotation, point, transforms[i])
