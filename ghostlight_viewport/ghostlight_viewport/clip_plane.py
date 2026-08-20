"""Clip-plane state holder for the two-axis cross-section.

The viewport supports up to two simultaneous clip planes (slot ``a`` and slot
``b``).  Each plane is the half-space ``a*x + b*y + c*z + d > 0`` (discard).
A per-plane ``invert`` flag flips the kept side.  A ``symmetric`` flag mirrors
plane ``a`` through the origin into plane ``b`` so a single slider carves a
slab around the origin.

Shaders receive both planes as separate ``vec4`` uniforms (`u_clip_plane` /
`u_clip_plane_b`).  Disabled = ``(0,0,0,0)`` so the dot-product cheap-skip in
the shader is reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


Plane = tuple[float, float, float, float]
RayClipMode = Literal["segment", "origin"]


def _normalise(plane: Plane) -> Optional[Plane]:
    n = np.asarray(plane[:3], dtype=np.float64)
    nl = float(np.linalg.norm(n))
    if nl < 1e-9:
        return None
    n /= nl
    return (float(n[0]), float(n[1]), float(n[2]), float(plane[3]) / nl)


def _maybe_invert(plane: Optional[Plane], invert: bool) -> Optional[Plane]:
    if plane is None:
        return None
    if not invert:
        return plane
    return (-plane[0], -plane[1], -plane[2], -plane[3])


def _mirror_through_origin(plane: Optional[Plane]) -> Optional[Plane]:
    """Reflect ``a*x+b*y+c*z+d > 0`` through the origin: ``a*x+b*y+c*z-d > 0``
    points the opposite way with mirrored offset.  Use to form the second half
    of a symmetric slab around the origin from one user-set plane."""
    if plane is None:
        return None
    return (-plane[0], -plane[1], -plane[2], -plane[3])


@dataclass
class ClipPlaneState:
    """Up to two clip planes plus invert / symmetric flags.

    Axis-named helpers populate slot ``a`` (X) or ``b`` (Y) for clarity.
    """

    a_plane: Optional[Plane] = None       # slot A (X-axis by convention)
    b_plane: Optional[Plane] = None       # slot B (Y-axis by convention)
    a_invert: bool = False
    b_invert: bool = False
    symmetric: bool = False
    # How clip planes affect rays.  ``"segment"`` (default) chops each ray's
    # line draws at the plane (Liang-Barsky in ``bundle_to_segments``).
    # ``"origin"`` instead drops a ray entirely when its launch point lies in
    # the discard half-space, drawing surviving rays in full — useful when
    # you want the cross-section to select which rays you see rather than
    # which fragments of them.  Lens geometry is always cut at the plane.
    ray_clip_mode: RayClipMode = "segment"

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.a_plane is not None or self.b_plane is not None

    def clear(self) -> None:
        self.a_plane = None
        self.b_plane = None
        self.a_invert = False
        self.b_invert = False
        self.symmetric = False
        self.ray_clip_mode = "segment"

    # ------------------------------------------------------------------
    # Axis-named setters
    # ------------------------------------------------------------------
    def set_x(self, d: float) -> None:
        """X-axis clip plane at world ``x = -d``: discard ``x + d > 0``."""
        self.a_plane = _normalise((1.0, 0.0, 0.0, float(d)))

    def set_y(self, d: float) -> None:
        """Y-axis clip plane at world ``y = -d``: discard ``y + d > 0``."""
        self.b_plane = _normalise((0.0, 1.0, 0.0, float(d)))

    def clear_x(self) -> None:
        self.a_plane = None

    def clear_y(self) -> None:
        self.b_plane = None

    # ------------------------------------------------------------------
    # Uniform extraction
    # ------------------------------------------------------------------
    def _resolved_planes(self) -> tuple[Optional[Plane], Optional[Plane]]:
        """Return the two planes after applying invert + symmetric flags.

        When ``symmetric`` is on, slot B is overridden with the mirror of
        slot A through the origin (so one slider drives both walls of a slab).
        """
        a = _maybe_invert(self.a_plane, self.a_invert)
        if self.symmetric:
            b = _mirror_through_origin(a)
        else:
            b = _maybe_invert(self.b_plane, self.b_invert)
        return a, b

    def uniform_vec4(self) -> Plane:
        """Return slot-A uniform.  Disabled state is ``(0,0,0,0)``."""
        a, _ = self._resolved_planes()
        return a if a is not None else (0.0, 0.0, 0.0, 0.0)

    def uniform_vec4_b(self) -> Plane:
        """Return slot-B uniform.  Disabled state is ``(0,0,0,0)``."""
        _, b = self._resolved_planes()
        return b if b is not None else (0.0, 0.0, 0.0, 0.0)

    def uniform_vec4_pair(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Return both planes as one 8-tuple, useful for shader uniform unpack."""
        a = self.uniform_vec4()
        b = self.uniform_vec4_b()
        return (*a, *b)

    def both_planes(self) -> list[Plane]:
        """Return the active planes as a list (empty if both disabled).  CPU
        clipping (Liang-Barsky for rays) iterates this."""
        a, b = self._resolved_planes()
        out: list[Plane] = []
        if a is not None:
            out.append(a)
        if b is not None:
            out.append(b)
        return out
