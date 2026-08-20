"""View-cube renderer geometry + face → preset mapping.

Provides the per-vertex data for a small 3D cube drawn in the top-right
corner of the viewport.  Picking and rendering use the same VBO; the widget
sets per-face uniform tints.

Face index → preset name (consumed by :class:`OrthographicCamera.set_preset`):

    0: +X   1: -X
    2: +Y   3: -Y
    4: +Z   5: -Z
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Face index → preset name fired when that face is clicked in the pick FBO.
# X is swapped because the X-axis VIEW_PRESETS use the opposite convention
# from Y/Z (camera at -X for "+x", vs camera at +Y for "+y") — combined with
# the gizmo's intentionally-reversed Z projection, the cube face that *renders*
# at the +X position is actually the -X-encoded face, so clicking it should
# navigate to "+x".  Y/Z need no swap: their rotation convention and the
# reverse-Z projection cancel out.
FACE_PRESETS = ["-x", "+x", "+y", "-y", "+z", "-z"]
FACE_COLORS = [
    (0.78, 0.35, 0.35),  # +X warm
    (0.55, 0.25, 0.25),
    (0.35, 0.78, 0.45),  # +Y green
    (0.25, 0.55, 0.30),
    (0.35, 0.55, 0.85),  # +Z blue
    (0.25, 0.40, 0.65),
]


# --- Edge-slop hit testing ---------------------------------------------------
#
# When the camera is aligned with an axis, four of the six cube faces become
# edge-on and project to zero pixels — so the user can't click them without
# first nudging the view away from the axis.  EDGE_SLOP_PX widens each hidden
# face's clickable area into a small disc around its projected centroid (just
# outside the visible cube silhouette).  Set EDGE_SLOP_ENABLED = False to
# disable the feature entirely and revert to strict pixel-accurate picking.
EDGE_SLOP_ENABLED = True
EDGE_SLOP_PX = 14


def face_corners_local() -> np.ndarray:
    """Return the 4 corner vertices of each cube face, shape ``(6, 4, 3)``.

    Faces are ordered to match :data:`FACE_PRESETS`.  Within each face the
    corners trace the quad in order (suitable for a ``GL_LINE_LOOP`` outline
    or a centroid average).
    """
    s = 0.5
    return np.array([
        [(+s, -s, -s), (+s, +s, -s), (+s, +s, +s), (+s, -s, +s)],  # +X
        [(-s, +s, -s), (-s, -s, -s), (-s, -s, +s), (-s, +s, +s)],  # -X
        [(-s, +s, -s), (-s, +s, +s), (+s, +s, +s), (+s, +s, -s)],  # +Y
        [(+s, -s, -s), (+s, -s, +s), (-s, -s, +s), (-s, -s, -s)],  # -Y
        [(-s, -s, +s), (+s, -s, +s), (+s, +s, +s), (-s, +s, +s)],  # +Z
        [(+s, -s, -s), (-s, -s, -s), (-s, +s, -s), (+s, +s, -s)],  # -Z
    ], dtype=np.float32)


def face_centroids_local() -> np.ndarray:
    """Return the centroid of each cube face in cube-local coords, shape ``(6, 3)``."""
    return face_corners_local().mean(axis=1)


def hidden_face_slop_picks(
    camera_azimuth: float,
    camera_elevation: float,
    viewport_w: int,
    viewport_h: int,
    dpr: float = 1.0,
    corner_px: int = 120,
    margin_px: int = 16,
) -> list[tuple[int, float, float, float]]:
    """Slop pick targets for cube faces that are NOT the visible front face.

    Returns a list of ``(face_index, x_px, y_px, radius_px)`` tuples in
    physical-pixel coordinates with Y measured from the TOP of the widget —
    matching the click coords that :meth:`LensViewport._pick_at` receives.
    A click within ``radius_px`` of any returned centre snaps to that face,
    even if no pick fragment was actually rendered there (edge-on case).

    Returns an empty list when :data:`EDGE_SLOP_ENABLED` is False so the
    caller can guard with a cheap module-level check at runtime.
    """
    if not EDGE_SLOP_ENABLED or EDGE_SLOP_PX <= 0:
        return []

    vp, scissor_log = gizmo_view_proj(
        camera_azimuth, camera_elevation,
        viewport_w, viewport_h,
        corner_px=corner_px, margin_px=margin_px,
    )

    sx_log, sy_log_bot, sw_log, sh_log = scissor_log
    sx_phys = sx_log * dpr
    sw_phys = sw_log * dpr
    sh_phys = sh_log * dpr
    # gizmo_view_proj's scissor Y is measured from the BOTTOM (GL convention);
    # _pick_at uses TOP-down Qt coords.  Flip here so callers don't have to.
    sy_log_top = float(viewport_h) - (sy_log_bot + sh_log)
    sy_phys_top = sy_log_top * dpr

    centroids = face_centroids_local()
    projected: list[Optional[np.ndarray]] = []
    for c in centroids:
        clip = vp @ np.array([c[0], c[1], c[2], 1.0], dtype=np.float32)
        if abs(float(clip[3])) < 1e-9:
            projected.append(None)
        else:
            projected.append(clip[:3] / clip[3])

    # The face with the smallest NDC-Z wins GL_LESS in the pick pass — that's
    # the visible front face the user can already click pixel-accurately, so
    # don't generate a slop target for it.
    front_idx = -1
    front_z = float("inf")
    for i, ndc in enumerate(projected):
        if ndc is None:
            continue
        z = float(ndc[2])
        if z < front_z:
            front_z = z
            front_idx = i

    radius_phys = float(EDGE_SLOP_PX) * float(dpr)
    targets: list[tuple[int, float, float, float]] = []
    for i, ndc in enumerate(projected):
        if i == front_idx or ndc is None:
            continue
        px = sx_phys + (float(ndc[0]) * 0.5 + 0.5) * sw_phys
        # NDC Y points up, screen Y points down.
        py = sy_phys_top + (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * sh_phys
        targets.append((i, float(px), float(py), radius_phys))
    return targets


def build_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, normals, face_colors_per_vertex).

    Each face is two triangles, 6 vertices, with constant normal + colour.
    Shape: ``(36, 3)`` positions / normals / colours.  Faces are emitted in
    the same order as :data:`FACE_PRESETS`, so the picking pass can map a
    drawn triangle range back to a preset by integer division.
    """
    s = 0.5
    faces = [
        # +X: x=+s
        ((+s, -s, -s), (+s, +s, -s), (+s, +s, +s), (+s, -s, +s), (1, 0, 0), 0),
        # -X
        ((-s, +s, -s), (-s, -s, -s), (-s, -s, +s), (-s, +s, +s), (-1, 0, 0), 1),
        # +Y: y=+s
        ((-s, +s, -s), (-s, +s, +s), (+s, +s, +s), (+s, +s, -s), (0, 1, 0), 2),
        # -Y
        ((+s, -s, -s), (+s, -s, +s), (-s, -s, +s), (-s, -s, -s), (0, -1, 0), 3),
        # +Z
        ((-s, -s, +s), (+s, -s, +s), (+s, +s, +s), (-s, +s, +s), (0, 0, 1), 4),
        # -Z
        ((+s, -s, -s), (-s, -s, -s), (-s, +s, -s), (+s, +s, -s), (0, 0, -1), 5),
    ]

    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    cols: list[tuple[float, float, float]] = []
    for p0, p1, p2, p3, n, fi in faces:
        col = FACE_COLORS[fi]
        for tri in ((p0, p1, p2), (p0, p2, p3)):
            for p in tri:
                verts.append(p)
                norms.append(n)
                cols.append(col)

    return (
        np.array(verts, dtype=np.float32),
        np.array(norms, dtype=np.float32),
        np.array(cols, dtype=np.float32),
    )


def gizmo_view_proj(camera_azimuth: float, camera_elevation: float,
                     viewport_w: int, viewport_h: int,
                     corner_px: int = 120, margin_px: int = 16) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (view_proj 4x4 float32, scissor (x, y, w, h) in pixels) for the
    view-cube overlay in the top-right corner.

    The view-cube uses the same orientation as the main camera (so users
    visually map between them), but its translation is fixed (centred at the
    origin) and the projection is ortho to fit the cube.
    """
    import math

    az = math.radians(float(camera_azimuth))
    el = math.radians(float(camera_elevation))

    cx, sx = math.cos(el), math.sin(el)
    cy, sy = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0, 0],
                   [0, cx, -sx, 0],
                   [0, sx, cx, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy, 0],
                   [0, 1, 0, 0],
                   [-sy, 0, cy, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
    view = Rx @ Ry

    # Ortho fit the [-0.9, 0.9] cube extents in screen-space.
    # NOTE: m[2][2] = -1/half (the unconventional sign) is intentional — it
    # makes the cube face that's geometrically BEHIND the gizmo's origin-camera
    # win the depth test, which matches what the main camera *sees* at each
    # preset (the main camera is translated to +Y / +Z / etc. and looks back
    # at origin, so the world-side face the user "should" see corresponds to
    # the back-facing face in the gizmo's rotation-only view).  The X-axis
    # asymmetry from this hack is corrected via FACE_PRESETS, not here.
    half = 0.9
    proj = np.array([[1 / half, 0, 0, 0],
                     [0, 1 / half, 0, 0],
                     [0, 0, -1 / half, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    vp = proj @ view

    # Scissor in pixels: top-right
    x = max(0, int(viewport_w) - corner_px - margin_px)
    y = max(0, int(viewport_h) - corner_px - margin_px)
    return vp, (x, y, corner_px, corner_px)
