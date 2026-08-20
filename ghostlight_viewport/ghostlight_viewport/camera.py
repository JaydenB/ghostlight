"""Orthographic camera with Maya-style mouse bindings and view-cube snaps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _mat_translate(x: float, y: float, z: float) -> np.ndarray:
    m = _mat_identity()
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def _mat_rotate_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = _mat_identity()
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def _mat_rotate_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    m = _mat_identity()
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _mat_ortho(left: float, right: float, bottom: float, top: float,
                near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


# ---------------------------------------------------------------------------
# Camera state
# ---------------------------------------------------------------------------

# Preset names map to (azimuth_deg, elevation_deg).  The default view
# (``"+x"``) looks down +X with +Y up — optical axis runs left-to-right in
# the widget.
#
# Convention: each named preset positions the camera ON its named axis,
# looking back at the origin (e.g. "+y" → camera at +Y, looking -Y, so the
# scene reads as "viewed from above").  Note the X-axis presets predate
# this convention and instead position the camera on the OPPOSITE axis
# (e.g. "+x" → camera at -X looking +X); the gizmo's FACE_PRESETS table
# compensates so cube clicks still feel right.
VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "+x":  ( 90.0,   0.0),
    "-x":  (-90.0,   0.0),
    "+y":  (  0.0,  90.0),
    "-y":  (  0.0, -90.0),
    "+z":  (  0.0,   0.0),
    "-z":  (180.0,   0.0),
    "iso": ( 45.0,  30.0),
}


@dataclass
class OrthographicCamera:
    """Trackball-style orthographic camera with pivot + ortho_height."""

    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    azimuth: float = 90.0       # degrees; rotation about world Y, "+X side" by default
    elevation: float = 0.0      # degrees
    dist: float = 100.0         # mm from pivot (only affects near/far)
    pan: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    ortho_height: float = 60.0  # mm of vertical extent visible
    viewport_w: int = 1
    viewport_h: int = 1

    # Input sensitivity tuning — tweak per host preference.
    orbit_sensitivity: float = 0.25   # degrees per pixel
    pan_sensitivity: float = 0.8      # multiplier on the px==view-px mapping
    dolly_sensitivity: float = 0.002  # exp() exponent per pixel
    wheel_sensitivity: float = 0.90   # zoom base per wheel detent

    def aspect(self) -> float:
        return max(1.0, self.viewport_w) / max(1.0, self.viewport_h)

    def view_matrix(self) -> np.ndarray:
        # World transform: translate by -pivot, rotate elev around X, rotate
        # az around Y, then translate by (-pan_x, -pan_y, -dist).
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        T_pivot = _mat_translate(-self.pivot[0], -self.pivot[1], -self.pivot[2])
        R = _mat_rotate_x(el) @ _mat_rotate_y(az)
        T_pan = _mat_translate(-float(self.pan[0]), -float(self.pan[1]), -float(self.dist))
        return (T_pan @ R @ T_pivot).astype(np.float32)

    def projection_matrix(self) -> np.ndarray:
        half_h = max(self.ortho_height, 1e-3) * 0.5
        half_w = half_h * self.aspect()
        near = -max(2.0 * self.dist, 1.0)
        far = max(2.0 * self.dist, 1.0)
        return _mat_ortho(-half_w, half_w, -half_h, half_h, near, far)

    def view_projection(self) -> np.ndarray:
        return self.projection_matrix() @ self.view_matrix()

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def fit_to_bbox(self, mn: np.ndarray, mx: np.ndarray, *, margin: float = 1.1) -> None:
        mn = np.asarray(mn, dtype=np.float32)
        mx = np.asarray(mx, dtype=np.float32)
        center = (mn + mx) * 0.5
        diag = float(np.linalg.norm(mx - mn))
        if diag < 1e-6:
            diag = 1.0
        self.pivot = center.astype(np.float32)
        self.dist = float(diag) * 2.0
        self.pan = np.zeros(2, dtype=np.float32)

        # Project the eight bbox corners into camera space so the framing
        # tracks the actual visible silhouette under the current view angle,
        # not the world-space diagonal. For a long thin lens viewed side-on
        # the diagonal is dominated by the optical axis (which is into the
        # screen), so fitting to the diagonal makes the lens look distant.
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        R = (_mat_rotate_x(el) @ _mat_rotate_y(az))[:3, :3]
        corners = np.array(
            [
                [x, y, z]
                for x in (float(mn[0]), float(mx[0]))
                for y in (float(mn[1]), float(mx[1]))
                for z in (float(mn[2]), float(mx[2]))
            ],
            dtype=np.float32,
        ) - center
        view_corners = corners @ R.T
        half_h = float(np.max(np.abs(view_corners[:, 1])))
        half_w = float(np.max(np.abs(view_corners[:, 0])))
        half_h_for_width = half_w / max(self.aspect(), 1e-6)
        target_half_h = max(half_h, half_h_for_width)
        if target_half_h < 1e-6:
            target_half_h = diag * 0.5
        self.ortho_height = 2.0 * target_half_h * margin

    def reset_view(self, mn: np.ndarray, mx: np.ndarray) -> None:
        self.fit_to_bbox(mn, mx)
        self.azimuth, self.elevation = VIEW_PRESETS["+x"]

    def set_preset(self, preset: str) -> None:
        key = preset.lower()
        if key not in VIEW_PRESETS:
            raise ValueError(
                f"unknown view preset {preset!r}; "
                f"expected one of {sorted(VIEW_PRESETS)}"
            )
        self.azimuth, self.elevation = VIEW_PRESETS[key]

    # ------------------------------------------------------------------
    # Maya-style mouse handling
    # ------------------------------------------------------------------

    def orbit(self, dx_px: float, dy_px: float) -> None:
        s = self.orbit_sensitivity
        self.azimuth = (self.azimuth + dx_px * s) % 360.0
        self.elevation = max(-89.0, min(89.0, self.elevation + dy_px * s))

    def pan_drag(self, dx_px: float, dy_px: float) -> None:
        # Scene follows the cursor; sensitivity tunes the px->view-px ratio.
        if self.viewport_h <= 0:
            return
        scale = self.ortho_height / max(1.0, self.viewport_h) * self.pan_sensitivity
        self.pan[0] -= float(dx_px) * scale
        self.pan[1] += float(dy_px) * scale  # screen Y is down, view Y is up

    def dolly(self, dy_px: float) -> None:
        self.ortho_height = max(
            0.1, self.ortho_height * math.exp(dy_px * self.dolly_sensitivity)
        )

    def wheel(self, delta_steps: float) -> None:
        # `delta_steps` is e.g. QWheelEvent.angleDelta().y() / 120
        self.ortho_height = max(
            0.1, self.ortho_height * (self.wheel_sensitivity ** delta_steps)
        )

    # ------------------------------------------------------------------
    # State save/restore
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "pivot": [float(x) for x in self.pivot],
            "azimuth": float(self.azimuth),
            "elevation": float(self.elevation),
            "dist": float(self.dist),
            "pan": [float(x) for x in self.pan],
            "ortho_height": float(self.ortho_height),
        }

    def load_state(self, d: dict) -> None:
        self.pivot = np.array(d.get("pivot", [0, 0, 0]), dtype=np.float32)
        self.azimuth = float(d.get("azimuth", 90.0))
        self.elevation = float(d.get("elevation", 0.0))
        self.dist = float(d.get("dist", 100.0))
        self.pan = np.array(d.get("pan", [0, 0]), dtype=np.float32)
        self.ortho_height = float(d.get("ortho_height", 60.0))
