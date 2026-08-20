"""Background gradient tracks the camera as a directional skybox.

Each pixel synthesises a world-space view direction from the camera basis
plus a small pseudo-FOV, and the gradient is driven by that direction's Y.
So orbiting (or pitching down/up) actually changes what the screen shows,
instead of just rotating a 2D world-up vector that collapses when looking
straight along Y.
"""

from __future__ import annotations

import pathlib

import numpy as np


def test_blit_frag_uses_camera_basis_uniforms():
    frag = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ghostlight_viewport" / "shaders" / "blit.frag"
    )
    src = frag.read_text(encoding="utf-8")
    assert "uniform vec3 u_cam_forward" in src
    assert "uniform vec3 u_cam_right" in src
    assert "uniform vec3 u_cam_up" in src
    assert "uniform float u_sky_fov_scale" in src
    # The gradient must be driven by the per-pixel world direction's Y, not
    # by NDC Y directly — otherwise looking straight down would still show
    # the same gradient as side view.
    assert "dir.y" in src or "normalize(" in src


def test_widget_uploads_camera_basis():
    widget = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ghostlight_viewport" / "widget.py"
    )
    src = widget.read_text(encoding="utf-8")
    for name in ("u_cam_forward", "u_cam_right", "u_cam_up", "u_sky_fov_scale"):
        assert name in src, f"widget does not upload {name}"


def test_orbit_yields_different_camera_forward():
    """Orbiting / pitching must change the camera forward vector in world space."""
    from ghostlight_viewport.camera import OrthographicCamera

    def forward(cam: OrthographicCamera) -> tuple[float, float, float]:
        V = cam.view_matrix()
        f = -V[2, :3]
        return float(f[0]), float(f[1]), float(f[2])

    cam = OrthographicCamera(viewport_w=512, viewport_h=512)
    cam.azimuth = 90.0
    cam.elevation = 0.0
    side = forward(cam)

    cam.azimuth = 0.0
    cam.elevation = 90.0
    top = forward(cam)

    assert side != top, "camera forward did not change after orbit"
    # Top-down view's forward should point predominantly along -Y in world.
    assert top[1] < -0.9, f"top view forward should be ~-Y, got {top}"
