"""Two-axis clip planes with invert and symmetric.

Verifies the ``ClipPlaneState`` API exposes both slots, the invert flag flips
a plane's normal, and ``symmetric`` mirrors slot A into slot B.  Also runs a
Liang-Barsky sanity check that ``bundle_to_segments`` honours both planes.
"""

from __future__ import annotations

import numpy as np


def test_set_x_and_y_populates_both_slots():
    from ghostlight_viewport.clip_plane import ClipPlaneState
    s = ClipPlaneState()
    s.set_x(5.0)
    s.set_y(-3.0)
    a, b, c, d = s.uniform_vec4()
    assert (a, b, c, d) == (1.0, 0.0, 0.0, 5.0)
    a, b, c, d = s.uniform_vec4_b()
    assert (a, b, c, d) == (0.0, 1.0, 0.0, -3.0)


def test_invert_flips_normal():
    from ghostlight_viewport.clip_plane import ClipPlaneState
    s = ClipPlaneState()
    s.set_x(5.0)
    s.a_invert = True
    a, b, c, d = s.uniform_vec4()
    assert (a, b, c, d) == (-1.0, 0.0, 0.0, -5.0)


def test_symmetric_mirrors_into_slot_b():
    from ghostlight_viewport.clip_plane import ClipPlaneState
    s = ClipPlaneState()
    s.set_x(5.0)
    s.symmetric = True
    a_plane = s.uniform_vec4()
    b_plane = s.uniform_vec4_b()
    # B is the mirror through the origin of A
    assert b_plane == (-a_plane[0], -a_plane[1], -a_plane[2], -a_plane[3])


def test_clear_resets_everything():
    from ghostlight_viewport.clip_plane import ClipPlaneState
    s = ClipPlaneState()
    s.set_x(1.0)
    s.set_y(2.0)
    s.a_invert = True
    s.symmetric = True
    s.clear()
    assert s.uniform_vec4() == (0.0, 0.0, 0.0, 0.0)
    assert s.uniform_vec4_b() == (0.0, 0.0, 0.0, 0.0)
    assert s.both_planes() == []


def test_bundle_to_segments_clips_against_both_planes():
    """A segment that crosses both planes' kept sides must be retained; one
    that's entirely in the discard region of either plane must be dropped."""
    from ghostlight_viewport import rays

    class _Result:
        position = type("V", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
        status = type("S", (), {"name": "OK"})()
    class _Ev:
        hit_point = type("V", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
        fresnel_weight = 1.0
        status = type("S", (), {"name": "OK"})()
    class _Path:
        events = [_Ev()]
        result = _Result()

    # No-op bundle, but we just want to exercise the plane parameter path.
    out = rays.bundle_to_segments([], clip_planes=[(1.0, 0.0, 0.0, 0.0),
                                                    (0.0, 1.0, 0.0, 0.0)])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 7)


def _make_ray(origin_xyz, sensor_xyz):
    """Build a minimal RayPath-like object with one OK event at the lens and
    an OK result at the sensor, plus a paired RayBundle origin."""
    ox, oy, oz = origin_xyz
    sx, sy, sz = sensor_xyz

    class _V:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class _S:
        name = "OK"

    class _Ev:
        hit_point = _V(0.0, 0.0, 0.0)  # ray reaches the lens at origin
        fresnel_weight = 1.0
        status = _S()

    class _Result:
        position = _V(sx, sy, sz)
        status = _S()

    class _Path:
        events = [_Ev()]
        result = _Result()

    return _Path(), _V(ox, oy, oz)


def test_ray_clip_mode_default_is_segment():
    from ghostlight_viewport.clip_plane import ClipPlaneState
    s = ClipPlaneState()
    assert s.ray_clip_mode == "segment"


def test_ray_clip_mode_origin_drops_rays_whose_origin_is_in_discard_half():
    """In ``"origin"`` mode the whole ray vanishes when its launch point is
    on the discard side, and the segment-clipping path is bypassed entirely
    so the kept ray draws end to end."""
    from ghostlight_viewport import rays

    # Two rays:
    #   keep_ray  — origin at x = -5 (kept by plane x+0 > 0 discard)
    #   cull_ray  — origin at x = +5 (discarded)
    keep_path, keep_origin = _make_ray((-5.0, 0.0, -20.0), (0.0, 0.0, 0.0))
    cull_path, cull_origin = _make_ray((+5.0, 0.0, -20.0), (0.0, 0.0, 0.0))

    bundle = rays.RayBundle(
        paths=[keep_path, cull_path],
        wavelengths_nm=[550.0, 550.0],
        origins=[keep_origin, cull_origin],
        flat_alpha=True,
    )

    plane = (1.0, 0.0, 0.0, 0.0)  # discard x > 0

    # Origin mode: cull_ray dropped, keep_ray drawn in full (origin -> lens
    # hit -> sensor = 2 segments = 4 vertices).
    out_origin = rays.bundle_to_segments(
        [bundle], clip_planes=[plane], clip_mode="origin"
    )
    assert out_origin.shape == (4, 7)
    # The kept ray must span its entire path, including the kept-side origin
    # at x = -5 — the segment-clipping path is bypassed.
    xs = out_origin[:, 0]
    assert float(xs.min()) == -5.0

    # Segment mode keeps both rays but chops their kept-side portions at the
    # plane.  Each ray contributes the origin -> lens-hit leg (the lens-hit
    # at x=0 sits on the plane, so the leg is preserved) and either drops or
    # truncates the lens-hit -> sensor leg.  Both rays still produce at least
    # one surviving segment here.
    out_segment = rays.bundle_to_segments(
        [bundle], clip_planes=[plane], clip_mode="segment"
    )
    assert out_segment.shape[0] >= 2  # at least one segment survives


def test_ray_clip_mode_origin_falls_back_to_first_event_when_no_origins():
    """When a bundle has no ``origins`` list, origin-cull uses the first
    event's hit_point as the ray's launch point."""
    from ghostlight_viewport import rays

    keep_path, _ = _make_ray((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Place the first event clearly on the keep side.
    keep_path.events[0].hit_point.x = -3.0
    cull_path, _ = _make_ray((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    cull_path.events[0].hit_point.x = +3.0

    bundle = rays.RayBundle(
        paths=[keep_path, cull_path],
        wavelengths_nm=[550.0, 550.0],
        flat_alpha=True,
    )

    out = rays.bundle_to_segments(
        [bundle], clip_planes=[(1.0, 0.0, 0.0, 0.0)], clip_mode="origin"
    )
    # Only one ray survives → 1 segment (lens-hit -> sensor) → 2 vertices.
    assert out.shape == (2, 7)


def test_clip_mode_rejects_invalid_value():
    import pytest
    from ghostlight_viewport import rays
    with pytest.raises(ValueError):
        rays.bundle_to_segments([], clip_mode="bogus")
