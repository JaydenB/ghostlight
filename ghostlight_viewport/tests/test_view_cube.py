"""View-cube face mapping covers all six axes.

The screen-space hit test that only mapped to +x/-x/+y/-y has been removed.
``gizmo.FACE_PRESETS`` must include all six face presets in the order the
picking pass writes them.
"""

from __future__ import annotations


def test_all_six_face_presets_present():
    from ghostlight_viewport import gizmo
    assert len(gizmo.FACE_PRESETS) == 6
    assert set(gizmo.FACE_PRESETS) == {"+x", "-x", "+y", "-y", "+z", "-z"}


def test_face_preset_order_matches_build_cube():
    """build_cube emits triangles face-by-face in the same order as
    ``FACE_COLORS``/``FACE_PRESETS``; verify each 6-vertex chunk carries the
    expected face colour (one constant colour per face, in FACE_COLORS order)."""
    import numpy as np
    from ghostlight_viewport import gizmo
    _verts, _normals, colors = gizmo.build_cube()
    # 6 vertices per face, in order 0..5
    for face in range(6):
        chunk = colors[face * 6:(face + 1) * 6]
        expected = np.array(gizmo.FACE_COLORS[face], dtype=np.float32)
        assert np.allclose(chunk, expected), (
            f"face {face} vertex chunk has colour {chunk[0]} vs {expected}"
        )


def test_camera_set_preset_supports_all_faces():
    from ghostlight_viewport.camera import OrthographicCamera, VIEW_PRESETS
    cam = OrthographicCamera()
    for preset in ("+x", "-x", "+y", "-y", "+z", "-z"):
        cam.set_preset(preset)
        az, el = VIEW_PRESETS[preset]
        assert cam.azimuth == az
        assert cam.elevation == el
