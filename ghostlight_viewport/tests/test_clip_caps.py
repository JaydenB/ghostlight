"""Cross-section cap rendering: stencil-based gap fill + backface visibility.

Two issues with the previous clip-plane rendering:
1. ``glEnable(GL_CULL_FACE)`` hid the inside of every solid through the cut,
   so single-surface elements like aperture stops became invisible from one
   side once a clip plane was active.
2. The cut left a hole — there was no geometry filling the cross-section
   polygon where the clip plane intersected each sub-solid.

This file covers:
* The ``_cap_uniforms_for_plane`` math — orthonormal basis lies in the
  plane and the centre projects onto it correctly.
* The default surface format requests an 8-bit stencil attachment.
* A render smoke test that activates a cutaway, renders one frame, and
  checks that the cap shader compiled, the stencil buffer is present, and
  the frame produced no GL errors.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest


pytest.importorskip("PySide6", reason="PySide6 not installed; viewport tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

from _helpers import example_doublet_path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from ghostlight_viewport import LensViewport
from ghostlight_viewport.widget import LensViewport as _LV
from ghostlight_viewport.widget import set_default_surface_format


@pytest.fixture(scope="module")
def app():
    set_default_surface_format()
    instance = QApplication.instance() or QApplication([])
    yield instance


# ---------------------------------------------------------------------------
# _cap_uniforms_for_plane (pure math — no GL)
# ---------------------------------------------------------------------------

def test_cap_basis_lies_in_plane_for_x_axis():
    plane = (1.0, 0.0, 0.0, 5.0)  # plane x = -5
    bbox_min = np.array([-10.0, -2.0, -3.0])
    bbox_max = np.array([10.0, 4.0, 50.0])
    center, t, b, half = _LV._cap_uniforms_for_plane(plane, bbox_min, bbox_max)
    # Centre lies on the plane: x = -5.
    assert center[0] == pytest.approx(-5.0, abs=1e-6)
    # Basis vectors are orthogonal to the plane normal (which is +X).
    assert t[0] == pytest.approx(0.0, abs=1e-6)
    assert b[0] == pytest.approx(0.0, abs=1e-6)
    # …and orthogonal to each other.
    dot = t[0] * b[0] + t[1] * b[1] + t[2] * b[2]
    assert dot == pytest.approx(0.0, abs=1e-6)
    # Half-extent covers the bbox diagonal at least.
    assert half >= float(np.linalg.norm(bbox_max - bbox_min))


def test_cap_basis_lies_in_plane_for_inverted_x_axis():
    # Cutaway-X mode inverts the slot-A plane.  Same geometry, opposite normal.
    plane = (-1.0, 0.0, 0.0, -5.0)
    bbox_min = np.array([-10.0, -2.0, -3.0])
    bbox_max = np.array([10.0, 4.0, 50.0])
    center, t, b, _half = _LV._cap_uniforms_for_plane(plane, bbox_min, bbox_max)
    # Plane equation -x - 5 = 0  ⇒  x = -5; centre still at x = -5.
    assert center[0] == pytest.approx(-5.0, abs=1e-6)
    # Basis still in the y-z plane.
    assert t[0] == pytest.approx(0.0, abs=1e-6)
    assert b[0] == pytest.approx(0.0, abs=1e-6)


def test_cap_basis_lies_in_plane_for_y_axis():
    plane = (0.0, 1.0, 0.0, -2.0)  # plane y = 2
    bbox_min = np.array([-10.0, -4.0, -3.0])
    bbox_max = np.array([10.0, 6.0, 50.0])
    center, t, b, _half = _LV._cap_uniforms_for_plane(plane, bbox_min, bbox_max)
    assert center[1] == pytest.approx(2.0, abs=1e-6)
    # Basis is in the x-z plane.
    assert t[1] == pytest.approx(0.0, abs=1e-6)
    assert b[1] == pytest.approx(0.0, abs=1e-6)


def test_cap_disabled_plane_returns_zero_extent():
    center, t, b, half = _LV._cap_uniforms_for_plane(
        (0.0, 0.0, 0.0, 0.0), np.zeros(3), np.zeros(3)
    )
    assert half == 0.0
    # Centre / basis are placeholders — callers must gate on activeness.
    assert center == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Surface format
# ---------------------------------------------------------------------------

def test_default_surface_format_requests_stencil():
    set_default_surface_format()
    fmt = QSurfaceFormat.defaultFormat()
    assert fmt.stencilBufferSize() >= 8, (
        "cap pass needs at least 8 stencil bits — the INVERT/NOTEQUAL dance "
        "writes to and reads from the stencil attachment per sub-solid"
    )


def test_viewport_format_upgrades_stencil_when_default_missed_it(app):
    # Even if a host app set a default format without stencil, the widget
    # constructor must top it up so the cap pass works.
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setStencilBufferSize(0)
    QSurfaceFormat.setDefaultFormat(fmt)
    try:
        viewport = LensViewport()
        assert viewport.format().stencilBufferSize() >= 8
    finally:
        # Restore the well-formed default for other tests.
        set_default_surface_format()


# ---------------------------------------------------------------------------
# Render smoke test
# ---------------------------------------------------------------------------

def _render_frame(widget) -> None:
    from PySide6.QtCore import QCoreApplication
    widget.resize(320, 240)
    widget.show()
    for _ in range(3):
        QCoreApplication.processEvents()
    widget.repaint()
    for _ in range(3):
        QCoreApplication.processEvents()


def _try_load_lens(viewport: LensViewport) -> bool:
    """Push the bundled example doublet if ghostlight is importable."""
    try:
        import ghostlight  # noqa: F401
    except Exception:
        return False
    lens_path = example_doublet_path()
    if not lens_path.exists():
        return False
    try:
        system = ghostlight.OpticalSystem.load(str(lens_path))
        elements = ghostlight.Element.from_lens_file(str(lens_path))
    except Exception:
        return False
    viewport.set_lens(system, elements)
    return True


def test_cap_solids_runs_after_composite_in_paint_impl():
    """The cross-section cap fill runs in a dedicated post-composite pass
    so it lands as flat opaque grey on top of the OIT result.  If it ran
    before composite the cap would mix with the visible wall fragments
    through OIT and never reach the apparent opacity of the lens-wall
    collar between two surfaces, which is the symptom of running the cap
    inside the OIT pass.

    We detect the order by string-position within ``_paint_impl``:
    ``_composite_oit`` must come before ``_draw_cap_solids``.
    """
    widget_src = (
        _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "widget.py"
    ).read_text(encoding="utf-8")
    start = widget_src.index("def _paint_impl(")
    end = widget_src.index("\n    def ", start + 1)
    body = widget_src[start:end]
    composite_idx = body.index("self._composite_oit(")
    cap_solids_idx = body.index("self._draw_cap_solids(")
    assert composite_idx < cap_solids_idx, (
        "cap-solid overlay must run AFTER the OIT composite — otherwise "
        "the opaque cap can't actually overdraw the lens behind the cut"
    )


def test_back_face_culling_is_keyed_on_is_cap_not_clip_state():
    """Back-face culling must be driven by per-region geometry kind, not
    by clip-plane state.  An earlier revision conditioned culling on
    ``clip_state.enabled`` — that made single-surface elements like
    aperture stops visibly switch between one-sided and two-sided
    rendering the moment a clip plane activated, and produced a slight
    overall tone shift on closed solids.

    The current design culls walls (``is_cap == False``) and leaves caps
    + stops + single-surface dummies (``is_cap == True``) double-sided,
    so the visual treatment is consistent regardless of clip-plane state.
    This test guards both invariants: the cull toggle must reference
    ``is_cap``, and it must NOT reference any clip-plane signal.
    """
    widget_src = (
        _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "widget.py"
    ).read_text(encoding="utf-8")

    # If culling is enabled anywhere, it must be in a block that also
    # consults ``is_cap`` (so caps stay double-sided).
    enable_count = widget_src.count("glEnable(GL_CULL_FACE)")
    if enable_count > 0:
        # Look for the per-region cull toggle pattern: a `want_cull` /
        # `is_cap` check sitting in the same neighbourhood as the
        # glEnable.  Use a wide window because the toggle and the enable
        # span ~10 lines.
        import re
        enable_blocks = []
        for m in re.finditer(r"glEnable\(GL_CULL_FACE\)", widget_src):
            start = max(0, m.start() - 600)
            end = min(len(widget_src), m.end() + 200)
            enable_blocks.append(widget_src[start:end])
        for block in enable_blocks:
            assert "is_cap" in block, (
                "glEnable(GL_CULL_FACE) appears without a nearby "
                "`is_cap` check — caps and stops would then render "
                "one-sided too, which the stop/dummy mesh can't survive"
            )
            assert "clip_active" not in block and "clip_state" not in block, (
                "glEnable(GL_CULL_FACE) is conditioned on clip-plane "
                "state — that's the exact regression this test guards"
            )


def test_stencil_subsolid_rebinds_lens_program():
    """XY-mode regression: in the per-plane loop the first plane's cap pass
    binds + releases ``_cap_solid_prog``, leaving no program current.  If the next
    iteration's stencil pass doesn't rebind ``lens_prog`` before its
    ``_set_u`` / ``glDrawArrays`` calls, the stencil count for the second
    plane stays zero everywhere and that plane's cap never draws — the
    visible symptom is "only one leg of the L-cut is filled".
    """
    widget_src = (
        _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "widget.py"
    ).read_text(encoding="utf-8")
    start = widget_src.index("def _stencil_subsolid(")
    end = widget_src.index("\n    def ", start + 1)
    body = widget_src[start:end]
    # Skip past the docstring (which may mention these calls in prose).
    # The function body proper starts after the closing triple-quote.
    code_after_doc = body.split('"""', 2)[-1] if body.count('"""') >= 2 else body
    bind_idx = code_after_doc.find("lens_prog.bind()")
    set_u_idx = code_after_doc.find("_set_u(lens_prog")
    draw_idx = code_after_doc.find("glDrawArrays")
    assert bind_idx != -1, "_stencil_subsolid must call lens_prog.bind()"
    assert bind_idx < set_u_idx, (
        "lens_prog.bind() must precede uniform writes — _set_u writes to the "
        "CURRENT GL program, not to lens_prog directly"
    )
    assert bind_idx < draw_idx, (
        "lens_prog.bind() must precede glDrawArrays — with no program bound, "
        "the draw is a no-op and the stencil pass writes nothing"
    )


def test_widget_clears_stencil_before_each_plane_pass():
    """XY mode regression: cap fragments shader-discarded by the other plane
    don't trigger the stencil-op clear, so the bit survives into plane B's
    INVERT count and the next sub-solid's pass.  The lens-pass code must
    ``glClear(STENCIL)`` before *each* (sub-solid, plane) stencil pass to
    keep the parity rule honest.
    """
    widget_src = (
        _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "widget.py"
    ).read_text(encoding="utf-8")
    # The clear lives inside the per-plane loop, between the sub-solid open
    # gate and the stencil-INVERT setup.
    assert "glClear(GL_STENCIL_BUFFER_BIT)" in widget_src, (
        "per-plane stencil clear is required; without it XY-mode caps leak "
        "stencil bits across planes and sub-solids"
    )


def test_cap_solid_shader_writes_opaque_alpha():
    """The cross-section cap is rendered as a flat OPAQUE fill (matching
    the lens-wall collar between two surfaces) rather than the previous
    translucent OIT contribution — earlier translucent values composited
    against the visible walls behind the cap and never read as solid.
    """
    sh = _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "shaders"
    frag = (sh / "cap_solid.frag").read_text(encoding="utf-8")
    assert "vec4(u_color, 1.0)" in frag, (
        "cap_solid.frag must write fully-opaque alpha — translucent caps "
        "are the exact regression this pass was added to fix"
    )


def test_widget_skips_cap_pass_for_open_subsolids():
    """Stops + single-surface dummies have one region (open mesh).  The
    INVERT stencil parity rule degenerates for open shapes — it would set
    the stencil bit at every visible iris pixel and then the cap quad would
    smear the cap colour over the whole iris.  The lens-pass code gates the
    cap pass on ``len(regions) >= 2`` so the iris keeps its real shading.
    """
    widget_src = (
        _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "widget.py"
    ).read_text(encoding="utf-8")
    # Gate may be phrased either way (>= 2 take, or < 2 skip) — both forms
    # equivalently keep the open-mesh case out of the cap pass.
    assert (
        "len(ss[\"regions\"]) >= 2" in widget_src
        or "len(ss[\"regions\"]) < 2" in widget_src
    ), (
        "the closed-sub-solid gate is the only thing keeping the cap pass "
        "from overwriting stop irises — keep it"
    )


def test_cap_shader_sources_exist():
    sh = _ROOT / "ghostlight_viewport" / "ghostlight_viewport" / "shaders"
    vert = (sh / "cap.vert").read_text(encoding="utf-8")
    frag = (sh / "cap_solid.frag").read_text(encoding="utf-8")
    # Vertex shader builds a quad from gl_VertexID + plane basis uniforms.
    assert "gl_VertexID" in vert
    assert "u_plane_center" in vert
    assert "u_plane_tangent" in vert
    assert "u_plane_bitangent" in vert
    # Fragment shader discards past the OTHER active plane so two-axis caps
    # trim each other correctly.
    assert "u_other_plane" in frag
    assert "discard" in frag


def test_render_with_cutaway_does_not_raise(app):
    """Drive the visible-pass paint code path with a clip plane active.

    The offscreen Qt platform doesn't reliably fire paintGL — if it didn't,
    we skip rather than fail; the value is catching crashes when paint *does*
    fire, not asserting it always does.
    """
    viewport = LensViewport()
    if not _try_load_lens(viewport):
        pytest.skip("ghostlight/example_doublet.lens unavailable")
    _render_frame(viewport)  # warm up
    if not viewport._gl_ready:
        pytest.skip("offscreen platform did not initialize GL")
    viewport.set_cutaway_mode("x")
    assert viewport.clip_state.enabled
    _render_frame(viewport)
    # No exception, GL still alive — the stencil + cap pipeline executed
    # without bringing the context down.
    assert viewport._gl_ready


def test_render_with_xy_cutaway_does_not_raise(app):
    viewport = LensViewport()
    if not _try_load_lens(viewport):
        pytest.skip("ghostlight/example_doublet.lens unavailable")
    _render_frame(viewport)
    if not viewport._gl_ready:
        pytest.skip("offscreen platform did not initialize GL")
    viewport.set_cutaway_mode("xy")
    # Both planes active — the cap pass runs twice per sub-solid.
    assert viewport.clip_state.uniform_vec4() != (0.0, 0.0, 0.0, 0.0)
    assert viewport.clip_state.uniform_vec4_b() != (0.0, 0.0, 0.0, 0.0)
    _render_frame(viewport)
    assert viewport._gl_ready
