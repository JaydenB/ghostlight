"""LensViewport — the QOpenGLWidget subclass.

Owns the GL context, compiles shaders, manages VBOs/VAOs/FBOs, and dispatches
mouse / keyboard events to the camera + picking + selection state.
"""

from __future__ import annotations

import ctypes
import math
import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import (
    QImage,
    QMouseEvent,
    QKeyEvent,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from . import gizmo as gizmo_mod
from . import picking, rays as rays_mod
from .camera import OrthographicCamera, VIEW_PRESETS
from .clip_plane import ClipPlaneState
from .colors import PALETTE
from .info_bar import InfoBar
from .scene import Scene
from .selection import SelectionState
from .sensor import CalibratedSensorSpec, SensorSpec
from .shaders import load as load_shader
from .viewport_toolbar import ViewportToolbar


# ---------------------------------------------------------------------------
# Minimal GL constants we need (avoid PyOpenGL dependency)
# ---------------------------------------------------------------------------

GL_COLOR_BUFFER_BIT    = 0x4000
GL_DEPTH_BUFFER_BIT    = 0x0100
GL_STENCIL_BUFFER_BIT  = 0x0400

GL_DEPTH_TEST          = 0x0B71
GL_BLEND               = 0x0BE2
GL_SCISSOR_TEST        = 0x0C11
GL_CULL_FACE           = 0x0B44
GL_STENCIL_TEST        = 0x0B90

GL_TRIANGLES           = 0x0004
GL_LINES               = 0x0001
GL_LINE_LOOP           = 0x0002

GL_UNSIGNED_BYTE       = 0x1401
GL_FLOAT               = 0x1406

GL_SRC_ALPHA              = 0x0302
GL_ONE_MINUS_SRC_ALPHA    = 0x0303
GL_ONE                    = 0x0001  # alias for blending; value-collides with GL_LINES (intentional)

GL_RGBA                = 0x1908
GL_LESS                = 0x0201
GL_LEQUAL              = 0x0203
GL_FRONT               = 0x0404
GL_BACK                = 0x0405

GL_TEXTURE_2D          = 0x0DE1
GL_TEXTURE0            = 0x84C0
GL_RGBA16F             = 0x881A
GL_FRAMEBUFFER         = 0x8D40

GL_KEEP                = 0x1E00
GL_ZERO_OP             = 0x0000  # GL_ZERO for stencil ops (same value as GL_ZERO blend factor)
GL_INVERT              = 0x150A
GL_ALWAYS              = 0x0207
GL_NOTEQUAL            = 0x0205

GL_LINE_SMOOTH         = 0x0B20
GL_LINE_SMOOTH_HINT    = 0x0C52
GL_NICEST              = 0x1102


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERR_LOG_PATH = os.path.join(
    os.path.expanduser("~"), ".ghostlight_viewport_errors.log"
)
_ERR_LOG_PRINTED_HEADER = False


def _log_gl_info(widget) -> None:
    """Write a banner with GL driver info to the error log on each init."""
    try:
        from PySide6.QtGui import QOpenGLContext
        ctx = QOpenGLContext.currentContext()
        actual_version = "<no context>"
        actual_profile = "<no context>"
        if ctx is not None:
            actfmt = ctx.format()
            actual_version = f"{actfmt.majorVersion()}.{actfmt.minorVersion()}"
            actual_profile = str(actfmt.profile())

        widget_fmt = widget.format()
        req_version = f"{widget_fmt.majorVersion()}.{widget_fmt.minorVersion()}"
        req_profile = str(widget_fmt.profile())

        info = (
            f"\n--- ghostlight_viewport init at pid={os.getpid()} ---\n"
            f"Requested format: {req_version} profile={req_profile}\n"
            f"Actual context:   {actual_version} profile={actual_profile}\n"
        )
        with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(info)
    except Exception:
        pass


def install_global_error_logging() -> None:
    """Route Python exceptions + Qt warnings to ``~/.ghostlight_viewport_errors.log``.

    Qt's event loop silently swallows exceptions raised from Python overrides
    of virtual methods (paintGL, mousePressEvent, ...) — this hook captures
    them post-mortem.  Also routes ``qDebug``/``qWarning``/``qCritical``
    messages through Qt's message handler.

    Safe to call multiple times; only the first installs the hooks.
    """
    import sys as _sys
    if getattr(install_global_error_logging, "_installed", False):
        return
    install_global_error_logging._installed = True

    previous_excepthook = _sys.excepthook
    def hook(exc_type, exc, tb):
        try:
            import traceback as _tb
            msg = "".join(_tb.format_exception(exc_type, exc, tb))
            with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write("[unhandled exception]\n" + msg + "\n")
            _sys.stderr.write("[ghostlight_viewport] " + msg)
        except Exception:
            pass
        previous_excepthook(exc_type, exc, tb)
    _sys.excepthook = hook

    try:
        from PySide6.QtCore import qInstallMessageHandler
        def qt_msg_handler(mode, ctx, msg):
            try:
                with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(f"[qt:{mode}] {msg}\n")
            except Exception:
                pass
        qInstallMessageHandler(qt_msg_handler)
    except Exception:
        pass


def _log_gl_error(stage: str) -> None:
    """Always print + append the current exception to a log file.

    Qt swallows Python exceptions raised from QOpenGLWidget overrides — they
    print to stderr only sometimes, and never reach the user's terminal in
    GUI-launched contexts.  Writing to a known log file is the only reliable
    way to surface the crash post-mortem.
    """
    global _ERR_LOG_PRINTED_HEADER
    import sys
    import traceback as _tb
    tb = _tb.format_exc()
    msg = f"[ghostlight_viewport:{stage}]\n{tb}\n"
    try:
        sys.stderr.write(msg)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
            if not _ERR_LOG_PRINTED_HEADER:
                fh.write("=" * 60 + "\n")
                _ERR_LOG_PRINTED_HEADER = True
            fh.write(msg)
    except Exception:
        pass


def _setup_attribute(prog, name: str, vbo, components: int, stride: int, offset: int) -> None:
    """Wire a VBO into a vertex attribute via QOpenGLShaderProgram.setAttributeBuffer.

    PySide6's raw ``glVertexAttribPointer`` silently fails to bind attributes
    (subsequent draws use NaN-poisoned vertices and get clipped, producing a
    blank framebuffer with no errors).  The high-level Qt helper
    ``setAttributeBuffer`` accepts an integer offset and a numeric attribute
    location and reliably wires up the attribute.

    Must be called while the target VAO is bound.  The VBO is bound
    internally and left unbound on return.
    """
    loc = prog.attributeLocation(name)
    if loc < 0:
        return
    vbo.bind()
    prog.enableAttributeArray(loc)
    prog.setAttributeBuffer(loc, GL_FLOAT, int(offset), int(components), int(stride))
    vbo.release()


def _voidp(offset: int):
    """Wrap a buffer offset as ctypes.c_void_p for PySide6's GL bindings.

    PySide6's QOpenGLFunctions rejects bare ints for the GLvoid* argument of
    glVertexAttribPointer; the binding only accepts ctypes.c_void_p (or bytes
    for client-side arrays).

    NOTE: ``glDrawElements`` with a c_void_p offset is *broken* in PySide6 —
    the driver treats the offset as a CPU pointer and crashes with
    STATUS_STACK_BUFFER_OVERRUN.  This file therefore uses ``glDrawArrays``
    with unrolled vertex data exclusively; the EBO path is intentionally
    absent.  See ``_upload_element_buffers``.
    """
    return ctypes.c_void_p(int(offset))


def _set_u(prog, name: str, *values):
    """setUniformValue wrapper that handles PySide6's binding quirks.

    PySide6's ``QOpenGLShaderProgram.setUniformValue`` has multiple silent
    bugs:

    1. Name-based overloads (``name: bytes``) fail overload resolution even
       with documented argument shapes.
    2. ``setUniformValue(loc, float)`` issues GL_INVALID_OPERATION inside the
       driver because the binding dispatches to ``glUniform1i`` rather than
       ``glUniform1f``.

    For scalar float values we therefore drop down to the raw GL call.
    Matrix and vec2/3/4 forms (where the location-int overload works) go
    through setUniformValue as before.
    """
    if not values:
        return
    loc = prog.uniformLocation(name)
    if loc < 0:
        return  # uniform optimised out by the compiler — silent skip
    if len(values) == 1:
        v = values[0]
        if isinstance(v, float):
            gl = _gl()
            if gl is not None:
                gl.glUniform1f(loc, float(v))
            return
        if isinstance(v, int):
            gl = _gl()
            if gl is not None:
                gl.glUniform1i(loc, int(v))
            return
        prog.setUniformValue(loc, v)
        return
    prog.setUniformValue(loc, *values)


def _gl():
    """Return the GL ES 2.0 subset function object for the current context."""
    from PySide6.QtGui import QOpenGLContext
    ctx = QOpenGLContext.currentContext()
    return ctx.functions() if ctx is not None else None


_MITER_STRIDE_FLOATS = 10   # this(3) + prev(3) + next(3) + side(1)
_MITER_STRIDE_BYTES = _MITER_STRIDE_FLOATS * 4

# Six-vertex pattern that turns one polyline segment into two triangles.
# Index 0 picks the segment's *start* polyline vertex, index 1 picks the
# *end* vertex; each output vertex carries (this, prev_of_this,
# next_of_this, side) so the shader can compute its miter offset.
_QUAD_PICK_START  = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
_QUAD_SIDE        = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)


def _expand_polyline_to_miter_quads(loop: np.ndarray) -> np.ndarray:
    """Closed polyline ``V[0..N-1]`` → ``(6*N, 10)`` miter-joined quad buffer.

    Each segment ``(V[i], V[i+1 mod N])`` becomes 2 triangles.  Every
    output vertex carries its own polyline neighbours so the shader can
    bisect the two adjacent segment directions and offset along the
    miter normal — without that, adjacent segments meet at slightly
    different angles and the joints show as a high-frequency wiggle on
    tessellated rim circles.
    """
    loop = np.asarray(loop, dtype=np.float32)
    n = loop.shape[0]
    if n < 2:
        return np.zeros((0, _MITER_STRIDE_FLOATS), dtype=np.float32)

    prev_v = np.roll(loop, +1, axis=0)
    next_v = np.roll(loop, -1, axis=0)

    # Stack the start- and end-of-segment attributes, so per-output-vertex
    # we pick column 0 (start) or 1 (end) based on the 6-vertex pattern.
    starts = np.stack([loop, prev_v, next_v], axis=1)             # (N, 3, 3)
    ends   = np.stack(
        [np.roll(loop, -1, axis=0),
         np.roll(prev_v, -1, axis=0),
         np.roll(next_v, -1, axis=0)],
        axis=1,
    )                                                              # (N, 3, 3)
    pair = np.stack([starts, ends], axis=1)                        # (N, 2, 3, 3)

    pick = _QUAD_PICK_START
    selected = pair[:, pick, :, :]                                 # (N, 6, 3, 3)
    flattened = selected.reshape(n, 6, 9)
    sides = np.broadcast_to(_QUAD_SIDE[None, :, None], (n, 6, 1))
    packed = np.concatenate([flattened, sides], axis=2)            # (N, 6, 10)
    return packed.reshape(-1, _MITER_STRIDE_FLOATS).astype(np.float32)


def _expand_disjoint_segments_to_miter_quads(
    starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Disjoint ``(M, 3)`` start/end pairs → ``(6*M, 10)`` quad buffer.

    Used for clip-plane cross-section segments where each segment has
    no polyline context.  Sets ``prev == this`` at each start vertex
    and ``next == this`` at each end vertex; the shader detects those
    degeneracies and falls back to a perpendicular butt cap, so the
    same fragment path handles both rim loops and disjoint segments.
    """
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    m = starts.shape[0]
    if m == 0:
        return np.zeros((0, _MITER_STRIDE_FLOATS), dtype=np.float32)

    # At the start vertex of each segment: prev = self (degenerate), next = end.
    # At the end vertex:                    prev = start,             next = self.
    start_attrs = np.stack([starts, starts, ends], axis=1)         # (M, 3, 3)
    end_attrs   = np.stack([ends,   starts, ends], axis=1)         # (M, 3, 3)
    pair = np.stack([start_attrs, end_attrs], axis=1)              # (M, 2, 3, 3)

    selected = pair[:, _QUAD_PICK_START, :, :]                     # (M, 6, 3, 3)
    flattened = selected.reshape(m, 6, 9)
    sides = np.broadcast_to(_QUAD_SIDE[None, :, None], (m, 6, 1))
    packed = np.concatenate([flattened, sides], axis=2)            # (M, 6, 10)
    return packed.reshape(-1, _MITER_STRIDE_FLOATS).astype(np.float32)


class _GLBuffer:
    """Tiny wrapper around QOpenGLBuffer that also tracks element count."""

    def __init__(self, buftype):
        self.buf = QOpenGLBuffer(buftype)
        self.buf.create()
        self.count: int = 0

    def upload(self, data: np.ndarray) -> None:
        self.buf.bind()
        b = np.ascontiguousarray(data).tobytes()
        self.buf.allocate(b, len(b))
        self.count = int(data.shape[0]) if data.ndim > 0 else 0
        self.buf.release()

    def bind(self) -> None:
        # Called by ``_setup_attribute`` so the wrapper can be passed in place
        # of the bare QOpenGLBuffer.
        self.buf.bind()

    def release(self) -> None:
        self.buf.release()

    def destroy(self) -> None:
        self.buf.destroy()


# ---------------------------------------------------------------------------
# Default surface format helper
# ---------------------------------------------------------------------------

# Auto-install the global handlers on import so any user gets reliable error
# capture without needing to call install_global_error_logging() explicitly.
install_global_error_logging()


def set_default_surface_format() -> None:
    """Request a GL 3.3 core profile with depth + stencil before QApplication.

    Call this *before* constructing the QApplication for the most reliable
    behaviour.  Idempotent.

    The 8-bit stencil attachment backs the cross-section cap pass: each
    sub-solid's cut polygon is built by INVERT-ing stencil along the clip
    plane, then filled by the cap-quad shader.  Without a stencil buffer the
    cap pass silently no-ops and the clip cut shows as a hole.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


# ---------------------------------------------------------------------------
# LensViewport
# ---------------------------------------------------------------------------

class LensViewport(QOpenGLWidget):
    """3D viewport for a :class:`ghostlight.OpticalSystem`.

    Push lens data with :meth:`set_lens`, ray traces with
    :meth:`set_trace_results`, and sensor extents with :meth:`set_sensor`.
    Emits :sig:`elementSelected` on click.
    """

    elementSelected = Signal(object)
    surfaceSelected = Signal(object)  # int (global surface index) or None
    # Plain (unmodified) right-click on a pickable element/surface. Payload is
    # a dict {mode, element_index, element, surface_index, global_pos}; a host
    # wires this to a context menu. Not emitted in "none" mode or when the
    # context menu is disabled (read-only viewports).
    contextMenuRequested = Signal(object)
    viewChanged = Signal(object)
    clipPlaneChanged = Signal(object)
    contextReady = Signal()

    _CYCLE_TOLERANCE_PX = 4

    # Duration of the gizmo / preset-driven camera transition, in milliseconds.
    # Set to 0 (or anything <= 0) to snap instantly.  ~250ms reads as a clear
    # "I'm moving here" without feeling slow.
    view_transition_duration_ms: int = 250

    def __init__(self, parent=None):
        super().__init__(parent)
        # Apply a sensible format if the application didn't set one globally.
        fmt = QSurfaceFormat.defaultFormat()
        if fmt.version() < (3, 3):
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            fmt.setStencilBufferSize(8)
            fmt.setSamples(4)
        elif fmt.stencilBufferSize() < 8:
            # Default format was set (e.g. by set_default_surface_format) but
            # is missing the stencil bits the cap pass needs.  Top up.
            fmt.setStencilBufferSize(8)
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # State
        self.camera = OrthographicCamera()
        self.clip_state = ClipPlaneState()
        self.selection = SelectionState()
        self.scene = Scene()
        self.sensor: Optional[SensorSpec] = None
        self.calibrated_sensor: Optional[CalibratedSensorSpec] = None
        self._show_calibrated_sensor: bool = False
        self.ray_bundles: list[rays_mod.RayBundle] = []
        self._system = None
        self._show_axes = True
        self._show_view_cube = True
        # Element centre-of-rotation markers. On by default — they only draw
        # for elements that actually declare a non-zero pivot, so the default
        # costs nothing on an on-axis lens and means a user who sets a pivot
        # sees where it is without hunting for a menu item.
        self._show_pivots = True
        self._bg_top = PALETTE["background_top"]
        self._bg_bottom = PALETTE["background_bottom"]
        # Pseudo-FOV that controls how strongly the directional sky gradient
        # curves across the frame.  ~0.7 gives a noticeable wrap without
        # exaggerating it for an ortho viewport.
        self._sky_fov_scale = 0.7

        # GL resources, created in initializeGL
        self._gl_ready = False
        self._lens_prog: Optional[QOpenGLShaderProgram] = None
        self._ray_prog: Optional[QOpenGLShaderProgram] = None
        self._sensor_prog: Optional[QOpenGLShaderProgram] = None
        self._gizmo_prog: Optional[QOpenGLShaderProgram] = None
        self._blit_prog: Optional[QOpenGLShaderProgram] = None
        self._cap_solid_prog: Optional[QOpenGLShaderProgram] = None
        # OIT (order-independent transparency) — see _ensure_oit_fbo /
        # _composite_oit / shaders/composite.frag for the per-pixel
        # additive-accumulator approach used by the lens pass.
        self._composite_prog: Optional[QOpenGLShaderProgram] = None
        self._composite_vao: Optional[QOpenGLVertexArrayObject] = None
        # Stroke outlines for rim seams + clip-plane cross-sections.  Static
        # rim buffers are populated per sub-solid in _upload_element_buffers;
        # the cross-section buffer is rebuilt on demand whenever the clip
        # plane state changes.
        self._outline_prog: Optional[QOpenGLShaderProgram] = None
        self._cross_section_vbo: Optional[_GLBuffer] = None
        self._cross_section_vao: Optional[QOpenGLVertexArrayObject] = None
        self._cross_section_count: int = 0
        self._cross_section_cache_key: Optional[tuple] = None
        self._oit_fbo: Optional[QOpenGLFramebufferObject] = None
        self._oit_fbo_size: tuple[int, int] = (0, 0)
        self._element_buffers: list[dict] = []
        self._ray_vbo: Optional[_GLBuffer] = None
        self._ray_vao: Optional[QOpenGLVertexArrayObject] = None
        self._ray_segments: int = 0
        self._sensor_quad_vbo: Optional[_GLBuffer] = None
        self._sensor_quad_vao: Optional[QOpenGLVertexArrayObject] = None
        self._sensor_border_vbo: Optional[_GLBuffer] = None
        self._sensor_border_vao: Optional[QOpenGLVertexArrayObject] = None
        self._cal_sensor_disk_vbo: Optional[_GLBuffer] = None
        self._cal_sensor_disk_ibo: Optional[_GLBuffer] = None
        self._cal_sensor_disk_vao: Optional[QOpenGLVertexArrayObject] = None
        self._cal_sensor_disk_count: int = 0
        self._cal_sensor_ring_vbo: Optional[_GLBuffer] = None
        self._cal_sensor_ring_vao: Optional[QOpenGLVertexArrayObject] = None
        self._cal_sensor_ring_count: int = 0
        self._gizmo_vbo: Optional[_GLBuffer] = None
        self._gizmo_normal_vbo: Optional[_GLBuffer] = None
        self._gizmo_color_vbo: Optional[_GLBuffer] = None
        self._gizmo_vao: Optional[QOpenGLVertexArrayObject] = None
        self._gizmo_outline_vbo: Optional[_GLBuffer] = None
        self._gizmo_outline_vao: Optional[QOpenGLVertexArrayObject] = None
        # Index of the gizmo face under the cursor (visible or slop-hit),
        # or None when not hovering the cube.  Drives the highlight tint
        # in _draw_view_cube.
        self._gizmo_hover_face: Optional[int] = None
        self._fullscreen_vao: Optional[QOpenGLVertexArrayObject] = None
        self._axes_vbo: Optional[_GLBuffer] = None
        self._axes_vao: Optional[QOpenGLVertexArrayObject] = None
        self._axes_segments: int = 0
        self._pivot_vbo: Optional[_GLBuffer] = None
        self._pivot_vao: Optional[QOpenGLVertexArrayObject] = None
        self._pivot_segments: int = 0
        self._pick_fbo: Optional[QOpenGLFramebufferObject] = None
        self._pick_fbo_size: tuple[int, int] = (0, 0)

        # Mouse state
        self._mouse_button: Optional[Qt.MouseButton] = None
        self._mouse_modifiers: Qt.KeyboardModifiers = Qt.NoModifier
        self._mouse_last: Optional[QPoint] = None
        self._mouse_press_pos: Optional[QPoint] = None
        self._dragging_camera = False

        # Whether a plain right-click emits contextMenuRequested. Read-only
        # viewports (e.g. the optimization preview) turn this off; "none"
        # selection mode also suppresses it independently.
        self._context_menu_enabled = True

        # Picking buffer dirty flag — recompute lazily on click
        self._pick_dirty = True

        # Click-through cycle: repeated clicks near the same logical pixel
        # walk from the front-most element to the furthest by accumulating an
        # exclude set across renders.  Reset whenever the click position
        # moves more than `_CYCLE_TOLERANCE_PX`.
        self._cycle_pos: Optional[QPoint] = None
        self._cycle_excluded: set[int] = set()
        self._cycle_snapshot: Optional[tuple] = None

        # Camera transition animation (gizmo / preset clicks).  ``_view_anim``
        # holds (start_az, start_el, target_az, target_el, QElapsedTimer).
        # The timer ticks at ~60Hz and lerps az/el until elapsed reaches
        # ``view_transition_duration_ms``.
        self._view_anim_timer: Optional[QTimer] = None
        self._view_anim: Optional[tuple] = None

        # Floating overlay toolbar (top-right, left of the gizmo).  Pure Qt
        # child widget; never touches GL directly — it calls the existing
        # public setters which already route through `_refresh_gl`.
        self._toolbar = ViewportToolbar(self)
        self._toolbar.cutawayChanged.connect(self._apply_cutaway)
        self._toolbar.selectionModeChanged.connect(self._on_selection_mode_changed)
        self._toolbar.raise_()
        self._position_toolbar()

        # Bottom-centered info-bar overlay (lens metrics text).  Pure Qt
        # child like the toolbar; the host pushes a formatted string via
        # ``set_info_text`` and we hide whenever that string is empty.
        self._info_bar = InfoBar(self)
        self._info_bar.raise_()
        self._position_info_bar()

    # ------------------------------------------------------------------
    # Overlay layout
    # ------------------------------------------------------------------

    def _position_toolbar(self) -> None:
        # Reserve the gizmo's corner: corner_px=120 + margin_px=16 (defaults
        # of gizmo.gizmo_view_proj); +8 px gap so the toolbar doesn't kiss
        # the cube's hit area.
        gizmo_reserve = 120 + 16
        tb = self._toolbar
        tb.adjustSize()
        tw, th = tb.width(), tb.height()
        x = max(0, self.width() - gizmo_reserve - tw - 8)
        y = 16
        tb.setGeometry(x, y, tw, th)

    def _position_info_bar(self) -> None:
        bar = self._info_bar
        if not bar.isVisible():
            # Still set a geometry so when it becomes visible the size is
            # current — adjustSize updates it on set_text but the position
            # math below needs valid width/height.
            bar.adjustSize()
        bw, bh = bar.width(), bar.height()
        margin_bottom = 12
        x = max(0, (self.width() - bw) // 2)
        y = max(0, self.height() - bh - margin_bottom)
        bar.setGeometry(x, y, bw, bh)

    # ------------------------------------------------------------------
    # Public API — data
    # ------------------------------------------------------------------

    def _refresh_gl(self, *uploads) -> None:
        """Run ``_upload_*`` helpers with this widget's GL context current.

        Public setters fire from outside ``paintGL`` (signal handlers, host
        app callbacks), but ``QOpenGLWidget`` releases its context between
        paint events. Calling raw GL through ``QOpenGLBuffer`` /
        ``QOpenGLVertexArrayObject`` / ``QOpenGLShaderProgram`` with no
        current context silently writes to nothing — then the next paint
        dereferences invalid IDs and segfaults inside the driver. Always
        route post-init uploads through here.
        """
        if not self._gl_ready:
            return
        self.makeCurrent()
        try:
            for fn in uploads:
                fn()
        finally:
            self.doneCurrent()
        self.update()

    def set_lens(
        self,
        system,
        elements: list,
        *,
        fit_view: bool = True,
        ghost_solo_surface_indices: set | frozenset | None = None,
    ) -> None:
        """Push a system + element list into the viewport.

        ``ghost_solo_surface_indices`` (optional) is the set of global
        surface indices the user has marked ghost-solo in the designer.
        Elements containing a solo'd surface render with a brighter
        accent tint + near-opaque alpha. None / empty produces the
        normal lens render with no solo highlight.
        """
        self._system = system
        self.scene.rebuild(
            system,
            elements,
            ghost_solo_surface_indices=ghost_solo_surface_indices,
        )
        self.scene.expand_bbox_with_sensor(self.sensor)
        if self._show_calibrated_sensor:
            self.scene.expand_bbox_with_calibrated_sensor(self.calibrated_sensor)
        if fit_view:
            self.reset_view()
        self._pick_dirty = True
        # Hover may still point at an Element that no longer belongs to the
        # scene; clear it so the next move re-resolves against the new set.
        self.selection.clear_hover()
        self._refresh_gl(
            self._upload_element_buffers,
            self._upload_axes_buffers,
            self._upload_pivot_buffers,
        )

    def update_surface(self, surface_index: int) -> None:
        if self._system is None:
            return
        # Find the element that owns this surface and refresh just that one.
        for se in self.scene.elements:
            try:
                indices = se.element.resolve_surfaces(self._system)
            except KeyError:
                continue
            if surface_index in indices:
                self.scene.update_element_at(self._system, se.index, se.element)
                self.scene.expand_bbox_with_sensor(self.sensor)
                if self._show_calibrated_sensor:
                    self.scene.expand_bbox_with_calibrated_sensor(self.calibrated_sensor)
                self._pick_dirty = True
                # Pivot markers are sized off the scene bbox, which
                # ``update_element_at`` can move, so they refresh alongside
                # the element geometry.
                self._refresh_gl(
                    self._upload_element_buffers, self._upload_pivot_buffers
                )
                return

    def set_trace_results(self, bundles: list) -> None:
        self.ray_bundles = list(bundles)
        self._refresh_gl(self._upload_ray_buffers)

    def clear_trace_results(self) -> None:
        self.ray_bundles = []
        self._refresh_gl(self._upload_ray_buffers)

    def set_sensor(self, sensor: Optional[SensorSpec]) -> None:
        self.sensor = sensor
        self.scene.expand_bbox_with_sensor(sensor)
        self._refresh_gl(
            self._upload_sensor_buffers,
            self._upload_axes_buffers,
            self._upload_pivot_buffers,
        )

    def set_sensor_size(self, half_w: float, half_h: float) -> None:
        """Resize the sensor quad in-place, keeping the existing z and label.

        Useful for hosts that drive sensor dimensions independently of the
        lens calibration (e.g. matching the DCC's render output spec).  No-op
        when no sensor is set.
        """
        if self.sensor is None:
            return
        self.sensor = SensorSpec(
            half_w=float(half_w),
            half_h=float(half_h),
            pixel_w=self.sensor.pixel_w,
            pixel_h=self.sensor.pixel_h,
            label=self.sensor.label,
        )
        self.scene.expand_bbox_with_sensor(self.sensor)
        self._refresh_gl(
            self._upload_sensor_buffers,
            self._upload_axes_buffers,
            self._upload_pivot_buffers,
        )

    def set_calibrated_sensor(self, calibration) -> None:
        """Set the circular 'calibrated sensor' overlay.

        Accepts a :class:`ghostlight.LensCalibration` (translated via
        :meth:`CalibratedSensorSpec.from_calibration`) or a pre-built
        :class:`CalibratedSensorSpec`.  Pass ``None`` to clear.  Visibility is
        independently controlled via :meth:`set_show_calibrated_sensor`.
        """
        if calibration is None:
            self.calibrated_sensor = None
        elif isinstance(calibration, CalibratedSensorSpec):
            self.calibrated_sensor = calibration
        else:
            self.calibrated_sensor = CalibratedSensorSpec.from_calibration(calibration)
        if self._show_calibrated_sensor:
            self.scene.expand_bbox_with_calibrated_sensor(self.calibrated_sensor)
        self._refresh_gl(
            self._upload_calibrated_sensor_buffers,
            self._upload_axes_buffers,
            self._upload_pivot_buffers,
        )

    def set_show_calibrated_sensor(self, visible: bool) -> None:
        """Toggle visibility of the calibrated-sensor circle."""
        visible = bool(visible)
        if visible == self._show_calibrated_sensor:
            return
        self._show_calibrated_sensor = visible
        if visible:
            self.scene.expand_bbox_with_calibrated_sensor(self.calibrated_sensor)
        self._refresh_gl(self._upload_axes_buffers, self._upload_pivot_buffers)

    # ------------------------------------------------------------------
    # Public API — view
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        self._cancel_view_animation()
        self.camera.reset_view(self.scene.bbox_min, self.scene.bbox_max)
        self.viewChanged.emit(self.camera.state())
        self.update()

    def set_view(self, preset: str) -> None:
        key = preset.lower()
        if key not in VIEW_PRESETS:
            self.camera.set_preset(preset)  # raises ValueError with the right msg
            return
        target_az, target_el = VIEW_PRESETS[key]
        if key in ("+y", "-y"):
            # For top/bottom snaps, azimuth at el=±90 is just screen rotation
            # around the vertical axis — snapping it to the preset's value
            # spins the world for no spatial reason and disorients the user.
            # Round to the nearest 90° so we still land on an axis-aligned
            # heading (max ~45° rotation) without the big flip.
            target_az = round(float(self.camera.azimuth) / 90.0) * 90.0
        if int(self.view_transition_duration_ms) <= 0:
            self._cancel_view_animation()
            self.camera.azimuth = float(target_az)
            self.camera.elevation = float(target_el)
            self.viewChanged.emit(self.camera.state())
            self.update()
            return
        self._animate_to_view(float(target_az), float(target_el))

    def set_camera_state(self, state: dict) -> None:
        self._cancel_view_animation()
        self.camera.load_state(state)
        self.viewChanged.emit(self.camera.state())
        self.update()

    def camera_state(self) -> dict:
        return self.camera.state()

    def _animate_to_view(self, target_az: float, target_el: float) -> None:
        """Lerp the camera's azimuth/elevation to the target over
        :attr:`view_transition_duration_ms` ms.  Picks the shortest angular
        path on azimuth so e.g. +x → -x goes through the nearer side, not the
        long way around.  An in-flight animation gets replaced; the new one
        starts from wherever the camera is at this instant.
        """
        start_az = float(self.camera.azimuth)
        start_el = float(self.camera.elevation)
        # Wrap delta to [-180, 180] so we always go the short way.
        delta_az = ((float(target_az) - start_az + 540.0) % 360.0) - 180.0
        resolved_target_az = start_az + delta_az
        if self._view_anim_timer is None:
            self._view_anim_timer = QTimer(self)
            self._view_anim_timer.setInterval(16)  # ~60 Hz
            self._view_anim_timer.timeout.connect(self._step_view_animation)
        else:
            self._view_anim_timer.stop()
        elapsed = QElapsedTimer()
        elapsed.start()
        self._view_anim = (
            start_az, start_el,
            resolved_target_az, float(target_el),
            elapsed,
        )
        self._view_anim_timer.start()

    def _step_view_animation(self) -> None:
        if self._view_anim is None:
            if self._view_anim_timer is not None:
                self._view_anim_timer.stop()
            return
        start_az, start_el, target_az, target_el, elapsed = self._view_anim
        duration = max(1, int(self.view_transition_duration_ms))
        t = min(1.0, float(elapsed.elapsed()) / float(duration))
        # Smoothstep for ease-in-out — no extra import, no overshoot.
        eased = t * t * (3.0 - 2.0 * t)
        az = start_az + (target_az - start_az) * eased
        el = start_el + (target_el - start_el) * eased
        if t >= 1.0:
            # Snap to exact preset to avoid float drift accumulating across
            # repeated transitions.
            self.camera.azimuth = target_az % 360.0
            self.camera.elevation = target_el
            self._view_anim_timer.stop()
            self._view_anim = None
        else:
            self.camera.azimuth = az % 360.0
            self.camera.elevation = el
        self.viewChanged.emit(self.camera.state())
        self.update()

    def _cancel_view_animation(self) -> None:
        if self._view_anim_timer is not None:
            self._view_anim_timer.stop()
        self._view_anim = None

    # ------------------------------------------------------------------
    # Public API — cross-section
    # ------------------------------------------------------------------

    def set_clip_plane_x(self, d: float) -> None:
        """Set the X-axis clip plane: discard ``x + d > 0``."""
        self.clip_state.set_x(float(d))
        self._after_clip_change()

    def set_clip_plane_y(self, d: float) -> None:
        """Set the Y-axis clip plane: discard ``y + d > 0``."""
        self.clip_state.set_y(float(d))
        self._after_clip_change()

    def clear_clip_plane_x(self) -> None:
        self.clip_state.clear_x()
        self._after_clip_change()

    def clear_clip_plane_y(self) -> None:
        self.clip_state.clear_y()
        self._after_clip_change()

    def set_clip_plane_invert_x(self, on: bool) -> None:
        self.clip_state.a_invert = bool(on)
        self._after_clip_change()

    def set_clip_plane_invert_y(self, on: bool) -> None:
        self.clip_state.b_invert = bool(on)
        self._after_clip_change()

    def set_clip_plane_symmetric(self, on: bool) -> None:
        self.clip_state.symmetric = bool(on)
        self._after_clip_change()

    def set_ray_clip_mode(self, mode: str) -> None:
        """Choose how active clip planes affect rays.

        ``"segment"`` (default) chops each line draw at the plane.
        ``"origin"`` keeps a ray in full when its launch point is on the
        kept side and drops it entirely otherwise.  Lens geometry continues
        to cross-section at the plane in either mode.
        """
        if mode not in ("segment", "origin"):
            raise ValueError(
                f"ray_clip_mode must be 'segment' or 'origin', got {mode!r}"
            )
        if self.clip_state.ray_clip_mode == mode:
            return
        self.clip_state.ray_clip_mode = mode
        self._refresh_gl(self._upload_ray_buffers)

    def clear_clip_plane(self) -> None:
        self.clip_state.clear()
        self._after_clip_change()

    def _apply_cutaway(self, mode: str) -> None:
        """Translate a ``ViewportToolbar`` cutaway selection into clip planes.

        The plane is anchored at the midpoint of the current scene bbox on
        the chosen axis.  Re-read each time so updated geometry (new lens,
        sensor) reflows the cut.  Sign convention: ``ClipPlaneState.set_x(d)``
        stores plane ``(1,0,0,d)``, discarding ``x > -d``.

        The X cut is inverted (``a_invert=True``) so both ``"x"`` and the X
        portion of ``"xy"`` keep the +X half of the lens — the opposite of
        the bare ``set_x`` direction.  Y is not inverted.
        """
        if mode == "none":
            self.clear_clip_plane()
            return
        if mode not in ("x", "y", "xy"):
            raise ValueError(f"unknown cutaway mode: {mode!r}")
        mn = self.scene.bbox_min
        mx = self.scene.bbox_max
        mid_x = float((mn[0] + mx[0]) * 0.5)
        mid_y = float((mn[1] + mx[1]) * 0.5)
        cs = self.clip_state
        cs.clear()
        if mode == "x":
            cs.set_x(-mid_x)
            cs.a_invert = True
        elif mode == "y":
            cs.set_y(-mid_y)
        else:  # "xy"
            cs.set_x(-mid_x)
            cs.a_invert = True
            cs.set_y(-mid_y)
        self._after_clip_change()

    def _after_clip_change(self) -> None:
        self._refresh_gl(self._upload_ray_buffers)
        # Signal: emit the pair (slot A, slot B) so listeners see both planes.
        self.clipPlaneChanged.emit({
            "x": None if self.clip_state.uniform_vec4() == (0.0, 0.0, 0.0, 0.0) else self.clip_state.uniform_vec4(),
            "y": None if self.clip_state.uniform_vec4_b() == (0.0, 0.0, 0.0, 0.0) else self.clip_state.uniform_vec4_b(),
        })

    # ------------------------------------------------------------------
    # Public API — selection
    # ------------------------------------------------------------------

    def selected_element(self):
        return self.selection.element

    def selected_surface(self) -> Optional[int]:
        """Global surface index currently selected, or ``None``."""
        return self.selection.surface

    def clear_selection(self) -> None:
        had_element = self.selection.element is not None
        had_surface = self.selection.surface is not None
        if self.selection.clear():
            if had_element:
                self.elementSelected.emit(None)
            if had_surface:
                self.surfaceSelected.emit(None)
            self.update()

    def set_selected_element(self, element) -> None:
        """Programmatically set the selected element without emitting.

        Mirror of click-driven selection but does NOT fire ``elementSelected``
        — call this from a host syncing selection in from another panel to
        avoid a feedback loop.  Clears any surface selection: surface
        selection always implies a specific element + cap pair, so changing
        the element directly invalidates the previous surface pick.
        """
        if element is None:
            if self.selection.clear():
                self.update()
            return
        changed = self.selection.set_element(element)
        # Surface selection is element-scoped; switching element implicitly
        # drops the surface unless the host immediately re-supplies one.
        if self.selection.clear_surface():
            changed = True
        if changed:
            self.update()

    def set_selected_surface(self, surface_index: Optional[int]) -> None:
        """Programmatically set the selected surface without emitting.

        Does NOT update ``selection.element`` — pair it with
        :meth:`set_selected_element` when the host knows the owning element
        (e.g. the optical-editor tree resolves both before pushing).
        """
        changed = self.selection.set_surface(
            None if surface_index is None else int(surface_index)
        )
        if changed:
            self.update()

    # ------------------------------------------------------------------
    # Public API — cosmetic
    # ------------------------------------------------------------------

    def set_background(self, top, bottom) -> None:
        self._bg_top = tuple(top)
        self._bg_bottom = tuple(bottom)
        self.update()

    def set_show_axes(self, on: bool) -> None:
        self._show_axes = bool(on)
        self.update()

    def set_info_text(self, text: Optional[str]) -> None:
        """Set the bottom-overlay info text.  ``None`` or empty hides the bar."""
        self._info_bar.set_text(text)
        self._position_info_bar()

    def set_show_view_cube(self, on: bool) -> None:
        self._show_view_cube = bool(on)
        self.update()

    def set_show_pivots(self, on: bool) -> None:
        """Show / hide the element centre-of-rotation markers."""
        self._show_pivots = bool(on)
        self.update()

    def show_pivots(self) -> bool:
        return self._show_pivots

    # ------------------------------------------------------------------
    # Public API — overlay toolbar
    # ------------------------------------------------------------------

    def cutaway_mode(self) -> str:
        return self._toolbar.cutaway_value()

    def set_cutaway_mode(self, mode: str) -> None:
        """Set the cutaway selection AND apply the matching clip planes.

        Updates the toolbar without firing its signal — `_apply_cutaway` is
        called explicitly so a programmatic setter and a menu click both end
        up running the same code path exactly once.
        """
        self._toolbar.set_cutaway(mode, emit=False)
        self._apply_cutaway(mode)

    def context_menu_enabled(self) -> bool:
        return self._context_menu_enabled

    def set_context_menu_enabled(self, on: bool) -> None:
        """Enable/disable right-click ``contextMenuRequested`` emission.

        A generic input-capability toggle (like selection mode): read-only
        viewports turn it off so a right-click never offers editing actions.
        """
        self._context_menu_enabled = bool(on)

    def selection_mode(self) -> str:
        return self._toolbar.selection_mode()

    def set_selection_mode(self, mode: str) -> None:
        """Set the click-time selection mode.

        Switching modes always clears any existing selection: the previous
        pick was made under a different rule (element vs surface) and
        carrying it across would leave the viewport in a state the user
        didn't ask for.  Use ``emit=False`` on the toolbar so the toolbar's
        own signal doesn't fire — `_on_selection_mode_changed` will be
        called once explicitly below.
        """
        self._toolbar.set_selection_mode(mode, emit=False)
        self._on_selection_mode_changed(mode)

    def _on_selection_mode_changed(self, mode: str) -> None:
        # Clear any current element + surface selection: per the design we
        # always start fresh after a mode switch.  Hover also goes away —
        # the rules for what's hoverable depend on the mode.
        had_element = self.selection.element is not None
        had_surface = self.selection.surface is not None
        cleared = self.selection.clear()
        cleared = self.selection.clear_hover() or cleared
        if had_element:
            self.elementSelected.emit(None)
        if had_surface:
            self.surfaceSelected.emit(None)
        if cleared:
            self.update()

    def screenshot(self) -> QImage:
        return self.grabFramebuffer()

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        try:
            _log_gl_info(self)
            self._compile_shaders()
            self._build_static_buffers()
            self._upload_element_buffers()
            self._upload_ray_buffers()
            self._upload_sensor_buffers()
            self._upload_calibrated_sensor_buffers()
            self._upload_axes_buffers()
            self._upload_pivot_buffers()
            if self.scene.elements or self.sensor is not None:
                self.reset_view()
            self._gl_ready = True
            self.contextReady.emit()
        except Exception:
            _log_gl_error("initializeGL")
            raise

    def resizeEvent(self, ev) -> None:
        # Qt widget resize (logical pixels).  Reposition the overlay toolbar
        # here — `resizeGL` runs in the GL viewport-resize path and isn't the
        # right hook for Qt child layout.
        super().resizeEvent(ev)
        if getattr(self, "_toolbar", None) is not None:
            self._position_toolbar()
        if getattr(self, "_info_bar", None) is not None:
            self._position_info_bar()

    def resizeGL(self, w: int, h: int) -> None:
        self.camera.viewport_w = int(w)
        self.camera.viewport_h = int(h)
        # Picking FBO resized lazily
        self._pick_dirty = True

    def paintGL(self) -> None:
        try:
            self._paint_impl()
        except Exception:
            _log_gl_error("paintGL")
            raise

    def _paint_impl(self) -> None:
        gl = _gl()
        if gl is None:
            return
        w = max(1, self.width())
        h = max(1, self.height())
        self.camera.viewport_w = w
        self.camera.viewport_h = h

        stage = "setup"
        try:
            # Clear with opaque alpha so Qt's compositor doesn't read the
            # widget as transparent on systems that respect framebuffer alpha.
            gl.glClearColor(
                float(self._bg_bottom[0]),
                float(self._bg_bottom[1]),
                float(self._bg_bottom[2]),
                1.0,
            )
            gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDisable(GL_BLEND)
            # Disable depth writes during the fullscreen background pass so
            # the depth buffer stays at 1.0 (cleared) for the lens pass to
            # write into.  Without this, the bg triangle's z=0 would fill the
            # depth buffer and reject every subsequent fragment.
            gl.glDepthMask(0)
            stage = "background"
            self._draw_background()
            gl.glDepthMask(1)

            # Depth pre-pass — lay down the front-most lens depth into the
            # default FBO so the post-composite outline pass can be culled
            # by occluding geometry.  Runs on the default FBO (still bound)
            # before we hand off to the OIT FBO for the colour pass; the
            # lens fragment shader's clip-plane discards apply, so cut-away
            # geometry doesn't contribute depth and the outlines on the
            # newly-visible interior aren't hidden.
            stage = "depth_prepass"
            self._draw_lens_depth_prepass(self.camera.view_projection())

            # Lens pass — Weighted-Blended-style additive OIT.
            #
            # Per-primitive sort (sub-solid or region centroid, even with
            # two-pass back-face culling per sub-solid) cannot give correct
            # per-pixel ordering when sub-solids' projected depth ranges
            # overlap — pixels of a back element kept popping in front of
            # closer elements on rotation.  Instead we render every lens
            # fragment additively into an RGBA16F accumulator (one extra
            # FBO) and reconstruct an alpha-weighted average colour +
            # saturating visibility in composite.frag.  Additive blend is
            # commutative so draw order doesn't matter — the result is
            # per-pixel correct regardless of how the sub-solids project.
            #
            # ``_draw_rays`` / ``_draw_axes`` reset depth state below so
            # rays/axes remain visible through transparent glass.
            self._ensure_oit_fbo()
            self._oit_fbo.bind()
            gl.glViewport(0, 0, *self._oit_fbo_size)
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDepthMask(0)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_ONE, GL_ONE)
            gl.glDisable(GL_CULL_FACE)

            vp = self.camera.view_projection()
            cam_pos = self._camera_world_pos()

            # Sensor is drawn after the composite (sensor.frag stays on the
            # legacy α-blend path because it lives at one fixed plane and
            # has no overlapping-translucent-with-itself issue).  No
            # interleaving needed under OIT.
            stage = "lens"
            self._draw_lens(vp, cam_pos)

            # Rebind the QOpenGLWidget's backing FBO and composite the OIT
            # accumulator onto the background we drew earlier.  Qt's
            # ``QOpenGLFramebufferObject.release()`` binds FBO 0, which is
            # NOT the widget's framebuffer — use ``defaultFramebufferObject``
            # to get the actual id, or the composite/rays/axes would draw
            # into a black screen the user never sees.
            self._oit_fbo.release()
            default_fbo = self.defaultFramebufferObject()
            gl.glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)
            # OIT FBO size == default FBO size (both physical pixels), so
            # the same viewport works for the composite pass and every
            # overlay that follows.
            gl.glViewport(0, 0, *self._oit_fbo_size)
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDepthMask(0)
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(GL_CULL_FACE)
            stage = "composite"
            self._composite_oit()
            # Cross-section cap fill — opaque flat grey on top of the OIT
            # composite, stencil-masked to each closed sub-solid's cut
            # polygon.  Drawn before outlines so the cross-section stroke
            # lands on top of the cap fill.
            stage = "cap_solids"
            self._draw_cap_solids(vp)
            # Stroke outlines for rim seams (and clip-plane cross-sections
            # when a cut is active).  Drawn after the composite so the
            # lines sit on top of the lens fill colour, and depth-tested
            # against the pre-pass depth so a stroke behind another lens
            # is correctly hidden.
            stage = "outline"
            self._draw_lens_outlines(vp)
            stage = "sensor"
            self._draw_sensor(vp)
            self._draw_calibrated_sensor(vp)

            # Rays and axes overlay everything — they're conceptually
            # diagnostic markers, not part of the depth-sorted scene.  Turn
            # depth test/culling off so a ray segment behind glass still
            # shows through.
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDisable(GL_CULL_FACE)
            stage = "rays"
            self._draw_rays(vp)
            stage = "axes"
            self._draw_axes(vp)
            stage = "pivots"
            self._draw_pivots(vp)
            if self._show_view_cube:
                stage = "view_cube"
                self._draw_view_cube()
        except Exception:
            _log_gl_error(f"paint:{stage}")
            raise

    # ------------------------------------------------------------------
    # Internal: shader compilation + static buffers
    # ------------------------------------------------------------------

    def _compile_shaders(self) -> None:
        def make(vsrc: str, fsrc: str) -> QOpenGLShaderProgram:
            prog = QOpenGLShaderProgram(self)
            if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vsrc):
                raise RuntimeError(f"vertex shader failed:\n{prog.log()}")
            if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fsrc):
                raise RuntimeError(f"fragment shader failed:\n{prog.log()}")
            if not prog.link():
                raise RuntimeError(f"shader link failed:\n{prog.log()}")
            return prog

        self._lens_prog   = make(load_shader("lens.vert"),   load_shader("lens.frag"))
        self._ray_prog    = make(load_shader("ray.vert"),    load_shader("ray.frag"))
        self._sensor_prog = make(load_shader("sensor.vert"), load_shader("sensor.frag"))
        self._gizmo_prog  = make(load_shader("gizmo.vert"),  load_shader("gizmo.frag"))
        self._blit_prog   = make(load_shader("blit.vert"),   load_shader("blit.frag"))
        self._cap_solid_prog = make(load_shader("cap.vert"), load_shader("cap_solid.frag"))
        self._composite_prog = make(load_shader("composite.vert"),
                                    load_shader("composite.frag"))
        self._outline_prog = make(load_shader("lens_outline.vert"),
                                  load_shader("lens_outline.frag"))


    def _build_static_buffers(self) -> None:
        # Fullscreen pass uses a hardcoded triangle in the vertex shader
        # (gl_VertexID), so we only need a dummy VAO bound to satisfy core
        # profile.  See blit.vert.
        self._fullscreen_vao = QOpenGLVertexArrayObject(self)
        self._fullscreen_vao.create()

        # Gizmo cube — position (3) + normal (3) + color (3) across three VBOs.
        gv, gn, gc = gizmo_mod.build_cube()
        self._gizmo_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._gizmo_vbo.upload(gv)
        self._gizmo_normal_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._gizmo_normal_vbo.upload(gn)
        self._gizmo_color_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._gizmo_color_vbo.upload(gc)
        self._gizmo_vao = QOpenGLVertexArrayObject(self)
        self._gizmo_vao.create()
        self._gizmo_vao.bind()
        _setup_attribute(self._gizmo_prog, "a_position", self._gizmo_vbo, 3, 12, 0)
        _setup_attribute(self._gizmo_prog, "a_normal",   self._gizmo_normal_vbo, 3, 12, 0)
        _setup_attribute(self._gizmo_prog, "a_color",    self._gizmo_color_vbo,  3, 12, 0)
        self._gizmo_vao.release()

        # Separate VAO for picking the gizmo.  We use lens_prog for the pick
        # pass (see workaround in _render_pick_buffer), so this VAO is wired
        # against lens_prog's attribute locations.  lens.vert expects both
        # a_position and a_normal — the gizmo VBO only has positions, so we
        # point a_normal at the same data (it's unused when u_unlit=1).
        self._gizmo_pick_vao = QOpenGLVertexArrayObject(self)
        self._gizmo_pick_vao.create()
        self._gizmo_pick_vao.bind()
        _setup_attribute(self._lens_prog, "a_position", self._gizmo_vbo, 3, 12, 0)
        _setup_attribute(self._lens_prog, "a_normal",   self._gizmo_vbo, 3, 12, 0)
        self._gizmo_pick_vao.release()

        # Outline geometry for the hovered face: 24 corners (6 faces × 4),
        # drawn as GL_LINE_LOOP per face.  Only position is wired; gizmo.frag
        # outputs a constant orange when u_highlight=1, so the disabled
        # a_normal/a_color attributes (default vec3(0)) are irrelevant.  The
        # outline is the *visible* feedback for slop-hovered faces — those
        # project to zero pixels in the face draw so the u_highlight tint
        # alone has nothing to draw on; the line loop projects to a line
        # along the cube silhouette either way.
        outline = gizmo_mod.face_corners_local().reshape(-1, 3).astype(np.float32)
        self._gizmo_outline_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._gizmo_outline_vbo.upload(outline)
        self._gizmo_outline_vao = QOpenGLVertexArrayObject(self)
        self._gizmo_outline_vao.create()
        self._gizmo_outline_vao.bind()
        _setup_attribute(self._gizmo_prog, "a_position",
                          self._gizmo_outline_vbo, 3, 12, 0)
        self._gizmo_outline_vao.release()

    # ------------------------------------------------------------------
    # Internal: per-element buffer upload
    # ------------------------------------------------------------------

    def _upload_element_buffers(self) -> None:
        """Upload one VBO per region using glDrawArrays (no EBO).

        Each :class:`SceneElement` holds one or more sub-solids (one per
        closed glass region in a cemented n-let).  Each sub-solid is further
        decomposed into regions — front cap, back cap, and the two wall
        halves attributed to the nearer cap surface (singlets and stops
        produce a single region).  Each region gets its own VAO so the lens
        pass can highlight a single optical surface and the picking pass can
        encode (element, surface) per pixel.  Sub-solids stay grouped so the
        renderer can depth-sort cemented n-lets correctly.

        Indexed drawing via glDrawElements is broken in PySide6's
        QOpenGLFunctions binding — passing ``ctypes.c_void_p(0)`` for the
        indices offset triggers STATUS_STACK_BUFFER_OVERRUN inside the
        driver.  We unroll the index buffer into a flat vertex array at
        upload time so the draw call can use ``glDrawArrays`` instead.

        Layout of ``self._element_buffers[i]``:
            ``{"scene_elem": SceneElement, "subsolids": list[subsolid_entry]}``
        where each ``subsolid_entry`` is
            ``{"centroid": (x, y, z), "regions": list[region_entry]}``
        and each ``region_entry`` is
            ``{"vbo": _GLBuffer, "vao": QOpenGLVertexArrayObject,
               "draw_count": int, "surface_index": int, "is_cap": bool}``.
        Empty elements produce an empty ``subsolids`` list.
        """
        for entry in self._element_buffers:
            for ss in entry["subsolids"]:
                for region in ss["regions"]:
                    region["vbo"].destroy()
                    region["vao"].destroy()
                outline_vbo = ss.get("outline_vbo")
                outline_vao = ss.get("outline_vao")
                if outline_vbo is not None:
                    outline_vbo.destroy()
                if outline_vao is not None:
                    outline_vao.destroy()
        self._element_buffers = []

        for se in self.scene.elements:
            subsolid_entries: list[dict] = []
            for ss in se.subsolids:
                region_entries: list[dict] = []
                for region in ss.regions:
                    if region.vertex_count == 0 or region.indices.size == 0:
                        continue
                    # Unroll: for each index, fetch the vertex + normal + kind.
                    # Resulting buffer interleaves position (3) + normal (3) +
                    # kind (1) = 7 floats per unrolled vertex.  ``kind`` is 0
                    # for cap surfaces, 1 for side-wall vertices — used by
                    # lens.frag to shade the connecting wall grey while keeping
                    # caps tinted.
                    pos = region.vertices[region.indices]
                    nrm = region.normals[region.indices]
                    knd = region.kinds[region.indices].reshape(-1, 1)
                    interleaved = np.concatenate([pos, nrm, knd], axis=1).astype(np.float32)

                    vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
                    vbo.upload(interleaved)

                    vao = QOpenGLVertexArrayObject(self)
                    vao.create()
                    vao.bind()
                    _setup_attribute(self._lens_prog, "a_position", vbo, 3, 28, 0)
                    _setup_attribute(self._lens_prog, "a_normal",   vbo, 3, 28, 12)
                    _setup_attribute(self._lens_prog, "a_kind",     vbo, 1, 28, 24)
                    vao.release()

                    if region.vertex_count > 0:
                        rc = region.vertices.mean(axis=0)
                        region_centroid = (float(rc[0]), float(rc[1]), float(rc[2]))
                    else:
                        region_centroid = (0.0, 0.0, 0.0)
                    region_entries.append({
                        "vbo": vbo,
                        "vao": vao,
                        "draw_count": int(interleaved.shape[0]),
                        "surface_index": int(region.surface_index),
                        "is_cap": bool(region.is_cap),
                        "centroid": region_centroid,
                    })
                if region_entries:
                    # Stroke outline buffer for the sub-solid's rim loops.
                    # Each closed loop is unrolled into GL_LINES (consecutive
                    # vertex pairs = one segment) so a single draw call covers
                    # all loops with no extra glDrawArrays per loop.
                    outline_vbo, outline_vao, outline_count = \
                        self._build_outline_buffer(ss.rim_loops)
                    subsolid_entries.append({
                        "centroid": ss.centroid,
                        "regions": region_entries,
                        "outline_vbo": outline_vbo,
                        "outline_vao": outline_vao,
                        "outline_count": outline_count,
                    })

            self._element_buffers.append({
                "scene_elem": se,
                "subsolids": subsolid_entries,
            })

        # Outline buffer for clip-plane cross-sections depends on per-region
        # geometry — bust the cache when the underlying meshes change.
        self._cross_section_cache_key = None

    def _build_outline_buffer(
        self, loops: list[np.ndarray]
    ) -> tuple[Optional["_GLBuffer"], Optional["QOpenGLVertexArrayObject"], int]:
        """Pack a list of closed polylines into a miter-joined quad VBO.

        Each polyline segment becomes 6 vertices (2 triangles); each
        vertex carries its two polyline neighbours so the shader can
        offset along the bisector of the two adjacent edges' perpendiculars
        — without that, every joint shows as a high-frequency wiggle on
        tessellated rim circles.  Returns ``(None, None, 0)`` when there
        is nothing to draw.
        """
        prog = self._outline_prog
        if prog is None or not loops:
            return None, None, 0
        chunks: list[np.ndarray] = []
        for loop in loops:
            loop = np.asarray(loop, dtype=np.float32)
            if loop.shape[0] < 2:
                continue
            chunks.append(_expand_polyline_to_miter_quads(loop))
        chunks = [c for c in chunks if c.shape[0] > 0]
        if not chunks:
            return None, None, 0
        packed = np.concatenate(chunks, axis=0).astype(np.float32)

        vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.upload(packed)
        vao = QOpenGLVertexArrayObject(self)
        vao.create()
        vao.bind()
        self._wire_outline_attributes(vbo)
        vao.release()
        return vbo, vao, int(packed.shape[0])

    def _wire_outline_attributes(self, vbo: "_GLBuffer") -> None:
        """Bind the miter-stroke attribute layout onto the currently-bound VAO."""
        prog = self._outline_prog
        if prog is None:
            return
        # Layout: a_this(3) + a_prev(3) + a_next(3) + a_side(1) = 10 floats.
        stride = _MITER_STRIDE_BYTES
        _setup_attribute(prog, "a_this", vbo, 3, stride, 0)
        _setup_attribute(prog, "a_prev", vbo, 3, stride, 12)
        _setup_attribute(prog, "a_next", vbo, 3, stride, 24)
        _setup_attribute(prog, "a_side", vbo, 1, stride, 36)

    def _ensure_cross_section_buffer(
        self,
        plane_a: tuple[float, float, float, float],
        plane_b: tuple[float, float, float, float],
        a_active: bool,
        b_active: bool,
    ) -> int:
        """Rebuild the clip-plane cross-section outline VBO when state changes.

        For every active plane, slice every region's triangle mesh and
        collect the resulting line segments.  The fragment shader's
        ``CLIP_EPS`` tolerance lets these segments survive the same plane's
        discard test — for the *other* plane the normal discard rule
        applies, so a segment generated against plane A is trimmed
        automatically wherever it leaves the half-space kept by plane B.
        """
        from . import geometry as _geometry
        key = (plane_a, plane_b, a_active, b_active)
        if key == self._cross_section_cache_key:
            return self._cross_section_count
        self._cross_section_cache_key = key

        if not (a_active or b_active):
            self._cross_section_count = 0
            return 0

        prog = self._outline_prog
        if prog is None:
            return 0

        chunks: list[np.ndarray] = []
        for entry in self._element_buffers:
            se = entry["scene_elem"]
            for ss in se.subsolids:
                for region in ss.regions:
                    if region.vertex_count == 0 or region.indices.size == 0:
                        continue
                    if a_active:
                        chunks.append(_geometry.slice_triangles_by_plane(
                            region.vertices, region.indices, plane_a))
                    if b_active:
                        chunks.append(_geometry.slice_triangles_by_plane(
                            region.vertices, region.indices, plane_b))
        chunks = [c for c in chunks if c.size > 0]
        if not chunks:
            if self._cross_section_vbo is not None:
                # Keep the buffer around — re-uploading zero bytes on every
                # frame churns the driver allocator.  count=0 already gates
                # the draw call.
                pass
            self._cross_section_count = 0
            return 0
        # ``slice_triangles_by_plane`` returns (2*M, 3) — endpoint pairs.
        # Cross-section segments are disjoint (no shared polyline context),
        # so they use the degenerate-neighbour helper; the shader detects
        # prev == this / next == this and falls back to a butt cap.
        raw = np.concatenate(chunks, axis=0).astype(np.float32)
        seg_starts = raw[0::2]
        seg_ends = raw[1::2]
        packed = _expand_disjoint_segments_to_miter_quads(seg_starts, seg_ends)

        if self._cross_section_vbo is None:
            self._cross_section_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._cross_section_vao = QOpenGLVertexArrayObject(self)
            self._cross_section_vao.create()
        self._cross_section_vbo.upload(packed)
        # Re-wire attributes every rebuild — the underlying buffer may
        # have been re-allocated by ``upload``.
        self._cross_section_vao.bind()
        self._wire_outline_attributes(self._cross_section_vbo)
        self._cross_section_vao.release()
        self._cross_section_count = int(packed.shape[0])
        return self._cross_section_count

    def _upload_axes_buffers(self) -> None:
        """Build 3 colored line segments from world origin along +X/+Y/+Z.

        Length is tied to the scene's bbox so the gizmo scales with the lens;
        all three axes share the same length per the request.
        """
        mn = np.asarray(self.scene.bbox_min, dtype=np.float32)
        mx = np.asarray(self.scene.bbox_max, dtype=np.float32)
        extent = float(np.max(mx - mn)) if mx.size else 1.0
        length = max(1.0, extent * 0.4)
        cx = PALETTE["axis_x"]
        cy = PALETTE["axis_y"]
        cz = PALETTE["axis_z"]
        # 6 vertices × (pos[3] + rgba[4]) = 28-byte stride, matching ray.vert.
        data = np.array([
            0.0, 0.0, 0.0,  cx[0], cx[1], cx[2], 1.0,
            length, 0.0, 0.0,  cx[0], cx[1], cx[2], 1.0,
            0.0, 0.0, 0.0,  cy[0], cy[1], cy[2], 1.0,
            0.0, length, 0.0,  cy[0], cy[1], cy[2], 1.0,
            0.0, 0.0, 0.0,  cz[0], cz[1], cz[2], 1.0,
            0.0, 0.0, length,  cz[0], cz[1], cz[2], 1.0,
        ], dtype=np.float32)

        if self._axes_vbo is None:
            self._axes_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._axes_vao = QOpenGLVertexArrayObject(self)
            self._axes_vao.create()
        self._axes_vbo.upload(data.reshape(-1, 7))
        self._axes_segments = 6

        self._axes_vao.bind()
        _setup_attribute(self._ray_prog, "a_position", self._axes_vbo, 3, 28, 0)
        _setup_attribute(self._ray_prog, "a_color",    self._axes_vbo, 4, 28, 12)
        self._axes_vao.release()

    def _upload_pivot_buffers(self) -> None:
        """Build a 3-axis cross at each element centre of rotation.

        One concatenated ``GL_LINES`` batch for every marker, sharing the ray
        program's ``pos[3] + rgba[4]`` layout. Sized off the scene bbox like
        the world axes but at a much smaller fraction — this is a marker, not
        a second gizmo, and at axis scale it would swamp the lens.
        """
        points = list(getattr(self.scene, "pivot_points", ()) or ())
        if not points:
            self._pivot_segments = 0
            return

        mn = np.asarray(self.scene.bbox_min, dtype=np.float32)
        mx = np.asarray(self.scene.bbox_max, dtype=np.float32)
        extent = float(np.max(mx - mn)) if mx.size else 1.0
        arm = max(0.5, extent * 0.03)
        c = PALETTE["pivot_marker"]

        verts = []
        for p in points:
            px, py, pz = (float(p[0]), float(p[1]), float(p[2]))
            for dx, dy, dz in ((arm, 0.0, 0.0), (0.0, arm, 0.0), (0.0, 0.0, arm)):
                verts.extend((px - dx, py - dy, pz - dz, c[0], c[1], c[2], 1.0))
                verts.extend((px + dx, py + dy, pz + dz, c[0], c[1], c[2], 1.0))
        data = np.array(verts, dtype=np.float32)

        if self._pivot_vbo is None:
            self._pivot_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._pivot_vao = QOpenGLVertexArrayObject(self)
            self._pivot_vao.create()
        self._pivot_vbo.upload(data.reshape(-1, 7))
        self._pivot_segments = len(points) * 6

        self._pivot_vao.bind()
        _setup_attribute(self._ray_prog, "a_position", self._pivot_vbo, 3, 28, 0)
        _setup_attribute(self._ray_prog, "a_color",    self._pivot_vbo, 4, 28, 12)
        self._pivot_vao.release()

    def _upload_ray_buffers(self) -> None:
        data = rays_mod.bundle_to_segments(
            self.ray_bundles,
            clip_planes=self.clip_state.both_planes(),
            clip_mode=self.clip_state.ray_clip_mode,
        )
        if self._ray_vbo is None:
            self._ray_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._ray_vao = QOpenGLVertexArrayObject(self)
            self._ray_vao.create()
        self._ray_vbo.upload(data)
        self._ray_segments = int(data.shape[0])

        self._ray_vao.bind()
        _setup_attribute(self._ray_prog, "a_position", self._ray_vbo, 3, 28, 0)
        _setup_attribute(self._ray_prog, "a_color",    self._ray_vbo, 4, 28, 12)
        self._ray_vao.release()

    def _upload_sensor_buffers(self) -> None:
        if self.sensor is None:
            self._sensor_quad_vbo = None
            self._sensor_border_vbo = None
            return

        gl = _gl()
        verts, _idx = self.sensor.build_quad()
        # Build as two triangles directly (6 verts) for simpler draw
        order = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        tri = verts[order]
        if self._sensor_quad_vbo is None:
            self._sensor_quad_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._sensor_quad_vao = QOpenGLVertexArrayObject(self)
            self._sensor_quad_vao.create()
        self._sensor_quad_vbo.upload(tri)
        self._sensor_quad_vao.bind()
        _setup_attribute(self._sensor_prog, "a_position",
                          self._sensor_quad_vbo, 3, 12, 0)
        self._sensor_quad_vao.release()

        border = self.sensor.build_border()
        if self._sensor_border_vbo is None:
            self._sensor_border_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._sensor_border_vao = QOpenGLVertexArrayObject(self)
            self._sensor_border_vao.create()
        self._sensor_border_vbo.upload(border)
        self._sensor_border_vao.bind()
        _setup_attribute(self._sensor_prog, "a_position",
                          self._sensor_border_vbo, 3, 12, 0)
        self._sensor_border_vao.release()

    def _upload_calibrated_sensor_buffers(self) -> None:
        if self.calibrated_sensor is None:
            self._cal_sensor_disk_count = 0
            self._cal_sensor_ring_count = 0
            return

        verts, indices = self.calibrated_sensor.build_disk()
        # Expand to a flat triangle list — matches the sensor program's quad
        # path (no element-array drawing elsewhere in this file).
        tri = verts[indices]
        if self._cal_sensor_disk_vbo is None:
            self._cal_sensor_disk_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._cal_sensor_disk_vao = QOpenGLVertexArrayObject(self)
            self._cal_sensor_disk_vao.create()
        self._cal_sensor_disk_vbo.upload(tri)
        self._cal_sensor_disk_count = int(tri.shape[0])
        self._cal_sensor_disk_vao.bind()
        _setup_attribute(self._sensor_prog, "a_position",
                          self._cal_sensor_disk_vbo, 3, 12, 0)
        self._cal_sensor_disk_vao.release()

        ring = self.calibrated_sensor.build_circle()
        if self._cal_sensor_ring_vbo is None:
            self._cal_sensor_ring_vbo = _GLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._cal_sensor_ring_vao = QOpenGLVertexArrayObject(self)
            self._cal_sensor_ring_vao.create()
        self._cal_sensor_ring_vbo.upload(ring)
        self._cal_sensor_ring_count = int(ring.shape[0])
        self._cal_sensor_ring_vao.bind()
        _setup_attribute(self._sensor_prog, "a_position",
                          self._cal_sensor_ring_vbo, 3, 12, 0)
        self._cal_sensor_ring_vao.release()

    # ------------------------------------------------------------------
    # Internal: drawing helpers
    # ------------------------------------------------------------------

    def _camera_world_pos(self) -> np.ndarray:
        """Camera world position derived from inverting the view matrix."""
        V = self.camera.view_matrix()
        Vi = np.linalg.inv(V)
        return Vi[:3, 3]

    def _draw_background(self) -> None:
        prog = self._blit_prog
        prog.bind()
        _set_u(prog, "u_top",
            float(self._bg_top[0]), float(self._bg_top[1]), float(self._bg_top[2]))
        _set_u(prog, "u_bottom",
            float(self._bg_bottom[0]), float(self._bg_bottom[1]), float(self._bg_bottom[2]))

        # Camera world-space basis from the rotation part of the view matrix.
        # The view matrix V is orthonormal (rotation + translation only), so
        # the rows of its 3x3 block are the world-space directions of the
        # camera's right/up/back axes.  The camera looks down its own -Z, so
        # forward = -row 2.
        V = self.camera.view_matrix()
        cam_right = V[0, :3].astype(np.float32)
        cam_up = V[1, :3].astype(np.float32)
        cam_forward = (-V[2, :3]).astype(np.float32)
        _set_u(prog, "u_cam_forward",
            float(cam_forward[0]), float(cam_forward[1]), float(cam_forward[2]))
        _set_u(prog, "u_cam_right",
            float(cam_right[0]), float(cam_right[1]), float(cam_right[2]))
        _set_u(prog, "u_cam_up",
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]))
        _set_u(prog, "u_sky_fov_scale", float(self._sky_fov_scale))
        _set_u(prog, "u_aspect", float(self.camera.aspect()))

        # The blit vertex shader emits a fullscreen triangle from gl_VertexID,
        # so no VBO/VAO is needed.  A dummy VAO must still be bound in core
        # profile though, since draw calls require a VAO.
        self._fullscreen_vao.bind()
        _gl().glDrawArrays(GL_TRIANGLES, 0, 3)
        self._fullscreen_vao.release()
        prog.release()

    @staticmethod
    def _cap_uniforms_for_plane(plane: tuple, bbox_min, bbox_max) -> tuple:
        """Return ``(center, tangent, bitangent, half_extent)`` for a cap quad
        on ``plane`` sized to cover the scene bbox from any view angle.

        Returns identity-ish placeholders when ``plane`` is the (0,0,0,0)
        disabled state — callers gate the cap pass on activeness so the values
        are never read.
        """
        a, b, c, d = plane
        n2 = a * a + b * b + c * c
        if n2 < 0.5:
            return (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                0.0,
            )
        n = np.array([a, b, c], dtype=np.float64)
        mn = np.asarray(bbox_min, dtype=np.float64)
        mx = np.asarray(bbox_max, dtype=np.float64)
        bcenter = (mn + mx) * 0.5
        # Project bbox centre onto the plane to anchor the quad where the cut
        # is most likely to land.
        center = bcenter - (float(np.dot(bcenter, n)) + float(d)) * n
        # Tangent basis: pick the world axis least parallel to ``n`` so the
        # cross is numerically stable.
        abs_n = np.abs(n)
        if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
            helper = np.array([1.0, 0.0, 0.0])
        elif abs_n[1] <= abs_n[2]:
            helper = np.array([0.0, 1.0, 0.0])
        else:
            helper = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(n, helper)
        t_len = float(np.linalg.norm(tangent))
        if t_len < 1e-9:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / t_len
        bitangent = np.cross(n, tangent)
        # Diagonal of the bbox is the worst-case half-extent the cut polygon
        # could occupy on the plane; 1.5× for a safety margin against ortho
        # zoom/pan extending the visible scene region.
        diag = float(np.linalg.norm(mx - mn))
        half = max(diag, 1.0) * 1.5
        return (
            (float(center[0]), float(center[1]), float(center[2])),
            (float(tangent[0]), float(tangent[1]), float(tangent[2])),
            (float(bitangent[0]), float(bitangent[1]), float(bitangent[2])),
            float(half),
        )

    def _stencil_subsolid(self, lens_prog, ss_entry: dict, plane: tuple) -> None:
        """Render a sub-solid's regions with ``plane`` discarding past-it
        fragments and stencil INVERT toggling on every surviving face.

        After this pass, the stencil bit is set at every pixel where the
        camera ray crosses ``plane`` inside the sub-solid's closed body
        (parity rule: an even number of original solid faces survive at pixels
        wholly on one side of the plane, an odd number where the plane
        actually cuts).  Caller is responsible for stencil state, color/depth
        masks, and culling.

        We rebind ``lens_prog`` here even if it looks unnecessary at the call
        site: in XY mode the previous plane's cap pass binds and releases
        ``cap_prog`` (Qt calls ``glUseProgram(0)`` on release), leaving no
        program current.  Subsequent ``_set_u`` / ``glDrawArrays`` calls would
        silently no-op against the null program — stencil count stays zero,
        the cap's NOTEQUAL test fails everywhere, and the second plane's cap
        disappears entirely.  A redundant bind is the cheapest insurance.
        """
        lens_prog.bind()
        gl = _gl()
        _set_u(lens_prog, "u_clip_plane",
            float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))
        _set_u(lens_prog, "u_clip_plane_b", 0.0, 0.0, 0.0, 0.0)
        _set_u(lens_prog, "u_unlit", 1)
        _set_u(lens_prog, "u_tint", 0.0, 0.0, 0.0)
        _set_u(lens_prog, "u_alpha", 0.0)
        for region in ss_entry["regions"]:
            region["vao"].bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, region["draw_count"])
            region["vao"].release()

    def _draw_lens(self, vp: np.ndarray, cam_pos: np.ndarray) -> None:
        prog = self._lens_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        _set_u(prog, "u_model", _to_qmatrix(np.eye(4, dtype=np.float32)))
        wall = PALETTE["lens_wall"]
        sel_color = PALETTE["selection_outline"]
        hover_color = PALETTE["hover_outline"]
        sel_elem = self.selection.element
        sel_surface = self.selection.surface
        hover_elem = self.selection.hover
        hover_surface = self.selection.hover_surface
        # Surface-level highlights take priority over element-level ones.
        # When a surface is selected (or hovered), we want the user's eye
        # drawn to that specific cap — so the rest of the owning element
        # stays in its normal tint instead of all glowing orange.
        elem_select_suppressed = sel_surface is not None
        elem_hover_suppressed = hover_surface is not None
        ax, bx, cx, dx, ay, by, cy, dy = self.clip_state.uniform_vec4_pair()
        plane_a = (ax, bx, cx, dx)
        plane_b = (ay, by, cy, dy)
        a_active = (ax * ax + bx * bx + cx * cx) > 0.5
        b_active = (ay * ay + by * by + cy * cy) > 0.5
        clip_active = a_active or b_active
        _set_u(prog, "u_clip_plane", float(ax), float(bx), float(cx), float(dx))
        _set_u(prog, "u_clip_plane_b", float(ay), float(by), float(cy), float(dy))

        # Order-independent transparency: every region draws exactly once
        # (no sort, no back-face culling) into the OIT accumulator that
        # paint_impl has bound — additive blend GL_ONE, GL_ONE — and
        # composite.frag turns that into the final per-pixel colour and
        # visibility.  Closed sub-solids deliberately render BOTH faces
        # so a camera ray that passes through the solid contributes twice
        # to the accumulator, matching the two surface interactions a
        # ray actually has.  Open meshes (stops, single-surface dummies)
        # contribute once.
        #
        # Per-surface highlight (surface-mode selection) is applied per
        # region inside the draw loop so only the picked cap is recoloured;
        # the rest of the owning sub-solid stays in its normal tint.

        def _apply_highlight_color(color: tuple, alpha: float) -> None:
            _set_u(prog, "u_tint",
                float(color[0]), float(color[1]), float(color[2]))
            _set_u(prog, "u_wall_tint",
                float(color[0]), float(color[1]), float(color[2]))
            _set_u(prog, "u_alpha", float(alpha))
            # Highlights drive cap + wall with the same colour and alpha so
            # the whole element/surface reads as a single bright unit.
            _set_u(prog, "u_wall_alpha", float(alpha))
            _set_u(prog, "u_unlit", 0)

        def _apply_normal(se: "SceneElement") -> None:
            is_stop = getattr(se.element.kind, "name", "GLASS") == "STOP"
            tint = se.tint
            _set_u(prog, "u_tint",
                float(tint[0]), float(tint[1]), float(tint[2]))
            _set_u(prog, "u_wall_tint",
                float(wall[0]), float(wall[1]), float(wall[2]))
            # Cap (blue tinted optical surface) alpha: half the transparency
            # of se.alpha, i.e. α' = (1 + α) / 2 so a default 0.75 reads as
            # 0.875.  Side-wall alpha is cranked to 4.0 — that's not a
            # mistake.  composite.frag turns Σα into visibility via
            # 1 − exp(−Σα), and a closed sub-solid's wall hits a ray twice
            # (entry + exit), so 2·4.0 = 8.0 maps to ~99.97% visibility:
            # the grey collar reads as fully solid in the composite.  Stops
            # use α=1 on both (driven via the u_unlit=1 path) which the
            # picker also leans on for opaque-mask behaviour.
            cap_alpha = 1.0 if is_stop else 0.5 * (1.0 + float(se.alpha))
            wall_alpha = 1.0 if is_stop else 4.0
            _set_u(prog, "u_alpha", cap_alpha)
            _set_u(prog, "u_wall_alpha", wall_alpha)
            _set_u(prog, "u_unlit", 1 if is_stop else 0)

        gl = _gl()
        # Cross-section cap fills are drawn in a separate post-composite pass
        # (_draw_cap_solids) so they read as flat opaque grey rather than going
        # through the OIT accumulator — see that method for details.
        # Per-region uniform state is keyed on a short tuple so we only re-set
        # uniforms when the visual state actually changes.  Five priority
        # bands: surface-select > surface-hover > element-select > element-
        # hover > normal.  Surface bands suppress element bands on the same
        # element so the user gets a sharp single-cap highlight instead of
        # the whole element glowing.
        last_state_key: Optional[tuple] = None
        # Walls render with back-face culling so the grey collar reads as a
        # true solid — without it the back wall blends through under OIT
        # and the surface boundaries are hard to tell apart.  Caps stay
        # double-sided so the closed solid still contributes its two-hit
        # accumulation for correct curved-glass shading.
        cull_enabled = False
        gl.glDisable(GL_CULL_FACE)
        try:
            gl.glCullFace(GL_BACK)
        except AttributeError:
            pass
        for entry in self._element_buffers:
            se = entry["scene_elem"]
            # Muted elements render as outline-only via _draw_lens_outlines.
            if se.muted:
                continue
            for ss in entry["subsolids"]:
                is_elem_selected = (
                    not elem_select_suppressed
                    and sel_elem is not None
                    and se.element is sel_elem
                )
                is_elem_hovered = (
                    not elem_hover_suppressed
                    and not is_elem_selected
                    and hover_elem is not None
                    and se.element is hover_elem
                )

                for region in ss["regions"]:
                    is_sel_cap = (
                        sel_surface is not None
                        and region["is_cap"]
                        and region["surface_index"] == sel_surface
                    )
                    is_hover_cap = (
                        not is_sel_cap
                        and hover_surface is not None
                        and region["is_cap"]
                        and region["surface_index"] == hover_surface
                    )
                    if is_sel_cap:
                        state_key = ("sel_surface",)
                    elif is_hover_cap:
                        state_key = ("hover_surface",)
                    elif is_elem_selected:
                        state_key = ("elem_select", id(se))
                    elif is_elem_hovered:
                        state_key = ("elem_hover", id(se))
                    else:
                        state_key = ("normal", id(se))
                    if state_key != last_state_key:
                        if is_sel_cap:
                            _apply_highlight_color(sel_color, 0.95)
                        elif is_hover_cap:
                            _apply_highlight_color(hover_color, 0.95)
                        elif is_elem_selected:
                            _apply_highlight_color(sel_color, 0.85)
                        elif is_elem_hovered:
                            _apply_highlight_color(hover_color, 0.85)
                        else:
                            _apply_normal(se)
                        last_state_key = state_key
                    # Walls: cull back faces.  Caps: keep both faces so the
                    # closed-solid two-hit OIT contribution still happens.
                    want_cull = not region["is_cap"]
                    if want_cull != cull_enabled:
                        if want_cull:
                            gl.glEnable(GL_CULL_FACE)
                        else:
                            gl.glDisable(GL_CULL_FACE)
                        cull_enabled = want_cull
                    region["vao"].bind()
                    gl.glDrawArrays(GL_TRIANGLES, 0, region["draw_count"])
                    region["vao"].release()

        prog.release()

    def _draw_cap_quad_solid(self, plane: tuple, other_plane: tuple,
                              cap_uniforms: tuple, vp: np.ndarray,
                              color: tuple) -> None:
        """Draw the cap quad on ``plane`` as flat opaque ``color``,
        stencil-masked to the cut polygon.  Expects stencil state to be
        set to consume the INVERT bits (NOTEQUAL with ZERO replace).
        """
        center, tangent, bitangent, half = cap_uniforms
        prog = self._cap_solid_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        _set_u(prog, "u_plane_center", float(center[0]), float(center[1]), float(center[2]))
        _set_u(prog, "u_plane_tangent",
            float(tangent[0]), float(tangent[1]), float(tangent[2]))
        _set_u(prog, "u_plane_bitangent",
            float(bitangent[0]), float(bitangent[1]), float(bitangent[2]))
        _set_u(prog, "u_half_extent", float(half))
        _set_u(prog, "u_other_plane",
            float(other_plane[0]), float(other_plane[1]),
            float(other_plane[2]), float(other_plane[3]))
        _set_u(prog, "u_color", float(color[0]), float(color[1]), float(color[2]))
        gl = _gl()
        self._fullscreen_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 6)
        self._fullscreen_vao.release()
        prog.release()

    def _draw_cap_solids(self, vp: np.ndarray) -> None:
        """Opaque flat-grey overlay for clip-plane cross-section caps.

        Runs on the default FBO AFTER ``_composite_oit`` so the cap reads
        as a flat solid grey instead of mixing with the visible wall
        fragments via the OIT accumulator.  Uses the same stencil INVERT
        parity trick as the old in-OIT cap pass — the default FBO carries
        the 8-bit stencil attachment ``set_default_surface_format``
        requests, so a fresh stencil clear here doesn't collide with
        anything else.

        Caller contract: default FBO is bound, viewport is set, depth
        prepass already populated the depth buffer.  Selection / hover
        highlights override the grey with the selection / hover palette
        colour so the cap stays consistent with the rest of the element.
        """
        gl = _gl()
        if (gl is None
                or self._cap_solid_prog is None
                or self._lens_prog is None):
            return
        if not self._element_buffers:
            return
        ax, bx, cx, dx, ay, by, cy, dy = self.clip_state.uniform_vec4_pair()
        plane_a = (ax, bx, cx, dx)
        plane_b = (ay, by, cy, dy)
        a_active = (ax * ax + bx * bx + cx * cx) > 0.5
        b_active = (ay * ay + by * by + cy * cy) > 0.5
        if not (a_active or b_active):
            return

        wall = PALETTE["lens_wall"]
        sel_color = PALETTE["selection_outline"]
        hover_color = PALETTE["hover_outline"]
        sel_elem = self.selection.element
        sel_surface = self.selection.surface
        hover_elem = self.selection.hover
        hover_surface = self.selection.hover_surface
        elem_select_suppressed = sel_surface is not None
        elem_hover_suppressed = hover_surface is not None

        cap_uni_a = (
            self._cap_uniforms_for_plane(
                plane_a, self.scene.bbox_min, self.scene.bbox_max
            )
            if a_active else None
        )
        cap_uni_b = (
            self._cap_uniforms_for_plane(
                plane_b, self.scene.bbox_min, self.scene.bbox_max
            )
            if b_active else None
        )

        gl.glEnable(GL_STENCIL_TEST)
        try:
            gl.glClearStencil(0)
        except AttributeError:
            pass
        # Depth test ALWAYS + depth write ON so the cap lands on top of
        # the OIT composite AND stamps the clip-plane depth into the
        # default FBO's depth buffer.  The subsequent outline pass then
        # depth-tests rim strokes against the cap: rim segments behind
        # the cap (further from the camera) fail GL_LESS and stay
        # hidden, while the cross-section stroke wins because
        # lens_outline.vert biases its depth slightly toward the camera.
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_ALWAYS)
        gl.glDepthMask(1)
        gl.glDisable(GL_CULL_FACE)

        for entry in self._element_buffers:
            se = entry["scene_elem"]
            # Muted elements have no fill, so their cross-section cap
            # would read as a floating grey plate with no host — skip.
            if se.muted:
                continue
            for ss in entry["subsolids"]:
                if len(ss["regions"]) < 2:
                    # Open mesh (stops, single-surface dummies) — the
                    # parity-INVERT rule degenerates, no cap to fill.
                    continue

                is_elem_selected = (
                    not elem_select_suppressed
                    and sel_elem is not None
                    and se.element is sel_elem
                )
                is_elem_hovered = (
                    not elem_hover_suppressed
                    and not is_elem_selected
                    and hover_elem is not None
                    and se.element is hover_elem
                )
                if is_elem_selected:
                    cap_color = sel_color
                elif is_elem_hovered:
                    cap_color = hover_color
                else:
                    cap_color = wall

                plane_passes = (
                    (plane_a, plane_b, cap_uni_a, a_active),
                    (plane_b, plane_a, cap_uni_b, b_active),
                )
                for plane, other_plane, cap_uni, plane_is_active in plane_passes:
                    if not plane_is_active:
                        continue
                    gl.glStencilMask(0xFF)
                    gl.glClear(GL_STENCIL_BUFFER_BIT)
                    # Stencil-only sub-pass.
                    gl.glColorMask(False, False, False, False)
                    gl.glDisable(GL_BLEND)
                    gl.glStencilFunc(GL_ALWAYS, 0, 0xFF)
                    gl.glStencilOp(GL_KEEP, GL_INVERT, GL_INVERT)
                    self._stencil_subsolid(self._lens_prog, ss, plane)
                    # Solid-fill sub-pass — opaque, no blend.
                    gl.glColorMask(True, True, True, True)
                    gl.glDisable(GL_BLEND)
                    gl.glStencilFunc(GL_NOTEQUAL, 0, 0xFF)
                    gl.glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO_OP)
                    self._draw_cap_quad_solid(plane, other_plane, cap_uni,
                                               vp, cap_color)

        gl.glStencilFunc(GL_ALWAYS, 0, 0xFF)
        gl.glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        gl.glDisable(GL_STENCIL_TEST)

    def _draw_lens_depth_prepass(self, vp: np.ndarray) -> None:
        """Populate the default FBO's depth buffer with the front-most lens
        depth so the later outline pass can be hidden by occluding geometry.

        Uses the same ``lens_prog`` + per-region VAOs as the regular lens
        pass.  Colour writes are masked off so this contributes nothing to
        the pixel — only depth is laid down.  The fragment shader's
        clip-plane discards still apply (discarded fragments don't write
        depth), so an outline behind a cut-away region is correctly
        unoccluded.

        Caller contract (set by ``_paint_impl`` before invoking us): default
        FBO is bound, viewport set to default-FBO size, depth buffer is
        cleared to 1.0, no stencil state required.
        """
        gl = _gl()
        if gl is None or self._lens_prog is None:
            return
        prog = self._lens_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        _set_u(prog, "u_model", _to_qmatrix(np.eye(4, dtype=np.float32)))
        # Drive the lens fragment shader's clip-plane discard.  Colour
        # outputs are masked off so the rest of the uniforms don't matter,
        # but we still need a valid u_unlit so the shader doesn't try the
        # OIT premultiplied branch (the framebuffer here isn't RGBA16F).
        ax, bx, cx, dx, ay, by, cy, dy = self.clip_state.uniform_vec4_pair()
        _set_u(prog, "u_clip_plane", float(ax), float(bx), float(cx), float(dx))
        _set_u(prog, "u_clip_plane_b", float(ay), float(by), float(cy), float(dy))
        _set_u(prog, "u_unlit", 1)
        _set_u(prog, "u_tint", 0.0, 0.0, 0.0)
        _set_u(prog, "u_wall_tint", 0.0, 0.0, 0.0)
        _set_u(prog, "u_alpha", 1.0)
        _set_u(prog, "u_wall_alpha", 1.0)

        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_LESS)
        gl.glDepthMask(1)
        gl.glColorMask(False, False, False, False)
        gl.glDisable(GL_BLEND)
        gl.glDisable(GL_CULL_FACE)

        for entry in self._element_buffers:
            # Muted elements skip depth so they don't occlude the
            # elements behind them — the outline pass still draws.
            if entry["scene_elem"].muted:
                continue
            for ss in entry["subsolids"]:
                for region in ss["regions"]:
                    region["vao"].bind()
                    gl.glDrawArrays(GL_TRIANGLES, 0, region["draw_count"])
                    region["vao"].release()

        gl.glColorMask(True, True, True, True)
        prog.release()

    def _draw_lens_outlines(self, vp: np.ndarray) -> None:
        """Stroke the rim seams and clip-plane cross-sections.

        Runs after the OIT composite with depth testing enabled against the
        depth buffer populated by ``_draw_lens_depth_prepass`` — outlines
        behind glass are culled, outlines on visible surfaces survive.

        Stroke width is enforced by per-vertex screen-space quad expansion
        in ``lens_outline.vert`` (not ``glLineWidth`` — core-profile drivers
        cap that at 1.0 regardless of what you pass).
        """
        gl = _gl()
        if gl is None or self._outline_prog is None:
            return
        ax, bx, cx, dx, ay, by, cy, dy = self.clip_state.uniform_vec4_pair()
        plane_a = (float(ax), float(bx), float(cx), float(dx))
        plane_b = (float(ay), float(by), float(cy), float(dy))
        a_active = (ax * ax + bx * bx + cx * cx) > 0.5
        b_active = (ay * ay + by * by + cy * cy) > 0.5

        # Cross-section line geometry is dynamic — rebuild when the clip
        # state changes since the previous frame, no-op otherwise.
        self._ensure_cross_section_buffer(plane_a, plane_b, a_active, b_active)

        prog = self._outline_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        _set_u(prog, "u_model", _to_qmatrix(np.eye(4, dtype=np.float32)))
        _set_u(prog, "u_clip_plane", *plane_a)
        _set_u(prog, "u_clip_plane_b", *plane_b)
        # Near-black stroke reads cleanly on both the blue-tinted glass
        # and the dark sky/background.
        _set_u(prog, "u_color", 0.04, 0.05, 0.06)
        _set_u(prog, "u_alpha", 0.95)
        # Drives the vertex-shader quad expansion: framebuffer size in
        # physical pixels (matches the OIT FBO == default FBO sizing) and
        # the stroke's half-thickness in those same pixels.  Change the
        # half-width number to make every outline thicker or thinner.
        fb_w, fb_h = self._oit_fbo_size
        _set_u(prog, "u_viewport_size_px",
               float(max(1, fb_w)), float(max(1, fb_h)))
        _set_u(prog, "u_line_half_width_px", 1.5)

        gl.glEnable(GL_DEPTH_TEST)
        # GL_LESS + depth write: the front-most outline segment at each
        # pixel wins and seals depth so all the other 63 segments tiling
        # the same rim silhouette (each oriented slightly differently in
        # screen space) fail the test instead of stacking their AA halos
        # into a wiggly rosette.  The vertex shader's small depth bias
        # keeps these outlines beating the lens body in the pre-pass.
        gl.glDepthFunc(GL_LESS)
        gl.glDepthMask(1)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(GL_CULL_FACE)

        # Rim outlines — one draw call per sub-solid keeps state-changes low.
        for entry in self._element_buffers:
            for ss in entry["subsolids"]:
                vao = ss.get("outline_vao")
                count = int(ss.get("outline_count", 0))
                if vao is None or count <= 0:
                    continue
                vao.bind()
                gl.glDrawArrays(GL_TRIANGLES, 0, count)
                vao.release()

        # Clip-plane cross-section strokes.
        if self._cross_section_count > 0 and self._cross_section_vao is not None:
            self._cross_section_vao.bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, self._cross_section_count)
            self._cross_section_vao.release()

        gl.glDisable(GL_DEPTH_TEST)
        # Leave depth state quiescent for the downstream sensor/rays/axes
        # passes — they all assume depth writes are off going in.
        gl.glDepthMask(0)
        prog.release()

    def _draw_sensor(self, vp: np.ndarray) -> None:
        if self.sensor is None or self._sensor_quad_vao is None:
            return
        prog = self._sensor_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        # Sensor is never clipped — clip planes only affect rays + lens elements.
        fill = PALETTE["sensor_fill"]
        border = PALETTE["sensor_border"]
        _set_u(prog, "u_fill", float(fill[0]), float(fill[1]), float(fill[2]))
        _set_u(prog, "u_border", float(border[0]), float(border[1]), float(border[2]))
        _set_u(prog, "u_alpha", 0.30)
        _set_u(prog, "u_is_border", 0.0)

        gl = _gl()
        self._sensor_quad_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 6)
        self._sensor_quad_vao.release()

        _set_u(prog, "u_is_border", 1.0)
        self._sensor_border_vao.bind()
        gl.glDrawArrays(GL_LINES, 0, 8)
        self._sensor_border_vao.release()
        prog.release()

    def _draw_calibrated_sensor(self, vp: np.ndarray) -> None:
        if (
            not self._show_calibrated_sensor
            or self.calibrated_sensor is None
            or self._cal_sensor_disk_vao is None
            or self._cal_sensor_disk_count == 0
        ):
            return
        prog = self._sensor_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        # Same dark fill as the rectangular sensor, with a black outline so
        # the calibrated disk reads as a distinct overlay rather than a second
        # yellow-bordered sensor.
        fill = PALETTE["sensor_fill"]
        border = PALETTE["calibrated_sensor_border"]
        _set_u(prog, "u_fill", float(fill[0]), float(fill[1]), float(fill[2]))
        _set_u(prog, "u_border", float(border[0]), float(border[1]), float(border[2]))
        _set_u(prog, "u_alpha", 0.30)
        _set_u(prog, "u_is_border", 0.0)

        gl = _gl()
        self._cal_sensor_disk_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, self._cal_sensor_disk_count)
        self._cal_sensor_disk_vao.release()

        if self._cal_sensor_ring_vao is not None and self._cal_sensor_ring_count > 0:
            _set_u(prog, "u_is_border", 1.0)
            self._cal_sensor_ring_vao.bind()
            gl.glDrawArrays(GL_LINES, 0, self._cal_sensor_ring_count)
            self._cal_sensor_ring_vao.release()
        prog.release()

    def _draw_axes(self, vp: np.ndarray) -> None:
        if not self._show_axes or self._axes_vao is None or self._axes_segments < 2:
            return
        prog = self._ray_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        gl = _gl()
        # Solid alpha for axis lines so the red/green/blue read cleanly even
        # when they cross transparent glass — additive (ray) blending would
        # wash them out against the background.
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            gl.glLineWidth(3.0)
        except Exception:
            pass
        self._axes_vao.bind()
        gl.glDrawArrays(GL_LINES, 0, self._axes_segments)
        self._axes_vao.release()
        prog.release()

    def _draw_pivots(self, vp: np.ndarray) -> None:
        if (
            not self._show_pivots
            or self._pivot_vao is None
            or self._pivot_segments < 2
        ):
            return
        prog = self._ray_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        gl = _gl()
        # Solid alpha, matching _draw_axes: additive blending would wash the
        # marker out wherever it crosses glass, which is exactly where a
        # centre of rotation usually sits.
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            gl.glLineWidth(2.0)
        except Exception:
            pass
        self._pivot_vao.bind()
        gl.glDrawArrays(GL_LINES, 0, self._pivot_segments)
        self._pivot_vao.release()
        prog.release()

    def _draw_rays(self, vp: np.ndarray) -> None:
        if self._ray_vao is None or self._ray_segments < 2:
            return
        prog = self._ray_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))

        gl = _gl()
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        try:
            gl.glLineWidth(1.5)
        except Exception:
            pass
        self._ray_vao.bind()
        gl.glDrawArrays(GL_LINES, 0, self._ray_segments)
        self._ray_vao.release()
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        prog.release()

    def _draw_view_cube(self) -> None:
        gl = _gl()
        vp, scissor = gizmo_mod.gizmo_view_proj(
            self.camera.azimuth, self.camera.elevation,
            self.width(), self.height()
        )
        # Qt's paintGL framebuffer is PHYSICAL pixels (logical × dpr).  Scale
        # the logical scissor/viewport coords by dpr so the cube lands at the
        # same Qt-logical position on the screen as in the pick FBO.
        dpr = float(self.devicePixelRatioF()) or 1.0
        x = int(round(scissor[0] * dpr))
        y = int(round(scissor[1] * dpr))
        w = int(round(scissor[2] * dpr))
        h = int(round(scissor[3] * dpr))
        gl.glEnable(GL_SCISSOR_TEST)
        gl.glScissor(x, y, w, h)
        gl.glViewport(x, y, w, h)
        # Depth writes are disabled by the OIT/composite passes and never
        # restored by _draw_rays/_draw_axes.  Re-enable before glClear so the
        # clear actually lands (clears are masked by glDepthMask), and so each
        # per-face draw below updates the depth buffer for the next one — else
        # the last face drawn wins every pixel regardless of geometry.
        gl.glDepthMask(1)
        gl.glClear(GL_DEPTH_BUFFER_BIT)
        gl.glDisable(GL_BLEND)
        gl.glEnable(GL_DEPTH_TEST)

        prog = self._gizmo_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        # Per-face draw so we can tint just the hovered face.  Each face is
        # 6 contiguous vertices (two triangles) in the same order as
        # FACE_PRESETS — see gizmo.build_cube.  For visible faces this is
        # what gives the user the "I'm about to click this" feedback; for
        # slop-hovered faces the tint has nothing to draw on (zero pixels)
        # and the line-loop outline below does the work instead.
        hover_face = self._gizmo_hover_face
        self._gizmo_vao.bind()
        for face_idx in range(6):
            h = 0.55 if hover_face == face_idx else 0.0
            _set_u(prog, "u_highlight", float(h))
            gl.glDrawArrays(GL_TRIANGLES, face_idx * 6, 6)
        self._gizmo_vao.release()

        # Edge outline for the hovered face.  Drawn after the cube so the
        # line wins over the triangle fragments at the silhouette pixels.
        if (hover_face is not None
                and self._gizmo_outline_vao is not None):
            _set_u(prog, "u_highlight", 1.0)
            self._gizmo_outline_vao.bind()
            try:
                gl.glLineWidth(2.5)
            except Exception:
                # Some drivers reject widths != 1 in core profile; the 1px
                # outline is still a useful (if subtle) highlight.
                pass
            gl.glDrawArrays(GL_LINE_LOOP, hover_face * 4, 4)
            try:
                gl.glLineWidth(1.0)
            except Exception:
                pass
            self._gizmo_outline_vao.release()

        prog.release()

        gl.glDisable(GL_SCISSOR_TEST)
        # Restore the physical-size viewport that paintGL started with.
        gl.glViewport(0, 0,
                       int(round(self.width() * dpr)),
                       int(round(self.height() * dpr)))

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _ensure_pick_fbo(self) -> None:
        # Size the pick FBO in PHYSICAL pixels (logical × devicePixelRatio).
        # Qt's QOpenGLWidget renders the visible scene to a physical-sized
        # framebuffer, and the gizmo's visible position is computed in those
        # physical-pixel coordinates.  Mirroring that here keeps the visible
        # cube and the pick-FBO cube at the exact same on-screen position so
        # a click maps to the same pixel in both.
        dpr = float(self.devicePixelRatioF()) or 1.0
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        if self._pick_fbo is not None and self._pick_fbo_size == (w, h):
            return
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fmt.setSamples(0)
        GL_RGBA8 = 0x8058
        fmt.setInternalTextureFormat(GL_RGBA8)
        self._pick_fbo = QOpenGLFramebufferObject(w, h, fmt)
        self._pick_fbo_size = (w, h)

    def _ensure_oit_fbo(self) -> None:
        """Allocate / resize the additive-accumulator FBO the lens pass
        renders into.  One RGBA16F colour attachment + combined
        depth/stencil — the depth half is unused (lens pass keeps depth
        test off) but the stencil half is needed for the clip-plane cap
        INVERT/NOTEQUAL passes.  Sized in physical pixels to match the
        default framebuffer (so composite.frag's texelFetch lines up).
        """
        dpr = float(self.devicePixelRatioF()) or 1.0
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        if self._oit_fbo is not None and self._oit_fbo_size == (w, h):
            return
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        fmt.setSamples(0)
        fmt.setInternalTextureFormat(GL_RGBA16F)
        self._oit_fbo = QOpenGLFramebufferObject(w, h, fmt)
        self._oit_fbo_size = (w, h)

    def _composite_oit(self) -> None:
        """Composite the OIT accumulator onto the currently-bound FBO.

        Expects: blend on, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, depth
        test off, default FBO bound, viewport set to default-FBO size.
        Reads the RGBA16F accum texture, derives per-pixel
        alpha-weighted colour + saturating visibility (see
        composite.frag), and blends it over whatever's already in the
        framebuffer (the sky/background pass).
        """
        gl = _gl()
        prog = self._composite_prog
        prog.bind()
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, self._oit_fbo.texture())
        _set_u(prog, "u_accum", 0)
        # Reuse the existing empty fullscreen VAO that blit.vert uses —
        # composite.vert generates its three vertices from gl_VertexID, so
        # any bound VAO satisfies the core-profile draw-call requirement.
        self._fullscreen_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        self._fullscreen_vao.release()
        gl.glBindTexture(GL_TEXTURE_2D, 0)
        prog.release()

    def _render_pick_buffer(self, *,
                            exclude_indices: Optional[set[int]] = None) -> None:
        gl = _gl()
        self._pick_fbo.bind()
        gl.glViewport(0, 0, *self._pick_fbo_size)
        # Depth state setup MUST come before the clear: glClear honours the
        # current depth mask, so we have to enable writes first or the depth
        # buffer keeps its stale (or undefined-on-first-use) values and every
        # subsequent fragment fails the depth test against garbage.
        #
        # An earlier iteration disabled depth test in the pick pass entirely
        # and relied on per-element / per-region back-to-front sorting plus
        # last-write-wins.  That works as long as one region's fragments
        # cleanly win against everything that came before — but for curved
        # caps that's not true: a single region's vertices span a range of
        # view-space depths, and the centroid sort can't tell which cap's
        # fragment is actually closer at a given pixel.  In a 3/4 view where
        # a back cap is half-exposed past a front cap, the front cap's mesh
        # rasterized fragments at depths that overlapped the back cap's
        # visible silhouette and stole pixels the user was hovering on.
        # Per-fragment depth test is the only thing that handles this
        # correctly.
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_LEQUAL)
        gl.glDepthMask(1)
        gl.glDisable(GL_BLEND)
        # Back-face culling OFF to mirror the visible pass.  A click on a
        # backface (a single-surface iris from the "wrong" side, or the
        # inside of a glass body exposed by a clip plane) must resolve to
        # the surface it belongs to; with culling on, those fragments aren't
        # in the pick FBO and the click falls through to whatever's behind.
        # Per-region depth-sort + LEQUAL still gives "closest visible
        # fragment wins" per pixel.
        gl.glDisable(GL_CULL_FACE)
        gl.glDisable(GL_SCISSOR_TEST)
        # Clear to (0, 0, 0, 0).  decode_pixel treats r=g=b=0 + tag=0 as
        # "empty pixel — no element rendered here", which is correct now that
        # encode_element_id offsets indices by +1 (so element 0 encodes to
        # (1, 0, 0, 0)).
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        try:
            gl.glClearDepthf(1.0)
        except AttributeError:
            # Some PySide6 builds expose only glClearDepth (double).
            gl.glClearDepth(1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        vp = self.camera.view_projection()
        cam_pos = self._camera_world_pos()
        # WORKAROUND: use lens_prog for the pick pass.  A slim, dedicated pick
        # program failed to produce ANY fragments when drawing the element VAO
        # despite all locations/uniforms appearing correct — the lens VAO +
        # lens_prog combination is the only one that demonstrably writes
        # fragments on this PySide6 + Windows setup.  lens.frag's ``u_unlit=1``
        # branch outputs ``vec4(u_tint, u_alpha)`` directly, which is exactly
        # the picking encoder's wire format when we feed the encoded RGB to
        # ``u_tint`` and the tag byte (as a 0..1 float) to ``u_alpha``.
        prog = self._lens_prog
        prog.bind()
        _set_u(prog, "u_view_proj", _to_qmatrix(vp))
        _set_u(prog, "u_model", _to_qmatrix(np.eye(4, dtype=np.float32)))
        # Clip planes MUST match the visible pass — otherwise the user can pick
        # geometry that's currently hidden by a cross-section cut, or fail to
        # pick the cap face that's actually under the cursor.
        ax, bx, cx, dx, ay, by, cy, dy = self.clip_state.uniform_vec4_pair()
        _set_u(prog, "u_clip_plane",   float(ax), float(bx), float(cx), float(dx))
        _set_u(prog, "u_clip_plane_b", float(ay), float(by), float(cy), float(dy))

        # Sort back-to-front in VIEW space at the REGION level so the
        # front-most cap wins in the pick FBO, not just the front-most
        # element.  Sorting only at element- or sub-solid-level meant that
        # for a cemented n-let (one element, multiple sub-solids) or for
        # any element with both a front and a back cap, the regions drew
        # in fixed storage order: back cap painted after front cap, so the
        # back surface won every pixel where they overlapped — surface
        # picks went through the front to land on the back.  Per-region
        # depth-sort with last-write-wins fixes both pick-through and the
        # "biased toward the cap centre" symptom (which was the same back
        # cap winning the centre, with edges falling through to the front
        # only because the back cap didn't project there).
        view = self.camera.view_matrix()
        def _region_depth(centroid: tuple) -> float:
            return (
                view[2, 0] * float(centroid[0])
                + view[2, 1] * float(centroid[1])
                + view[2, 2] * float(centroid[2])
                + view[2, 3]
            )

        flat_regions: list[tuple[float, int, dict]] = []
        for se in self.scene.elements:
            if exclude_indices is not None and se.index in exclude_indices:
                continue
            entry = self._element_buffers[se.index]
            if not entry["subsolids"]:
                continue
            for ss in entry["subsolids"]:
                for region in ss["regions"]:
                    flat_regions.append(
                        (_region_depth(region["centroid"]), se.index, region)
                    )
        flat_regions.sort(key=lambda t: t[0])

        # Encode one (element, surface) pair per region so a click resolves
        # not just to the owning element but to the specific optical
        # surface under the cursor — wall halves carry the nearer cap's
        # surface index so wall clicks snap to that surface.  Encoded RGB
        # → u_tint; tag byte → u_alpha; lens.frag with u_unlit=1 emits
        # ``vec4(u_tint, u_alpha)`` — exactly the picking pixel format.
        for _depth, el_idx, region in flat_regions:
            r, g, b, a = picking.encode_element_surface_id(
                el_idx, region["surface_index"]
            )
            _set_u(prog, "u_tint",  float(r), float(g), float(b))
            _set_u(prog, "u_alpha", float(a))
            _set_u(prog, "u_unlit", 1)
            region["vao"].bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, region["draw_count"])
            region["vao"].release()

        # View-cube draws use the same logical scissor as the visible
        # _draw_view_cube, scaled by devicePixelRatio so the cube lands at
        # the same physical-pixel rectangle as the on-screen widget.
        if self._show_view_cube and self._gizmo_pick_vao is not None:
            gizmo_vp, scissor = gizmo_mod.gizmo_view_proj(
                self.camera.azimuth, self.camera.elevation,
                self.width(), self.height()
            )
            dpr = float(self.devicePixelRatioF()) or 1.0
            x = int(round(scissor[0] * dpr))
            y = int(round(scissor[1] * dpr))
            w = int(round(scissor[2] * dpr))
            h = int(round(scissor[3] * dpr))
            gl.glEnable(GL_SCISSOR_TEST)
            gl.glScissor(x, y, w, h)
            gl.glViewport(x, y, w, h)
            gl.glClear(GL_DEPTH_BUFFER_BIT)
            # Enable depth test so the FRONT-FACING cube face wins when
            # multiple faces project to the same pixel.  Without this, the
            # last-drawn face (face 5, "-Z") would always overwrite the
            # visibly-front face → cube clicks intermittently snap to "-Z"
            # regardless of which face the user actually clicked.
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(GL_LESS)
            # Keep using lens_prog: the gizmo cube vertices are at NDC-ish
            # extents in their own world frame, and lens.vert's
            # u_view_proj * u_model * vec4(a_position, 1.0) handles them.
            # gizmo VBO interleaves position only at offset 0, stride 12.
            _set_u(prog, "u_view_proj", _to_qmatrix(gizmo_vp))
            _set_u(prog, "u_model", _to_qmatrix(np.eye(4, dtype=np.float32)))
            # Disable clip planes for the cube: the lens-system clip planes
            # are defined in lens world coords (often d > 0.5), and the cube
            # vertices live in [-0.5, 0.5] in their own frame.  Without this
            # reset, any active clip plane satisfies the discard test on the
            # cube and the pick FBO ends up empty in the corner — visible
            # cube is unaffected because _draw_view_cube uses _gizmo_prog,
            # which has no clip-plane logic.
            _set_u(prog, "u_clip_plane",   0.0, 0.0, 0.0, 0.0)
            _set_u(prog, "u_clip_plane_b", 0.0, 0.0, 0.0, 0.0)
            self._gizmo_pick_vao.bind()
            for face_idx in range(6):
                r, g, b, a = picking.encode_view_cube_face(face_idx)
                _set_u(prog, "u_tint",  float(r), float(g), float(b))
                _set_u(prog, "u_alpha", float(a))
                _set_u(prog, "u_unlit", 1)
                gl.glDrawArrays(GL_TRIANGLES, face_idx * 6, 6)
            self._gizmo_pick_vao.release()
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDisable(GL_SCISSOR_TEST)
            gl.glViewport(0, 0, *self._pick_fbo_size)

        prog.release()

    def _pick_at(self, x: int, y: int, *,
                 exclude_indices: Optional[set[int]] = None) -> dict:
        # Mouse events run outside paintGL/initializeGL — Qt does NOT keep
        # the GL context current at that point.  Every GL call below would
        # silently fail with GL_INVALID_OPERATION (verified via per-step
        # glGetError diagnostics: glBindVertexArray and glDrawArrays both
        # logged 0x502 because the context wasn't bound, so the entire pick
        # pipeline produced zero fragments).  makeCurrent() activates the
        # context for this thread; the matching doneCurrent() at the bottom
        # releases it so Qt's compositor can resume control.
        self.makeCurrent()
        try:
            self._ensure_pick_fbo()
            self._render_pick_buffer(exclude_indices=exclude_indices)
            gl = _gl()
            rgba = bytearray(4)
            gl.glReadPixels(int(x), int(self._pick_fbo_size[1] - y), 1, 1,
                             GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            self._pick_fbo.release()
            # Restore viewport to the physical default framebuffer size.
            gl.glViewport(0, 0, self._pick_fbo_size[0], self._pick_fbo_size[1])
        finally:
            self.doneCurrent()
        result = picking.decode_pixel((rgba[0], rgba[1], rgba[2], rgba[3]))
        # Edge-slop fallback: when nothing was rendered at the click pixel,
        # see if the cursor sits inside the slop disc of a hidden cube face.
        # Lets the user pick an axis-aligned side without first nudging the
        # view off-axis to make the face have non-zero pixels.
        if result["is_empty"] and self._show_view_cube:
            slop_face = self._gizmo_slop_face_at(x, y)
            if slop_face is not None:
                return {
                    "tag": picking.TAG_VIEW_CUBE,
                    "element_index": None,
                    "surface_index": None,
                    "face_index": int(slop_face),
                    "is_empty": False,
                }
        return result

    def _gizmo_slop_face_at(self, x: int, y: int) -> Optional[int]:
        """Return the face index of the hidden gizmo face nearest ``(x, y)``,
        or ``None`` if the cursor is outside every slop disc.  Coords are in
        the same space :meth:`_pick_at` receives (physical pixels, top-down).
        """
        dpr = float(self.devicePixelRatioF()) or 1.0
        targets = gizmo_mod.hidden_face_slop_picks(
            self.camera.azimuth, self.camera.elevation,
            self.width(), self.height(), dpr=dpr,
        )
        best_face: Optional[int] = None
        best_dist = float("inf")
        for face_idx, tx, ty, radius in targets:
            d = math.hypot(float(x) - tx, float(y) - ty)
            if d <= radius and d < best_dist:
                best_dist = d
                best_face = face_idx
        return best_face

    # ------------------------------------------------------------------
    # Hover preview
    # ------------------------------------------------------------------

    def _set_gizmo_hover(self, face_index: Optional[int]) -> bool:
        """Mutate ``_gizmo_hover_face`` and return ``True`` if it changed."""
        if face_index == self._gizmo_hover_face:
            return False
        self._gizmo_hover_face = face_index
        return True

    def _is_in_gizmo_region(self, x_log: int, y_log: int) -> bool:
        """Whether logical pixel ``(x_log, y_log)`` falls inside the gizmo's
        clickable bounding box (corner + edge slop).  Cheap rectangle test
        used to gate the expensive pick when the cursor isn't near the cube.
        """
        if not self._show_view_cube:
            return False
        corner_px = 120
        margin_px = 16
        slop = (gizmo_mod.EDGE_SLOP_PX
                if gizmo_mod.EDGE_SLOP_ENABLED else 0)
        w, h = self.width(), self.height()
        left = w - corner_px - margin_px - slop
        right = w - margin_px + slop
        top = margin_px - slop
        bottom = margin_px + corner_px + slop
        return (left <= x_log <= right) and (top <= y_log <= bottom)

    def _update_hover(self, x: int, y: int) -> None:
        """Pick at the cursor and update the hover preview.

        Picks whenever the cursor is over either a scene element (when the
        selection mode allows it) OR over the view-cube's clickable region
        (always, since the cube is a camera control independent of selection
        mode).  Element/surface hover follows the selection-mode rules;
        gizmo-face hover is set whenever a cube face — visible or slop — is
        under the cursor.
        """
        if not self._gl_ready:
            return
        mode = self.selection_mode()
        scene_hoverable = bool(self.scene.elements) and mode != "none"
        gizmo_hoverable = self._is_in_gizmo_region(x, y)
        if not scene_hoverable and not gizmo_hoverable:
            elem_changed = self.selection.clear_hover()
            gizmo_changed = self._set_gizmo_hover(None)
            if elem_changed or gizmo_changed:
                self.update()
            return

        dpr = float(self.devicePixelRatioF()) or 1.0
        fbx = int(round(x * dpr))
        fby = int(round(y * dpr))
        info = self._pick_at(fbx, fby)

        # View-cube hover: independent of selection mode.
        new_gizmo_face: Optional[int] = None
        if (info["tag"] == picking.TAG_VIEW_CUBE
                and info["face_index"] is not None):
            face = int(info["face_index"])
            if 0 <= face < len(gizmo_mod.FACE_PRESETS):
                new_gizmo_face = face
        gizmo_changed = self._set_gizmo_hover(new_gizmo_face)

        # Element/surface hover.
        new_element = None
        new_surface = None
        if (scene_hoverable
                and not info["is_empty"]
                and info["tag"] == picking.TAG_ELEMENT_BODY):
            idx = info["element_index"]
            if idx is not None and 0 <= idx < len(self.scene.elements):
                new_element = self.scene.elements[idx].element
                if mode == "surface":
                    picked = info.get("surface_index")
                    new_surface = None if picked is None else int(picked)
        elem_changed = self.selection.set_hover(new_element)
        elem_changed = (self.selection.set_hover_surface(new_surface)
                         or elem_changed)
        if elem_changed or gizmo_changed:
            self.update()

    def leaveEvent(self, event) -> None:
        # Cursor left the viewport — drop the hover so the highlight doesn't
        # linger while the user is interacting with other panels (e.g. the
        # optical design editor).
        changed = self.selection.clear_hover()
        changed = self._set_gizmo_hover(None) or changed
        if changed:
            self.update()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._mouse_button = event.button()
        self._mouse_modifiers = event.modifiers()
        self._mouse_last = event.position().toPoint()
        self._mouse_press_pos = event.position().toPoint()
        self._dragging_camera = False
        # A press commits the cursor to either a click (resolved on release)
        # or a drag — either way the hover preview is no longer useful and
        # would just clash with the upcoming selection/drag affordance.
        changed = self.selection.clear_hover()
        changed = self._set_gizmo_hover(None) or changed
        if changed:
            self.update()
        if event.modifiers() & Qt.AltModifier:
            btn = event.button()
            if btn == Qt.LeftButton:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif btn == Qt.MiddleButton:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif btn == Qt.RightButton:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mouse_last is None:
            # No button held — pure hover.  Picking on every move is the
            # simple approach; Qt throttles native mouse-move delivery so
            # this stays bounded.
            pos = event.position().toPoint()
            self._update_hover(pos.x(), pos.y())
            super().mouseMoveEvent(event)
            return
        pos = event.position().toPoint()
        dx = float(pos.x() - self._mouse_last.x())
        dy = float(pos.y() - self._mouse_last.y())
        self._mouse_last = pos

        mods = event.modifiers()
        if mods & Qt.AltModifier and self._mouse_button is not None:
            if not self._dragging_camera:
                # First drag step after a press — kill any preset transition
                # so the manual orbit/pan/dolly starts cleanly from here.
                self._cancel_view_animation()
            self._dragging_camera = True
            if self._mouse_button == Qt.LeftButton:
                self.camera.orbit(dx, dy)
            elif self._mouse_button == Qt.MiddleButton:
                self.camera.pan_drag(dx, dy)
            elif self._mouse_button == Qt.RightButton:
                self.camera.dolly(-dx)
            self.viewChanged.emit(self.camera.state())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # Treat as click only if mouse didn't move significantly and Alt wasn't held
        was_click = (
            self._mouse_press_pos is not None
            and not self._dragging_camera
            and (event.button() == Qt.LeftButton)
        )
        if was_click:
            pos = event.position().toPoint()
            dist = (pos - self._mouse_press_pos).manhattanLength()
            if dist <= 4:
                self._handle_click(pos.x(), pos.y())

        # Plain right-click (no Alt — Alt+Right is camera dolly), no drag:
        # offer a context menu on the picked element/surface. Modifiers are
        # read from press time so a released Alt-dolly still counts as Alt.
        was_rclick = (
            self._mouse_press_pos is not None
            and not self._dragging_camera
            and event.button() == Qt.RightButton
            and not (self._mouse_modifiers & Qt.AltModifier)
        )
        if was_rclick:
            pos = event.position().toPoint()
            if (pos - self._mouse_press_pos).manhattanLength() <= 4:
                self._handle_context_click(
                    pos.x(), pos.y(), event.globalPosition().toPoint()
                )

        self._mouse_button = None
        self._mouse_last = None
        self._mouse_press_pos = None
        self._dragging_camera = False
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = float(event.angleDelta().y()) / 120.0
        self.camera.wheel(delta)
        self.viewChanged.emit(self.camera.state())
        self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Home:
            self.reset_view()
            event.accept()
            return
        if key == Qt.Key.Key_F:
            if self.selection.element is not None:
                # Frame the selected element
                for se in self.scene.elements:
                    if se.element is self.selection.element:
                        if se.mesh.vertex_count > 0:
                            mn = se.mesh.vertices.min(axis=0)
                            mx = se.mesh.vertices.max(axis=0)
                            self.camera.fit_to_bbox(mn, mx)
                            self.viewChanged.emit(self.camera.state())
                            self.update()
                        break
            else:
                # Nothing selected: frame everything (sensor included).
                self.camera.fit_to_bbox(self.scene.bbox_min, self.scene.bbox_max)
                self.viewChanged.emit(self.camera.state())
                self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Click resolution
    # ------------------------------------------------------------------

    def _pick_cycle_snapshot(self) -> tuple:
        """Identifier for the camera+clip+scene state.

        If this changes between two clicks at the same screen position, the
        pixel now refers to different world geometry, so the cycle restarts.
        """
        cs = self.camera.state()
        return (
            cs["azimuth"], cs["elevation"], cs["dist"],
            tuple(cs["pivot"]), tuple(cs["pan"]), cs["ortho_height"],
            self.clip_state.uniform_vec4_pair(),
            len(self.scene.elements),
        )

    def _handle_click(self, x: int, y: int) -> None:
        # Click coords from Qt are LOGICAL widget pixels.  The pick FBO is
        # sized in PHYSICAL pixels so the gizmo's on-screen position matches
        # between the visible draw and the pick render — scale click coords
        # by devicePixelRatio to read the right FBO pixel.
        dpr = float(self.devicePixelRatioF()) or 1.0
        fbx = int(round(x * dpr))
        fby = int(round(y * dpr))

        mode = self.selection_mode()

        # Decide whether this click continues the current cycle or starts a
        # new one.  A "cycle" is a sequence of clicks at nearly the same spot;
        # each step selects the next element behind the previously-selected
        # one (top-first → furthest).  Cycling is element-mode only; surface
        # mode treats each click as a fresh resolve so the user can step
        # between front and back caps without the cycle stealing one of them.
        snapshot = self._pick_cycle_snapshot()
        continue_cycle = (
            mode == "element"
            and self._cycle_pos is not None
            and (QPoint(x, y) - self._cycle_pos).manhattanLength()
                <= self._CYCLE_TOLERANCE_PX
            and self._cycle_snapshot == snapshot
        )
        if not continue_cycle:
            self._cycle_excluded = set()

        info = self._pick_at(
            fbx, fby,
            exclude_indices=self._cycle_excluded if continue_cycle else None,
        )
        tag = info["tag"]

        # View-cube face → camera preset (handled by picking pass now).
        # Always honoured regardless of selection mode — the cube is a
        # camera control, not part of element/surface picking.
        if tag == picking.TAG_VIEW_CUBE and info["face_index"] is not None:
            face_idx = info["face_index"]
            self._cycle_pos = None
            self._cycle_excluded = set()
            if 0 <= face_idx < len(gizmo_mod.FACE_PRESETS):
                self.set_view(gizmo_mod.FACE_PRESETS[face_idx])
            return

        # ``none`` mode disables all selection changes from viewport clicks
        # (the host can still drive selection in via set_selected_element /
        # set_selected_surface).  We bail out AFTER the view-cube check so
        # the cube still rotates the camera.
        if mode == "none":
            self._cycle_pos = None
            self._cycle_excluded = set()
            return

        # Empty result.  If we were cycling and ran out of things behind,
        # wrap by clearing the exclude set and re-picking from the front.
        if info["is_empty"] or info["element_index"] is None:
            if continue_cycle and self._cycle_excluded:
                self._cycle_excluded = set()
                info = self._pick_at(fbx, fby)
                tag = info["tag"]

        if info["is_empty"] or info["element_index"] is None:
            # Truly nothing under the cursor.  Reset cycle and clear selection.
            self._cycle_pos = None
            self._cycle_excluded = set()
            had_element = self.selection.element is not None
            had_surface = self.selection.surface is not None
            if self.selection.clear():
                if had_element:
                    self.elementSelected.emit(None)
                if had_surface:
                    self.surfaceSelected.emit(None)
                self.update()
            return

        idx = info["element_index"]
        if not (0 <= idx < len(self.scene.elements)):
            return
        se = self.scene.elements[idx]
        picked_surface = info.get("surface_index")

        # Record this click as the active cycle anchor and mark this element
        # as already-seen so the next click steps past it (element-mode
        # cycling only; surface mode skips this).
        self._cycle_pos = QPoint(x, y) if mode == "element" else None
        self._cycle_snapshot = snapshot
        if mode == "element":
            self._cycle_excluded.add(idx)

        if mode == "surface":
            # Surface mode picks the surface under the cursor; the owning
            # element is updated alongside so a host listening to element
            # signals still has the right context.
            updated = False
            if self.selection.set_element(se.element):
                self.elementSelected.emit(se.element)
                updated = True
            new_surface = (
                None if picked_surface is None else int(picked_surface)
            )
            if self.selection.set_surface(new_surface):
                self.surfaceSelected.emit(new_surface)
                updated = True
            if updated:
                self.update()
            return

        # Element mode (default).  Surface selection (if any) is dropped:
        # the host signalled an element-level click and surface state would
        # leave the highlight ambiguous.
        had_surface = self.selection.surface is not None
        element_changed = self.selection.set_element(se.element)
        surface_cleared = self.selection.clear_surface()
        if element_changed:
            self.elementSelected.emit(se.element)
        if had_surface and surface_cleared:
            self.surfaceSelected.emit(None)
        if element_changed or surface_cleared:
            self.update()

    def _handle_context_click(self, x: int, y: int, global_pos: QPoint) -> None:
        """Resolve the element/surface under a right-click and emit
        :sig:`contextMenuRequested` with the picked identity.

        Also updates the viewport's own selection (and emits the usual
        selection signals) so the highlight agrees with the menu. Bails
        silently when the context menu is disabled, in "none" mode, or on
        empty / non-element picks (blank space, view cube)."""
        if not self._context_menu_enabled:
            return
        mode = self.selection_mode()
        if mode == "none":
            return

        # Same DPR scaling as _handle_click: click coords are logical pixels,
        # the pick FBO is sized in physical pixels.
        dpr = float(self.devicePixelRatioF()) or 1.0
        fbx = int(round(x * dpr))
        fby = int(round(y * dpr))
        info = self._pick_at(fbx, fby)

        if (
            info["is_empty"]
            or info["tag"] != picking.TAG_ELEMENT_BODY
            or info["element_index"] is None
        ):
            return
        idx = info["element_index"]
        if not (0 <= idx < len(self.scene.elements)):
            return
        se = self.scene.elements[idx]
        picked_surface = info.get("surface_index")

        # Sync selection so the highlight matches the menu target. Mirrors
        # the non-cycling tail of _handle_click.
        if mode == "surface":
            if self.selection.set_element(se.element):
                self.elementSelected.emit(se.element)
            new_surface = (
                None if picked_surface is None else int(picked_surface)
            )
            if self.selection.set_surface(new_surface):
                self.surfaceSelected.emit(new_surface)
        else:
            had_surface = self.selection.surface is not None
            element_changed = self.selection.set_element(se.element)
            surface_cleared = self.selection.clear_surface()
            if element_changed:
                self.elementSelected.emit(se.element)
            if had_surface and surface_cleared:
                self.surfaceSelected.emit(None)
        self.update()

        self.contextMenuRequested.emit({
            "mode": mode,
            "element_index": int(idx),
            "element": se.element,
            "surface_index": (
                None if picked_surface is None else int(picked_surface)
            ),
            "global_pos": global_pos,
        })



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_qmatrix(mat: np.ndarray):
    """Convert a 4x4 numpy matrix to a QMatrix4x4 for setUniformValue."""
    from PySide6.QtGui import QMatrix4x4
    m = np.asarray(mat, dtype=np.float32)
    return QMatrix4x4(*[float(v) for v in m.flatten().tolist()])


