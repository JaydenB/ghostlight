"""Body widget for the ``sourceflare`` panel type.

Renders the project's lens with an extended (area) light source: a
geometric shape — point, circle, rectangle or square — of angular size
in degrees, centered on a draggable marker.  The shape is sampled into
weighted angular offsets (``ghostlight.source_sampling``) and rendered in
chunks through ``OpticalSystem.render_source_flare``; partial sums are
displayed as they accumulate, so the image refines progressively and a
source drag or lens edit aborts the remaining chunks instead of
blocking on a full-quality frame.

Threading is the designer's standard render pattern (daemon worker + 50 ms
poller + epoch discard) with one addition: a render *serial* bumped on
every new request, checked between chunks so an in-flight progressive pass
stops as soon as its inputs are stale.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import ghostlight
from ghostlight import source_sampling

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from ..optical_editor import surface_actions
from ..project import Project
from ..settings import AppSettings
from .. import lens_metrics as lm_mod
from ..render_common import (
    DRAFT_PRESET,
    EXPOSURE_STOPS_MIN,
    FLARE_GAIN,
    HIGH_PLUS_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    MIN_SURFACES,
    POLL_INTERVAL_MS,
    PUPIL_JITTER,
    SLIDER_MAX,
    SLIDER_MIN,
    SLIDER_SCALE,
    SOURCE_RGB,
    ExposureDialog,
    FlareCanvas,
    RenderSettings,
    RenderSettingsDialog,
    attach_spinbox_scrubber,
)
from .. import viewtransform as vt
from ..export import writers
from ..vignette import VignetteController
from . import export_worker, motion_patterns
from .export_dialog import ExportAnimationDialog, ExportOptions

_log = logging.getLogger("ghostlight_designer.sourceflare_panel")

# Samples per render_source_flare call. Small enough that the first
# partial appears quickly and an aborted pass wastes at most one chunk;
# large enough that per-call overhead (probe passes, launch latency)
# stays amortized.
CHUNK_SIZE = 8

SHAPE_POINT = "point"
SHAPE_CIRCLE = "circle"
SHAPE_RECT = "rect"
SHAPE_SQUARE = "square"
SHAPE_POLYGON = "polygon"

# (key, combo label). W is the primary size for every extended shape (circle
# diameter, square side, polygon across-corners); H is rect-only; rotation
# orients rect/square/polygon; sides is polygon-only.
SHAPES = [
    (SHAPE_POINT, "Point"),
    (SHAPE_CIRCLE, "Circle"),
    (SHAPE_RECT, "Rectangle"),
    (SHAPE_SQUARE, "Square"),
    (SHAPE_POLYGON, "Polygon"),
]

# Shapes whose orientation matters (a disk is rotation-invariant).
_ROTATABLE = {SHAPE_RECT, SHAPE_SQUARE, SHAPE_POLYGON}

SIZE_DEG_MIN = 0.01
SIZE_DEG_MAX = 45.0
SAMPLES_MIN = 1
SAMPLES_MAX = 1024
SIDES_MIN = 3
SIDES_MAX = 12

# This panel's exposure works in half-stop clicks: the dialog snaps every
# input to the 0.5 grid and Auto-Expose rounds its metered value onto the
# same grid, with the ceiling raised to +90 st (the shared dialog default
# is ±20 in 0.25 steps).
EXPOSURE_SNAP_STEP = 0.5
EXPOSURE_MAX_STOPS = 90.0

# Matte-box (front-of-lens rectangular flag) inline controls. Half-extents are
# measured from the optical axis in millimetres; the flag plane sits
# ``z_front_mm`` ahead of the front vertex. The inline control drives symmetric
# blades — one Width and one Height half-extent map onto the renderer's
# left=right / top=bottom MatteBox. Disabled leaves the render config's matte
# box off, a total no-op in the tracer.
MATTE_HALF_MM_MIN = 0.5
MATTE_HALF_MM_MAX = 300.0
MATTE_Z_MM_MIN = 1.0
MATTE_Z_MM_MAX = 1000.0


@dataclass(frozen=True)
class _MatteBoxState:
    """Immutable snapshot of the inline matte-box controls, safe to hand to
    the render worker thread. Symmetric blades: ``half_w_mm`` drives
    left=right, ``half_h_mm`` drives top=bottom."""
    enabled: bool = False
    half_w_mm: float = 30.0
    half_h_mm: float = 20.0
    z_front_mm: float = 60.0


def build_shape_offsets(
    shape: str,
    size_w_deg: float,
    size_h_deg: float,
    samples: int,
    rotation_deg: float = 0.0,
    n_sides: int = 6,
) -> np.ndarray:
    """Sample a shape spec into (N, 3) [d_angle_x, d_angle_y, weight] rows.

    Sizes are full angular extents in degrees (circle: diameter; rect:
    width × height; square: side; polygon: across-corners diameter).
    ``rotation_deg`` orients rect/square/polygon; a point ignores everything.
    """
    if shape == SHAPE_POINT:
        return source_sampling.sample_point()
    n = max(SAMPLES_MIN, min(SAMPLES_MAX, int(samples)))
    half_w = math.radians(max(SIZE_DEG_MIN, float(size_w_deg))) * 0.5
    half_h = math.radians(max(SIZE_DEG_MIN, float(size_h_deg))) * 0.5
    if shape == SHAPE_CIRCLE:
        return source_sampling.sample_disk(half_w, n=n)  # rotation-invariant
    if shape == SHAPE_SQUARE:
        offsets = source_sampling.sample_square(half_w, n=n)
    elif shape == SHAPE_POLYGON:
        offsets = source_sampling.sample_polygon(half_w, int(n_sides), n=n)
    else:
        offsets = source_sampling.sample_rect(half_w, half_h, n=n)
    return source_sampling.rotate_offsets(offsets, math.radians(float(rotation_deg)))


def build_flare_config(
    sx: float,
    sy: float,
    half_w: float,
    half_h: float,
    settings: RenderSettings,
    matte: "_MatteBoxState",
    *,
    ghost_filter: Optional["ghostlight.GhostFilter"] = None,
) -> "ghostlight.PointFlareConfig":
    """Build the :class:`ghostlight.PointFlareConfig` this panel renders with.

    Extracted from :meth:`SourceFlarePanelBody._do_render_chunk` so the ghost
    explorer's brightness-metering pass configures its render exactly the way
    the display render is configured — same gain, colour space, accelerators,
    matte box — and only overrides ``aov_mode`` on top. One builder, so the two
    can't drift apart into disagreeing about which ghosts exist or how bright
    they are.
    """
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = sx
    cfg.source_y = sy
    cfg.source_r = SOURCE_RGB[0]
    cfg.source_g = SOURCE_RGB[1]
    cfg.source_b = SOURCE_RGB[2]
    cfg.flare_gain = FLARE_GAIN
    # Scene-linear ACEScg output, read as such by the designer view transform.
    cfg.output_cs = ghostlight.OutputColorSpace.ACESCG
    cfg.ray_grid = int(settings.ray_grid)
    cfg.spectral_samples = int(settings.spectral_samples)
    cfg.pupil_jitter = PUPIL_JITTER
    cfg.sensor_half_w = half_w
    cfg.sensor_half_h = half_h
    # Ghost sampling accelerators (user-controllable via Render Settings).
    cfg.cull_dead_pairs = bool(settings.cull_dead_pairs)
    cfg.concentrate_samples = bool(settings.concentrate_samples)
    cfg.adaptive_sample_budgets = bool(settings.adaptive_sample_budgets)
    cfg.adaptive_density_boost = float(settings.adaptive_density_boost)
    # Aperture-diffraction starburst (a separate additive layer). For an
    # extended source it convolves with the source shape — one starburst per
    # sampled offset — so it composites linearly with the progressive chunks.
    cfg.diffraction.starburst = bool(settings.starburst)
    _engine = getattr(ghostlight, "StarburstEngine", None)
    if _engine is not None:
        cfg.diffraction.starburst_engine = (
            _engine.MDFT if settings.starburst_engine == "mdft" else _engine.SPRITE)
    cfg.diffraction.scale_trim = float(settings.starburst_scale_trim)
    cfg.diffraction.starburst_gain = float(settings.starburst_gain)
    cfg.diffraction.spectral_samples = int(settings.starburst_spectral)
    cfg.diffraction.starburst_grid = int(settings.starburst_grid)
    cfg.diffraction.starburst_grid_cap = int(settings.starburst_grid_cap)
    # Physical veiling glare: a separate additive layer folded into the
    # metered flare. Guarded so an older binding without the veil fields
    # simply renders no veil.
    if hasattr(cfg.diffraction, "veil"):
        cfg.diffraction.veil = bool(settings.veil)
        cfg.diffraction.veil_gain = float(settings.veil_gain)
        cfg.diffraction.veil_spread = float(settings.veil_spread)
        cfg.diffraction.veil_falloff = float(settings.veil_falloff)
    # Physical ghost-edge diffraction: HURB bends each ghost ray that passes
    # a hard edge into a soft chromatic glow. Guarded so an older binding
    # without the hurb fields / enum renders no kick.
    if hasattr(cfg.diffraction, "hurb"):
        cfg.diffraction.hurb = bool(settings.hurb)
        _kick = getattr(ghostlight, "HurbKickDistribution", None)
        if _kick is not None:
            cfg.diffraction.hurb_kick = (
                _kick.GAUSSIAN if settings.hurb_kick == "gaussian"
                else _kick.LORENTZIAN)
    # Front-of-lens matte box (a baffle): an inline source control, not
    # a Render-Settings option. It clips entrance rays (tracer + starburst /
    # diffraction pupils), so it composites into every layer. Symmetric
    # blades drive the renderer's left=right / top=bottom. Mutate-and-assign
    # the existing MatteBox (the ghostlight.MatteBox class isn't always re-exported
    # at module top level, but the config member always is). Guarded so an
    # older binding without matte_box simply renders no occluder.
    if matte.enabled and hasattr(cfg.diffraction, "matte_box"):
        mb = cfg.diffraction.matte_box
        mb.enabled = True
        mb.z_front_mm = float(matte.z_front_mm)
        mb.left = mb.right = float(matte.half_w_mm)
        mb.top = mb.bottom = float(matte.half_h_mm)
        cfg.diffraction.matte_box = mb
    # Film-gate flare (mechanical): the aperture plate's cut edge scatters
    # light that would land just outside the frame back into it. Its own
    # additive layer, folded into the metered flare alongside the veil.
    # Mutate-and-assign as the matte box does — ghostlight.GateConfig isn't reliably
    # re-exported at module top level, but the config member always is. Guarded
    # so an older binding without the gate field simply renders no gate.
    if hasattr(cfg, "gate"):
        gt = cfg.gate
        gt.enabled = bool(settings.gate)
        gt.standoff_mm = float(settings.gate_standoff)
        gt.roughness_rad = float(settings.gate_roughness)
        gt.gain = float(settings.gate_gain)
        cfg.gate = gt
    if ghost_filter is not None:
        cfg.ghost_filter = ghost_filter
    return cfg


class SourceFlareCanvas(FlareCanvas):
    """The shared flare canvas plus a dashed outline of the source shape.

    The outline is a list of (fx, fy) points in image-fractional
    coordinates (same space as the source marker), mapped individually
    through the angle→screen transform so a rotated/curved shape is drawn
    faithfully.  ``None`` hides it (point shape or no calibration yet).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._outline: Optional[list] = None

    def set_shape_outline(self, points: Optional[list]) -> None:
        self._outline = points
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._image is None or not self._outline:
            return
        r = self._image_rect()
        poly = QPolygonF()
        for fx, fy in self._outline:
            poly.append(QPointF(r.left() + fx * r.width(), r.top() + fy * r.height()))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 170, 60, 200), 1.0)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)


class SourceFlarePanelBody(QWidget):
    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._app_settings = settings

        # Off-centre by default: an on-axis source puts every ghost on top of
        # the source itself, so the flare only reads as a flare once the source
        # is away from centre. Upper-right quadrant, matching how a practical is
        # usually framed. "Recenter Source" still returns to (0.5, 0.5).
        self._sx: float = 0.75
        self._sy: float = 0.25
        self._shape: str = SHAPE_POINT
        self._size_w_deg: float = 1.0
        self._size_h_deg: float = 1.0
        self._rotation_deg: float = 0.0
        self._n_sides: int = 6
        self._samples: int = 64

        # Front-of-lens matte box (inline source control, NOT a Render-Settings
        # option). Off by default so the render is byte-identical until enabled.
        self._matte: _MatteBoxState = _MatteBoxState()

        # Per-panel viewer exposure in stops (2**stops linear pre-multiply,
        # applied before the designer-wide ACES view transform); default 0.0.
        self._exposure_stops: float = 0.0
        # Set when a freshly loaded lens has reset the exposure to 0: the next
        # frame that lands auto-meters itself (see _poll_results), so a new lens
        # opens on a viewable image instead of a black or blown-out one. Armed
        # for the first render of this panel too.
        self._auto_expose_pending: bool = True
        self._auto_render: bool = True
        # Desqueeze on by default: a source flare is judged on its shape, and an
        # anamorphic lens's render is only the right shape once unsqueezed. The
        # factor is resolved from the lens at the end of __init__ (and refreshed
        # whenever the system changes); spherical lenses give 1.0, a no-op.
        self._desqueeze: bool = True
        self._squeeze_factor: float = 1.0

        self._is_active: bool = False
        self._dirty_pending: bool = False

        self._settings: RenderSettings = MID_PRESET
        self._latest_hwc: Optional[np.ndarray] = None
        # Flare (ghost) layer of the latest frame, without the starburst —
        # the Auto-Expose metering reference so the starburst's hot core can't
        # hijack it (see auto_expose / _worker).
        self._latest_flare_hwc: Optional[np.ndarray] = None
        self._cc_dialog: Optional[ExposureDialog] = None
        # Live handle to the modeless render-settings dialog (if open) so a
        # second menu invocation raises it instead of spawning a duplicate.
        self._settings_dialog: Optional[RenderSettingsDialog] = None

        # Threading state (see module docstring). ``_serial`` marks the
        # newest request; a worker whose captured serial falls behind
        # aborts between chunks.
        self._lock = threading.Lock()
        self._pending: bool = False
        self._busy: bool = False
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._epoch: int = 0
        self._serial: int = 0

        # Animation-export state. While ``_exporting`` is True the live
        # progressive renderer is suspended (see the guards in
        # ``_maybe_launch`` / ``request_render``) so the batch worker owns the
        # GPU; the modal progress dialog keeps the poll timer firing underneath
        # it, so the guard is what stops it relaunching mid-export.
        self._exporting: bool = False
        self._export_options: ExportOptions = ExportOptions()
        self._export_thread: Optional[QThread] = None
        self._export_worker: Optional[export_worker.AnimationWorker] = None
        self._export_cancel: Optional[threading.Event] = None
        self._export_progress: Optional[QProgressDialog] = None
        self._export_idle_waits: int = 0

        # Angle→screen mapping snapshot from the last dispatch, used to
        # draw the shape overlay: (tan_w, tan_h) where
        # screen_frac = 0.5 + 0.5 * tan(angle) / tan_*.
        # (angle_x, angle_y, screen_x, screen_y, jacobian4, half_w, half_h)
        # from the renderer's own source-map solve; see _refresh below.
        self._overlay_map: Optional[Tuple[float, float, float, float,
                                          Tuple[float, float, float, float],
                                          float, float]] = None

        self._canvas = SourceFlareCanvas(self)
        self._canvas.set_source(self._sx, self._sy)
        self._canvas.sourceDragged.connect(self._on_canvas_drag)

        # Vignette overlay (regions no primary ray can reach). Off by default,
        # toggled from the View menu.
        self._vignette = VignetteController(project, self._canvas, parent=self)

        self._status = QLabel("", self)
        self._status.setStyleSheet("color: #aaa; padding: 2px 6px;")

        self._suppress_widget_signals: bool = False
        self._slider_x, self._spin_x = self._build_axis_row()
        self._slider_y, self._spin_y = self._build_axis_row()
        self._slider_x.valueChanged.connect(self._on_slider_x)
        self._slider_y.valueChanged.connect(self._on_slider_y)
        self._spin_x.valueChanged.connect(self._on_spin_x)
        self._spin_y.valueChanged.connect(self._on_spin_y)

        shape_grid = self._build_shape_grid()
        matte_grid = self._build_matte_grid()

        # Ctrl+MMB value-scrubbing on every spinbox in the panel (the
        # triggers parent to their spinbox, so the list is just to keep
        # them discoverable/tested — not for lifetime).
        self._scrubbers = [
            attach_spinbox_scrubber(self._spin_x, label="Source X"),
            attach_spinbox_scrubber(self._spin_y, label="Source Y"),
            attach_spinbox_scrubber(self._spin_size_w, label="Width (°)"),
            attach_spinbox_scrubber(self._spin_size_h, label="Height (°)"),
            attach_spinbox_scrubber(self._spin_rot, label="Rotation (°)"),
            attach_spinbox_scrubber(self._spin_sides, label="Sides"),
            attach_spinbox_scrubber(self._spin_samples, label="Samples"),
            attach_spinbox_scrubber(self._spin_matte_w, label="Matte width (mm)"),
            attach_spinbox_scrubber(self._spin_matte_h, label="Matte height (mm)"),
            attach_spinbox_scrubber(self._spin_matte_z, label="Matte distance (mm)"),
        ]

        matte_sep = QFrame(self)
        matte_sep.setFrameShape(QFrame.HLine)
        matte_sep.setFrameShadow(QFrame.Sunken)

        # Wrapped in one container so the View menu's "Matte Box Controls"
        # toggle can hide the whole group (sep + grid) as a unit; a bare
        # QGridLayout has no visibility of its own to hide.
        self._matte_container = QWidget(self)
        matte_container_layout = QVBoxLayout(self._matte_container)
        matte_container_layout.setContentsMargins(0, 0, 0, 0)
        matte_container_layout.setSpacing(0)
        matte_container_layout.addWidget(matte_sep)
        matte_container_layout.addLayout(matte_grid)
        self._matte_container.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._canvas, 1)
        outer.addWidget(self._status)
        outer.addLayout(self._row_layout("X", self._slider_x, self._spin_x))
        outer.addLayout(self._row_layout("Y", self._slider_y, self._spin_y))
        outer.addLayout(shape_grid)
        outer.addWidget(self._matte_container)
        outer.addLayout(self._build_render_row())

        self._sync_source_widgets()
        self._sync_shape_widget_enabled()
        self._sync_matte_widget_enabled()

        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)
        project.ghostSoloChanged.connect(self._on_system_modified)
        project.systemSetupChanged.connect(self._on_system_setup_changed)
        self._app_settings.autoUpdateChanged.connect(
            self._on_global_auto_update_changed
        )
        self._app_settings.viewTransformChanged.connect(
            self._on_view_transform_changed
        )

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start()

        # Desqueeze defaults on, so seed the factor from the current lens.
        self._refresh_squeeze_factor()
        self._push_squeeze_to_canvas()

        self._refresh_status_for_lens()
        self._dirty_pending = True

    # ------------------------------------------------------------------
    # Public API used by the View menu
    # ------------------------------------------------------------------

    @property
    def auto_render(self) -> bool:
        return self._auto_render

    def set_auto_render(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._auto_render:
            return
        self._auto_render = enabled
        if enabled:
            self.request_render()
        else:
            self._status.setText("Auto-render off")

    def request_render(self) -> None:
        if self._exporting:
            # The batch exporter owns the GPU; note the request so the live
            # view refreshes once the export finishes (see _finish_export).
            self._dirty_pending = True
            return
        if not self._lens_eligible():
            self._canvas.clear_image(self._placeholder_for_lens())
            self._status.setText(self._placeholder_for_lens())
            return
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        self._request()

    def recenter_source(self) -> None:
        self._sx = 0.5
        self._sy = 0.5
        self._sync_source_widgets()
        self._canvas.set_source(self._sx, self._sy)
        self._update_overlay()
        self.request_render()

    @property
    def vignette_overlay(self) -> bool:
        return self._vignette.enabled

    def set_vignette_overlay(self, enabled: bool) -> None:
        """Toggle the half-red vignette overlay (regions no primary ray can
        reach). Enabling kicks a background probe if the cached mask is stale;
        disabling just hides it. Display-only; independent of the render."""
        self._vignette.set_enabled(bool(enabled))

    @property
    def matte_controls_visible(self) -> bool:
        # See spot_diagram's ``settings_visible`` for why isHidden() (the
        # explicit visibility setting) and not isVisible() (on-screen now).
        return not self._matte_container.isHidden()

    def set_matte_controls_visible(self, visible: bool) -> None:
        """Show/hide the inline matte-box controls. Purely a layout toggle —
        the matte-box state itself (enabled/size) is unaffected either way."""
        self._matte_container.setVisible(bool(visible))

    @property
    def desqueeze(self) -> bool:
        return self._desqueeze

    def set_desqueeze(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._desqueeze:
            return
        self._desqueeze = enabled
        if enabled:
            self._refresh_squeeze_factor()
        self._push_squeeze_to_canvas()

    def _refresh_squeeze_factor(self) -> None:
        sensor = self._project.system_setup.sensor
        try:
            metrics = lm_mod.compute_metrics(
                self._project.system,
                float(sensor.width_mm),
                float(sensor.height_mm),
            )
        except Exception:
            _log.exception("SourceFlarePanelBody: compute_metrics failed")
            metrics = None
        self._squeeze_factor = float(metrics.squeeze) if metrics is not None else 1.0

    def _push_squeeze_to_canvas(self) -> None:
        self._canvas.set_squeeze(self._squeeze_factor if self._desqueeze else 1.0)

    # ------------------------------------------------------------------
    # Exposure + view transform
    # ------------------------------------------------------------------

    def open_exposure_dialog(self) -> None:
        if self._cc_dialog is not None and self._cc_dialog.isVisible():
            self._cc_dialog.raise_()
            self._cc_dialog.activateWindow()
            return
        dlg = ExposureDialog(
            self._exposure_stops,
            parent=self,
            stops_max=EXPOSURE_MAX_STOPS,
            stops_step=EXPOSURE_SNAP_STEP,
            snap_to_step=True,
        )
        dlg.stopsChanged.connect(self._on_exposure_stops_changed)
        dlg.finished.connect(self._on_exposure_dialog_finished)
        self._cc_dialog = dlg
        dlg.show()

    def _on_exposure_dialog_finished(self, _result: int) -> None:
        self._cc_dialog = None

    def _on_exposure_stops_changed(self, stops: float) -> None:
        self._exposure_stops = float(stops)
        self._redisplay()

    def auto_expose(self) -> None:
        # Meter the FLARE (ghost), not the composite: the starburst core is
        # orders of magnitude brighter, so metering the composite exposes for that
        # core and buries the flare. Exposing for the flare lets the star core clip
        # to white like a real sunstar. Falls back to the composite when no flare
        # layer is cached or the starburst is off (they're identical).
        if self._latest_hwc is None:
            return
        meter_src = (self._latest_flare_hwc
                     if self._latest_flare_hwc is not None else self._latest_hwc)
        stops = vt.meter_auto_stops(meter_src)
        # Same half-stop grid + cap the exposure dialog enforces, so the
        # metered value round-trips through the spinbox unchanged.
        stops = round(stops / EXPOSURE_SNAP_STEP) * EXPOSURE_SNAP_STEP
        stops = min(max(stops, EXPOSURE_STOPS_MIN), EXPOSURE_MAX_STOPS)
        self._exposure_stops = stops
        if self._cc_dialog is not None:
            self._cc_dialog.set_stops(stops)
        self._redisplay()

    def reset_exposure(self) -> None:
        self._exposure_stops = 0.0
        if self._cc_dialog is not None:
            self._cc_dialog.set_stops(0.0)
        self._redisplay()

    def _on_view_transform_changed(self) -> None:
        self._redisplay()

    def _redisplay(self) -> None:
        if self._latest_hwc is None:
            return
        try:
            spec = vt.spec_from_settings(self._app_settings)
            display = vt.apply_view(self._latest_hwc, self._exposure_stops, spec)
        except vt.ViewTransformError as exc:
            self._status.setText(f"View transform error: {exc}")
            return
        self._canvas.set_image(vt.to_qimage(display))
        self._status.setText(self._final_status_text())

    # ------------------------------------------------------------------
    # Render settings dialog + presets
    # ------------------------------------------------------------------

    @property
    def settings(self) -> RenderSettings:
        return self._settings

    def apply_settings(self, new_settings: RenderSettings) -> None:
        new_settings = new_settings.clamp()
        if new_settings == self._settings:
            return
        self._settings = new_settings
        self._latest_hwc = None
        if self._lens_eligible():
            self.request_render()

    def apply_preset_draft(self) -> None:
        self.apply_settings(DRAFT_PRESET)

    def apply_preset_mid(self) -> None:
        self.apply_settings(MID_PRESET)

    def apply_preset_high(self) -> None:
        self.apply_settings(HIGH_PRESET)

    def apply_preset_high_plus(self) -> None:
        self.apply_settings(HIGH_PLUS_PRESET)

    def open_render_settings_dialog(self) -> None:
        """Open the modeless render-settings dialog. Every change applies
        live (:meth:`apply_settings`), so the user can toggle passes and scrub
        values while watching the panel re-render — no need to close it."""
        if self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        dlg = RenderSettingsDialog(self._settings, parent=self)
        dlg.settingsChanged.connect(self.apply_settings)
        dlg.finished.connect(self._on_settings_dialog_finished)
        self._settings_dialog = dlg
        dlg.show()

    def _on_settings_dialog_finished(self, _result: int) -> None:
        # Drop our reference so the next open builds fresh.
        self._settings_dialog = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._is_active = True
        self._vignette.set_active(True)
        if self._dirty_pending:
            self.request_render()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._is_active = False
        self._vignette.set_active(False)
        self._dirty_pending = True

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_axis_row(self) -> Tuple[QSlider, QDoubleSpinBox]:
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(int(SLIDER_MIN * SLIDER_SCALE), int(SLIDER_MAX * SLIDER_SCALE))
        slider.setSingleStep(int(0.01 * SLIDER_SCALE))
        slider.setPageStep(int(0.1 * SLIDER_SCALE))

        spin = MathDoubleSpinBox(self)
        spin.setRange(SLIDER_MIN, SLIDER_MAX)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setKeyboardTracking(False)
        spin.setFixedWidth(96)
        return slider, spin

    def _row_layout(self, name: str, slider: QSlider, spin: QDoubleSpinBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(6)
        label = QLabel(name, self)
        label.setFixedWidth(14)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        return row

    def _build_shape_grid(self) -> QGridLayout:
        """Compact 2-column grid of the shape controls.

        Scrubbers are attached centrally in ``__init__`` (not here) so
        every spinbox in the panel gets one uniformly.
        """
        self._shape_combo = QComboBox(self)
        for key, label in SHAPES:
            self._shape_combo.addItem(label, key)
        self._shape_combo.setCurrentIndex(
            next(i for i, (k, _) in enumerate(SHAPES) if k == self._shape)
        )
        self._shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        self._shape_combo.setToolTip(
            "Source geometry. Sizes are angular extents in degrees — the "
            "source sits at infinity, so a 0.53° circle matches the sun."
        )

        self._spin_size_w = MathDoubleSpinBox(self)
        self._spin_size_w.setRange(SIZE_DEG_MIN, SIZE_DEG_MAX)
        self._spin_size_w.setDecimals(2)
        self._spin_size_w.setSingleStep(0.1)
        self._spin_size_w.setSuffix(" °")
        self._spin_size_w.setValue(self._size_w_deg)
        self._spin_size_w.setKeyboardTracking(False)
        self._spin_size_w.setToolTip(
            "Angular width in degrees (circle: diameter; square: side; "
            "polygon: across-corners)."
        )
        self._spin_size_w.valueChanged.connect(self._on_size_w_changed)

        self._spin_size_h = MathDoubleSpinBox(self)
        self._spin_size_h.setRange(SIZE_DEG_MIN, SIZE_DEG_MAX)
        self._spin_size_h.setDecimals(2)
        self._spin_size_h.setSingleStep(0.1)
        self._spin_size_h.setSuffix(" °")
        self._spin_size_h.setValue(self._size_h_deg)
        self._spin_size_h.setKeyboardTracking(False)
        self._spin_size_h.setToolTip("Angular height in degrees (rectangle only).")
        self._spin_size_h.valueChanged.connect(self._on_size_h_changed)

        self._spin_rot = MathDoubleSpinBox(self)
        self._spin_rot.setRange(0.0, 360.0)
        self._spin_rot.setDecimals(1)
        self._spin_rot.setSingleStep(5.0)
        self._spin_rot.setSuffix(" °")
        self._spin_rot.setWrapping(True)
        self._spin_rot.setValue(self._rotation_deg)
        self._spin_rot.setKeyboardTracking(False)
        self._spin_rot.setToolTip(
            "Orientation of the shape in degrees (rectangle / square / "
            "polygon). A circle is rotation-invariant."
        )
        self._spin_rot.valueChanged.connect(self._on_rotation_changed)

        self._spin_sides = MathSpinBox(self)
        self._spin_sides.setRange(SIDES_MIN, SIDES_MAX)
        self._spin_sides.setValue(self._n_sides)
        self._spin_sides.setKeyboardTracking(False)
        self._spin_sides.setToolTip(
            "Number of sides of the regular polygon (3 = triangle, "
            "6 = hexagon), mirroring an aperture blade count."
        )
        self._spin_sides.valueChanged.connect(self._on_sides_changed)

        self._spin_samples = MathSpinBox(self)
        self._spin_samples.setRange(SAMPLES_MIN, SAMPLES_MAX)
        self._spin_samples.setValue(self._samples)
        self._spin_samples.setKeyboardTracking(False)
        self._spin_samples.setToolTip(
            "Points sampled on the shape and averaged. More → smoother "
            "extended-source look, longer render (progressive)."
        )
        self._spin_samples.valueChanged.connect(self._on_samples_changed)

        grid = QGridLayout()
        grid.setContentsMargins(6, 0, 6, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.addWidget(QLabel("Shape", self), 0, 0)
        grid.addWidget(self._shape_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("W", self), 1, 0)
        grid.addWidget(self._spin_size_w, 1, 1)
        grid.addWidget(QLabel("H", self), 1, 2)
        grid.addWidget(self._spin_size_h, 1, 3)
        grid.addWidget(QLabel("Rot", self), 2, 0)
        grid.addWidget(self._spin_rot, 2, 1)
        grid.addWidget(QLabel("Sides", self), 2, 2)
        grid.addWidget(self._spin_sides, 2, 3)
        grid.addWidget(QLabel("Samples", self), 3, 0)
        grid.addWidget(self._spin_samples, 3, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _build_matte_grid(self) -> QGridLayout:
        """Compact grid of the front-of-lens matte-box controls.

        A rectangular flag ahead of the front element that clips entrance rays
        (and the starburst / diffraction pupils), cutting flare from bright
        sources just off the frame — the inline equivalent of stopping a matte
        box's blades in. Symmetric blades: one Width and one Height half-extent
        drive the renderer's left=right / top=bottom. Scrubbers are attached
        centrally in ``__init__``.
        """
        self._matte_check = QCheckBox("Matte box", self)
        self._matte_check.setChecked(self._matte.enabled)
        self._matte_check.setToolTip(
            "Front-of-lens rectangular flag that clips rays entering the lens, "
            "cutting flare from bright sources just outside the frame — like "
            "closing a matte box's blades in. Off = no occluder (no-op)."
        )
        self._matte_check.toggled.connect(self._on_matte_enabled_changed)

        self._spin_matte_w = MathDoubleSpinBox(self)
        self._spin_matte_w.setRange(MATTE_HALF_MM_MIN, MATTE_HALF_MM_MAX)
        self._spin_matte_w.setDecimals(1)
        self._spin_matte_w.setSingleStep(1.0)
        self._spin_matte_w.setSuffix(" mm")
        self._spin_matte_w.setValue(self._matte.half_w_mm)
        self._spin_matte_w.setKeyboardTracking(False)
        self._spin_matte_w.setToolTip(
            "Horizontal half-aperture of the flag from the axis (mm): the left "
            "and right blades. Smaller = the flag closes in from the sides."
        )
        self._spin_matte_w.valueChanged.connect(self._on_matte_w_changed)

        self._spin_matte_h = MathDoubleSpinBox(self)
        self._spin_matte_h.setRange(MATTE_HALF_MM_MIN, MATTE_HALF_MM_MAX)
        self._spin_matte_h.setDecimals(1)
        self._spin_matte_h.setSingleStep(1.0)
        self._spin_matte_h.setSuffix(" mm")
        self._spin_matte_h.setValue(self._matte.half_h_mm)
        self._spin_matte_h.setKeyboardTracking(False)
        self._spin_matte_h.setToolTip(
            "Vertical half-aperture of the flag from the axis (mm): the top and "
            "bottom blades. Smaller = the flag closes in from top and bottom."
        )
        self._spin_matte_h.valueChanged.connect(self._on_matte_h_changed)

        self._spin_matte_z = MathDoubleSpinBox(self)
        self._spin_matte_z.setRange(MATTE_Z_MM_MIN, MATTE_Z_MM_MAX)
        self._spin_matte_z.setDecimals(1)
        self._spin_matte_z.setSingleStep(5.0)
        self._spin_matte_z.setSuffix(" mm")
        self._spin_matte_z.setValue(self._matte.z_front_mm)
        self._spin_matte_z.setKeyboardTracking(False)
        self._spin_matte_z.setToolTip(
            "Distance of the flag plane ahead of the front element (mm). "
            "Farther out = the same blade opening cuts a wider field angle."
        )
        self._spin_matte_z.valueChanged.connect(self._on_matte_z_changed)

        grid = QGridLayout()
        grid.setContentsMargins(6, 0, 6, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.addWidget(self._matte_check, 0, 0, 1, 4)
        grid.addWidget(QLabel("W", self), 1, 0)
        grid.addWidget(self._spin_matte_w, 1, 1)
        grid.addWidget(QLabel("H", self), 1, 2)
        grid.addWidget(self._spin_matte_h, 1, 3)
        grid.addWidget(QLabel("Dist", self), 2, 0)
        grid.addWidget(self._spin_matte_z, 2, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _build_render_row(self) -> QHBoxLayout:
        """Right-aligned 'Render ▾' drop-down: one entry per motion pattern
        (jumps straight to the export dialog with that pattern preselected)
        plus an 'Animation…' entry that opens the dialog on the last choice.

        The QToolButton + InstantPopup idiom mirrors the optical-editor Add
        button; the menu is stored on ``self`` so it isn't garbage-collected.
        """
        btn = QToolButton(self)
        btn.setText("Render")
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setPopupMode(QToolButton.InstantPopup)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(
            "Render an animation of the flare following a motion pattern and "
            "export it as a GIF, MOV, or JPEG / EXR sequence."
        )
        self._render_menu = QMenu(btn)
        for pat in motion_patterns.PATTERNS:
            label = pat.name + ("  (loop)" if pat.loop else "")
            act = QAction(label, self._render_menu)
            act.triggered.connect(
                lambda _checked=False, name=pat.name: self.export_animation(name)
            )
            self._render_menu.addAction(act)
        self._render_menu.addSeparator()
        act_dialog = QAction("Animation…", self._render_menu)
        act_dialog.triggered.connect(
            lambda _checked=False: self.export_animation(None)
        )
        self._render_menu.addAction(act_dialog)
        btn.setMenu(self._render_menu)
        self._render_button = btn

        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 2)
        row.addStretch(1)
        row.addWidget(btn)
        return row

    # ------------------------------------------------------------------
    # Shape state
    # ------------------------------------------------------------------

    def _on_shape_changed(self, index: int) -> None:
        key = self._shape_combo.itemData(index)
        if key == self._shape:
            return
        self._shape = key
        self._sync_shape_widget_enabled()
        self._update_overlay()
        if self._auto_render:
            self.request_render()

    def _on_size_w_changed(self, val: float) -> None:
        self._size_w_deg = float(val)
        self._update_overlay()
        if self._auto_render:
            self.request_render()

    def _on_size_h_changed(self, val: float) -> None:
        self._size_h_deg = float(val)
        self._update_overlay()
        if self._auto_render:
            self.request_render()

    def _on_samples_changed(self, val: int) -> None:
        self._samples = int(val)
        if self._auto_render:
            self.request_render()

    def _on_rotation_changed(self, val: float) -> None:
        self._rotation_deg = float(val)
        self._update_overlay()
        if self._auto_render and self._shape in _ROTATABLE:
            self.request_render()

    def _on_sides_changed(self, val: int) -> None:
        self._n_sides = int(val)
        self._update_overlay()
        if self._auto_render and self._shape == SHAPE_POLYGON:
            self.request_render()

    def _sync_shape_widget_enabled(self) -> None:
        is_point = self._shape == SHAPE_POINT
        self._spin_size_w.setEnabled(not is_point)
        self._spin_size_h.setEnabled(self._shape == SHAPE_RECT)
        self._spin_rot.setEnabled(self._shape in _ROTATABLE)
        self._spin_sides.setEnabled(self._shape == SHAPE_POLYGON)
        self._spin_samples.setEnabled(not is_point)

    # ------------------------------------------------------------------
    # Matte-box state
    # ------------------------------------------------------------------

    def _on_matte_enabled_changed(self, checked: bool) -> None:
        self._matte = replace(self._matte, enabled=bool(checked))
        self._sync_matte_widget_enabled()
        if self._auto_render:
            self.request_render()

    def _on_matte_w_changed(self, val: float) -> None:
        self._matte = replace(self._matte, half_w_mm=float(val))
        if self._auto_render and self._matte.enabled:
            self.request_render()

    def _on_matte_h_changed(self, val: float) -> None:
        self._matte = replace(self._matte, half_h_mm=float(val))
        if self._auto_render and self._matte.enabled:
            self.request_render()

    def _on_matte_z_changed(self, val: float) -> None:
        self._matte = replace(self._matte, z_front_mm=float(val))
        if self._auto_render and self._matte.enabled:
            self.request_render()

    def _sync_matte_widget_enabled(self) -> None:
        """The blade / distance spinboxes only apply when the matte box is on."""
        on = self._matte.enabled
        self._spin_matte_w.setEnabled(on)
        self._spin_matte_h.setEnabled(on)
        self._spin_matte_z.setEnabled(on)

    def _shape_label(self) -> str:
        if self._shape == SHAPE_POINT:
            return "point"
        if self._shape == SHAPE_RECT:
            base = f"rect {self._size_w_deg:.2f}×{self._size_h_deg:.2f}°"
        elif self._shape == SHAPE_SQUARE:
            base = f"square {self._size_w_deg:.2f}°"
        elif self._shape == SHAPE_POLYGON:
            base = f"{self._n_sides}-gon {self._size_w_deg:.2f}°"
        else:
            return f"circle {self._size_w_deg:.2f}°"
        if self._rotation_deg:
            base += f" @{self._rotation_deg:.0f}°"
        return base

    # ------------------------------------------------------------------
    # Shape overlay
    # ------------------------------------------------------------------

    def _shape_outline_offsets(self) -> Optional[list]:
        """Boundary vertices of the current shape as angular offsets (rad).

        Returns ``None`` for a point.  Circle is approximated by a 48-gon;
        rect/square/polygon corners are rotation-applied.
        """
        if self._shape == SHAPE_POINT:
            return None
        half_w = math.radians(max(SIZE_DEG_MIN, self._size_w_deg)) * 0.5
        rot = math.radians(self._rotation_deg)
        if self._shape == SHAPE_CIRCLE:
            steps = 48
            return [
                (half_w * math.cos(2.0 * math.pi * i / steps),
                 half_w * math.sin(2.0 * math.pi * i / steps))
                for i in range(steps)
            ]
        if self._shape == SHAPE_POLYGON:
            return [(float(x), float(y))
                    for x, y in source_sampling.polygon_vertices(half_w, self._n_sides, rot)]
        half_h = (math.radians(max(SIZE_DEG_MIN, self._size_h_deg)) * 0.5
                  if self._shape == SHAPE_RECT else half_w)
        corners = [(-half_w, -half_h), (half_w, -half_h),
                   (half_w, half_h), (-half_w, half_h)]
        c, s = math.cos(rot), math.sin(rot)
        return [(x * c - y * s, x * s + y * c) for x, y in corners]

    def _update_overlay(self) -> None:
        """Recompute the shape outline in image-fractional coordinates.

        Each boundary vertex is mapped through the solved angle→screen
        transform captured at the last dispatch, so rotated / curved shapes
        draw faithfully and land where the render puts them; hidden until a
        first render has run or for a point shape.
        """
        offsets = self._shape_outline_offsets()
        if offsets is None or self._overlay_map is None:
            self._canvas.set_shape_outline(None)
            return
        ax0, ay0, sx0, sy0, jac, half_w, half_h = self._overlay_map
        if half_w <= 0.0 or half_h <= 0.0:
            self._canvas.set_shape_outline(None)
            return
        jxx, jxy, jyx, jyy = jac
        # d(screen fraction)/d(angle): the Jacobian is in mm per radian, and a
        # millimetre is 1/(2·half) of the frame.
        pts = [
            (sx0 + 0.5 * (jxx * dx + jxy * dy) / half_w,
             sy0 + 0.5 * (jyx * dx + jyy * dy) / half_h)
            for dx, dy in offsets
        ]
        self._canvas.set_shape_outline(pts)

    # ------------------------------------------------------------------
    # Project signal handlers
    # ------------------------------------------------------------------

    def _on_system_replaced(self, _system) -> None:
        self._epoch += 1
        self._exposure_stops = 0.0
        # A different lens can sit many stops away from the old one, so meter
        # the first frame it produces rather than leaving the user at 0 stops.
        self._auto_expose_pending = True
        self._latest_hwc = None
        self._overlay_map = None
        self._update_overlay()
        self._vignette.invalidate()
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        self._refresh_status_for_lens()
        if self._lens_eligible():
            self.request_render()
        else:
            self._canvas.clear_image(self._placeholder_for_lens())

    def _on_system_modified(self) -> None:
        self._vignette.invalidate()
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _on_system_setup_changed(self) -> None:
        self._epoch += 1
        self._latest_hwc = None
        self._vignette.invalidate()
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _should_auto_render(self) -> bool:
        return self._auto_render and self._app_settings.auto_update_enabled()

    def _on_global_auto_update_changed(self, enabled: bool) -> None:
        if enabled and self._dirty_pending and self._auto_render:
            self.request_render()

    # ------------------------------------------------------------------
    # Source-position bookkeeping
    # ------------------------------------------------------------------

    def _on_canvas_drag(self, sx: float, sy: float) -> None:
        self._set_source(sx, sy)

    def _on_slider_x(self, raw: int) -> None:
        if self._suppress_widget_signals:
            return
        self._set_source(raw / SLIDER_SCALE, self._sy)

    def _on_slider_y(self, raw: int) -> None:
        if self._suppress_widget_signals:
            return
        self._set_source(self._sx, raw / SLIDER_SCALE)

    def _on_spin_x(self, val: float) -> None:
        if self._suppress_widget_signals:
            return
        self._set_source(float(val), self._sy)

    def _on_spin_y(self, val: float) -> None:
        if self._suppress_widget_signals:
            return
        self._set_source(self._sx, float(val))

    def _set_source(self, sx: float, sy: float) -> None:
        if sx == self._sx and sy == self._sy:
            return
        self._sx = float(sx)
        self._sy = float(sy)
        self._sync_source_widgets()
        self._canvas.set_source(self._sx, self._sy)
        self._update_overlay()
        if self._auto_render:
            self.request_render()

    def _sync_source_widgets(self) -> None:
        self._suppress_widget_signals = True
        try:
            self._slider_x.setValue(int(round(self._sx * SLIDER_SCALE)))
            self._slider_y.setValue(int(round(self._sy * SLIDER_SCALE)))
            self._spin_x.setValue(self._sx)
            self._spin_y.setValue(self._sy)
        finally:
            self._suppress_widget_signals = False

    # ------------------------------------------------------------------
    # Render dispatch
    # ------------------------------------------------------------------

    def _lens_eligible(self) -> bool:
        try:
            return self._project.system.num_surfaces() >= MIN_SURFACES
        except Exception:
            return False

    def _placeholder_for_lens(self) -> str:
        try:
            n = self._project.system.num_surfaces()
        except Exception:
            n = 0
        if n < MIN_SURFACES:
            return f"Add at least {MIN_SURFACES} surfaces to render"
        return "Load a lens to render"

    def _refresh_status_for_lens(self) -> None:
        if self._lens_eligible():
            self._status.setText("Rendering…")
        else:
            self._status.setText(self._placeholder_for_lens())

    def _resolve_render_dims(
        self, settings: Optional[RenderSettings] = None
    ) -> Tuple[int, int, float, float]:
        settings = settings if settings is not None else self._settings
        sensor = self._project.system_setup.sensor
        w_mm = float(sensor.width_mm) if sensor.width_mm > 0 else 24.0
        h_mm = float(sensor.height_mm) if sensor.height_mm > 0 else 16.0
        width_px = int(settings.width_px)
        height_px = max(8, int(round(width_px * (h_mm / w_mm))))
        return width_px, height_px, w_mm * 0.5, h_mm * 0.5

    def _final_status_text(self) -> str:
        w, h, _, _ = self._resolve_render_dims()
        n = 1 if self._shape == SHAPE_POINT else self._samples
        matte = ""
        if self._matte.enabled:
            matte = (f"  matte {self._matte.half_w_mm:.0f}×{self._matte.half_h_mm:.0f}"
                     f"@{self._matte.z_front_mm:.0f}mm")
        return (
            f"src ({self._sx:.3f}, {self._sy:.3f})  {self._shape_label()}  "
            f"n={n}  {self._exposure_stops:+.2f} st  {w}×{h}{matte}"
        )

    def _request(self) -> None:
        """Mark a render pending. State (source, shape, samples) is
        snapshotted at launch time on the main thread, so a burst of
        requests coalesces to one render of the newest state. Bumping
        the serial tells any in-flight progressive pass to abort after
        its current chunk."""
        was_idle = False
        with self._lock:
            was_idle = (not self._busy) and (not self._pending)
            self._pending = True
            self._serial += 1
        if was_idle and self._status.text() != "Rendering…":
            self._status.setText("Rendering…")
        self._maybe_launch()

    def _maybe_launch(self) -> None:
        if self._exporting:
            # The poll timer keeps ticking under the modal export dialog and
            # this is its last call; the guard is what stops the live worker
            # relaunching and fighting the batch exporter for the GPU.
            return
        if not self._is_active or not self._auto_render:
            return
        if not self._lens_eligible():
            return

        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True
            serial = self._serial

        width, height, half_w, half_h = self._resolve_render_dims()
        lens = self._project.system
        # Freeze calibration on the dispatch thread — one consistent
        # (lens, calib) snapshot for every chunk of this pass. Without it the
        # worker re-derives the calibration internally, and a surface mutated
        # on the main thread between those two reads leaves cal and lens
        # disagreeing, which shows up as wrong ghost positions.
        calib = None
        try:
            lens._check_invalidate()
            calib = lens.calibration()
        except Exception:
            _log.exception("SourceFlarePanelBody: calibration failed")
        try:
            ghost_filter = surface_actions.compute_ghost_filter(self._project)
        except Exception:
            _log.exception("SourceFlarePanelBody: compute_ghost_filter failed")
            ghost_filter = None

        # Refresh the angle→screen mapping snapshot for the overlay.
        #
        # Taken from the renderer's own solve rather than from a closed form
        # written out here.  This used to be
        #     screen_frac = 0.5 + 0.5·tan(angle)/(crop_scale·tan(max_half)),
        # which assumes a single focal length for the whole frame; on a
        # distorting or anamorphic lens that is several percent out, so the
        # outline drew somewhere the render did not put the source.
        #
        # The snapshot is (base angle, base screen position, Jacobian).  A solve
        # per outline vertex is not affordable — the outline is 32-64 vertices
        # refreshed on every scrub — but the local Jacobian is exact at the
        # source and first-order around it, over an offset of at most a degree
        # or so.  It is the same transform render_source_flare uses to place its
        # own per-sample splats, so the outline and the render cannot disagree.
        if calib is not None:
            try:
                sm = ghostlight._ghostlight._solve_source_map(
                    lens, calib, self._sx, self._sy, half_w, half_h)
                self._overlay_map = (
                    float(sm["angle_x"]), float(sm["angle_y"]),
                    float(sm["screen_x"]), float(sm["screen_y"]),
                    tuple(float(v) for v in sm["jacobian"]),
                    half_w, half_h,
                )
                self._update_overlay()
            except Exception:
                _log.exception("SourceFlarePanelBody: overlay map failed")

        offsets = build_shape_offsets(
            self._shape, self._size_w_deg, self._size_h_deg, self._samples,
            self._rotation_deg, self._n_sides,
        )
        chunks = source_sampling.chunk_offsets(offsets, CHUNK_SIZE)
        settings = self._settings
        matte = self._matte
        epoch = self._epoch
        # Snapshot exposure + view transform on the dispatch (GUI) thread.
        stops = self._exposure_stops
        try:
            view_spec = vt.spec_from_settings(self._app_settings)
        except vt.ViewTransformError:
            view_spec = None
        sx, sy = self._sx, self._sy
        threading.Thread(
            target=self._worker,
            args=(
                lens, calib, chunks, sx, sy, width, height, half_w, half_h,
                settings, matte, stops, view_spec, epoch, serial, ghost_filter,
            ),
            daemon=True,
        ).start()

    def _worker(
        self,
        lens: ghostlight.OpticalSystem,
        calib,
        chunks: List[np.ndarray],
        sx: float,
        sy: float,
        width: int,
        height: int,
        half_w: float,
        half_h: float,
        settings: RenderSettings,
        matte: "_MatteBoxState",
        stops: float,
        view_spec: Optional["vt.ViewTransformSpec"],
        epoch: int,
        serial: int,
        ghost_filter: Optional["ghostlight.GhostFilter"] = None,
    ) -> None:
        """Producer thread: renders the sample chunks in order, pushing
        a progressively refined frame after each one.

        The running weighted mean (accumulated sum ÷ weight rendered so far)
        keeps brightness constant while samples accumulate, so the preview
        doesn't pump — and because the viewer exposure is a fixed number of
        stops (not auto-metered per chunk), re-displaying that weighted mean is
        exact at any point. Between chunks the pass aborts if a newer request
        (serial) or lens state (epoch) exists — wasted GPU work is bounded by
        one chunk.
        """
        try:
            acc: Optional[np.ndarray] = None
            acc_flare: Optional[np.ndarray] = None
            weight_done = 0.0
            n_total = sum(int(c.shape[0]) for c in chunks)
            n_done = 0
            for i, chunk in enumerate(chunks):
                if serial != self._serial or epoch != self._epoch:
                    return  # stale — a newer pass is pending
                hwc, flare = self._do_render_chunk(
                    lens, calib, chunk, sx, sy, width, height, half_w, half_h,
                    settings, matte, ghost_filter=ghost_filter,
                )
                acc = hwc if acc is None else acc + hwc
                acc_flare = flare if acc_flare is None else acc_flare + flare
                weight_done += float(chunk[:, 2].sum())
                n_done += int(chunk.shape[0])
                norm = acc / weight_done if weight_done > 0.0 else acc
                norm_flare = (acc_flare / weight_done
                              if weight_done > 0.0 else acc_flare)
                qimg = None
                if view_spec is not None:
                    try:
                        display = vt.apply_view(norm, stops, view_spec)
                        qimg = vt.to_qimage(display)
                    except Exception:
                        _log.exception(
                            "SourceFlarePanelBody: view transform failed"
                        )
                is_final = i == len(chunks) - 1
                self._results.put(
                    (epoch, qimg, norm, norm_flare, n_done, n_total, is_final)
                )
        except Exception:
            _log.exception("SourceFlarePanelBody: render failed")
            self._results.put((epoch, None, None, None, 0, 0, True))
        finally:
            with self._lock:
                self._busy = False

    @staticmethod
    def _do_render_chunk(
        lens: ghostlight.OpticalSystem,
        calib,
        offsets: np.ndarray,
        sx: float,
        sy: float,
        width: int,
        height: int,
        half_w: float,
        half_h: float,
        settings: RenderSettings,
        matte: "_MatteBoxState",
        *,
        ghost_filter: Optional["ghostlight.GhostFilter"] = None,
        with_layers: bool = False,
    ) -> Tuple:
        """Render one sample chunk. Returns ``(hwc, flare)``; when
        ``with_layers`` is set also returns a third value — a dict of the raw
        per-pass components ``{"ghost", "starburst"?, "veil"?}`` (absent passes
        omitted) — for EXR AOV layer export. The two-value path is unchanged
        and byte-identical to before this argument existed."""
        cfg = build_flare_config(
            sx, sy, half_w, half_h, settings, matte, ghost_filter=ghost_filter,
        )
        if calib is not None:
            out = lens.render_source_flare(offsets, width, height, cfg, calib=calib)
        else:
            out = lens.render_source_flare(offsets, width, height, cfg)
        # Ghost is the raw geometric layer. The veil (a broad physical glow) and
        # the gate (a border streak, well under 1% of the source) both join the
        # METERED flare layer, so Auto-Expose accounts for them; the starburst
        # core is far too bright to meter against, so it stays out of `flare` and
        # only enters the composite. (Each *_to_hwc returns None when off.)
        ghost = ghostlight._arrays.ghost_to_hwc(out)
        veil = ghostlight._arrays.veil_to_hwc(out)
        gate = ghostlight._arrays.gate_to_hwc(out)
        flare = ghost if veil is None else ghost + veil
        if gate is not None:
            flare = flare + gate
        starburst = ghostlight._arrays.starburst_to_hwc(out)
        hwc = flare if starburst is None else flare + starburst
        if with_layers:
            # Raw per-pass components for EXR AOV layers (absent passes omitted).
            layers = {"ghost": ghost}
            if starburst is not None:
                layers["starburst"] = starburst
            if veil is not None:
                layers["veil"] = veil
            if gate is not None:
                layers["gate"] = gate
            return hwc, flare, layers
        return hwc, flare

    def _poll_results(self) -> None:
        """Drain producer results, paint the newest, relaunch if pending.

        The queue is FIFO, so the last drained valid frame is the
        newest; partial frames update the canvas and progress readout,
        the final frame also refreshes the status line.
        """
        latest_qimg: Optional[QImage] = None
        latest_hwc: Optional[np.ndarray] = None
        latest_flare: Optional[np.ndarray] = None
        latest_progress: Optional[Tuple[int, int]] = None
        latest_final = False
        had_error = False
        while not self._results.empty():
            epoch, qimg, hwc, flare, n_done, n_total, is_final = self._results.get()
            if epoch != self._epoch:
                continue
            if hwc is None:
                had_error = True
                continue
            latest_qimg = qimg  # None if the worker's view transform failed
            latest_hwc = hwc
            latest_flare = flare
            latest_progress = (n_done, n_total)
            latest_final = is_final

        if latest_hwc is not None:
            self._latest_hwc = latest_hwc
            self._latest_flare_hwc = latest_flare
            if self._auto_expose_pending and latest_final:
                # A newly loaded lens meters itself once its first render
                # completes. Deliberately on the FINAL frame only: the worker
                # bakes the dispatch-time stops into each chunk's QImage, so
                # metering mid-pass would be undone by the next chunk's paint.
                self._auto_expose_pending = False
                self.auto_expose()  # sets the stops and repaints
            elif latest_qimg is not None:
                self._canvas.set_image(latest_qimg)
            else:
                self._redisplay()

        with self._lock:
            still_busy = self._busy or self._pending
        if still_busy:
            if latest_progress is not None and not latest_final:
                self._status.setText(
                    f"Rendering… {latest_progress[0]}/{latest_progress[1]}"
                )
            elif not self._status.text().startswith("Rendering"):
                self._status.setText("Rendering…")
        elif had_error and latest_hwc is None:
            self._status.setText("Render failed (see log)")
        elif latest_qimg is not None:
            self._status.setText(self._final_status_text())

        self._maybe_launch()

    # ------------------------------------------------------------------
    # Animation export (Render ▾ dropdown + View menu)
    # ------------------------------------------------------------------

    def open_export_dialog(self) -> None:
        """View-menu entry point — open the export dialog on the last choice."""
        self.export_animation(None)

    def export_animation(self, pattern_name: Optional[str]) -> None:
        """Open the modal export dialog (``pattern_name`` preselects a pattern,
        or ``None`` keeps the last choice) and, on accept, kick off the export.

        Refuses while a lens isn't eligible or an export is already running."""
        if self._exporting:
            return
        if not self._lens_eligible():
            self._status.setText(self._placeholder_for_lens())
            return
        preselect = pattern_name or self._export_options.pattern_name or None
        # The factor is only kept live while the view toggle is on, so refresh
        # it here — the dialog labels itself with it either way.
        self._refresh_squeeze_factor()
        dlg = ExportAnimationDialog(
            self._app_settings,
            # Width comes from the exporter's own default, not the panel's live
            # render width (which is set for interactive speed, not delivery).
            preselect_pattern=preselect,
            squeeze_factor=self._squeeze_factor,
            default_desqueeze=self._desqueeze,
            dims_for_width=self._export_dims_for_width,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._export_options = dlg.result_options()
        self._begin_export()

    def _export_dims_for_width(self, width_px: int) -> Tuple[int, int]:
        """Rendered ``(width, height)`` an export width would produce — the
        dialog's "Output" label reads it so the size it quotes comes from the
        same resolver the export itself uses."""
        settings = replace(self._settings, width_px=int(width_px)).clamp()
        w, h, _hw, _hh = self._resolve_render_dims(settings)
        return int(w), int(h)

    def _begin_export(self) -> None:
        """Snapshot the render inputs on the GUI thread (mirrors the live
        dispatch in _maybe_launch), validate the target, suspend the live
        renderer, and wait for any in-flight live pass to drain before
        launching the batch worker."""
        opts = self._export_options
        pat = motion_patterns.get_pattern(opts.pattern_name)
        if pat is None:
            self._status.setText("Unknown motion pattern")
            return

        # Width override lives on a copy of the panel settings so panel state is
        # untouched; height keeps the sensor aspect via _resolve_render_dims.
        settings = self._settings
        if opts.width_px and int(opts.width_px) != int(settings.width_px):
            settings = replace(settings, width_px=int(opts.width_px)).clamp()
        width, height, half_w, half_h = self._resolve_render_dims(settings)

        spec = writers.WRITER_SPECS.get(opts.writer_key)
        if spec is None:
            self._status.setText("Unknown export format")
            return
        reason = writers.check_writer_available(opts.writer_key)
        if reason is not None:
            QMessageBox.warning(self, "Render Animation", reason)
            return

        lens = self._project.system
        calib = None
        try:
            lens._check_invalidate()
            calib = lens.calibration()
        except Exception:
            _log.exception("SourceFlarePanelBody: export calibration failed")
        try:
            ghost_filter = surface_actions.compute_ghost_filter(self._project)
        except Exception:
            _log.exception("SourceFlarePanelBody: export compute_ghost_filter failed")
            ghost_filter = None

        offsets = build_shape_offsets(
            self._shape, self._size_w_deg, self._size_h_deg, self._samples,
            self._rotation_deg, self._n_sides,
        )
        chunks = tuple(source_sampling.chunk_offsets(offsets, CHUNK_SIZE))
        total_weight = float(offsets[:, 2].sum())

        stops = self._exposure_stops
        try:
            view_spec = vt.spec_from_settings(self._app_settings)
        except vt.ViewTransformError:
            view_spec = None
        if spec.needs_display and view_spec is None:
            QMessageBox.warning(
                self, "Render Animation",
                "The display (view) transform is unavailable, so GIF / MOV / "
                "JPEG can't be written. Export an EXR sequence instead.",
            )
            return

        # De-squeeze is baked into the written frames when asked for. The motion
        # pattern's aspect stays the RENDERED aspect: the source position is a
        # sensor-space coordinate, and stretching the frame afterwards doesn't
        # move it.
        if opts.desqueeze:
            self._refresh_squeeze_factor()
        squeeze = self._squeeze_factor if opts.desqueeze else 1.0

        aspect = (float(width) / float(height)) if height else 1.0
        ctx = motion_patterns.MotionContext(
            start_sx=self._sx, start_sy=self._sy, aspect=aspect,
        )
        job = export_worker.AnimationJob(
            lens=lens, calib=calib, ghost_filter=ghost_filter, chunks=chunks,
            total_weight=total_weight, width=width, height=height,
            half_w=half_w, half_h=half_h, settings=settings, matte=self._matte,
            stops=stops, view_spec=view_spec, pattern=pat, ctx=ctx,
            n_frames=int(opts.n_frames), fps=float(opts.fps),
            writer_key=opts.writer_key, exr_layers=bool(opts.exr_layers),
            out_path=opts.out_path, squeeze=float(squeeze),
        )

        # Suspend the live renderer and abort any in-flight progressive pass.
        self._exporting = True
        with self._lock:
            self._pending = False
            self._serial += 1
        self._export_idle_waits = 0
        self._status.setText("Preparing export…")
        self._begin_export_when_idle(job)

    def _begin_export_when_idle(self, job: "export_worker.AnimationJob") -> None:
        """Poll (~50 ms) until the live worker has drained, then launch. Bounded
        to ~10 s so a wedged live pass can't hang the export indefinitely."""
        with self._lock:
            busy = self._busy
        if busy:
            self._export_idle_waits += 1
            if self._export_idle_waits > 200:  # ~10 s at 50 ms
                self._exporting = False
                self._status.setText("Export aborted (renderer busy)")
                QMessageBox.warning(
                    self, "Render Animation",
                    "The live renderer did not settle in time; export aborted. "
                    "Try again in a moment.",
                )
                return
            QTimer.singleShot(50, lambda: self._begin_export_when_idle(job))
            return
        self._launch_export_thread(job)

    def _apply_export_preview_squeeze(self, job: "export_worker.AnimationJob") -> None:
        """Export previews arrive already de-squeezed, so the canvas must stop
        applying its own stretch or the frames show twice the squeeze. Called
        once the export is committed to launching — :meth:`_finish_export`
        restores the panel's setting on every exit path from there."""
        if job.squeeze != 1.0:
            self._canvas.set_squeeze(1.0)

    def _launch_export_thread(self, job: "export_worker.AnimationJob") -> None:
        self._apply_export_preview_squeeze(job)
        self._export_cancel = threading.Event()
        self._export_thread = QThread(self)
        self._export_worker = export_worker.AnimationWorker(
            job, self._do_render_chunk, self._export_cancel,
        )
        self._export_worker.moveToThread(self._export_thread)
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.frameReady.connect(self._on_export_frame)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_thread.started.connect(self._export_worker.run)

        total = int(job.n_frames) * max(1, len(job.chunks))
        dlg = QProgressDialog("Rendering animation…", "Cancel", 0, total, self)
        dlg.setWindowTitle("Render Animation")
        dlg.setWindowModality(Qt.ApplicationModal)
        # Defeat the 4 s auto-show delay so the bar appears immediately, and keep
        # it up at 100% until we tear the thread down.
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.canceled.connect(self._on_export_cancel_clicked)
        dlg.setValue(0)
        self._export_progress = dlg
        dlg.show()

        self._export_thread.start()

    def _on_export_cancel_clicked(self) -> None:
        if self._export_cancel is not None:
            self._export_cancel.set()
        if self._export_progress is not None:
            self._export_progress.setLabelText("Cancelling…")

    def _on_export_progress(self, done: int, total: int, label: str) -> None:
        dlg = self._export_progress
        if dlg is None:
            return
        dlg.setLabelText(label)
        dlg.setMaximum(int(total))
        dlg.setValue(int(done))

    def _on_export_frame(self, qimg) -> None:
        if qimg is not None:
            self._canvas.set_image(qimg)

    def _on_export_finished(self, summary: str) -> None:
        self._finish_export(summary, ok=True)

    def _on_export_failed(self, message: str) -> None:
        if message == export_worker.CANCELLED:
            self._finish_export("Export cancelled", ok=True)
        else:
            self._finish_export(message, ok=False)

    def _finish_export(self, message: str, ok: bool) -> None:
        """Tear down the worker thread + progress dialog, clear export state,
        and restore the live view. Runs on the GUI thread (signal receiver)."""
        if self._export_progress is not None:
            self._export_progress.close()
            self._export_progress = None
        if self._export_thread is not None:
            self._export_thread.quit()
            self._export_thread.wait(2000)
            self._export_thread = None
        self._export_worker = None
        self._export_cancel = None
        self._exporting = False
        # Undo the preview's squeeze override (a no-op when it never fired).
        self._push_squeeze_to_canvas()
        self._status.setText(message)
        if not ok:
            QMessageBox.warning(self, "Render Animation", message)
        # Repaint the live frame the export displaced.
        self.request_render()


# Public module-level name for the panel's chunk renderer (a plain function —
# ``_do_render_chunk`` is a staticmethod). Sibling panels that want a
# source-flare chunk built from exactly the same PointFlareConfig — the ghost
# explorer — call this instead of reaching through the class, so the two panels
# can never drift in how they configure a render.
render_chunk = SourceFlarePanelBody._do_render_chunk
