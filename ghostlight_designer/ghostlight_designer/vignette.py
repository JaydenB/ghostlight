"""Chief-ray-independent vignette overlay for the flare render panels.

Highlights the sensor regions that **no primary ray can reach** — a target
position is red iff, at the field angle that images to it, not a single pupil
sample survives the aperture stop and every lens rim all the way to the sensor.
That is the lens's hard image circle: outside it the frame simply receives no
light, however you aim.

Why not the PSF panel's chief-ray status?  The PSF panel aims each cell's chief
ray *freely* (Newton-refines the field direction to land on the target), so its
survivor-mean fallback finds a grazing ray almost everywhere — "no ray at all"
essentially never fires and the overlay would be blank.  Here we instead probe
the pupil at the cell's **natural** field angle (the tan-linear inverse of its
sensor position, un-clamped past the calibrated field), which is exactly the
forward imaging geometry and yields the true hard cut-off.

Mechanically this is ``render_psf`` in ``CHIEF_CENTROID`` mode: that mode runs
one fixed-angle pupil probe per source and reports ``PSFCellStatus.DARK`` when
zero probe rays transmit — no Newton, so it stays cheap (~40-60 ms for a
48-long grid).  The rendered PSF tiles are discarded; only the per-cell status
is read.

The mask depends solely on lens geometry + sensor extent (not source position,
exposure, ray grid, …), so it is computed once per lens/sensor change, cached,
and recomputed off the GUI thread.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

import ghostlight

from .project import Project

_log = logging.getLogger("ghostlight_designer.vignette")

# Half-alpha red — matches the PSF panel's vignette tint so the same physical
# regions read the same colour across the PSF / point / source panels.
VIGNETTE_RGBA: Tuple[int, int, int, int] = (255, 40, 40, 128)

# Probe grid resolution along the sensor's long axis.  48 gives a crisp
# image-circle boundary that smooth-upscales to a clean soft edge, at ~40-60 ms.
GRID_LONG_CELLS = 48
GRID_MIN_CELLS = 8

# Sensor fallback (mm) when the project has no sensor set — mirrors the flare
# panels' ``_resolve_render_dims`` so the probed extent matches the render.
_FALLBACK_W_MM = 24.0
_FALLBACK_H_MM = 16.0

# A lens needs at least two surfaces for a calibration to exist.
_MIN_SURFACES = 2

# Poll cadence for draining the worker result (only runs while a compute is in
# flight; the timer stops itself when idle).
_POLL_INTERVAL_MS = 60


def sensor_half_extents(project: Project) -> Tuple[float, float]:
    """``(half_w_mm, half_h_mm)`` for the project sensor, with the same
    ``>0 else fallback`` rule the flare panels use so the probed sensor extent
    matches the rendered frame exactly."""
    sensor = project.system_setup.sensor
    w_mm = float(sensor.width_mm) if sensor.width_mm > 0 else _FALLBACK_W_MM
    h_mm = float(sensor.height_mm) if sensor.height_mm > 0 else _FALLBACK_H_MM
    return w_mm * 0.5, h_mm * 0.5


def _choose_grid(half_w: float, half_h: float) -> Tuple[int, int]:
    """Pick ``(nx, ny)`` so cells are ~square in sensor mm, long axis =
    :data:`GRID_LONG_CELLS`."""
    if half_w >= half_h:
        nx = GRID_LONG_CELLS
        ny = max(GRID_MIN_CELLS, int(round(GRID_LONG_CELLS * half_h / max(1e-6, half_w))))
    else:
        ny = GRID_LONG_CELLS
        nx = max(GRID_MIN_CELLS, int(round(GRID_LONG_CELLS * half_w / max(1e-6, half_h))))
    return nx, ny


# Coarse grid the traced map is solved on before being interpolated onto the
# cells, and the correction size that makes solving it worth the time.  Both
# measured (see _build_seeds).
_MAP_GRID = 5
_MAP_CORRECTION_FRAC = 0.005


def _cell_targets(nx: int, ny: int, half_w: float, half_h: float) -> np.ndarray:
    """Cell-centre sensor positions (mm), row-major, ``row 0 = top = -y_mm``,
    ``col 0 = left = -x_mm`` — the flare render's raster convention."""
    out = np.empty((ny * nx, 2), dtype=np.float32)
    k = 0
    for row in range(ny):
        y = 0.0 if ny == 1 else (-half_h + (row + 0.5) * (2.0 * half_h / ny))
        for col in range(nx):
            x = 0.0 if nx == 1 else (-half_w + (col + 0.5) * (2.0 * half_w / nx))
            out[k, 0] = x
            out[k, 1] = y
            k += 1
    return out


def _analytic_seeds(targets: np.ndarray, calib) -> np.ndarray:
    """The closed-form inverse: ``angle = atan(mm / f_eff)``.

    Correct only where the lens obeys ``h = f·tan(theta)``.  Kept because on a
    well-behaved lens it is not merely adequate but BETTER than interpolating a
    coarse solved grid — measured on double_gauss, where its seed error is
    0.007 deg against a 5x5 grid's 0.004 deg but a 3x3 grid's 0.012 deg.
    """
    cal_hw = max(1e-6, float(calib.sensor_half_w))
    cal_hh = max(1e-6, float(calib.sensor_half_h))
    f_w = cal_hw / max(1e-9, math.tan(float(calib.max_half_angle_h)))
    f_h = cal_hh / max(1e-9, math.tan(float(calib.max_half_angle_v)))
    seeds = np.empty_like(targets)
    seeds[:, 0] = np.arctan(targets[:, 0] / f_w)
    seeds[:, 1] = np.arctan(targets[:, 1] / f_h)
    return seeds


def _build_seeds(targets: np.ndarray, system, calib,
                 half_w: float, half_h: float) -> np.ndarray:
    """Per-cell field-angle seeds (rad) for the vignette probe.

    The probe is CHIEF_CENTROID: it samples the pupil AT the seed angle and has
    no aim step to correct it (see run_vignette).  So the seed is the answer,
    and a seed from the closed-form map puts the overlay's boundary wherever
    that map is wrong — up to 4% of the frame on a distorting anamorphic, which
    is about two cells of this grid painted the wrong colour.

    Solving every cell is not affordable: ~2000 cells at ~3 ms each.  Instead
    the traced map is solved on a coarse grid and interpolated, which measured
    26x better than the closed form on the Atlas (0.09 deg against 2.40 deg)
    for ~65 ms on the worker thread.

    That trade is only worth making when there is something to correct.  On a
    lens the closed form already fits, interpolating a coarse grid is the
    LARGER of the two errors, so two solves decide it: if the map's correction
    at the frame corner is under _MAP_CORRECTION_FRAC of the frame, the closed
    form is kept and nothing else is solved.  Not clamped to the calibrated
    field either way — cells past the sensor edge need their true (steeper)
    angle or the probe never reaches the vignetting regime.
    """
    fallback = _analytic_seeds(targets, calib)
    if system is None:
        return fallback

    def solve(x_mm, y_mm):
        return ghostlight._ghostlight._solve_source_map(
            system, calib,
            0.5 + 0.5 * x_mm / half_w, 0.5 + 0.5 * y_mm / half_h,
            half_w, half_h)

    try:
        # Is the closed form already right? Measured at the corner, where the
        # two maps differ most, and expressed in frame fractions through the
        # solve's own Jacobian rather than as a bare angle — a tenth of a degree
        # is a different amount of screen on a 25 mm lens and a 200 mm one.
        probe = solve(half_w, half_h)
        ana = _analytic_seeds(np.asarray([[half_w, half_h]], np.float32), calib)[0]
        d_ax = float(probe["angle_x"]) - float(ana[0])
        d_ay = float(probe["angle_y"]) - float(ana[1])
        jxx, jxy, jyx, jyy = (float(v) for v in probe["jacobian"])
        frac = max(abs(jxx * d_ax + jxy * d_ay) / (2.0 * half_w),
                   abs(jyx * d_ax + jyy * d_ay) / (2.0 * half_h))
        if frac < _MAP_CORRECTION_FRAC:
            return fallback

        n = _MAP_GRID
        gx = np.linspace(-half_w, half_w, n)
        gy = np.linspace(-half_h, half_h, n)
        node = np.empty((n, n, 2), dtype=np.float64)
        for j in range(n):
            for i in range(n):
                r = solve(float(gx[i]), float(gy[j]))
                node[j, i, 0] = float(r["angle_x"])
                node[j, i, 1] = float(r["angle_y"])
        if not np.isfinite(node).all():
            return fallback

        u = (targets[:, 0] - gx[0]) / (gx[-1] - gx[0]) * (n - 1)
        v = (targets[:, 1] - gy[0]) / (gy[-1] - gy[0]) * (n - 1)
        i0 = np.clip(u.astype(np.int32), 0, n - 2)
        j0 = np.clip(v.astype(np.int32), 0, n - 2)
        fu = (u - i0)[:, None]
        fv = (v - j0)[:, None]
        top = node[j0, i0] * (1.0 - fu) + node[j0, i0 + 1] * fu
        bot = node[j0 + 1, i0] * (1.0 - fu) + node[j0 + 1, i0 + 1] * fu
        return (top * (1.0 - fv) + bot * fv).astype(np.float32)
    except Exception:
        _log.exception("vignette: traced seed grid failed; using the closed form")
        return fallback


def plan_vignette(
    calib, half_w: float, half_h: float
) -> Optional[Tuple[int, int, np.ndarray]]:
    """Build ``(nx, ny, cell_targets_mm)`` for the probe (GUI thread).

    Only the cell geometry, which is arithmetic.  The field angles are solved in
    run_vignette on the worker thread — they cost tens of milliseconds and this
    runs on every geometry scrub.
    """
    if calib is None or not (half_w > 0.0) or not (half_h > 0.0):
        return None
    nx, ny = _choose_grid(half_w, half_h)
    return nx, ny, _cell_targets(nx, ny, half_w, half_h)


def run_vignette(
    system: ghostlight.OpticalSystem, nx: int, ny: int, targets: np.ndarray,
    calib=None, half_w: float = 0.0, half_h: float = 0.0
) -> Optional[np.ndarray]:
    """Probe the pupil at each seed field angle; return an ``(ny, nx)`` bool
    mask where ``True`` means *no primary ray reaches this sensor position*.

    Runs on a worker thread — ``render_psf`` releases the GIL for the CUDA work,
    and the seed solve below is why the seeds are built here rather than in
    plan_vignette.  Returns ``None`` on failure.

    ``targets`` are cell-centre sensor positions in mm.  With ``calib`` and the
    sensor extents given, the field angle for each is taken from the renderer's
    traced source map; without them the closed-form inverse is used, which is
    what the lens-free unit-test stubs exercise.
    """
    try:
        if calib is not None and half_w > 0.0 and half_h > 0.0:
            seeds = _build_seeds(targets, system, calib, half_w, half_h)
        else:
            seeds = np.ascontiguousarray(targets, dtype=np.float32)
        cfg = ghostlight.PSFConfig()
        cfg.grid_nx = int(nx)
        cfg.grid_ny = int(ny)
        # The rendered tiles are thrown away — only the per-cell status is read,
        # so keep every dimension minimal.
        cfg.tile_w = 2
        cfg.tile_h = 2
        cfg.tile_extent_mm = 0.05
        cfg.ray_grid = 4
        cfg.spectral_samples = 1
        cfg.pupil_jitter = 2
        # CHIEF_CENTROID probes the pupil at the seed angle (no Newton re-aim):
        # status == DARK iff zero probe rays transmit at that natural field.
        cfg.center_mode = ghostlight.PSFCenterMode.CHIEF_CENTROID
        out = system.render_psf(seeds, cfg)
        status = np.asarray(out["status"])
        dark = int(ghostlight.PSFCellStatus.DARK)
        return (status == dark).reshape(int(ny), int(nx))
    except Exception:
        _log.exception("run_vignette: render_psf probe failed")
        return None


def mask_to_qimage(mask: Optional[np.ndarray]) -> Optional[QImage]:
    """Pack an ``(ny, nx)`` bool mask into an owned RGBA8888 QImage: solid red
    everywhere, alpha 128 where vignetted and 0 elsewhere.

    Returns ``None`` when nothing is vignetted so the caller can skip the draw.
    RGB is red on every pixel (not just the masked ones) so smooth upscaling
    interpolates only the alpha channel — no dark fringe bleeds into the edge.
    """
    if mask is None or not bool(mask.any()):
        return None
    ny, nx = mask.shape
    rgba = np.empty((ny, nx, 4), dtype=np.uint8)
    rgba[..., 0] = VIGNETTE_RGBA[0]
    rgba[..., 1] = VIGNETTE_RGBA[1]
    rgba[..., 2] = VIGNETTE_RGBA[2]
    rgba[..., 3] = np.where(mask, VIGNETTE_RGBA[3], 0).astype(np.uint8)
    buf = np.ascontiguousarray(rgba)
    img = QImage(buf.data, nx, ny, nx * 4, QImage.Format_RGBA8888)
    # .copy() detaches from the numpy buffer that is about to be freed.
    return img.copy()


class VignetteController(QObject):
    """Owns the vignette overlay's compute + cache + canvas push for one panel.

    The panel drives it with three calls — :meth:`set_enabled` (menu toggle),
    :meth:`set_active` (show/hide), :meth:`invalidate` (lens/setup change) —
    and the controller decides when a fresh probe is actually needed.  It
    mirrors the panels' one-busy/one-pending threading with an epoch guard so
    a lens swap mid-compute drops the stale result.
    """

    # (n_vignetted_cells, n_total_cells) — emitted after each applied result.
    resultReady = Signal(int, int)

    def __init__(
        self,
        project: Project,
        canvas,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._canvas = canvas

        self._enabled = False   # menu toggle
        self._active = False    # panel visible
        self._dirty = True      # geometry changed since last successful probe

        self._lock = threading.Lock()
        self._busy = False
        self._pending = False
        self._epoch = 0
        self._results: "queue.SimpleQueue" = queue.SimpleQueue()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------
    # Public API driven by the panel
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        on = bool(on)
        if on == self._enabled:
            return
        self._enabled = on
        self._canvas.set_vignette_visible(on)
        if on:
            self._maybe_start()

    def set_active(self, on: bool) -> None:
        on = bool(on)
        if on == self._active:
            return
        self._active = on
        if on:
            self._maybe_start()

    def invalidate(self) -> None:
        """Geometry changed — the cached mask is stale.  Bumping the epoch
        discards any in-flight result computed against the old lens."""
        self._dirty = True
        self._epoch += 1
        self._maybe_start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _eligible(self) -> bool:
        try:
            return self._project.system.num_surfaces() >= _MIN_SURFACES
        except Exception:
            return False

    def _maybe_start(self) -> None:
        # Cheapest gates first so a disabled overlay costs nothing during a
        # scrub burst (invalidate() is called on every edit).
        if not (self._enabled and self._active and self._dirty):
            return
        if not self._eligible():
            self._canvas.set_vignette_image(None)
            return
        with self._lock:
            self._pending = True
        self._timer.start()
        self._launch()

    def _launch(self) -> None:
        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True
            epoch = self._epoch

        # Snapshot the calibration on the GUI thread (same rule as the flare
        # panels) so the seeds are built against one consistent lens moment;
        # the worker's render_psf re-derives internally, but an epoch mismatch
        # drops the result if the lens changed underneath.
        system = self._project.system
        half_w, half_h = sensor_half_extents(self._project)
        plan = None
        try:
            system._check_invalidate()
            calib = system.calibration()
            plan = plan_vignette(calib, half_w, half_h)
        except Exception:
            _log.exception("VignetteController: calibration/plan failed")

        if plan is None:
            with self._lock:
                self._busy = False
            self._canvas.set_vignette_image(None)
            return

        nx, ny, targets = plan
        # We are now computing the current geometry — clean unless a later
        # invalidate() dirties it again (which also bumps the epoch).
        self._dirty = False
        threading.Thread(
            target=self._worker,
            args=(system, nx, ny, targets, calib, half_w, half_h, epoch),
            daemon=True,
        ).start()

    def _worker(
        self,
        system: ghostlight.OpticalSystem,
        nx: int,
        ny: int,
        targets: np.ndarray,
        calib,
        half_w: float,
        half_h: float,
        epoch: int,
    ) -> None:
        try:
            mask = run_vignette(system, nx, ny, targets, calib, half_w, half_h)
            img = mask_to_qimage(mask)
            n_dark = int(mask.sum()) if mask is not None else 0
            self._results.put((epoch, img, n_dark, int(nx) * int(ny)))
        except Exception:
            _log.exception("VignetteController: worker failed")
            self._results.put((epoch, None, 0, int(nx) * int(ny)))
        finally:
            with self._lock:
                self._busy = False

    def _poll(self) -> None:
        latest: Optional[Tuple[Optional[QImage], int, int]] = None
        while not self._results.empty():
            epoch, img, n_dark, n_total = self._results.get()
            if epoch != self._epoch:
                continue
            latest = (img, n_dark, n_total)

        if latest is not None:
            img, n_dark, n_total = latest
            # img is None both on failure and on "nothing vignetted"; either
            # way clearing the overlay is correct.
            self._canvas.set_vignette_image(img)
            self.resultReady.emit(n_dark, n_total)

        with self._lock:
            still = self._busy or self._pending
        if still:
            self._launch()
        elif self._results.empty():
            self._timer.stop()
