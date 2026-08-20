"""Body widget for the ``psf`` panel type.

Renders a grid of geometric point-spread functions through the CUDA
PSF kernel.  Mirrors the source-flare panel's threading + visibility-gating
pattern and uses the same edit-settle debounce (PSF renders are cheap
per-frame but the source-angle grid is rebuilt every time a settings
knob moves, so coalescing scrubs is still a win).

Display knobs (per-tile normalisation, log-gain tone-mapping) operate
on the last rendered float buffer without firing a GPU re-render.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

import ghostlight

from ..project import Project
from ..settings import AppSettings
from .. import lens_metrics as lm_mod
from ..viewtransform import to_qimage
from .dialogs import (
    HIGH_PRESET,
    LOW_PRESET,
    PSFRenderSettings,
    PSFRenderSettingsDialog,
    ToneMappingDialog,
)

_log = logging.getLogger("ghostlight_designer.psf_panel")

# A lens needs at least two surfaces to even build a calibration; below
# that we show a placeholder.
MIN_SURFACES = 2

POLL_INTERVAL_MS = 50

# Edit-settle debounce.  Same rationale as the flare panels — coalesce
# fast-scrub edits into one render.  PSF renders are cheaper than a flare
# render but rebuilding the source-angle grid + chief-ray pre-pass per source
# adds up at large grid_n.
DEBOUNCE_MS = 350

# Display tone mapping defaults.  Matches the standalone demo so the
# panel and demo agree visually out of the box.
DEFAULT_LOG_GAIN = 50.0


def _composite_to_display(
    comp_hwc: np.ndarray,
    tile_w: int, tile_h: int,
    grid_nx: int, grid_ny: int,
    per_tile: bool,
    log_gain: float,
) -> np.ndarray:
    """log1p-compress + normalise the PSF composite for display.

    per_tile=True: normalise each tile's peak independently so off-axis
                   tiles (dimmer due to vignetting / cos⁴ falloff) are
                   individually visible at the cost of relative-
                   illumination information across tiles.
    per_tile=False: single global normalisation (centre dominates).

    Clamps negatives (sub-epsilon atomic-float artefacts) to zero before
    the log — otherwise log1p produces NaN.
    """
    pos = np.maximum(comp_hwc, 0.0)
    if per_tile and grid_nx > 0 and grid_ny > 0:
        lum = pos.sum(axis=-1)  # (H, W)
        peaks = np.zeros((grid_ny, grid_nx), dtype=np.float32)
        for gy in range(grid_ny):
            for gx in range(grid_nx):
                tile = lum[gy*tile_h:(gy+1)*tile_h, gx*tile_w:(gx+1)*tile_w]
                peaks[gy, gx] = max(float(tile.max()), 1e-30)
        peak_full = np.repeat(np.repeat(peaks, tile_h, axis=0), tile_w, axis=1)[..., None]
        norm = pos / peak_full
    else:
        peak = float(pos.max())
        if peak <= 0.0:
            return np.zeros_like(comp_hwc)
        norm = pos / peak

    gain = max(1.0, float(log_gain))
    disp = np.log1p(norm * gain) / np.log1p(gain)
    return np.clip(disp, 0.0, 1.0)


def _build_cells(
    grid_nx: int, grid_ny: int,
    calib,
    sensor_half_w_mm: float,
    sensor_half_h_mm: float,
    field_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (seeds, targets) for a sensor-cell PSF grid.

    Cells partition the capture sensor.  ``targets`` are the cell centres in
    sensor mm; ``seeds`` are the tan-linear inverse field angles (the engine's
    canonical anchor inversion), refined by the C++ aim solver so each cell's
    chief ray lands on its target.

    Raster convention (matching the flare panels): **row 0 = top
    of the displayed frame = −y_mm**, col 0 = left = −x_mm.  This replaces the
    old ``+y-up`` tile arrangement — the vertical-mirror fix lives here, in the
    row ordering, with the GPU kernel untouched.

    Targets are NOT capped to the lens image circle: a larger capture sensor
    pushes corner cells past what the lens can image, and the aim solver flags
    those cells (status != OK) so the panel can paint them vignetted.  The seed
    ratio is clamped only so ``atan`` stays finite for unreachable targets.
    """
    if grid_nx <= 0 or grid_ny <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    spread = max(0.0, min(1.0, float(field_fraction)))
    hw = float(sensor_half_w_mm) * spread
    hh = float(sensor_half_h_mm) * spread
    cal_hw = max(1e-6, float(calib.sensor_half_w))
    cal_hh = max(1e-6, float(calib.sensor_half_h))
    tan_h = math.tan(float(calib.max_half_angle_h))
    tan_v = math.tan(float(calib.max_half_angle_v))

    seeds = np.empty((grid_ny * grid_nx, 2), dtype=np.float32)
    targets = np.empty((grid_ny * grid_nx, 2), dtype=np.float32)
    k = 0
    for row in range(grid_ny):
        # row 0 = top of frame = −y_mm (raster order)
        y = 0.0 if grid_ny == 1 else (-hh + (row + 0.5) * (2.0 * hh / grid_ny))
        for col in range(grid_nx):
            x = 0.0 if grid_nx == 1 else (-hw + (col + 0.5) * (2.0 * hw / grid_nx))
            targets[k, 0] = x
            targets[k, 1] = y
            rx = max(-1.0, min(1.0, x / cal_hw))
            ry = max(-1.0, min(1.0, y / cal_hh))
            seeds[k, 0] = math.atan(rx * tan_h)
            seeds[k, 1] = math.atan(ry * tan_v)
            k += 1
    return seeds, targets


class PSFCanvas(QWidget):
    """Composite-PSF canvas, aspect-preserving, with a thin grid overlay
    between tiles so the layout is obvious at any zoom."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._placeholder: str = "Load a lens to render"
        self._grid_nx: int = 0
        self._grid_ny: int = 0
        self._tile_w: int = 0
        self._tile_h: int = 0
        # Per-cell aiming report (FIXED_TARGET mode).  status: (N,) uint8
        # PSFCellStatus (row-major, cell i → tile i % grid_nx, i // grid_nx);
        # pupil_fraction: (N,) float.  None → no overlay (legacy / not ready).
        self._status: Optional[np.ndarray] = None
        self._pupil_fraction: Optional[np.ndarray] = None
        # Horizontal display stretch (anamorphic de-squeeze). Applies to
        # the whole composite — since tiles are uniform, each tile gets
        # stretched by the same factor, which is what shows each PSF the
        # way the de-squeezed projected image would.
        self._squeeze: float = 1.0
        self.setMinimumSize(QSize(256, 256))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, img: Optional[QImage],
                  grid_nx: int, grid_ny: int,
                  tile_w: int, tile_h: int,
                  status: Optional[np.ndarray] = None,
                  pupil_fraction: Optional[np.ndarray] = None) -> None:
        self._image = img
        self._grid_nx = int(grid_nx)
        self._grid_ny = int(grid_ny)
        self._tile_w = int(tile_w)
        self._tile_h = int(tile_h)
        self._status = status
        self._pupil_fraction = pupil_fraction
        self.update()

    def clear_image(self, placeholder: str) -> None:
        self._image = None
        self._placeholder = placeholder
        self.update()

    def set_squeeze(self, squeeze: float) -> None:
        """Set the horizontal stretch factor for the displayed composite.

        ``1.0`` is no stretch. Non-finite or non-positive values fall
        back to 1.0. Grid-overlay lines scale with the rect, so tile
        boundaries stay aligned with the stretched composite.
        """
        s = float(squeeze)
        if not (s > 0.0) or s != s:
            s = 1.0
        if s == self._squeeze:
            return
        self._squeeze = s
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        wid = float(self.width())
        hgt = float(self.height())
        iw = float(self._image.width()) * self._squeeze
        ih = float(self._image.height())
        if iw <= 0.0 or ih <= 0.0:
            return QRectF(self.rect())
        scale = min(wid / iw, hgt / ih)
        out_w = iw * scale
        out_h = ih * scale
        x = (wid - out_w) / 2.0
        y = (hgt - out_h) / 2.0
        return QRectF(x, y, out_w, out_h)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 13))
        if self._image is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        r = self._image_rect()
        p.drawImage(r, self._image)

        # Vignette overlay — cells whose cell-centre primary ray is blocked get
        # a 0.5-alpha red tint (any partial PSF stays visible underneath); a
        # near-vignetted cell is annotated with its pupil transmission.  Drawn
        # under the grid lines so the tile boundaries stay crisp.
        self._paint_vignette_overlay(p, r)

        # Tile grid overlay — thin lines between tiles in the rendered
        # composite (skipped when grid is 1x1 — single high-res mode).
        if self._grid_nx > 1 or self._grid_ny > 1:
            pen = QPen(QColor(64, 64, 64))
            pen.setWidth(0)  # cosmetic; always 1 device pixel
            p.setPen(pen)
            if self._tile_w > 0:
                px_per_tile_x = r.width() / float(self._grid_nx)
                for gx in range(1, self._grid_nx):
                    x = r.x() + gx * px_per_tile_x
                    p.drawLine(int(round(x)), int(r.top()),
                               int(round(x)), int(r.bottom()))
            if self._tile_h > 0:
                px_per_tile_y = r.height() / float(self._grid_ny)
                for gy in range(1, self._grid_ny):
                    y = r.y() + gy * px_per_tile_y
                    p.drawLine(int(r.left()), int(round(y)),
                               int(r.right()), int(round(y)))

    # 0.5-alpha red over the whole cell — matches the user's "half the entire
    # cell with 0.5 red" spec.  Partial PSFs (CHIEF_VIGNETTED) render underneath.
    _VIGNETTE_RGBA = (255, 40, 40, 128)

    def _paint_vignette_overlay(self, p: QPainter, r: QRectF) -> None:
        status = self._status
        if status is None or self._grid_nx <= 0 or self._grid_ny <= 0:
            return
        n_cells = self._grid_nx * self._grid_ny
        if int(status.size) < n_cells:
            return
        ok = int(ghostlight.PSFCellStatus.OK)
        dark = int(ghostlight.PSFCellStatus.DARK)
        frac = self._pupil_fraction
        frac_max = float(frac.max()) if (frac is not None and frac.size and frac.max() > 0) else 0.0
        cell_w = r.width() / float(self._grid_nx)
        cell_h = r.height() / float(self._grid_ny)
        red = QColor(*self._VIGNETTE_RGBA)
        amber = QColor(255, 180, 84)
        ink = QColor(230, 230, 234)
        for i in range(n_cells):
            gx = i % self._grid_nx
            gy = i // self._grid_nx
            cell = QRectF(r.x() + gx * cell_w, r.y() + gy * cell_h, cell_w, cell_h)
            st = int(status[i])
            if st != ok:
                p.fillRect(cell, red)
                if st == dark:
                    p.setPen(ink)
                    p.drawText(cell, Qt.AlignCenter, "vignetted")
            elif (frac is not None and frac_max > 0.0
                  and float(frac[i]) < 0.5 * frac_max):
                # Chief passes but the pupil is clipping — annotate, don't tint.
                p.setPen(amber)
                p.drawText(cell.adjusted(3, 0, -3, -3),
                           Qt.AlignBottom | Qt.AlignHCenter,
                           f"{float(frac[i]) / frac_max * 100:.0f}% pupil")


class PSFPanelBody(QWidget):
    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        # Persistent app prefs — View → Auto-Update Panels gates auto
        # renders. Named ``_app_settings`` to avoid colliding with the
        # existing ``self._settings`` (per-panel render settings).
        self._app_settings = settings

        self._auto_render: bool = True

        # Visibility-gated rendering — same pattern as the other panels.
        self._is_active: bool = False
        self._dirty_pending: bool = False

        # Per-panel render settings (not persisted to AppSettings).
        self._settings: PSFRenderSettings = LOW_PRESET

        # Display state — not part of settings since these don't trigger
        # a GPU re-render, they just re-process the last rendered buffer.
        self._per_tile_norm: bool = False
        self._log_gain: float = DEFAULT_LOG_GAIN
        # Anamorphic de-squeeze toggle. Stretches the entire composite
        # horizontally by the lens's squeeze factor — each tile gets
        # the same stretch, previewing what each PSF would look like in
        # the projected (de-squeezed) image.
        self._desqueeze: bool = False
        self._squeeze_factor: float = 1.0

        # Latest raw float composite — kept so display-only toggles
        # (per-tile norm, log gain) can re-process without re-rendering.
        # Also kept: tile/grid dims, needed by the display function.
        self._latest_comp: Optional[np.ndarray] = None
        self._latest_dims: Optional[Tuple[int, int, int, int]] = None  # (grid_nx, grid_ny, tile_w, tile_h)
        # Per-cell aiming report (FIXED_TARGET mode) — drives the vignette
        # overlay and the status-bar count.  Re-applied on display-only toggles.
        self._latest_status: Optional[np.ndarray] = None
        self._latest_frac: Optional[np.ndarray] = None

        # Modeless tone-mapping dialog (if open).
        self._tone_dialog: Optional[ToneMappingDialog] = None
        # Live handle to the modeless render-settings dialog (if open) so a
        # second menu invocation raises it instead of spawning a duplicate.
        self._settings_dialog: Optional[PSFRenderSettingsDialog] = None

        # Threading state — same one-busy / one-pending model as the flare panels.
        self._lock = threading.Lock()
        self._pending: bool = False
        self._busy: bool = False
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._epoch: int = 0

        self._canvas = PSFCanvas(self)

        self._status = QLabel("", self)
        self._status.setStyleSheet("color: #aaa; padding: 2px 6px;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._canvas, 1)
        outer.addWidget(self._status)

        # Edit-settle debouncer (see the module docstring for the why).
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._on_debounce_timeout)

        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)
        project.systemSetupChanged.connect(self._on_system_setup_changed)
        self._app_settings.autoUpdateChanged.connect(
            self._on_global_auto_update_changed
        )

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start()

        self._refresh_status_for_lens()
        # __init__ runs before showEvent — mark dirty so the first show
        # triggers the initial render.
        self._dirty_pending = True

    # ------------------------------------------------------------------
    # Public API used by the View / Settings menus
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
            self._debounce.stop()
            self._status.setText("Auto-render off")

    def request_render(self) -> None:
        """Debounced render request — see the module docstring for the rationale."""
        if not self._lens_eligible():
            self._canvas.clear_image(self._placeholder_for_lens())
            self._status.setText(self._placeholder_for_lens())
            self._debounce.stop()
            return
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        if self._status.text() != "Edits settling…":
            self._status.setText("Edits settling…")
        self._debounce.start()

    def force_render_now(self) -> None:
        """Bypass the debounce and dispatch a worker on the spot."""
        if not self._lens_eligible():
            self._canvas.clear_image(self._placeholder_for_lens())
            self._status.setText(self._placeholder_for_lens())
            return
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        self._debounce.stop()
        self._dispatch()

    def _on_debounce_timeout(self) -> None:
        if not self._is_active or not self._auto_render:
            return
        if not self._lens_eligible():
            self._canvas.clear_image(self._placeholder_for_lens())
            self._status.setText(self._placeholder_for_lens())
            return
        self._dispatch()

    # ------------------------------------------------------------------
    # Display toggles (no GPU re-render)
    # ------------------------------------------------------------------

    @property
    def per_tile_norm(self) -> bool:
        return self._per_tile_norm

    def set_per_tile_norm(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._per_tile_norm:
            return
        self._per_tile_norm = enabled
        self._reapply_display()

    @property
    def log_gain(self) -> float:
        return self._log_gain

    def set_log_gain(self, value: float) -> None:
        value = max(1.0, float(value))
        if value == self._log_gain:
            return
        self._log_gain = value
        self._reapply_display()

    def _reapply_display(self) -> None:
        """Re-process the last raw buffer with current display settings.
        No GPU work — purely numpy on the cached composite."""
        if self._latest_comp is None or self._latest_dims is None:
            return
        grid_nx, grid_ny, tile_w, tile_h = self._latest_dims
        disp = _composite_to_display(
            self._latest_comp, tile_w, tile_h, grid_nx, grid_ny,
            per_tile=self._per_tile_norm,
            log_gain=self._log_gain,
        )
        self._canvas.set_image(to_qimage(disp), grid_nx, grid_ny, tile_w, tile_h,
                               status=self._latest_status,
                               pupil_fraction=self._latest_frac)
        self._refresh_status_for_render()

    @property
    def desqueeze(self) -> bool:
        return self._desqueeze

    def set_desqueeze(self, enabled: bool) -> None:
        """Toggle anamorphic de-squeeze display.

        Stretches the entire PSF composite horizontally by the lens's
        squeeze factor so each tile previews how its PSF would look in
        the projected image. Underlying render is unchanged.
        """
        enabled = bool(enabled)
        if enabled == self._desqueeze:
            return
        self._desqueeze = enabled
        if enabled:
            self._refresh_squeeze_factor()
        self._push_squeeze_to_canvas()

    def _refresh_squeeze_factor(self) -> None:
        """Recompute the cached squeeze factor — same source as the
        viewport info bar's ``×`` value. Falls back to 1.0 when the
        lens isn't ready or ``compute_metrics`` returns ``None``."""
        sensor = self._project.system_setup.sensor
        try:
            metrics = lm_mod.compute_metrics(
                self._project.system,
                float(sensor.width_mm),
                float(sensor.height_mm),
            )
        except Exception:
            _log.exception("PSFPanelBody: compute_metrics failed")
            metrics = None
        self._squeeze_factor = float(metrics.squeeze) if metrics is not None else 1.0

    def _push_squeeze_to_canvas(self) -> None:
        self._canvas.set_squeeze(self._squeeze_factor if self._desqueeze else 1.0)

    # ------------------------------------------------------------------
    # Render-affecting toggles (trigger a GPU re-render)
    # ------------------------------------------------------------------

    def set_monochromatic(self, enabled: bool) -> None:
        """Toggle single-wavelength mode.  Fires a render — explicit action."""
        enabled = bool(enabled)
        if enabled == self._settings.monochromatic:
            return
        self._settings = PSFRenderSettings(
            **{**self._settings.__dict__, "monochromatic": enabled}
        ).clamp()
        self._latest_comp = None
        if self._lens_eligible():
            self.force_render_now()

    # ------------------------------------------------------------------
    # Tone-mapping dialog
    # ------------------------------------------------------------------

    def open_tone_mapping_dialog(self) -> None:
        if self._tone_dialog is not None and self._tone_dialog.isVisible():
            self._tone_dialog.raise_()
            self._tone_dialog.activateWindow()
            return
        dlg = ToneMappingDialog(self._log_gain, parent=self)
        dlg.gainChanged.connect(self.set_log_gain)
        dlg.finished.connect(self._on_tone_dialog_finished)
        self._tone_dialog = dlg
        dlg.show()

    def _on_tone_dialog_finished(self, _result: int) -> None:
        self._tone_dialog = None

    # ------------------------------------------------------------------
    # Render settings dialog + presets
    # ------------------------------------------------------------------

    @property
    def settings(self) -> PSFRenderSettings:
        return self._settings

    def apply_settings(self, new_settings: PSFRenderSettings, immediate: bool = True) -> None:
        new_settings = new_settings.clamp()
        if new_settings == self._settings:
            return
        self._settings = new_settings
        self._latest_comp = None
        if self._lens_eligible():
            if immediate:
                # Preset swap is an explicit user action — render at once.
                self.force_render_now()
            else:
                # Live dialog edits (scrubbing, toggling) can arrive in a
                # burst; the debounce coalesces them so the expensive PSF-grid
                # render only fires once the value settles.
                self.request_render()

    def apply_preset_low(self) -> None:
        self.apply_settings(LOW_PRESET)

    def apply_preset_high(self) -> None:
        self.apply_settings(HIGH_PRESET)

    def open_render_settings_dialog(self) -> None:
        """Open the modeless render-settings dialog. Every change applies
        live via the debounced render path, so the user can toggle options and
        scrub values while watching the panel re-render — no need to close it."""
        if self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        dlg = PSFRenderSettingsDialog(self._settings, parent=self)
        dlg.settingsChanged.connect(self._on_settings_changed)
        dlg.finished.connect(self._on_settings_dialog_finished)
        self._settings_dialog = dlg
        dlg.show()

    def _on_settings_changed(self, new_settings: PSFRenderSettings) -> None:
        # Route through the debounce (immediate=False) — the PSF grid is
        # expensive, so coalesce a scrub burst into one render.
        self.apply_settings(new_settings, immediate=False)

    def _on_settings_dialog_finished(self, _result: int) -> None:
        # Drop our reference so the next open builds fresh.
        self._settings_dialog = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._is_active = True
        if self._dirty_pending:
            self.force_render_now()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._is_active = False
        self._dirty_pending = True
        self._debounce.stop()

    # ------------------------------------------------------------------
    # Project signal handlers
    # ------------------------------------------------------------------

    def _on_system_replaced(self, _system) -> None:
        self._epoch += 1
        self._latest_comp = None
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        self._refresh_status_for_lens()
        if self._lens_eligible():
            self.force_render_now()
        else:
            self._canvas.clear_image(self._placeholder_for_lens())

    def _on_system_modified(self) -> None:
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _on_system_setup_changed(self) -> None:
        self._epoch += 1
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _should_auto_render(self) -> bool:
        """Two-layer auto-render gate: both per-panel and global must be on."""
        return self._auto_render and self._app_settings.auto_update_enabled()

    def _on_global_auto_update_changed(self, enabled: bool) -> None:
        """User flipped View → Auto-Update Panels. Catch up if dirty."""
        if enabled and self._dirty_pending and self._auto_render:
            self.request_render()

    # ------------------------------------------------------------------
    # Dispatch
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

    def _refresh_status_for_render(self) -> None:
        s = self._settings
        n = s.grid_n
        mode = "mono" if s.monochromatic else "RGB"
        vign = ""
        if self._latest_status is not None:
            n_vign = int((self._latest_status != int(ghostlight.PSFCellStatus.OK)).sum())
            if n_vign > 0:
                total = int(self._latest_status.size)
                vign = f"  ·  {n_vign}/{total} vignetted"
        self._status.setText(
            f"PSF {n}×{n}  ·  tile {s.tile_extent_mm*1000:.1f}µm  ·  "
            f"ray {s.ray_grid}²  ·  λ {s.spectral_samples}  ·  σ {s.splat_sigma_um:.1f}µm  "
            f"·  {mode}{vign}"
        )

    def _dispatch(self) -> None:
        was_idle = False
        with self._lock:
            was_idle = (not self._busy) and (not self._pending)
            self._pending = True
        if was_idle and self._status.text() != "Rendering…":
            self._status.setText("Rendering…")
        self._maybe_launch()

    def _maybe_launch(self) -> None:
        if not self._is_active or not self._auto_render:
            return
        if not self._lens_eligible():
            return

        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True

        lens = self._project.system
        # Freeze the calibration on the dispatch thread for the same
        # reason as the flare panels: the worker shouldn't have the lens mutate
        # underneath the chief-ray pre-pass.
        try:
            lens._check_invalidate()
            calib = lens.calibration()
        except Exception:
            _log.exception("PSFPanelBody: calibration failed")
            calib = None

        # Snapshot the sensor dimensions on the GUI thread so the worker
        # sees a consistent value even if the user keeps editing System
        # Setup while the render is in flight.
        sensor = self._project.system_setup.sensor
        sensor_half_w_mm = max(0.0, float(sensor.width_mm) * 0.5)
        sensor_half_h_mm = max(0.0, float(sensor.height_mm) * 0.5)

        settings = self._settings
        epoch = self._epoch
        threading.Thread(
            target=self._worker,
            args=(lens, calib, settings, epoch,
                  sensor_half_w_mm, sensor_half_h_mm),
            daemon=True,
        ).start()

    def _worker(
        self,
        lens: ghostlight.OpticalSystem,
        calib,
        settings: PSFRenderSettings,
        epoch: int,
        sensor_half_w_mm: float,
        sensor_half_h_mm: float,
    ) -> None:
        try:
            comp, grid_nx, grid_ny, tile_w, tile_h, status, frac = self._do_render(
                lens, calib, settings,
                sensor_half_w_mm, sensor_half_h_mm,
            )
            self._results.put((epoch, comp, grid_nx, grid_ny, tile_w, tile_h, status, frac))
        except Exception:
            _log.exception("PSFPanelBody: render failed")
            self._results.put((epoch, None, 0, 0, 0, 0, None, None))
        finally:
            with self._lock:
                self._busy = False

    @staticmethod
    def _do_render(
        lens: ghostlight.OpticalSystem,
        calib,
        settings: PSFRenderSettings,
        sensor_half_w_mm: float,
        sensor_half_h_mm: float,
    ) -> Tuple[np.ndarray, int, int, int, int, np.ndarray, np.ndarray]:
        if calib is None:
            calib = lens.calibration()
        # Cells partition the user's chosen sensor (System Setup panel).  A
        # larger sensor pushes corner cells past the lens image circle, where
        # the aim solver flags them vignetted — so bigger sensors show more
        # vignetted cells, smaller ones fewer.  Falls back to the image circle
        # when no sensor is set: that is what the lens covers.  (Not
        # sensor_half_*, which is the covered field — the onset of vignetting,
        # far inside the circle on a lens that shades off gradually.)
        half_w = sensor_half_w_mm if sensor_half_w_mm > 0.0 else float(calib.image_circle_semi_w)
        half_h = sensor_half_h_mm if sensor_half_h_mm > 0.0 else float(calib.image_circle_semi_h)
        seeds, targets = _build_cells(
            settings.grid_n, settings.grid_n,
            calib, half_w, half_h,
            settings.field_fraction,
        )

        cfg = ghostlight.PSFConfig()
        cfg.grid_nx = int(settings.grid_n)
        cfg.grid_ny = int(settings.grid_n)
        cfg.tile_w = int(settings.tile_w)
        cfg.tile_h = int(settings.tile_h)
        cfg.tile_extent_mm = float(settings.tile_extent_mm)
        cfg.ray_grid = int(settings.ray_grid)
        cfg.spectral_samples = int(settings.spectral_samples)
        cfg.splat_sigma_um = float(settings.splat_sigma_um)
        cfg.monochromatic = bool(settings.monochromatic)
        cfg.pupil_jitter = 2  # Halton
        cfg.center_mode = ghostlight.PSFCenterMode.FIXED_TARGET

        out = lens.render_psf(seeds, cfg, targets_mm=targets)
        comp = np.stack([out["r"], out["g"], out["b"]], axis=-1)
        status = np.asarray(out["status"])              # (N,) uint8 PSFCellStatus
        pupil_fraction = np.asarray(out["pupil_fraction"])  # (N,) float
        return (comp,
                int(out["grid_nx"]), int(out["grid_ny"]),
                int(out["tile_w"]),  int(out["tile_h"]),
                status, pupil_fraction)

    def _poll_results(self) -> None:
        latest_comp: Optional[np.ndarray] = None
        latest_dims: Optional[Tuple[int, int, int, int]] = None
        latest_status: Optional[np.ndarray] = None
        latest_frac: Optional[np.ndarray] = None
        had_error: bool = False
        while not self._results.empty():
            epoch, comp, gnx, gny, tw, th, status, frac = self._results.get()
            if epoch != self._epoch:
                continue
            if comp is None:
                had_error = True
                continue
            latest_comp = comp
            latest_dims = (gnx, gny, tw, th)
            latest_status = status
            latest_frac = frac

        if latest_comp is not None and latest_dims is not None:
            self._latest_comp = latest_comp
            self._latest_dims = latest_dims
            self._latest_status = latest_status
            self._latest_frac = latest_frac
            grid_nx, grid_ny, tile_w, tile_h = latest_dims
            disp = _composite_to_display(
                latest_comp, tile_w, tile_h, grid_nx, grid_ny,
                per_tile=self._per_tile_norm,
                log_gain=self._log_gain,
            )
            self._canvas.set_image(to_qimage(disp),
                                   grid_nx, grid_ny, tile_w, tile_h,
                                   status=latest_status,
                                   pupil_fraction=latest_frac)

        with self._lock:
            still_busy = self._busy or self._pending
        if still_busy:
            if self._status.text() != "Rendering…":
                self._status.setText("Rendering…")
        elif had_error and latest_comp is None:
            self._status.setText("Render failed (see log)")
        elif latest_comp is not None:
            self._refresh_status_for_render()

        self._maybe_launch()
