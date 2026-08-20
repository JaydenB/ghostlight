"""Body widget for the ``textures`` panel type.

A diagnostic / authoring panel for the lens's raster-image inputs. In this
codebase there is a *single* raster mechanism — the ``APERTURE_IMAGE`` bitmap
carried on ``OpticalSystem.aperture_images`` parallel to the surfaces — that
serves three roles:

  * an **aperture matte** (hard binary silhouette) applied during the ray
    trace at every ``APERTURE_IMAGE`` surface, and
  * a **front-glass transmission / "dirt map"** folded into the diffraction
    pupil as a graded amplitude (dust / smudges diffract), gated by
    ``DiffractionConfig.use_surface_textures``.

This panel lets the user load an image onto a surface (authoring only — the
lens-file writer does not yet round-trip the path) and *visualise* it four
ways, all off one debug render:

  * Raw texture      — the loaded bitmap itself.
  * Composited pupil — ``A(u,v) = stop x cat's-eye x matte box x front texture``
                       (the exact thing the FFT starburst is built from).
  * Starburst sprite — the diffraction sprite that pupil produces.
  * PSF              — the energy-normalised point spread.

The three GPU views share one ``_ghostlight._render_starburst_debug`` call, so
switching the view combo is instant (no re-render).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import ghostlight

from ..math_spinbox import MathDoubleSpinBox
from ..project import Project
from ..settings import AppSettings
from .. import viewtransform as vt

_log = logging.getLogger("ghostlight_designer.textures_panel")

# A lens needs at least two surfaces before calibration / the starburst pass
# produce anything meaningful.
MIN_SURFACES = 2
POLL_INTERVAL_MS = 50

# Debug render size. The pupil / sprite / PSF are grid-sized (independent of
# this), so a small sensor buffer keeps the round-trip cheap.
DEBUG_RENDER_PX = 512
DEBUG_STARBURST_GRID = 512

# View modes (combo index → key).
VIEW_RAW = "raw"
VIEW_PUPIL = "pupil"
VIEW_SPRITE = "sprite"
VIEW_PSF = "psf"
_VIEW_ITEMS = [
    ("Raw texture", VIEW_RAW),
    ("Composited pupil  (stop x cat's-eye x matte x texture)", VIEW_PUPIL),
    ("Starburst sprite", VIEW_SPRITE),
    ("PSF", VIEW_PSF),
]

# Image file filter for the load dialog.
_IMAGE_FILTER = "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.exr);;All files (*)"


def _to_rgb(mono: np.ndarray) -> np.ndarray:
    """``(H, W)`` float → ``(H, W, 3)`` float in [0, 1] for :func:`vt.to_qimage`."""
    m = np.clip(np.asarray(mono, dtype=np.float32), 0.0, 1.0)
    return np.repeat(m[..., None], 3, axis=2)


def _tonemap_hdr(rgb: np.ndarray) -> np.ndarray:
    """Normalise an unbounded scene-linear array to a viewable [0, 1] gamma image."""
    a = np.asarray(rgb, dtype=np.float32)
    peak = float(a.max()) if a.size else 0.0
    if peak > 0.0:
        a = a / peak
    # Perceptual gamma so faint diffraction structure is visible.
    return np.power(np.clip(a, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)


class TextureCanvas(QWidget):
    """Minimal aspect-preserving image blitter (no source marker / overlays).

    Same paint contract as ``FlareCanvas`` but stripped to just the
    image — a texture / pupil view has no draggable light source.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._placeholder = "Load a lens to inspect its image apertures"
        self.setMinimumSize(QSize(256, 256))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, img: Optional[QImage]) -> None:
        self._image = img
        self.update()

    def clear_image(self, placeholder: str) -> None:
        self._image = None
        self._placeholder = placeholder
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        wid, hgt = float(self.width()), float(self.height())
        iw, ih = float(self._image.width()), float(self._image.height())
        if iw <= 0.0 or ih <= 0.0:
            return QRectF(self.rect())
        scale = min(wid / iw, hgt / ih)
        out_w, out_h = iw * scale, ih * scale
        return QRectF((wid - out_w) / 2.0, (hgt - out_h) / 2.0, out_w, out_h)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 13))
        if self._image is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        # Nearest-neighbour so a small mask reads as crisp pixels, not blurred.
        p.drawImage(self._image_rect(), self._image)


class TexturesPanelBody(QWidget):
    """Load + visualise per-surface aperture images and the diffraction pupil."""

    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._app_settings = settings

        # surface-combo index → surface index in the optical system.
        self._surface_map: list[int] = []
        # Sticky load-dialog directory; see _start_dir().
        self._last_dir: str = ""
        # Latest debug dict from _render_starburst_debug (pupil/sprite/psf).
        self._debug: Optional[dict] = None

        # --- threading state (worker → queue → poll timer) --------------
        self._results: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._busy = False
        self._pending = False
        self._epoch = 0
        self._is_active = False

        self._build_ui()

        # React to lens changes like the render panels do.
        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start()

        self._reload_surfaces()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- left: controls -------------------------------------------
        controls = QVBoxLayout()
        controls.setSpacing(8)

        src_box = QGroupBox("Image aperture", self)
        src_form = QFormLayout(src_box)

        self._surface_combo = QComboBox(self)
        self._surface_combo.currentIndexChanged.connect(self._on_surface_changed)
        src_form.addRow("Surface", self._surface_combo)

        self._semi_spin = MathDoubleSpinBox(self)
        self._semi_spin.setRange(0.001, 10000.0)
        self._semi_spin.setDecimals(3)
        self._semi_spin.setSuffix(" mm")
        self._semi_spin.setToolTip(
            "World-space radius the image boundary maps to (aperture_semi_diameter)."
        )
        src_form.addRow("Semi-diameter", self._semi_spin)

        self._aspect_spin = MathDoubleSpinBox(self)
        self._aspect_spin.setRange(0.01, 100.0)
        self._aspect_spin.setDecimals(3)
        self._aspect_spin.setValue(1.0)
        self._aspect_spin.setToolTip("X-axis stretch of the image (aperture_aspect).")
        src_form.addRow("Aspect", self._aspect_spin)

        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load image…", self)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._remove_btn = QPushButton("Remove", self)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._remove_btn)
        src_form.addRow(btn_row)

        self._info_label = QLabel("—", self)
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #9a9a9a;")
        src_form.addRow(self._info_label)

        controls.addWidget(src_box)

        # Diffraction / preview controls.
        diff_box = QGroupBox("Diffraction preview", self)
        diff_form = QFormLayout(diff_box)

        self._use_tex_check = QCheckBox("Fold front texture into pupil", self)
        self._use_tex_check.setChecked(True)
        self._use_tex_check.setToolTip(
            "DiffractionConfig.use_surface_textures — multiply the front-most "
            "image aperture into the pupil amplitude (dust / smudges diffract)."
        )
        self._use_tex_check.toggled.connect(lambda _=None: self._request_render())
        diff_form.addRow(self._use_tex_check)

        self._field_x = MathDoubleSpinBox(self)
        self._field_x.setRange(-2.0, 2.0)
        self._field_x.setSingleStep(0.05)
        self._field_x.setDecimals(3)
        self._field_x.setValue(0.5)
        self._field_x.valueChanged.connect(lambda _=None: self._request_render())
        self._field_y = MathDoubleSpinBox(self)
        self._field_y.setRange(-2.0, 2.0)
        self._field_y.setSingleStep(0.05)
        self._field_y.setDecimals(3)
        self._field_y.setValue(0.5)
        self._field_y.valueChanged.connect(lambda _=None: self._request_render())
        field_row = QHBoxLayout()
        field_row.addWidget(self._field_x)
        field_row.addWidget(self._field_y)
        diff_form.addRow("Field X / Y", field_row)
        diff_box.setToolTip(
            "Field position (fractional sensor, 0.5 = on-axis) the pupil is "
            "built at — off-axis reveals cat's-eye vignetting + throughput."
        )

        controls.addWidget(diff_box)

        self._refresh_btn = QPushButton("Refresh", self)
        self._refresh_btn.clicked.connect(self._request_render)
        controls.addWidget(self._refresh_btn)

        controls.addStretch(1)

        controls_w = QWidget(self)
        controls_w.setLayout(controls)
        controls_w.setFixedWidth(300)
        root.addWidget(controls_w)

        # ---- right: preview -------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)

        self._view_combo = QComboBox(self)
        for label, _key in _VIEW_ITEMS:
            self._view_combo.addItem(label)
        self._view_combo.currentIndexChanged.connect(lambda _=None: self._redisplay())
        right.addWidget(self._view_combo)

        self._canvas = TextureCanvas(self)
        right.addWidget(self._canvas, 1)

        self._status = QLabel("", self)
        self._status.setStyleSheet("color: #9a9a9a;")
        right.addWidget(self._status)

        root.addLayout(right, 1)

    # ------------------------------------------------------- surface state
    def _system(self) -> Optional[ghostlight.OpticalSystem]:
        try:
            return self._project.system
        except Exception:
            return None

    def _current_surface_index(self) -> int:
        i = self._surface_combo.currentIndex()
        if 0 <= i < len(self._surface_map):
            return self._surface_map[i]
        return -1

    def _reload_surfaces(self) -> None:
        """Repopulate the surface combo from the current lens."""
        sysm = self._system()
        self._surface_combo.blockSignals(True)
        self._surface_combo.clear()
        self._surface_map = []
        if sysm is not None:
            try:
                surfaces = list(sysm.surfaces)
            except Exception:
                surfaces = []
            for i, s in enumerate(surfaces):
                has_img = self._surface_has_image(sysm, i)
                is_stop = bool(getattr(s, "is_stop", False))
                tag = []
                if is_stop:
                    tag.append("stop")
                if has_img:
                    tag.append("image")
                suffix = f"  [{', '.join(tag)}]" if tag else ""
                self._surface_combo.addItem(f"Surface {i}{suffix}")
                self._surface_map.append(i)
        self._surface_combo.blockSignals(False)
        self._sync_surface_widgets()
        self._request_render()

    @staticmethod
    def _surface_has_image(sysm: ghostlight.OpticalSystem, idx: int) -> bool:
        try:
            imgs = sysm.aperture_images
            if idx >= len(imgs):
                return False
            return int(np.asarray(imgs[idx].pixels).size) > 0
        except Exception:
            return False

    def _sync_surface_widgets(self) -> None:
        """Seed the semi-diameter / aspect / info widgets for the selected surface."""
        sysm = self._system()
        idx = self._current_surface_index()
        has = sysm is not None and idx >= 0
        self._load_btn.setEnabled(has)
        eligible = has and self._surface_has_image(sysm, idx)
        self._remove_btn.setEnabled(bool(eligible))
        if not has:
            self._info_label.setText("—")
            return
        surf = sysm.surfaces[idx]
        semi = float(getattr(surf, "aperture_semi_diameter", 0.0))
        if semi <= 0.0:
            semi = float(getattr(surf, "semi_aperture", 0.0)) or 5.0
        self._semi_spin.blockSignals(True)
        self._semi_spin.setValue(semi)
        self._semi_spin.blockSignals(False)
        aspect = float(getattr(surf, "aperture_aspect", 1.0)) or 1.0
        self._aspect_spin.blockSignals(True)
        self._aspect_spin.setValue(aspect)
        self._aspect_spin.blockSignals(False)

        if eligible:
            img = sysm.aperture_images[idx]
            path = getattr(img, "source_path", "") or "(in-memory)"
            self._info_label.setText(
                f"{img.width}×{img.height}px\n{path}"
            )
        else:
            self._info_label.setText("No image on this surface.")

    # -------------------------------------------------------- authoring
    def _on_load_clicked(self) -> None:
        sysm = self._system()
        idx = self._current_surface_index()
        if sysm is None or idx < 0:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load aperture image", self._start_dir(), _IMAGE_FILTER
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        try:
            pixels = self._decode_image(path)
        except Exception as exc:  # noqa: BLE001 - surface any decode failure
            _log.exception("TexturesPanel: failed to decode %s", path)
            self._status.setText(f"Load failed: {exc}")
            return

        semi = float(self._semi_spin.value())
        aspect = float(self._aspect_spin.value())
        try:
            with self._project.edit("Assign aperture image"):
                surf = sysm.surfaces[idx]
                surf.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
                surf.aperture_semi_diameter = semi
                surf.aperture_aspect = aspect
                img = sysm.aperture_images[idx]
                img.source_path = path
                img.semi_diameter = semi
                img.pixels = pixels
        except Exception:
            _log.exception("TexturesPanel: assign failed")
            self._status.setText("Assign failed (see log)")
            return
        self._status.setText(f"Loaded {pixels.shape[1]}×{pixels.shape[0]} image.")
        # systemModified → _on_system_modified refreshes combo + render.

    def _on_remove_clicked(self) -> None:
        sysm = self._system()
        idx = self._current_surface_index()
        if sysm is None or idx < 0 or not self._surface_has_image(sysm, idx):
            return
        try:
            with self._project.edit("Remove aperture image"):
                surf = sysm.surfaces[idx]
                # Revert to a plain circular stop.
                surf.aperture_shape = int(ghostlight.ApertureShape.CIRCLE)
                img = sysm.aperture_images[idx]
                img.pixels = np.zeros((0, 0), dtype=np.float32)
                img.source_path = ""
        except Exception:
            _log.exception("TexturesPanel: remove failed")
            self._status.setText("Remove failed (see log)")
            return
        self._status.setText("Removed image aperture.")

    def _start_dir(self) -> str:
        """Directory the load dialog opens in.

        Sticky across loads within a session; first time round it points at the
        bundled examples in :mod:`ghostlight_designer.resources.textures` so the
        panel is usable without hunting for an image. Empty string = let Qt pick.
        """
        if self._last_dir:
            return self._last_dir
        from ..resources.textures import textures_dir

        return textures_dir() or ""

    @staticmethod
    def _decode_image(path: str) -> np.ndarray:
        """PIL-decode ``path`` to a single-channel float32 [0, 1] array.

        Mirrors ``OpticalSystem.load_aperture_images`` so an in-session load
        matches what the lens-file helper would produce.
        """
        from PIL import Image  # local import: PIL is an optional dependency

        with Image.open(path) as pil_img:
            arr = np.asarray(pil_img.convert("F"), dtype=np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        if peak > 1.5:
            arr = arr / 255.0
        return np.ascontiguousarray(arr, dtype=np.float32)

    # ------------------------------------------------------------ signals
    def _on_surface_changed(self, _idx: int) -> None:
        self._sync_surface_widgets()
        self._redisplay()

    def _on_system_replaced(self, _system) -> None:
        self._debug = None
        self._reload_surfaces()

    def _on_system_modified(self) -> None:
        self._sync_surface_widgets()
        # Rebuild the [image] tags without disturbing the selection.
        self._refresh_surface_tags()
        self._request_render()

    def _refresh_surface_tags(self) -> None:
        sysm = self._system()
        if sysm is None:
            return
        for combo_i, surf_i in enumerate(self._surface_map):
            has_img = self._surface_has_image(sysm, surf_i)
            is_stop = bool(getattr(sysm.surfaces[surf_i], "is_stop", False))
            tag = []
            if is_stop:
                tag.append("stop")
            if has_img:
                tag.append("image")
            suffix = f"  [{', '.join(tag)}]" if tag else ""
            self._surface_combo.setItemText(combo_i, f"Surface {surf_i}{suffix}")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._is_active = True
        self._request_render()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._is_active = False

    # ------------------------------------------------------------- render
    def _lens_eligible(self) -> bool:
        sysm = self._system()
        if sysm is None:
            return False
        try:
            return int(sysm.num_surfaces()) >= MIN_SURFACES
        except Exception:
            return False

    def _request_render(self) -> None:
        """Schedule a debug render; the poll timer dispatches the worker."""
        with self._lock:
            self._pending = True
        self._maybe_launch()

    def _maybe_launch(self) -> None:
        if not self._is_active or not self._lens_eligible():
            return
        debug_fn = self._debug_fn()
        if debug_fn is None:
            self._status.setText("Debug render unavailable in this build.")
            return

        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True
            self._epoch += 1
            epoch = self._epoch

        sysm = self._system()
        # Freeze lens + calibration on the GUI thread so the worker reads one
        # consistent snapshot (same rule as the flare panels).
        calib = None
        try:
            sysm._check_invalidate()
            calib = sysm.calibration()
        except Exception:
            _log.exception("TexturesPanel: calibration failed")

        cfg = self._build_config()
        w = h = DEBUG_RENDER_PX
        self._status.setText("Rendering…")
        threading.Thread(
            target=self._worker,
            args=(debug_fn, w, h, sysm, calib, cfg, epoch),
            daemon=True,
        ).start()

    def _build_config(self) -> "ghostlight.PointFlareConfig":
        cfg = ghostlight.PointFlareConfig()
        cfg.source_x = float(self._field_x.value())
        cfg.source_y = float(self._field_y.value())
        cfg.output_cs = ghostlight.OutputColorSpace.ACESCG
        # The debug entry point only fills pupil/sprite/psf when the starburst
        # pass actually runs, so force it on regardless of the flare panel's
        # setting. SPRITE engine populates sprite_rgb (MDFT does not).
        cfg.diffraction.starburst = True
        engine = getattr(ghostlight, "StarburstEngine", None)
        if engine is not None:
            cfg.diffraction.starburst_engine = engine.SPRITE
        cfg.diffraction.starburst_grid = DEBUG_STARBURST_GRID
        if hasattr(cfg.diffraction, "use_surface_textures"):
            cfg.diffraction.use_surface_textures = bool(self._use_tex_check.isChecked())
        return cfg

    @staticmethod
    def _debug_fn():
        ext = getattr(ghostlight, "_ghostlight", None)
        return getattr(ext, "_render_starburst_debug", None) if ext else None

    def _worker(self, debug_fn, w, h, sysm, calib, cfg, epoch) -> None:
        result = None
        try:
            if calib is not None:
                result = debug_fn(w, h, sysm, calib, cfg)
        except Exception:
            _log.exception("TexturesPanel: debug render failed")
        finally:
            self._results.put((epoch, result))
            with self._lock:
                self._busy = False

    def _poll_results(self) -> None:
        latest = None
        got = False
        while not self._results.empty():
            epoch, result = self._results.get()
            if epoch != self._epoch:
                continue
            latest = result
            got = True
        if got:
            self._debug = latest
            self._redisplay()
        with self._lock:
            still = self._busy or self._pending
        if not still:
            self._maybe_launch()

    # ------------------------------------------------------------ display
    def _current_view(self) -> str:
        i = self._view_combo.currentIndex()
        if 0 <= i < len(_VIEW_ITEMS):
            return _VIEW_ITEMS[i][1]
        return VIEW_RAW

    def _redisplay(self) -> None:
        mode = self._current_view()
        if mode == VIEW_RAW:
            self._show_raw_texture()
            return
        self._show_debug_view(mode)

    def _show_raw_texture(self) -> None:
        sysm = self._system()
        idx = self._current_surface_index()
        if sysm is None or idx < 0 or not self._surface_has_image(sysm, idx):
            self._canvas.clear_image("This surface has no image aperture.")
            self._status.setText("")
            return
        pixels = np.asarray(sysm.aperture_images[idx].pixels, dtype=np.float32)
        self._canvas.set_image(vt.to_qimage(_to_rgb(pixels)))
        self._status.setText(f"Raw texture · {pixels.shape[1]}×{pixels.shape[0]}px")

    def _show_debug_view(self, mode: str) -> None:
        dbg = self._debug
        if not dbg or int(dbg.get("grid", 0)) <= 0:
            self._canvas.clear_image("No diffraction data — press Refresh.")
            return
        grid = int(dbg["grid"])
        try:
            if mode == VIEW_PUPIL:
                arr = np.asarray(dbg["pupil"], dtype=np.float32).reshape(grid, grid)
                rgb = _to_rgb(arr)
                extra = f"throughput T={float(dbg.get('throughput', 1.0)):.3f}"
            elif mode == VIEW_SPRITE:
                arr = np.asarray(dbg["sprite_rgb"], dtype=np.float32).reshape(grid, grid, 3)
                rgb = _tonemap_hdr(arr)
                extra = "tonemapped"
            else:  # VIEW_PSF
                arr = np.asarray(dbg["psf"], dtype=np.float32).reshape(grid, grid)
                rgb = _tonemap_hdr(_to_rgb(arr))
                extra = "log-ish tonemap"
        except Exception:
            _log.exception("TexturesPanel: failed to build %s view", mode)
            self._canvas.clear_image("Failed to display view (see log).")
            return
        self._canvas.set_image(vt.to_qimage(rgb))
        self._status.setText(f"{self._view_combo.currentText()} · {grid}×{grid} · {extra}")
