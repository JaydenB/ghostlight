"""Dialogs + render-settings dataclass for the PSF panel.

The PSF renderer has more knobs than the flare renderers because the
output is a *grid of PSFs* rather than a single image:

* ``grid_n``                — tiles per axis (1 = single high-res PSF)
* ``tile_w`` / ``tile_h``   — per-tile pixel size
* ``tile_extent_mm``        — physical extent of one tile on the sensor
* ``ray_grid``              — entrance-pupil sample resolution
* ``spectral_samples``      — wavelengths sampled
* ``splat_sigma_um``        — Gaussian per-ray spot size (µm on sensor, physical)
* ``field_fraction``        — fraction of the calibrated max half-angle
                              the grid spans on each axis
* ``monochromatic``         — single-wavelength trace (no chromatic
                              spatial separation)

Display-only knobs (``per_tile_norm``, ``log_gain``) live on the body
itself, not in this dataclass — they don't trigger a re-render.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from ..render_common import attach_spinbox_scrubber


# ---------------------------------------------------------------------------
# Render settings dataclass + presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PSFRenderSettings:
    """Per-panel PSF-grid render parameters.

    Costs scale roughly as
    ``grid_n² × ray_grid² × spectral_samples × splat_sigma²``.
    The ``LOW`` preset targets a sub-100ms interactive scrub on a typical
    card; ``HIGH`` is the "looks good for screenshots" preset.
    """
    grid_n: int = 5
    tile_w: int = 96
    tile_h: int = 96
    tile_extent_mm: float = 0.5
    ray_grid: int = 192
    spectral_samples: int = 12
    splat_sigma_um: float = 4.0
    field_fraction: float = 0.7
    monochromatic: bool = False

    def clamp(self) -> "PSFRenderSettings":
        return replace(
            self,
            grid_n=max(1, min(15, int(self.grid_n))),
            tile_w=max(16, min(512, int(self.tile_w))),
            tile_h=max(16, min(512, int(self.tile_h))),
            tile_extent_mm=max(0.001, min(10.0, float(self.tile_extent_mm))),
            ray_grid=max(16, min(1024, int(self.ray_grid))),
            spectral_samples=max(1, min(48, int(self.spectral_samples))),
            splat_sigma_um=max(0.0, min(15.0, float(self.splat_sigma_um))),
            field_fraction=max(0.01, min(1.0, float(self.field_fraction))),
        )


# The splat is a physical size (µm), so LOW and HIGH share it — the only
# difference between the presets is fidelity (tile resolution, ray/spectral
# sampling), never the picture itself.
_SPLAT_UM = 4.0
LOW_PRESET = PSFRenderSettings(
    grid_n=5,
    tile_w=64,
    tile_h=64,
    tile_extent_mm=0.5,
    ray_grid=64,
    spectral_samples=8,
    splat_sigma_um=_SPLAT_UM,
    field_fraction=0.7,
)
HIGH_PRESET = PSFRenderSettings(
    grid_n=5,
    tile_w=256,
    tile_h=256,
    tile_extent_mm=0.5,
    ray_grid=256,
    spectral_samples=16,
    splat_sigma_um=_SPLAT_UM,
    field_fraction=0.7,
)


# ---------------------------------------------------------------------------
# Render settings dialog
# ---------------------------------------------------------------------------


class PSFRenderSettingsDialog(QDialog):
    """Modeless dialog for editing the per-panel PSF render settings.

    Applies live: every widget change emits :attr:`settingsChanged` with a
    fresh :class:`PSFRenderSettings`, so the panel re-renders as the user
    toggles options or scrubs values without having to close the dialog.

    Signals:
        settingsChanged(PSFRenderSettings): emitted whenever any control changes.
    """

    settingsChanged = Signal(object)

    def __init__(
        self,
        current: PSFRenderSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PSF Render Settings")
        # Modeless + Tool so the user can tweak settings and watch the panel
        # behind it re-render on each change (mirrors the tone-mapping dialog).
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        self._grid_n = MathSpinBox(self)
        self._grid_n.setRange(1, 15)
        self._grid_n.setValue(int(current.grid_n))
        self._grid_n.setToolTip(
            "Tiles per axis.  Total PSFs = N².  1 = single high-res PSF mode. "
            "Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._grid_n, label="Grid N")

        self._tile_w = MathSpinBox(self)
        self._tile_w.setRange(16, 512)
        self._tile_w.setSingleStep(16)
        self._tile_w.setValue(int(current.tile_w))
        self._tile_w.setSuffix(" px")
        self._tile_w.setToolTip("Per-tile pixel width.  Ctrl+MMB to scrub.")
        attach_spinbox_scrubber(self._tile_w, label="Tile width")

        self._tile_h = MathSpinBox(self)
        self._tile_h.setRange(16, 512)
        self._tile_h.setSingleStep(16)
        self._tile_h.setValue(int(current.tile_h))
        self._tile_h.setSuffix(" px")
        self._tile_h.setToolTip("Per-tile pixel height.  Ctrl+MMB to scrub.")
        attach_spinbox_scrubber(self._tile_h, label="Tile height")

        self._tile_extent = MathDoubleSpinBox(self)
        self._tile_extent.setRange(0.001, 10.0)
        self._tile_extent.setSingleStep(0.05)
        self._tile_extent.setDecimals(4)
        self._tile_extent.setSuffix(" mm")
        self._tile_extent.setValue(float(current.tile_extent_mm))
        self._tile_extent.setToolTip(
            "Physical extent of one tile on the sensor.  Should comfortably "
            "contain the worst-case off-axis PSF for your lens.  "
            "Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._tile_extent, label="Tile extent (mm)")

        self._ray_grid = MathSpinBox(self)
        self._ray_grid.setRange(16, 1024)
        self._ray_grid.setSingleStep(16)
        self._ray_grid.setValue(int(current.ray_grid))
        self._ray_grid.setToolTip(
            "Square-root of entrance-pupil samples per source.  Higher → "
            "smoother PSF, more GPU time.  Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._ray_grid, label="Ray Grid")

        self._spectral = MathSpinBox(self)
        self._spectral.setRange(1, 48)
        self._spectral.setValue(int(current.spectral_samples))
        self._spectral.setToolTip(
            "Wavelengths sampled.  Ignored when Monochromatic is on "
            "(then exactly 1 sample at d-line).  Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._spectral, label="Spectral samples")

        self._splat = MathDoubleSpinBox(self)
        self._splat.setRange(0.0, 15.0)
        self._splat.setSingleStep(0.5)
        self._splat.setDecimals(1)
        self._splat.setSuffix(" µm")
        self._splat.setValue(float(current.splat_sigma_um))
        self._splat.setToolTip(
            "Gaussian per-ray spot size, in micrometres on the sensor "
            "(physical — the rendered spot stays the same size at any tile "
            "resolution).  0 = single-pixel bilinear (raw geometric).  "
            "Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._splat, label="Splat sigma (µm)")

        self._field = MathDoubleSpinBox(self)
        self._field.setRange(0.01, 1.0)
        self._field.setSingleStep(0.05)
        self._field.setDecimals(3)
        self._field.setValue(float(current.field_fraction))
        self._field.setToolTip(
            "Fraction of the capture sensor the cell grid spans on each axis.  "
            "1.0 fills the sensor to its edges; lower values zoom the grid into "
            "the centre.  Corner cells past the lens image circle are flagged "
            "vignetted.  Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._field, label="Field fraction")

        self._mono = QCheckBox(self)
        self._mono.setChecked(bool(current.monochromatic))
        self._mono.setToolTip(
            "On: single-wavelength (d-line) trace — no lateral-chromatic "
            "spatial separation in the output.  Off: full polychromatic."
        )

        form = QFormLayout()
        form.addRow("Grid N:", self._grid_n)
        form.addRow("Tile width:", self._tile_w)
        form.addRow("Tile height:", self._tile_h)
        form.addRow("Tile extent:", self._tile_extent)
        form.addRow("Ray grid:", self._ray_grid)
        form.addRow("Spectral samples:", self._spectral)
        form.addRow("Splat sigma:", self._splat)
        form.addRow("Field fraction:", self._field)
        form.addRow("Monochromatic:", self._mono)

        info = QLabel(
            "Changes apply live. Settings persist for the lifetime of this "
            "panel only.",
            self,
        )
        info.setStyleSheet("color: #888;")
        info.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.close)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(info)
        outer.addWidget(buttons)

        # Wire every control to re-emit the full settings. Connected last, after
        # all initial setValue/setChecked calls above, so construction doesn't
        # fire spurious change signals.
        for spin in (
            self._grid_n, self._tile_w, self._tile_h, self._tile_extent,
            self._ray_grid, self._spectral, self._splat, self._field,
        ):
            spin.valueChanged.connect(self._emit_changed)
        self._mono.toggled.connect(self._emit_changed)

    def _emit_changed(self, *_args) -> None:
        self.settingsChanged.emit(self.result_settings())

    def result_settings(self) -> PSFRenderSettings:
        return PSFRenderSettings(
            grid_n=int(self._grid_n.value()),
            tile_w=int(self._tile_w.value()),
            tile_h=int(self._tile_h.value()),
            tile_extent_mm=float(self._tile_extent.value()),
            ray_grid=int(self._ray_grid.value()),
            spectral_samples=int(self._spectral.value()),
            splat_sigma_um=float(self._splat.value()),
            field_fraction=float(self._field.value()),
            monochromatic=bool(self._mono.isChecked()),
        ).clamp()


# ---------------------------------------------------------------------------
# Tone-mapping dialog (display-only, no re-render)
# ---------------------------------------------------------------------------


class ToneMappingDialog(QDialog):
    """Modeless dialog for the display log-gain.

    Display-only — emits :sig:`gainChanged` whenever the spinbox changes
    so the body can re-process the *last rendered* float buffer without
    firing a GPU render. This is the PSF panel's own diagnostic tone map
    (peak-normalised log1p), intentionally outside the designer-wide ACES
    view transform.
    """

    gainChanged = Signal(float)

    def __init__(self, current_gain: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PSF Display Tone Mapping")
        self.setModal(False)

        self._gain = MathDoubleSpinBox(self)
        self._gain.setRange(1.0, 2000.0)
        self._gain.setSingleStep(1.0)
        self._gain.setDecimals(1)
        self._gain.setValue(float(current_gain))
        self._gain.setToolTip(
            "log1p compression gain.  Lower (1-10) = Zemax-style compact "
            "PSFs with wings fading to black.  Higher (200-2000) = lifts "
            "faint chromatic halos into view but also amplifies ray-hit "
            "noise.  Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._gain, label="Log gain")
        self._gain.valueChanged.connect(self._on_value_changed)

        form = QFormLayout()
        form.addRow("Log gain:", self._gain)

        info = QLabel("Display-only.  No GPU re-render.", self)
        info.setStyleSheet("color: #888;")

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(info)
        outer.addWidget(buttons)

    def set_gain(self, value: float) -> None:
        blocked = self._gain.blockSignals(True)
        try:
            self._gain.setValue(float(value))
        finally:
            self._gain.blockSignals(blocked)

    def _on_value_changed(self, value: float) -> None:
        self.gainChanged.emit(float(value))
