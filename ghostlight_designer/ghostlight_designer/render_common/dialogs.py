"""Shared render dialogs — color-correction (exposure) and per-panel
render settings.

Both dialogs are scoped to a single panel instance; they are not
persisted to disk and reset when the panel is destroyed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# Allowed starburst FFT grid sizes (power of two). Bigger = spikes reach further
# before the square window clips them, at ~4x render cost per step up.
_STARBURST_GRID_CHOICES = (512, 1024, 2048, 4096)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from .spinbox_scrub import attach_spinbox_scrubber


# Viewer exposure is edited in stops (log2 linear gain), matching a
# compositor's viewer. ±20 stops covers the full HDR span of flare output.
EXPOSURE_STOPS_MIN = -20.0
EXPOSURE_STOPS_MAX = 20.0
EXPOSURE_STOPS_STEP = 0.25


class _SnapMathDoubleSpinBox(MathDoubleSpinBox):
    """``MathDoubleSpinBox`` that pins every value to the ``singleStep()``
    grid — typed numbers, evaluated expressions, programmatic ``setValue``
    calls (which is also the scrubber's write path) all round to the
    nearest step multiple."""

    def _snap(self, value: float) -> float:
        step = self.singleStep()
        if step <= 0.0:
            return float(value)
        return round(float(value) / step) * step

    def setValue(self, value: float) -> None:  # type: ignore[override]
        super().setValue(self._snap(value))

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        return self._snap(super().valueFromText(text))

    def _format_result(self, value: float) -> str | None:
        # Snap before the range check so an expression landing just past a
        # limit (e.g. `90.2` with max 90) rounds onto it instead of reverting.
        return super()._format_result(self._snap(value))


# ---------------------------------------------------------------------------
# Render settings dataclass + presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderSettings:
    """Per-panel render parameters.

    ``width_px`` is the rendered image width; height is derived from
    the project sensor's aspect ratio at render time. ``ray_grid`` is
    the sqrt of total rays per render. ``spectral_samples`` is the
    number of wavelengths sampled.
    """
    width_px: int = 192
    ray_grid: int = 256
    spectral_samples: int = 4
    # Ghost-pipeline sampling accelerators — mirror the ghostlight C++ defaults.
    #   cull_dead_pairs      : skip ghost pairs whose rays never reach the
    #                          sensor (lossless speed-up).
    #   concentrate_samples  : lay each ghost's rays inside its survivor region
    #                          instead of the whole pupil (much cleaner at the
    #                          same ray grid).
    #   adaptive_sample_budgets : scale each ghost's ray COUNT to its
    #                          concentrated size — harvests the oversampling of
    #                          tiny ghosts as render speed. Requires
    #                          concentration.
    #   adaptive_density_boost : adaptive's quality dial — rays kept relative to
    #                          the concentrated density. 8 = near-lossless vs
    #                          concentration; lower = faster/grainier. Only
    #                          applies when adaptive is on.
    cull_dead_pairs: bool = True
    concentrate_samples: bool = True
    adaptive_sample_budgets: bool = True
    adaptive_density_boost: float = 8.0
    # Aperture-diffraction starburst (a separate additive layer).
    #   starburst          : master switch. Off by default — it is one of the
    #                          three extra layers the quality ladder keeps out of
    #                          Draft/Mid/High and turns on together in High+.
    #   starburst_scale_trim : artistic size multiplier on the physical pattern
    #                          (1.0 = physically scaled — often only a few pixels
    #                          across at preview resolution; raise to enlarge).
    #   starburst_gain     : brightness multiplier (1.0 = the source's own flux).
    #   starburst_spectral : wavelengths integrated for dispersion (0 = inherit
    #                          spectral_samples; 16-32 gives smooth rainbow spikes).
    #   starburst_grid     : FFT grid side (power of two). The pattern is bounded
    #                          by this square window, so a bigger grid = the
    #                          spikes reach further before the edge (less "boxed"
    #                          clipping) — at ~4x the cost per doubling. 1024 is
    #                          the interactive default; High+ uses 2048.
    #   starburst_grid_cap : ceiling the renderer auto-grows the grid TO when the
    #                          sensor is larger than the lens image circle — an
    #                          undersized lens makes the fixed-size pattern a small
    #                          tile whose FFT-period edge (the "box") is visible.
    #                          8192 (default) buries it on IMAX-70 even for the more-
    #                          spread aberrated star; raise to 16384 for larger
    #                          sensors / higher preview res (~4x cost per doubling;
    #                          needs the VRAM). Matched sensors never auto-grow.
    #   starburst_engine   : "sprite" (fast FFT-sprite path, default) or "mdft"
    #                          (the exact matrix-DFT engine — evaluates the
    #                          diffraction integral directly at the sensor pixels,
    #                          so a moving source stays crisp instead of shimmering
    #                          as the fixed sprite is resampled). "mdft" on High+.
    starburst: bool = False
    starburst_engine: str = "sprite"
    starburst_scale_trim: float = 8.0
    starburst_gain: float = 1.0
    starburst_spectral: int = 24
    starburst_grid: int = 1024
    starburst_grid_cap: int = 8192
    # Physical veiling glare — a separate additive layer carrying an
    # energy-conserving spectral glare-spread function (a broad radial halo
    # around each bright source). It joins the metered flare layer
    # (ghost + veil), so auto-exposure accounts for it.
    #   veil         : master switch. Off by default — one of the three extra
    #                  layers that only High+ turns on.
    #   veil_gain    : total halo energy as a fraction of the source's own flux
    #                  (0.03 ~ a subtle 3% wash; raise for a hazier lens).
    #   veil_spread  : GSF core radius as a fraction of the sensor half-height, so
    #                  the halo scales with the frame (larger = washes more of it).
    #   veil_falloff : radial power of the generalized Lorentzian tail. 1.0 = the
    #                  broadest (pure-Lorentzian) tails; 1.5 ~ a ~1/r^3 glare tail.
    veil: bool = False
    veil_gain: float = 0.03
    veil_spread: float = 0.12
    veil_falloff: float = 1.5
    # Physical ghost-edge diffraction — HURB (Heisenberg Uncertainty Ray
    # Bending). Every ghost ray that survives passing a hard edge (the stop, a
    # surface rim, a matte-box / baffle plane) gets a small random angular kick
    # σ = λ·K/d perpendicular to the nearest edge — the soft chromatic glow a real
    # lens throws around every edge. Direction-only (energy-conserving), envelope-only
    # (no fringes), chromatic (red kicks wider than blue).
    #   hurb      : master switch. Off by default — off keeps the ghost render
    #               byte-identical at zero added cost (the kernel is compile-time
    #               templated on it). Unlike the three additive layers it modifies
    #               the ghost pass itself rather than compositing over it, so the
    #               quality ladder turns it on from High upward.
    #   hurb_kick : kick distribution — "lorentzian" (default; heavy 1/θ² glare
    #               tails, hard-clamped) or "gaussian" (a softer, tighter falloff).
    hurb: bool = False
    hurb_kick: str = "lorentzian"
    # Film-gate flare (mechanical) — the aperture plate's cut edge
    # reflecting light that would land just outside the frame back into it. Only
    # fires when the source sits in a narrow band past the frame edge, and only
    # when the lens throws more image than the sensor uses; a source inside the
    # frame produces exactly nothing. Joins the metered flare layer, so
    # auto-exposure accounts for it (it peaks well under 1% of the source, so it
    # cannot hijack the meter the way the starburst core would).
    #   gate           : master switch. Off by default — no pass runs and the
    #                    render is byte-identical.
    #   gate_standoff  : mm from the plate's rear edge to the sensor. The lever
    #                    arm the scattered light travels, so it sets how far into
    #                    frame the streak reaches — and it moves which source
    #                    positions fire, because the capture band sits at
    #                    standoff*slope past the edge.
    #   gate_roughness : how rough the machined land is (radians). 0 is a perfect
    #                    mirror, which folds a point source back to a point; this
    #                    is the knob that turns the fold into a streak.
    #   gate_gain      : multiplier on the physical brightness. The layer is faint
    #                    by nature, so this is what makes it readable.
    # Plate thickness, wall reflectance, lobe model and groove angle are real
    # config fields but are left at their validated defaults rather than exposed.
    gate: bool = False
    gate_standoff: float = 5.0
    gate_roughness: float = 0.08
    gate_gain: float = 1.0

    def clamp(self) -> "RenderSettings":
        return replace(
            self,
            width_px=max(32, min(1024, int(self.width_px))),
            ray_grid=max(16, min(2048, int(self.ray_grid))),
            spectral_samples=max(1, min(32, int(self.spectral_samples))),
            adaptive_density_boost=max(1.0, min(64.0, float(self.adaptive_density_boost))),
            starburst_engine=(self.starburst_engine
                              if self.starburst_engine in ("sprite", "mdft") else "sprite"),
            starburst_scale_trim=max(1.0, min(200.0, float(self.starburst_scale_trim))),
            starburst_gain=max(0.0, min(1.0e6, float(self.starburst_gain))),
            starburst_spectral=max(0, min(64, int(self.starburst_spectral))),
            starburst_grid=_clamp_pow2(self.starburst_grid, 256, 4096),
            starburst_grid_cap=_clamp_pow2(self.starburst_grid_cap, 256, 16384),
            veil_gain=max(0.0, min(1.0, float(self.veil_gain))),
            veil_spread=max(1.0e-3, min(4.0, float(self.veil_spread))),
            veil_falloff=max(1.0, min(3.0, float(self.veil_falloff))),
            hurb_kick=(self.hurb_kick if self.hurb_kick in ("lorentzian", "gaussian")
                       else "lorentzian"),
            gate_standoff=max(0.0, min(100.0, float(self.gate_standoff))),
            gate_roughness=max(0.0, min(0.5, float(self.gate_roughness))),
            gate_gain=max(0.0, min(1.0e6, float(self.gate_gain))),
        )


def _clamp_pow2(value: int, lo: int, hi: int) -> int:
    """Round ``value`` down to a power of two within ``[lo, hi]``.

    The C++ starburst FFT requires a power-of-two grid; snapping here keeps the
    dialog spinbox forgiving (any value maps to the nearest legal grid at or
    below it) rather than erroring in the renderer.
    """
    v = max(int(lo), min(int(hi), int(value)))
    p = int(lo)
    while p * 2 <= v:
        p *= 2
    return p


# The quality ladder is deliberately split in two.
#
# Draft -> Mid -> High vary ONE thing: how well the ghost pass is sampled
# (output resolution, ray grid, wavelength count). Every extra layer stays off,
# so stepping up the ladder makes the same picture cleaner instead of adding
# passes — which is what makes the three comparable while dialling a lens in.
# High also switches HURB on: it is not an additive layer but a refinement of
# the ghost pass itself (each ray passing a hard edge gets its physical
# diffraction kick), so it belongs with the highest-fidelity ghost render.
#
# High+ is High plus the three whole-frame extras — the aperture starburst, the
# veiling glare and the gate flare. They are the expensive, look-defining passes
# (and the starburst's FFT window can leave a visible square "box" on a sensor
# larger than the lens's image circle), so they arrive together, on request.
DRAFT_PRESET = RenderSettings(
    width_px=192, ray_grid=256, spectral_samples=4,
)
MID_PRESET = RenderSettings(
    width_px=224, ray_grid=384, spectral_samples=10,
)
HIGH_PRESET = RenderSettings(
    width_px=256, ray_grid=512, spectral_samples=16, hurb=True,
)
# High+ inherits High's sampling verbatim, so the only difference the user sees
# between the two is the extra layers — a High+ render can never be a WORSE
# ghost render than High.
HIGH_PLUS_PRESET = replace(
    HIGH_PRESET,
    starburst=True, starburst_engine="mdft", starburst_grid=2048,
    veil=True,
    gate=True,
)


# ---------------------------------------------------------------------------
# Exposure dialog (viewer gain in stops)
# ---------------------------------------------------------------------------


class ExposureDialog(QDialog):
    """Modeless dialog for the per-panel viewer exposure, in stops.

    Exposure is a linear pre-multiply (``* 2**stops``) applied *before* the
    designer-wide ACES view transform — identical to Nuke's Viewer gain. It is
    display-only and never fed to the renderer.

    ``stops_min`` / ``stops_max`` / ``stops_step`` override the default range
    and increment; ``snap_to_step=True`` additionally rounds every input —
    typed, scrubbed, or programmatic — to the nearest ``stops_step`` multiple.

    Signals:
        stopsChanged(float): emitted as the user adjusts the spinbox.
    """

    stopsChanged = Signal(float)

    def __init__(
        self,
        current_stops: float,
        parent: Optional[QWidget] = None,
        *,
        stops_min: float = EXPOSURE_STOPS_MIN,
        stops_max: float = EXPOSURE_STOPS_MAX,
        stops_step: float = EXPOSURE_STOPS_STEP,
        snap_to_step: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exposure")
        # Modeless so the user can adjust exposure while watching renders
        # update in the panel behind it.
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        spin_cls = _SnapMathDoubleSpinBox if snap_to_step else MathDoubleSpinBox
        self._spin = spin_cls(self)
        self._spin.setRange(stops_min, stops_max)
        self._spin.setSingleStep(stops_step)
        self._spin.setDecimals(2)
        self._spin.setSuffix(" st")
        self._spin.setValue(float(current_stops))
        self._spin.setToolTip(
            "Viewer exposure in stops (2^stops linear gain, applied before the "
            "ACES view transform). Ctrl+MMB to scrub at varying sensitivities."
        )
        # Signal-to-signal: valueChanged(float) -> stopsChanged.
        self._spin.valueChanged.connect(self.stopsChanged)
        attach_spinbox_scrubber(self._spin, label="Exposure (stops)")

        close_btn = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        close_btn.rejected.connect(self.close)

        form = QFormLayout()
        form.addRow("Exposure:", self._spin)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(close_btn)

    def set_stops(self, stops: float) -> None:
        """Reflect a programmatically-changed exposure (e.g. after the
        'Auto-Expose' action metered the latest frame). Suppresses
        stopsChanged so the callsite doesn't see its own echo."""
        self._spin.blockSignals(True)
        try:
            self._spin.setValue(float(stops))
        finally:
            self._spin.blockSignals(False)


# ---------------------------------------------------------------------------
# Render settings dialog
# ---------------------------------------------------------------------------


class RenderSettingsDialog(QDialog):
    """Modeless dialog for editing the per-panel render settings.

    Applies live: every widget change emits :attr:`settingsChanged` with a
    fresh :class:`RenderSettings`, so the panel re-renders as the user toggles
    options or scrubs values without having to close the dialog. It is also
    readable in one shot via :meth:`result_settings`.

    ``allow_diffraction_layers=False`` forces the two *additive* layers — the
    aperture starburst and the veiling glare — off and locks their controls.
    Panels that render the geometric ghost layer alone (the ghost explorer)
    pass it so the dialog can't offer a switch that the panel would override.

    Signals:
        settingsChanged(RenderSettings): emitted whenever any control changes.
    """

    settingsChanged = Signal(object)

    def __init__(
        self,
        current: RenderSettings,
        parent: Optional[QWidget] = None,
        *,
        allow_diffraction_layers: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Render Settings")
        # Modeless + Tool so the user can tweak settings and watch the panel
        # behind it re-render on each change (mirrors ExposureDialog).
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        # starburst_grid_cap has no widget in this dialog, so preserve the incoming
        # value verbatim through result_settings() — otherwise a live settings edit
        # would silently reset a user-raised cap to the RenderSettings default and
        # change the auto-extent grid growth mid-session.
        self._starburst_grid_cap = int(current.starburst_grid_cap)

        self._width = MathSpinBox(self)
        self._width.setRange(32, 1024)
        self._width.setSingleStep(16)
        self._width.setValue(int(current.width_px))
        self._width.setSuffix(" px")
        self._width.setToolTip(
            "Width of the rendered image in pixels. Height tracks the "
            "project sensor's aspect ratio. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._width, label="Width (px)")

        self._ray_grid = MathSpinBox(self)
        self._ray_grid.setRange(16, 2048)
        self._ray_grid.setSingleStep(16)
        self._ray_grid.setValue(int(current.ray_grid))
        self._ray_grid.setToolTip(
            "Square root of total rays per render. Higher → less noise, "
            "more GPU time. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._ray_grid, label="Ray Grid")

        self._spectral = MathSpinBox(self)
        self._spectral.setRange(1, 32)
        self._spectral.setValue(int(current.spectral_samples))
        self._spectral.setToolTip(
            "Number of wavelengths sampled. Higher → smoother chromatic "
            "dispersion, more GPU time. Ctrl+MMB to scrub.\n"
            "Note: exactly 3 selects a legacy RGB fast-path that bypasses the "
            "sampling accelerators below (culling / concentration / adaptive)."
        )
        attach_spinbox_scrubber(self._spectral, label="Spectral")

        # --- ghost sampling accelerators --------------------------------------
        self._cull = QCheckBox("Cull dead ghost pairs", self)
        self._cull.setChecked(bool(current.cull_dead_pairs))
        self._cull.setToolTip(
            "Skip ghost pairs whose rays never reach the sensor. Lossless "
            "speed-up; only turn off to debug a missing ghost."
        )

        self._concentrate = QCheckBox("Concentrate samples", self)
        self._concentrate.setChecked(bool(current.concentrate_samples))
        self._concentrate.setToolTip(
            "Lay each ghost's rays inside its survivor region instead of the "
            "whole pupil — much cleaner at the same ray grid. Off = classic "
            "full-pupil sampling (noisier)."
        )

        self._adaptive = QCheckBox("Adaptive sample budgets", self)
        self._adaptive.setChecked(bool(current.adaptive_sample_budgets))
        self._adaptive.setToolTip(
            "Scale each ghost's ray count to its concentrated size — harvests "
            "the oversampling of tiny ghosts as render speed. Requires "
            "concentration. Off = concentration quality at full budget (slower)."
        )

        self._boost = MathDoubleSpinBox(self)
        self._boost.setRange(1.0, 64.0)
        self._boost.setSingleStep(1.0)
        self._boost.setDecimals(1)
        self._boost.setValue(float(current.adaptive_density_boost))
        self._boost.setToolTip(
            "Adaptive quality dial: rays kept relative to the concentrated "
            "density. 8 = near-lossless vs concentration; lower = faster and "
            "grainier. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._boost, label="Adaptive Boost")

        # adaptive requires concentration; boost only applies to adaptive.
        self._concentrate.toggled.connect(self._sync_sampling_enabled)
        self._adaptive.toggled.connect(self._sync_sampling_enabled)

        # --- aperture-diffraction starburst -----------------------------------
        self._starburst = QCheckBox("Aperture starburst (diffraction)", self)
        self._starburst.setChecked(bool(current.starburst))
        self._starburst.setToolTip(
            "Render the aperture's Fraunhofer diffraction pattern (the sunstar) "
            "around the source as a separate additive layer. Uses the lens's "
            "stop shape and the calibrated f-number for a physically-scaled "
            "pattern."
        )

        self._sb_trim = MathDoubleSpinBox(self)
        self._sb_trim.setRange(1.0, 200.0)
        self._sb_trim.setSingleStep(1.0)
        self._sb_trim.setDecimals(1)
        self._sb_trim.setValue(float(current.starburst_scale_trim))
        self._sb_trim.setToolTip(
            "Artistic size multiplier. 1.0 = physically scaled, which at preview "
            "resolution is often only a few pixels across; raise to enlarge the "
            "star without changing its structure. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._sb_trim, label="Starburst Size")

        self._sb_gain = MathDoubleSpinBox(self)
        self._sb_gain.setRange(0.0, 1000.0)
        self._sb_gain.setSingleStep(0.5)
        self._sb_gain.setDecimals(2)
        self._sb_gain.setValue(float(current.starburst_gain))
        self._sb_gain.setToolTip(
            "Starburst brightness. 1.0 renders the source's diffracted image at "
            "its own flux; raise to emphasise the spikes. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._sb_gain, label="Starburst Gain")

        self._sb_spectral = MathSpinBox(self)
        self._sb_spectral.setRange(0, 64)
        self._sb_spectral.setValue(int(current.starburst_spectral))
        self._sb_spectral.setToolTip(
            "Wavelengths integrated for the starburst's chromatic dispersion. "
            "0 = inherit the spectral samples above; 16-32 give smooth rainbow "
            "spikes. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._sb_spectral, label="Starburst Spectral")

        self._sb_grid = QComboBox(self)
        for g in _STARBURST_GRID_CHOICES:
            self._sb_grid.addItem(str(g), g)
        cur_grid = _clamp_pow2(current.starburst_grid, 256, 4096)
        idx = self._sb_grid.findData(cur_grid)
        self._sb_grid.setCurrentIndex(idx if idx >= 0 else 1)  # default to 1024
        self._sb_grid.setToolTip(
            "Starburst FFT grid. The pattern is bounded by this square window, so "
            "a larger grid lets the spikes reach further before they clip at the "
            "edge — at roughly 4x the render cost per step up. 2048 (the High+ "
            "preset) roughly doubles the spike reach vs 1024."
        )

        self._sb_engine = QComboBox(self)
        for label, key in (("Sprite (fast)", "sprite"), ("MDFT (exact)", "mdft")):
            self._sb_engine.addItem(label, key)
        eng_idx = self._sb_engine.findData(
            current.starburst_engine if current.starburst_engine in ("sprite", "mdft")
            else "sprite")
        self._sb_engine.setCurrentIndex(eng_idx if eng_idx >= 0 else 0)  # default Sprite
        self._sb_engine.setToolTip(
            "Starburst engine. Sprite is the fast FFT-sprite path. MDFT evaluates "
            "the diffraction integral exactly at the sensor pixels — flux-matched "
            "to Sprite but resample-free, so a moving source stays crisp instead "
            "of shimmering. MDFT is what the High+ preset uses."
        )

        self._starburst.toggled.connect(self._sync_starburst_enabled)

        # --- physical veiling glare -------------------------------------------
        self._veil = QCheckBox("Veiling glare (physical glow)", self)
        self._veil.setChecked(bool(current.veil))
        self._veil.setToolTip(
            "Render the broad, soft halo a real lens throws around every bright "
            "source (surface-reflection / stray-light veiling glare) as an energy-"
            "conserving spectral glare-spread function. This is the physical glow "
            "that replaced the old box-blur 'haze'; it joins the metered flare "
            "layer, so auto-exposure accounts for it."
        )

        self._veil_gain = MathDoubleSpinBox(self)
        self._veil_gain.setRange(0.0, 1.0)
        self._veil_gain.setSingleStep(0.01)
        self._veil_gain.setDecimals(3)
        self._veil_gain.setValue(float(current.veil_gain))
        self._veil_gain.setToolTip(
            "Total halo energy as a fraction of the source's own flux (0.03 ~ a "
            "subtle 3% wash; raise for a hazier / older lens). Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._veil_gain, label="Veil Gain")

        self._veil_spread = MathDoubleSpinBox(self)
        self._veil_spread.setRange(1.0e-3, 4.0)
        self._veil_spread.setSingleStep(0.02)
        self._veil_spread.setDecimals(3)
        self._veil_spread.setValue(float(current.veil_spread))
        self._veil_spread.setToolTip(
            "Core radius of the halo as a fraction of the sensor half-height, so "
            "it scales with the frame. Larger washes more of the frame. Ctrl+MMB."
        )
        attach_spinbox_scrubber(self._veil_spread, label="Veil Spread")

        self._veil_falloff = MathDoubleSpinBox(self)
        self._veil_falloff.setRange(1.0, 3.0)
        self._veil_falloff.setSingleStep(0.1)
        self._veil_falloff.setDecimals(2)
        self._veil_falloff.setValue(float(current.veil_falloff))
        self._veil_falloff.setToolTip(
            "Radial falloff power of the glare tail: 1.0 = the broadest (pure-"
            "Lorentzian) wings; 1.5 ~ a ~1/r^3 tail; higher = a tighter halo. "
            "Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._veil_falloff, label="Veil Falloff")

        self._veil.toggled.connect(self._sync_veil_enabled)

        # --- physical ghost-edge diffraction: HURB ----------------------------
        self._hurb = QCheckBox("Ghost-edge glow (HURB)", self)
        self._hurb.setChecked(bool(current.hurb))
        self._hurb.setToolTip(
            "Bend each ghost ray that passes a hard edge (the stop, a surface rim, "
            "a matte-box / baffle plane) by a small random angle σ = λ·K/d — the "
            "soft chromatic glow a real lens throws around every edge. It is "
            "energy-conserving and chromatic (red spreads wider than blue). Off by "
            "default and zero-cost when off (the ghost kernel is compile-time "
            "templated on it), so it is a pure opt-in quality pass."
        )

        self._hurb_kick = QComboBox(self)
        for label, key in (("Lorentzian (glow tails)", "lorentzian"),
                           ("Gaussian (soft)", "gaussian")):
            self._hurb_kick.addItem(label, key)
        kick_idx = self._hurb_kick.findData(
            current.hurb_kick if current.hurb_kick in ("lorentzian", "gaussian")
            else "lorentzian")
        self._hurb_kick.setCurrentIndex(kick_idx if kick_idx >= 0 else 0)  # default Lorentzian
        self._hurb_kick.setToolTip(
            "HURB kick distribution. Lorentzian gives the heavy 1/θ² tails of real "
            "lens glare (hard-clamped so a near-grazing ray can't fly off); Gaussian "
            "is a softer, tighter falloff. Lorentzian is the physical default."
        )

        self._hurb.toggled.connect(self._sync_hurb_enabled)

        # --- film-gate flare (mechanical) -------------------------------------
        self._gate = QCheckBox("Gate flare", self)
        self._gate.setChecked(bool(current.gate))
        self._gate.setToolTip(
            "Scatter light off the aperture plate's cut edge — the streak a real "
            "camera throws in from the frame border when a source sits just "
            "outside it. Only fires when the source lands in a narrow band past "
            "the edge AND the lens covers more than the sensor uses; a source "
            "inside the frame produces exactly nothing. Off by default and "
            "byte-identical when off, since no pass runs at all."
        )

        self._gate_standoff = MathDoubleSpinBox(self)
        self._gate_standoff.setRange(0.0, 100.0)
        self._gate_standoff.setDecimals(2)
        self._gate_standoff.setSingleStep(0.5)
        self._gate_standoff.setSuffix(" mm")
        self._gate_standoff.setKeyboardTracking(False)
        self._gate_standoff.setValue(float(current.gate_standoff))
        self._gate_standoff.setToolTip(
            "Distance from the plate's rear edge to the sensor. This is the lever "
            "arm the scattered light travels, so it sets how far into frame the "
            "streak reaches — and it moves WHICH source positions fire, because "
            "the capture band sits that far past the edge. A film gate sits under "
            "a millimetre away and gives a hairline; a few mm gives the streak. "
            "Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._gate_standoff, label="Gate Standoff")

        self._gate_roughness = MathDoubleSpinBox(self)
        self._gate_roughness.setRange(0.0, 0.5)
        self._gate_roughness.setDecimals(3)
        self._gate_roughness.setSingleStep(0.01)
        self._gate_roughness.setKeyboardTracking(False)
        self._gate_roughness.setValue(float(current.gate_roughness))
        self._gate_roughness.setToolTip(
            "How rough the machined land is, in radians. At 0 the wall is a "
            "perfect mirror and folds a point source back to a point; raising it "
            "is what turns the fold into a streak. Direction-only, so it spreads "
            "the light rather than adding any. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._gate_roughness, label="Gate Roughness")

        self._gate_gain = MathDoubleSpinBox(self)
        self._gate_gain.setRange(0.0, 1.0e6)
        self._gate_gain.setDecimals(2)
        self._gate_gain.setSingleStep(1.0)
        self._gate_gain.setKeyboardTracking(False)
        self._gate_gain.setValue(float(current.gate_gain))
        self._gate_gain.setToolTip(
            "Multiplier on the gate layer's physical brightness. At 1.0 it carries "
            "its traced fraction of the source's flux, which is well under 1% — "
            "faint by nature, so this is what makes it readable. Ctrl+MMB to scrub."
        )
        attach_spinbox_scrubber(self._gate_gain, label="Gate Gain")

        self._gate.toggled.connect(self._sync_gate_enabled)

        # Panels that only render the geometric ghost layer lock the additive
        # layers off. Unchecking before the live-update wiring below means
        # result_settings() reports them off without emitting anything; the
        # dependent controls grey out via the _sync_* calls further down.
        if not allow_diffraction_layers:
            for check, why in (
                (self._starburst, "starburst"),
                (self._veil, "veiling glare"),
                (self._gate, "gate flare"),
            ):
                check.setChecked(False)
                check.setEnabled(False)
                check.setToolTip(
                    f"This panel renders the geometric ghost layer only, so the "
                    f"{why} pass is unavailable here. Use the Source Flare "
                    f"Renderer panel for the full composite."
                )

        form = QFormLayout()
        form.addRow("Resolution width:", self._width)
        form.addRow("Ray grid:", self._ray_grid)
        form.addRow("Spectral samples:", self._spectral)
        form.addRow(self._cull)
        form.addRow(self._concentrate)
        form.addRow(self._adaptive)
        form.addRow("Adaptive boost:", self._boost)
        form.addRow(self._starburst)
        form.addRow("Starburst engine:", self._sb_engine)
        form.addRow("Starburst size (×):", self._sb_trim)
        form.addRow("Starburst gain:", self._sb_gain)
        form.addRow("Starburst spectral:", self._sb_spectral)
        form.addRow("Starburst grid:", self._sb_grid)
        form.addRow(self._veil)
        form.addRow("Veil gain:", self._veil_gain)
        form.addRow("Veil spread:", self._veil_spread)
        form.addRow("Veil falloff:", self._veil_falloff)
        form.addRow(self._hurb)
        form.addRow("HURB kick:", self._hurb_kick)
        form.addRow(self._gate)
        form.addRow("Gate standoff:", self._gate_standoff)
        form.addRow("Gate roughness:", self._gate_roughness)
        form.addRow("Gate gain:", self._gate_gain)
        self._sync_sampling_enabled()
        self._sync_starburst_enabled()
        self._sync_veil_enabled()
        self._sync_hurb_enabled()
        self._sync_gate_enabled()

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
        # all initial setValue/setChecked/setCurrentIndex calls above, so the
        # construction doesn't fire spurious change signals.
        self._wire_live_updates()

    def _wire_live_updates(self) -> None:
        """Emit :attr:`settingsChanged` on any control change so the panel
        re-renders live. Spinboxes fire on ``valueChanged`` (which the value
        scrubber also drives, one step per drag notch), checkboxes on
        ``toggled``, combos on ``currentIndexChanged``."""
        for spin in (
            self._width, self._ray_grid, self._spectral, self._boost,
            self._sb_trim, self._sb_gain, self._sb_spectral,
            self._veil_gain, self._veil_spread, self._veil_falloff,
            self._gate_standoff, self._gate_roughness, self._gate_gain,
        ):
            spin.valueChanged.connect(self._emit_changed)
        for check in (
            self._cull, self._concentrate, self._adaptive, self._starburst,
            self._veil, self._hurb, self._gate,
        ):
            check.toggled.connect(self._emit_changed)
        for combo in (self._sb_engine, self._sb_grid, self._hurb_kick):
            combo.currentIndexChanged.connect(self._emit_changed)

    def _emit_changed(self, *_args) -> None:
        self.settingsChanged.emit(self.result_settings())

    def _sync_sampling_enabled(self) -> None:
        """Reflect the C++ dependency in the UI: adaptive budgets require
        concentration, and the boost dial only applies when adaptive is on."""
        concentrate = self._concentrate.isChecked()
        self._adaptive.setEnabled(concentrate)
        self._boost.setEnabled(concentrate and self._adaptive.isChecked())

    def _sync_starburst_enabled(self) -> None:
        """The starburst controls only apply when the starburst is on."""
        on = self._starburst.isChecked()
        self._sb_engine.setEnabled(on)
        self._sb_trim.setEnabled(on)
        self._sb_gain.setEnabled(on)
        self._sb_spectral.setEnabled(on)
        self._sb_grid.setEnabled(on)

    def _sync_veil_enabled(self) -> None:
        """The veil controls only apply when the veil is on."""
        on = self._veil.isChecked()
        self._veil_gain.setEnabled(on)
        self._veil_spread.setEnabled(on)
        self._veil_falloff.setEnabled(on)

    def _sync_hurb_enabled(self) -> None:
        """The HURB kick-distribution combo only applies when HURB is on."""
        self._hurb_kick.setEnabled(self._hurb.isChecked())

    def _sync_gate_enabled(self) -> None:
        """The gate controls only apply when the gate pass is on."""
        on = self._gate.isChecked()
        self._gate_standoff.setEnabled(on)
        self._gate_roughness.setEnabled(on)
        self._gate_gain.setEnabled(on)

    def result_settings(self) -> RenderSettings:
        return RenderSettings(
            width_px=int(self._width.value()),
            ray_grid=int(self._ray_grid.value()),
            spectral_samples=int(self._spectral.value()),
            cull_dead_pairs=bool(self._cull.isChecked()),
            concentrate_samples=bool(self._concentrate.isChecked()),
            adaptive_sample_budgets=bool(self._adaptive.isChecked()),
            adaptive_density_boost=float(self._boost.value()),
            starburst=bool(self._starburst.isChecked()),
            starburst_engine=str(self._sb_engine.currentData() or "sprite"),
            starburst_scale_trim=float(self._sb_trim.value()),
            starburst_gain=float(self._sb_gain.value()),
            starburst_spectral=int(self._sb_spectral.value()),
            starburst_grid=int(self._sb_grid.currentData()),
            starburst_grid_cap=int(self._starburst_grid_cap),
            veil=bool(self._veil.isChecked()),
            veil_gain=float(self._veil_gain.value()),
            veil_spread=float(self._veil_spread.value()),
            veil_falloff=float(self._veil_falloff.value()),
            hurb=bool(self._hurb.isChecked()),
            hurb_kick=str(self._hurb_kick.currentData() or "lorentzian"),
            gate=bool(self._gate.isChecked()),
            gate_standoff=float(self._gate_standoff.value()),
            gate_roughness=float(self._gate_roughness.value()),
            gate_gain=float(self._gate_gain.value()),
        ).clamp()
