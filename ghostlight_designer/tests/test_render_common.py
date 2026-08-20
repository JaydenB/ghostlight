"""Unit tests for the shared render machinery in
:mod:`ghostlight_designer.render_common` — the per-panel render settings, the
quality-preset ladder, and the two dialogs every flare panel opens.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QAbstractSpinBox, QApplication

from ghostlight_designer.render_common import (
    DRAFT_PRESET,
    HIGH_PLUS_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    ExposureDialog,
    RenderSettings,
    RenderSettingsDialog,
)


def _destroy(dlg) -> None:
    """Destroy a dialog deterministically inside the owning test.

    Its spinboxes carry scrubber plumbing (a hidden QTreeView + adapter model
    each); letting dialogs pile up in the deleteLater queue and die during a
    later test's event processing corrupts the heap on Windows/PySide6.
    """
    dlg.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()


# ------------------------------------------------------------------ settings
def test_render_settings_clamp_to_safe_ranges():
    """User-provided values are clamped so the dialog can't push the
    renderer into a degenerate state."""
    rs = RenderSettings(width_px=5, ray_grid=2, spectral_samples=0).clamp()
    assert rs.width_px >= 32
    assert rs.ray_grid >= 16
    assert rs.spectral_samples >= 1

    rs = RenderSettings(width_px=99999, ray_grid=99999, spectral_samples=99).clamp()
    assert rs.width_px <= 1024
    assert rs.ray_grid <= 2048
    assert rs.spectral_samples <= 32


def test_hurb_settings_roundtrip_and_clamp():
    """HURB is off in a bare RenderSettings with the Lorentzian kick; an
    unknown kick string clamps back to the physical Lorentzian default."""
    assert RenderSettings().hurb is False
    assert RenderSettings().hurb_kick == "lorentzian"
    assert RenderSettings(hurb=True, hurb_kick="gaussian").clamp().hurb_kick == "gaussian"
    assert RenderSettings(hurb_kick="bogus").clamp().hurb_kick == "lorentzian"


# ------------------------------------------------------------------- presets
def test_quality_ladder_gets_strictly_more_expensive():
    """Draft -> Mid -> High must climb along every sampling axis, otherwise a
    step of the ladder would be pointless."""
    for lo, hi in ((DRAFT_PRESET, MID_PRESET), (MID_PRESET, HIGH_PRESET)):
        assert lo.width_px <= hi.width_px
        assert lo.ray_grid < hi.ray_grid
        assert lo.spectral_samples < hi.spectral_samples


def test_the_ladder_differs_in_sampling_alone_up_to_high():
    """The whole point of Draft/Mid/High: stepping up makes the SAME picture
    cleaner. Nothing but the sampling knobs (and HURB, which refines the ghost
    pass rather than compositing over it) may vary across the three."""
    from dataclasses import fields, replace

    sampling = {"width_px", "ray_grid", "spectral_samples", "hurb"}
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET):
        normalized = replace(
            preset,
            **{f.name: getattr(DRAFT_PRESET, f.name)
               for f in fields(RenderSettings) if f.name in sampling},
        )
        assert normalized == DRAFT_PRESET, (
            f"preset differs from Draft outside the sampling knobs: {preset}"
        )


def test_no_ladder_preset_below_high_plus_runs_an_extra_layer():
    """The three whole-frame extras arrive together, in High+ only."""
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET):
        assert preset.starburst is False
        assert preset.veil is False
        assert preset.gate is False
    assert (HIGH_PLUS_PRESET.starburst, HIGH_PLUS_PRESET.veil,
            HIGH_PLUS_PRESET.gate) == (True, True, True)


def test_every_preset_clamps_to_itself():
    """A preset must never resurrect a heavier default when re-clamped."""
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET, HIGH_PLUS_PRESET):
        assert preset.clamp() == preset


# -------------------------------------------------------------------- dialog
def test_render_settings_dialog_is_modeless_and_emits_live(qapp):
    """The settings dialog applies live: it is modeless and emits
    settingsChanged with a fresh RenderSettings on every widget change."""
    dlg = RenderSettingsDialog(MID_PRESET)
    try:
        assert dlg.isModal() is False
        seen: list[RenderSettings] = []
        dlg.settingsChanged.connect(seen.append)

        # A spinbox change (the same setValue path the value scrubber drives).
        dlg._width.setValue(MID_PRESET.width_px + 32)
        assert seen, "changing a spinbox should emit settingsChanged"
        assert seen[-1].width_px == MID_PRESET.width_px + 32

        # A checkbox toggle also emits.
        before = len(seen)
        dlg._starburst.setChecked(not MID_PRESET.starburst)
        assert len(seen) > before
        assert seen[-1].starburst == (not MID_PRESET.starburst)

        # A combo change (HURB kick) also emits.
        before = len(seen)
        dlg._hurb_kick.setCurrentIndex(1)
        assert len(seen) > before
    finally:
        _destroy(dlg)


def test_every_numeric_control_has_a_scrubber(qapp):
    """The value scrubber must cover every numeric spinbox in the popup —
    each attaches a _SpinBoxScrubTrigger event filter to itself as a child."""
    from ghostlight_designer.render_common.spinbox_scrub import _SpinBoxScrubTrigger

    dlg = RenderSettingsDialog(MID_PRESET)
    try:
        spinboxes = dlg.findChildren(QAbstractSpinBox)
        assert spinboxes, "dialog should contain spinboxes"
        for spin in spinboxes:
            triggers = spin.findChildren(_SpinBoxScrubTrigger)
            assert triggers, f"{spin.objectName() or spin} has no value scrubber"
    finally:
        _destroy(dlg)


def test_exposure_dialog_defaults_keep_the_shared_behaviour(qapp):
    """The sourceflare panel opts into half-stop snapping / a 90 st cap;
    the shared defaults (used by the ghost explorer) must stay +/-20 st in
    0.25 steps with no snapping."""
    dlg = ExposureDialog(0.0)
    try:
        spin = dlg._spin
        assert spin.minimum() == pytest.approx(-20.0)
        assert spin.maximum() == pytest.approx(20.0)
        assert spin.singleStep() == pytest.approx(0.25)
        spin.setValue(1.3)  # off any snap grid: must be kept verbatim
        assert spin.value() == pytest.approx(1.3)
    finally:
        _destroy(dlg)
