"""Designer wiring for the film-gate flare layer.

Adding a control to RenderSettingsDialog is a six-step contract — dataclass
field, clamp, widget, form row, live-update wiring, result_settings read-back —
and missing any one of them leaves the control silently dead. These cover the
two steps a reviewer cannot see by reading the constructor: that each control
actually emits, and that its value survives the round trip back out.
"""

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from ghostlight_designer.render_common import (
    DRAFT_PRESET,
    HIGH_PLUS_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    RenderSettings,
    RenderSettingsDialog,
)


def _destroy(dlg) -> None:
    """Destroy a dialog deterministically inside the owning test.

    The gate spinboxes carry scrubber plumbing (a hidden QTreeView + adapter
    model each); letting dialogs pile up in the deleteLater queue and die during
    a later test's event processing corrupts the heap on Windows/PySide6.
    """
    dlg.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()


# ------------------------------------------------------------------- defaults
def test_gate_is_off_in_every_preset_but_high_plus():
    """Off everywhere on the plain quality ladder, so stepping Draft -> Mid ->
    High never silently starts a pass that makes the render non-byte-identical.
    High+ is the one preset that opts in."""
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET):
        assert preset.gate is False
    assert HIGH_PLUS_PRESET.gate is True


def test_gate_defaults_match_the_validated_values():
    s = RenderSettings()
    assert s.gate is False
    assert s.gate_standoff == pytest.approx(5.0)
    assert s.gate_roughness == pytest.approx(0.08)
    assert s.gate_gain == pytest.approx(1.0)


def test_gate_settings_clamp():
    s = RenderSettings(gate_standoff=-5.0, gate_roughness=99.0, gate_gain=-2.0).clamp()
    assert s.gate_standoff == pytest.approx(0.0)
    assert s.gate_roughness == pytest.approx(0.5)
    assert s.gate_gain == pytest.approx(0.0)


def test_clamp_is_idempotent_for_every_preset():
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET, HIGH_PLUS_PRESET):
        assert preset.clamp() == preset


# ---------------------------------------------------------------------- dialog
def test_gate_controls_emit_live(qapp):
    dlg = RenderSettingsDialog(MID_PRESET)
    try:
        seen = []
        dlg.settingsChanged.connect(seen.append)

        dlg._gate.setChecked(True)
        assert seen and seen[-1].gate is True

        before = len(seen)
        dlg._gate_standoff.setValue(9.0)
        assert len(seen) > before
        assert seen[-1].gate_standoff == pytest.approx(9.0)

        before = len(seen)
        dlg._gate_roughness.setValue(0.2)
        assert len(seen) > before
        assert seen[-1].gate_roughness == pytest.approx(0.2)

        before = len(seen)
        dlg._gate_gain.setValue(4.0)
        assert len(seen) > before
        assert seen[-1].gate_gain == pytest.approx(4.0)
    finally:
        _destroy(dlg)


def test_gate_controls_grey_out_when_the_pass_is_off(qapp):
    dlg = RenderSettingsDialog(MID_PRESET)
    try:
        dlg._gate.setChecked(False)
        assert not dlg._gate_standoff.isEnabled()
        assert not dlg._gate_roughness.isEnabled()
        assert not dlg._gate_gain.isEnabled()

        dlg._gate.setChecked(True)
        assert dlg._gate_standoff.isEnabled()
        assert dlg._gate_roughness.isEnabled()
        assert dlg._gate_gain.isEnabled()
    finally:
        _destroy(dlg)


def test_gate_round_trips_through_result_settings(qapp):
    src = RenderSettings(gate=True, gate_standoff=7.5, gate_roughness=0.15,
                         gate_gain=3.0)
    dlg = RenderSettingsDialog(src)
    try:
        out = dlg.result_settings()
        assert out.gate is True
        assert out.gate_standoff == pytest.approx(7.5)
        assert out.gate_roughness == pytest.approx(0.15)
        assert out.gate_gain == pytest.approx(3.0)
    finally:
        _destroy(dlg)


def test_ghost_only_panels_lock_the_gate_off(qapp):
    """The ghost explorer renders one pair's geometric contribution, so a
    whole-frame pass belonging to no pair must be unavailable there."""
    dlg = RenderSettingsDialog(RenderSettings(gate=True),
                               allow_diffraction_layers=False)
    try:
        assert dlg._gate.isChecked() is False
        assert dlg._gate.isEnabled() is False
        assert dlg.result_settings().gate is False
    finally:
        _destroy(dlg)


def test_strip_layers_removes_the_gate():
    from ghostlight_designer.ghost_explorer_panel.body import _strip_layers

    stripped = _strip_layers(RenderSettings(starburst=True, veil=True, gate=True))
    assert stripped.starburst is False
    assert stripped.veil is False
    assert stripped.gate is False
