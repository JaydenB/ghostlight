"""Tests for the vignette overlay: the pure mask helpers, the controller's
launch/gating bookkeeping (no GPU), and — when a lens + CUDA are present — one
real end-to-end probe.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PySide6.QtGui import QImage

from ghostlight_designer import vignette
from ghostlight_designer.render_common import (
    DRAFT_PRESET,
    HIGH_PLUS_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    RenderSettings,
)
from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import SensorSettings

from _corpus import EXAMPLE_DOUBLET


def _example_lens_path() -> pathlib.Path:
    return EXAMPLE_DOUBLET


# ---------------------------------------------------------------------------
# The starburst is a High+ layer, not a default
# ---------------------------------------------------------------------------

def test_starburst_only_ships_in_the_high_plus_preset():
    """The quality ladder keeps the extra layers out of Draft/Mid/High so those
    three differ in ghost sampling alone; High+ is what turns them on."""
    assert RenderSettings().starburst is False
    for preset in (DRAFT_PRESET, MID_PRESET, HIGH_PRESET):
        assert preset.starburst is False
    assert HIGH_PLUS_PRESET.starburst is True


# ---------------------------------------------------------------------------
# Grid + seed geometry
# ---------------------------------------------------------------------------

def test_choose_grid_long_axis_and_squareness():
    # Landscape: long axis is nx = GRID_LONG_CELLS, cells stay ~square.
    nx, ny = vignette._choose_grid(18.0, 12.0)
    assert nx == vignette.GRID_LONG_CELLS
    assert ny == round(vignette.GRID_LONG_CELLS * 12.0 / 18.0)
    # Portrait: long axis flips to ny.
    nx, ny = vignette._choose_grid(6.0, 12.0)
    assert ny == vignette.GRID_LONG_CELLS
    assert nx == round(vignette.GRID_LONG_CELLS * 6.0 / 12.0)
    # Never collapses below the floor.
    nx, ny = vignette._choose_grid(100.0, 1.0)
    assert ny >= vignette.GRID_MIN_CELLS


class _FakeCalib:
    sensor_half_w = 10.0
    sensor_half_h = 10.0
    max_half_angle_h = 0.2
    max_half_angle_v = 0.2


def test_cell_targets_shape_and_raster_order():
    nx, ny = 8, 6
    tgt = vignette._cell_targets(nx, ny, 10.0, 10.0)
    assert tgt.shape == (nx * ny, 2)
    assert tgt.dtype == np.float32
    # Row 0 = top = -y ⇒ first row is negative in y; last row positive.
    assert tgt[0, 1] < 0.0
    assert tgt[-1, 1] > 0.0
    # Col 0 = left = -x ⇒ first cell negative in x.
    assert tgt[0, 0] < 0.0


def test_seeds_without_a_lens_fall_back_to_the_closed_form():
    """`system=None` (no lens to trace) must still produce usable seeds.

    The traced map needs a lens; the closed form keeps the overlay usable when
    no lens is available.
    """
    calib = _FakeCalib()
    tgt = vignette._cell_targets(8, 6, 10.0, 10.0)
    seeds = vignette._build_seeds(tgt, None, calib, 10.0, 10.0)
    assert seeds.shape == tgt.shape
    assert seeds.dtype == np.float32
    assert seeds[0, 0] < 0.0 and seeds[0, 1] < 0.0
    assert seeds[-1, 1] > 0.0


def test_traced_seed_grid_runs_without_falling_back(caplog, monkeypatch):
    """With a real lens the traced grid must actually be solved.

    `_build_seeds` degrades to the closed form on any exception and only logs
    it. The early-out for well-behaved lenses is disabled here to force the grid path; then
    the only way to reach the closed form is the failure guard, and the log
    is what proves it did not fire.
    """
    import logging

    import ghostlight

    lens_path = _example_lens_path()
    if not lens_path.exists():
        pytest.skip("sample lens not present")
    try:
        system = ghostlight.OpticalSystem.load(str(lens_path))
        system._check_invalidate()
        calib = system.calibration()
    except Exception:
        pytest.skip("lens failed to calibrate")

    monkeypatch.setattr(vignette, "_MAP_CORRECTION_FRAC", 0.0)
    half_w = float(calib.sensor_half_w)
    half_h = float(calib.sensor_half_h)
    tgt = vignette._cell_targets(8, 6, half_w, half_h)

    with caplog.at_level(logging.ERROR, logger="ghostlight_designer.vignette"):
        seeds = vignette._build_seeds(tgt, system, calib, half_w, half_h)
    assert not caplog.records, (
        "traced seed grid fell back to the closed form: "
        + "; ".join(r.getMessage() for r in caplog.records)
    )

    assert seeds.shape == tgt.shape
    assert seeds.dtype == np.float32
    assert np.isfinite(seeds).all()
    # Interpolated from a solved grid, so it must still be a sane field map:
    # monotonically signed across the frame and inside a half-turn.
    assert seeds[0, 0] < 0.0 and seeds[0, 1] < 0.0
    assert seeds[-1, 0] > 0.0 and seeds[-1, 1] > 0.0
    assert float(np.abs(seeds).max()) < np.pi / 2


def test_seeds_unclamped_past_calibrated_field():
    """A sensor larger than the calibrated extent must seed field angles
    steeper than max_half_angle — the whole point of the un-clamped probe."""
    calib = _FakeCalib()
    half_w = 3.0 * calib.sensor_half_w
    tgt = vignette._cell_targets(20, 1, half_w, 10.0)
    seeds = vignette._build_seeds(tgt, None, calib, half_w, 10.0)
    assert float(np.abs(seeds[:, 0]).max()) > calib.max_half_angle_h


# ---------------------------------------------------------------------------
# Mask → QImage packing
# ---------------------------------------------------------------------------

def test_mask_to_qimage_none_when_nothing_vignetted():
    assert vignette.mask_to_qimage(None) is None
    assert vignette.mask_to_qimage(np.zeros((4, 5), dtype=bool)) is None


def test_mask_to_qimage_red_with_alpha_where_vignetted():
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True   # top-left cell vignetted
    img = vignette.mask_to_qimage(mask)
    assert isinstance(img, QImage)
    assert img.width() == 4 and img.height() == 3
    r, g, b, a = vignette.VIGNETTE_RGBA
    # Vignetted pixel: solid red, half alpha.
    px = img.pixelColor(0, 0)
    assert (px.red(), px.green(), px.blue(), px.alpha()) == (r, g, b, a)
    # Clear pixel: fully transparent (RGB is still red so smooth-scaling the
    # alpha channel doesn't bleed a dark fringe, but alpha is 0).
    clear = img.pixelColor(3, 2)
    assert clear.alpha() == 0


# ---------------------------------------------------------------------------
# Controller gating (no GPU — thread launch is stubbed / lens calib is CPU)
# ---------------------------------------------------------------------------

class _StubCanvas:
    def __init__(self):
        self.visible = None
        self.images = []

    def set_vignette_visible(self, v):
        self.visible = v

    def set_vignette_image(self, img):
        self.images.append(img)


class _FakeThread:
    launches = 0

    def __init__(self, *_a, **_k):
        _FakeThread.launches += 1

    def start(self):
        pass


@pytest.fixture
def _eligible_project():
    lens_path = _example_lens_path()
    if not lens_path.exists():
        pytest.skip("sample lens not present")
    project = Project()
    project.load(str(lens_path))
    project.system_setup.sensor = SensorSettings(
        width_mm=36.0, height_mm=24.0, preset_name="Full Frame",
    )
    return project


def test_set_enabled_toggles_canvas_visibility(qapp, _eligible_project, monkeypatch):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(_eligible_project, canvas)
    ctrl.set_enabled(True)
    assert canvas.visible is True
    ctrl.set_enabled(False)
    assert canvas.visible is False


def test_disabled_controller_never_launches(qapp, _eligible_project, monkeypatch):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    _FakeThread.launches = 0
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(_eligible_project, canvas)
    ctrl.set_active(True)
    for _ in range(10):
        ctrl.invalidate()
    assert _FakeThread.launches == 0


def test_enabled_active_burst_collapses_to_one_inflight(
    qapp, _eligible_project, monkeypatch
):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    _FakeThread.launches = 0
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(_eligible_project, canvas)
    ctrl.set_active(True)
    ctrl.set_enabled(True)  # first launch
    for _ in range(20):
        ctrl.invalidate()   # busy → all collapse into one pending slot
    assert _FakeThread.launches == 1
    assert ctrl._busy is True
    assert ctrl._pending is True


def test_ineligible_lens_clears_overlay_and_does_not_launch(qapp, monkeypatch):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    _FakeThread.launches = 0
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(Project(), canvas)  # empty system
    ctrl.set_active(True)
    ctrl.set_enabled(True)
    assert _FakeThread.launches == 0
    assert canvas.images and canvas.images[-1] is None


def test_poll_applies_matching_epoch_result_to_canvas(
    qapp, _eligible_project, monkeypatch
):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(_eligible_project, canvas)
    ctrl.set_active(True)
    ctrl.set_enabled(True)
    epoch = ctrl._epoch
    dummy = QImage(2, 2, QImage.Format_RGBA8888)
    ctrl._results.put((epoch, dummy, 3, 100))
    got = {}
    ctrl.resultReady.connect(lambda nd, nt: got.update(nd=nd, nt=nt))
    ctrl._poll()
    assert canvas.images[-1] is dummy
    assert got == {"nd": 3, "nt": 100}


def test_poll_drops_stale_epoch_result(qapp, _eligible_project, monkeypatch):
    monkeypatch.setattr(vignette.threading, "Thread", _FakeThread)
    canvas = _StubCanvas()
    ctrl = vignette.VignetteController(_eligible_project, canvas)
    ctrl.set_active(True)
    ctrl.set_enabled(True)
    dummy = QImage(2, 2, QImage.Format_RGBA8888)
    ctrl._results.put((ctrl._epoch - 999, dummy, 1, 1))  # stale
    ctrl._poll()
    # Nothing new applied (only whatever the launch itself may have cleared).
    assert dummy not in canvas.images


# ---------------------------------------------------------------------------
# End-to-end probe (needs a lens + CUDA) — mirrors the manual validation
# ---------------------------------------------------------------------------

def test_real_probe_marks_dark_outside_image_circle():
    """An oversized sensor pushes the corners past the lens's image circle, so
    the probe must flag some cells DARK; a sensor within the circle flags none.
    Skips cleanly without a lens / GPU."""
    import ghostlight

    lens_path = _example_lens_path()
    if not lens_path.exists():
        pytest.skip("sample lens not present")
    try:
        system = ghostlight.OpticalSystem.load(str(lens_path))
        system._check_invalidate()
        calib = system.calibration()
    except Exception:
        pytest.skip("lens failed to calibrate")

    cal_hw = float(calib.sensor_half_w)
    cal_hh = float(calib.sensor_half_h)

    # Sensor well inside the calibrated image extent → expect no vignetting.
    plan = vignette.plan_vignette(calib, cal_hw * 0.5, cal_hh * 0.5)
    assert plan is not None
    nx, ny, targets = plan
    try:
        mask_in = vignette.run_vignette(system, nx, ny, targets, calib,
                                        cal_hw * 0.5, cal_hh * 0.5)
    except Exception:
        pytest.skip("GPU probe unavailable")
    if mask_in is None:
        pytest.skip("GPU probe unavailable")
    assert int(mask_in.sum()) == 0

    # Sensor far larger than the image circle → corners must go dark.
    plan = vignette.plan_vignette(calib, cal_hw * 4.0, cal_hh * 4.0)
    nx, ny, targets = plan
    mask_out = vignette.run_vignette(system, nx, ny, targets, calib,
                                     cal_hw * 4.0, cal_hh * 4.0)
    assert mask_out is not None and int(mask_out.sum()) > 0
    # Corner is dark, centre is reachable.
    assert bool(mask_out[0, 0]) is True
    assert bool(mask_out[ny // 2, nx // 2]) is False
