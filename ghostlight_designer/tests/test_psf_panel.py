"""Tests for the PSF panel — sensor-cell layout, raster orientation, and the
vignette overlay driven by the renderer's per-cell status.

Mirrors the flare panel tests: pure plumbing is checked without a
GPU render; one integration test does a real render and is skipped when no CUDA
device is present.
"""
from __future__ import annotations

import math
import pathlib
import time

import numpy as np
import pytest

import ghostlight
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage

from ghostlight_designer.project import Project
from ghostlight_designer.system_setup_data import SensorSettings
from ghostlight_designer.psf_panel.body import PSFCanvas, PSFPanelBody, _build_cells

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_DOUBLEGAUSS = _ROOT / "lenses" / "DoubleGauss.lens"


@pytest.fixture
def dg_calib():
    if not _DOUBLEGAUSS.exists():
        pytest.skip("DoubleGauss.lens not present")
    return ghostlight.OpticalSystem.load(str(_DOUBLEGAUSS)).calibration()


# ---------------------------------------------------------------- cell layout

def test_build_cells_raster_order(dg_calib):
    """Row 0 = top of frame = -y_mm; col 0 = left = -x_mm (raster convention)."""
    seeds, targets = _build_cells(5, 5, dg_calib, 12.0, 9.0, 1.0)
    assert seeds.shape == (25, 2) and targets.shape == (25, 2)
    # Top-left cell (index 0) is the most negative corner.
    assert targets[0, 0] < 0 and targets[0, 1] < 0
    # Top-right cell (index 4) is +x, still -y (top row).
    assert targets[4, 0] > 0 and targets[4, 1] < 0
    # Bottom row (index 20..24) is +y.
    assert targets[24, 1] > 0
    # Odd grid → centre cell on-axis.
    assert abs(targets[12, 0]) < 1e-6 and abs(targets[12, 1]) < 1e-6


def test_build_cells_partition_matches_sensor(dg_calib):
    """Targets are the cell centres of an even partition of the sensor rect."""
    half_w, half_h, n = 12.0, 9.0, 5
    seeds, targets = _build_cells(n, n, dg_calib, half_w, half_h, 1.0)
    pitch_x = 2 * half_w / n
    # Corner cell centre sits half a pitch inside the edge.
    assert targets[0, 0] == pytest.approx(-half_w + pitch_x / 2, rel=1e-5)
    # Cell x-positions are uniformly spaced.
    xs = targets[:n, 0]
    assert np.allclose(np.diff(xs), pitch_x, rtol=1e-5)


def test_build_cells_seed_is_tan_linear(dg_calib):
    """Each seed is the tan-linear inverse of its target (the canonical anchor)."""
    seeds, targets = _build_cells(3, 3, dg_calib, 10.0, 8.0, 1.0)
    cal_hw = float(dg_calib.sensor_half_w)
    tan_h = math.tan(float(dg_calib.max_half_angle_h))
    for i in range(9):
        want = math.atan((targets[i, 0] / cal_hw) * tan_h)
        assert seeds[i, 0] == pytest.approx(want, abs=1e-6)


def test_build_cells_field_fraction_shrinks(dg_calib):
    """field_fraction < 1 zooms the grid toward the centre."""
    _, full = _build_cells(5, 5, dg_calib, 12.0, 9.0, 1.0)
    _, half = _build_cells(5, 5, dg_calib, 12.0, 9.0, 0.5)
    assert abs(half[0, 0]) == pytest.approx(abs(full[0, 0]) * 0.5, rel=1e-5)


# ------------------------------------------------------------- overlay paint

def _paint_canvas(canvas, size=200) -> np.ndarray:
    img = QImage(QSize(size, size), QImage.Format_RGB888)
    img.fill(0)
    canvas.render(img)
    buf = img.constBits().tobytes()
    return np.frombuffer(buf, np.uint8).reshape(size, size, 3)


def test_canvas_vignette_overlay_paints_red(qapp):
    """A cell flagged != OK gets a red tint; an all-OK grid stays untinted."""
    canvas = PSFCanvas()
    canvas.resize(200, 200)
    # 2x2 grid, one dark cell (index 3).
    dummy = QImage(QSize(64, 64), QImage.Format_RGB888)
    dummy.fill(0)
    status = np.array([0, 0, 0, int(ghostlight.PSFCellStatus.DARK)], dtype=np.uint8)
    frac = np.array([0.3, 0.3, 0.3, 0.0], dtype=np.float32)
    canvas.set_image(dummy, 2, 2, 32, 32, status=status, pupil_fraction=frac)
    arr = _paint_canvas(canvas)
    # Red channel should dominate somewhere (the tinted cell).
    red_px = ((arr[:, :, 0] > 90) & (arr[:, :, 1] < 70) & (arr[:, :, 2] < 70)).sum()
    assert red_px > 100, "vignette tint not painted"

    # All-OK grid: no red tint.
    canvas.set_image(dummy, 2, 2, 32, 32,
                     status=np.zeros(4, np.uint8),
                     pupil_fraction=np.full(4, 0.3, np.float32))
    arr2 = _paint_canvas(canvas)
    red_px2 = ((arr2[:, :, 0] > 90) & (arr2[:, :, 1] < 70) & (arr2[:, :, 2] < 70)).sum()
    assert red_px2 == 0, "untinted grid should have no red"
    canvas.deleteLater()


def test_canvas_no_overlay_without_status(qapp):
    """Legacy render (status=None) paints no overlay and does not crash."""
    canvas = PSFCanvas()
    canvas.resize(200, 200)
    dummy = QImage(QSize(64, 64), QImage.Format_RGB888)
    dummy.fill(0)
    canvas.set_image(dummy, 2, 2, 32, 32)  # no status
    arr = _paint_canvas(canvas)
    red_px = ((arr[:, :, 0] > 90) & (arr[:, :, 1] < 70) & (arr[:, :, 2] < 70)).sum()
    assert red_px == 0
    canvas.deleteLater()


# ------------------------------------------------- display reapply (no GPU)

def test_reapply_display_uses_shared_qimage_encoder(qapp, isolated_settings):
    """The tone-map/per-tile toggle re-processes the cached composite through
    the shared ``viewtransform.to_qimage`` — with no GPU render. Guards the
    ``_reapply_display`` path (which the GPU test doesn't exercise)."""
    project = Project()
    body = PSFPanelBody(project, isolated_settings)
    try:
        grid_nx = grid_ny = 2
        tile_w = tile_h = 8
        comp = np.random.rand(grid_ny * tile_h, grid_nx * tile_w, 3).astype(np.float32)
        body._latest_comp = comp
        body._latest_dims = (grid_nx, grid_ny, tile_w, tile_h)
        body._latest_status = None
        body._latest_frac = None

        # Changing the display log-gain triggers _reapply_display -> to_qimage.
        body.set_log_gain(2.0)
        assert body._canvas._image is not None
        # Per-tile toggle exercises the same path.
        body.set_per_tile_norm(True)
        assert body._canvas._image is not None
    finally:
        body.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------- GPU integration

def _render_sync(body, qapp, timeout=30.0):
    prev = id(body._latest_status)
    body.force_render_now()
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        body._poll_results()
        if body._latest_status is not None and id(body._latest_status) != prev:
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(not ghostlight._cuda_available(), reason="no CUDA GPU")
def test_panel_render_flows_status_and_counts_vignetting(qapp, isolated_settings):
    if not _DOUBLEGAUSS.exists():
        pytest.skip("DoubleGauss.lens not present")
    project = Project()
    project.load(str(_DOUBLEGAUSS))
    body = PSFPanelBody(project, isolated_settings)
    try:
        body._is_active = True
        # A frame far wider than the lens can cover.  double_gauss images out to
        # roughly 38 mm off axis; at the default 5x5 grid and 0.7 field fraction
        # this puts the corner cells near 48 mm, where no ray reaches the sensor.
        # (IMAX-70 used to serve here, on the assumption that its corners fell
        # outside a ~21.7 mm image circle.  That figure is the calibrated covered
        # field, which measures axial pupil walk rather than the reach of the
        # lens; IMAX corners land near 24 mm and are comfortably imaged.)
        project.system_setup.sensor = SensorSettings(
            width_mm=140.0, height_mm=97.0, preset_name="Custom")
        project.mark_system_setup_modified()
        assert _render_sync(body, qapp), "render did not complete"
        assert body._latest_status is not None
        n_big = int((body._latest_status != int(ghostlight.PSFCellStatus.OK)).sum())
        assert n_big > 0
        assert "vignetted" in body._status.text()

        # Super-35 sits well inside the lens's reach → far fewer flagged cells.
        project.system_setup.sensor = SensorSettings(
            width_mm=24.89, height_mm=18.66, preset_name="Custom")
        project.mark_system_setup_modified()
        assert _render_sync(body, qapp)
        n_small = int((body._latest_status != int(ghostlight.PSFCellStatus.OK)).sum())
        assert n_small < n_big
    finally:
        body.deleteLater()
        qapp.processEvents()
