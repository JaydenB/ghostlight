"""Extended tests for LensCalibration."""

import math
import pytest
import ghostlight


# ---------------------------------------------------------------------------
# Sensor dimensions
# ---------------------------------------------------------------------------

def test_sensor_half_w_positive(simple_lens):
    cal = simple_lens.calibration()
    assert cal.sensor_half_w > 0.0


def test_sensor_half_h_positive(simple_lens):
    cal = simple_lens.calibration()
    assert cal.sensor_half_h > 0.0


def test_sensor_half_approx_equal_circular_aperture(simple_lens):
    """For a circular-aperture lens, sensor half-width ≈ half-height."""
    cal = simple_lens.calibration()
    ratio = cal.sensor_half_w / cal.sensor_half_h
    assert ratio == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# AOV / field angle
# ---------------------------------------------------------------------------

def test_aov_positive(simple_lens):
    cal = simple_lens.calibration()
    assert cal.max_half_angle_h > 0.0
    assert cal.max_half_angle_v > 0.0


def test_aov_less_than_90_degrees(simple_lens):
    """Half-angle must be less than 90° (π/2 rad) for a real lens."""
    cal = simple_lens.calibration()
    assert cal.max_half_angle_h < math.pi / 2
    assert cal.max_half_angle_v < math.pi / 2


def test_aov_reasonable_range(simple_lens):
    """Half-angle must be > 0.5° (0.009 rad) for any usable lens."""
    cal = simple_lens.calibration()
    assert cal.max_half_angle_h > 0.009


# ---------------------------------------------------------------------------
# Complex lens calibration
# ---------------------------------------------------------------------------

def test_doublegauss_calibration_succeeds(doublegauss_lens):
    """doublegauss.lens must calibrate without error."""
    cal = doublegauss_lens.calibration()
    assert isinstance(cal, ghostlight.LensCalibration)
    assert cal.sensor_half_w > 0.0


def test_cooketriplet_calibration_succeeds(cooketriplet_lens):
    cal = cooketriplet_lens.calibration()
    assert isinstance(cal, ghostlight.LensCalibration)
    assert cal.sensor_half_w > 0.0
