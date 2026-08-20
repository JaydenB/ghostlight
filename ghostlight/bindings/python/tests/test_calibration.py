"""Tests for LensCalibration and the Lens cache."""

import pytest
import ghostlight


def test_calibrate_returns_calibration(simple_lens):
    cal = simple_lens.calibration()
    assert isinstance(cal, ghostlight.LensCalibration)


def test_calibrate_sensor_half_positive(simple_lens):
    cal = simple_lens.calibration()
    assert cal.sensor_half_w > 0.0
    assert cal.sensor_half_h > 0.0


def test_calibrate_aov_positive(simple_lens):
    cal = simple_lens.calibration()
    assert cal.max_half_angle_h > 0.0
    assert cal.max_half_angle_v > 0.0


def test_calibration_cached(simple_lens):
    cal1 = simple_lens.calibration()
    cal2 = simple_lens.calibration()
    assert cal1 is cal2, "second call should return the same cached object"


def test_calibration_invalidated_on_surface_mutation(simple_lens):
    cal1 = simple_lens.calibration()
    # Mutate a surface to invalidate the geometry hash
    simple_lens.surfaces[0].radius = 999.0
    simple_lens.finalize()
    cal2 = simple_lens.calibration()
    assert cal1 is not cal2, "cache should have been invalidated"


def test_calibration_args_invalidate(simple_lens):
    cal1 = simple_lens.calibration(d_line_nm=550.0)
    cal2 = simple_lens.calibration(d_line_nm=587.56)
    # Different args -> different calibration
    assert cal1 is not cal2


def test_build_spectral_lambdas():
    lambdas = ghostlight.build_spectral_lambdas(10, 400.0, 700.0)
    assert len(lambdas) == 10
    assert all(400.0 <= lam <= 700.0 for lam in lambdas)
    # Cell-centred spacing: first should be 415, last 685
    assert lambdas[0] == pytest.approx(415.0)
    assert lambdas[-1] == pytest.approx(685.0)


def test_enumerate_ghost_pairs(simple_lens):
    pairs = simple_lens.ghost_pairs()
    assert len(pairs) > 0
    for p in pairs:
        assert 0 <= p.surf_a < p.surf_b < simple_lens.num_surfaces()
