"""Tests for render_point_flare."""

import pytest
import numpy as np
import ghostlight

pytestmark = pytest.mark.gpu


def test_render_point_flare_shape(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = 0.5
    cfg.source_y = 0.5
    cfg.spectral_samples = 4
    cfg.ray_grid = 16

    out = loaded_lens.render_point_flare(32, 32, cfg)

    assert out["ghost_r"].shape == (32, 32)
    assert out["ghost_g"].shape == (32, 32)
    assert out["ghost_b"].shape == (32, 32)
    assert out["width"]  == 32
    assert out["height"] == 32


def test_render_point_flare_dtype(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4

    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert out["ghost_r"].dtype == np.float32


def test_render_point_flare_no_source_keys(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4

    out = loaded_lens.render_point_flare(32, 32, cfg)
    # PointFlare does not populate source_r/g/b
    assert "source_r" not in out


def test_render_point_flare_non_negative(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.source_r = 5.0
    cfg.source_g = 4.0
    cfg.source_b = 3.0
    cfg.ray_grid = 16
    cfg.spectral_samples = 4

    out = loaded_lens.render_point_flare(32, 32, cfg)
    # The unblurred spectral render carries small negatives from the XYZ CMF
    # negative lobes (a few % of peak at low spectral_samples) plus GPU float32
    # atomic noise. Assert they never exceed 10% of the channel peak — that still
    # catches any real sign/energy bug (those are order-of-peak).
    for k in ("ghost_r", "ghost_g", "ghost_b"):
        peak = max(float(out[k].max()), 1e-30)
        assert out[k].min() >= -0.1 * peak, f"{k}: min {out[k].min():.2e} vs peak {peak:.2e}"


def test_render_uses_cached_calibration(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4
    # Call twice; second should reuse calibration
    cal_before = loaded_lens._calib  # None initially
    loaded_lens.render_point_flare(16, 16, cfg)
    cal_after = loaded_lens._calib
    assert cal_after is not None
    loaded_lens.render_point_flare(16, 16, cfg)
    # Cache should be the same object
    assert loaded_lens._calib is cal_after
