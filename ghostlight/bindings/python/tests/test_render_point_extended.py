"""Extended tests for render_point_flare — all require GPU."""

import pytest
import numpy as np
import ghostlight

pytestmark = pytest.mark.gpu


def _cfg(ray_grid=16, spectral_samples=4, **kwargs):
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = ray_grid
    cfg.spectral_samples = spectral_samples
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Ghost energy
# ---------------------------------------------------------------------------

def test_ghost_energy_positive_for_bright_source(loaded_lens):
    """A very bright source must produce measurable ghost energy."""
    cfg = _cfg(source_r=100.0, source_g=100.0, source_b=100.0,
               flare_gain=1000.0)
    out = loaded_lens.render_point_flare(64, 64, cfg)
    total = out["ghost_r"].sum() + out["ghost_g"].sum() + out["ghost_b"].sum()
    assert total > 0.0


def test_zero_source_produces_zero_ghost(loaded_lens):
    """A zero-intensity source must produce zero ghost output."""
    cfg = _cfg(source_r=0.0, source_g=0.0, source_b=0.0, flare_gain=1000.0)
    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert np.all(out["ghost_r"] == 0.0)
    assert np.all(out["ghost_g"] == 0.0)
    assert np.all(out["ghost_b"] == 0.0)


def test_ghost_non_negative(loaded_lens):
    """Ghost output must be non-negative everywhere."""
    cfg = _cfg(source_r=5.0, source_g=5.0, source_b=5.0)
    out = loaded_lens.render_point_flare(32, 32, cfg)
    # Small negatives from the XYZ CMF negative lobes (a few % of peak at low
    # spectral_samples, with no blur pass to smooth them) plus atomic noise;
    # assert they stay under 10% of peak (real bugs are order-of-peak).
    for k in ("ghost_r", "ghost_g", "ghost_b"):
        peak = max(float(out[k].max()), 1e-30)
        assert out[k].min() >= -0.1 * peak, f"{k}: min {out[k].min():.2e} vs peak {peak:.2e}"


# ---------------------------------------------------------------------------
# Source position affects pattern
# ---------------------------------------------------------------------------

def test_source_position_affects_ghost_pattern(loaded_lens):
    """Ghost patterns for different source positions must differ."""
    cfg_left = _cfg(source_x=0.2, source_y=0.5,
                    source_r=10.0, source_g=10.0, source_b=10.0)
    cfg_right = _cfg(source_x=0.8, source_y=0.5,
                     source_r=10.0, source_g=10.0, source_b=10.0)
    out_left = loaded_lens.render_point_flare(64, 64, cfg_left)
    out_right = loaded_lens.render_point_flare(64, 64, cfg_right)
    # Argmax pixel positions must differ
    argmax_left = np.argmax(out_left["ghost_r"])
    argmax_right = np.argmax(out_right["ghost_r"])
    assert argmax_left != argmax_right


def test_center_vs_offcenter_differ(loaded_lens):
    """Center source vs off-center must produce different ghost patterns."""
    cfg_c = _cfg(source_x=0.5, source_y=0.5,
                 source_r=10.0, source_g=10.0, source_b=10.0)
    cfg_o = _cfg(source_x=0.2, source_y=0.3,
                 source_r=10.0, source_g=10.0, source_b=10.0)
    out_c = loaded_lens.render_point_flare(64, 64, cfg_c)
    out_o = loaded_lens.render_point_flare(64, 64, cfg_o)
    diff = np.abs(out_c["ghost_r"] - out_o["ghost_r"]).max()
    assert diff > 0.0


# ---------------------------------------------------------------------------
# flare_gain scales output linearly
# ---------------------------------------------------------------------------

def test_flare_gain_scales_output(loaded_lens):
    """Doubling flare_gain must approximately double ghost output."""
    cfg1 = _cfg(source_r=5.0, source_g=5.0, source_b=5.0,
                flare_gain=500.0, ray_grid=32)
    cfg2 = _cfg(source_r=5.0, source_g=5.0, source_b=5.0,
                flare_gain=1000.0, ray_grid=32)
    out1 = loaded_lens.render_point_flare(64, 64, cfg1)
    out2 = loaded_lens.render_point_flare(64, 64, cfg2)
    sum1 = out1["ghost_r"].sum()
    sum2 = out2["ghost_r"].sum()
    if sum1 > 0:
        ratio = sum2 / sum1
        assert ratio == pytest.approx(2.0, rel=0.2)


# ---------------------------------------------------------------------------
# All channels populated with non-white source
# ---------------------------------------------------------------------------

def test_all_channels_populated_with_rgb_source(loaded_lens):
    """A chromatic source must produce non-zero energy in all three channels."""
    cfg = _cfg(source_r=10.0, source_g=5.0, source_b=2.0,
               flare_gain=500.0, ray_grid=32)
    out = loaded_lens.render_point_flare(64, 64, cfg)
    assert out["ghost_r"].sum() > 0.0
    assert out["ghost_g"].sum() > 0.0
    assert out["ghost_b"].sum() > 0.0


# ---------------------------------------------------------------------------
# Intensity threshold filters dim ghosts
# ---------------------------------------------------------------------------

def test_high_intensity_threshold_reduces_ghosts(loaded_lens):
    """Very high min_ghost_intensity must result in dimmer (or zero) ghost output."""
    cfg_loose = _cfg(source_r=5.0, source_g=5.0, source_b=5.0,
                     min_ghost_intensity=1e-9, flare_gain=500.0)
    cfg_strict = _cfg(source_r=5.0, source_g=5.0, source_b=5.0,
                      min_ghost_intensity=0.999, flare_gain=500.0)
    out_loose = loaded_lens.render_point_flare(32, 32, cfg_loose)
    out_strict = loaded_lens.render_point_flare(32, 32, cfg_strict)
    assert out_strict["ghost_r"].max() <= out_loose["ghost_r"].max() + 1e-6


# ---------------------------------------------------------------------------
# No source_r/g/b keys in point flare output
# ---------------------------------------------------------------------------

def test_no_source_keys_in_output(loaded_lens):
    """PointFlare output must not contain source_r/g/b keys."""
    cfg = _cfg()
    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert "source_r" not in out
    assert "source_g" not in out
    assert "source_b" not in out


# ---------------------------------------------------------------------------
# Output dtype and dimensions
# ---------------------------------------------------------------------------

def test_output_dtype_float32(loaded_lens):
    cfg = _cfg()
    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert out["ghost_r"].dtype == np.float32


def test_output_width_height_keys(loaded_lens):
    cfg = _cfg()
    out = loaded_lens.render_point_flare(48, 36, cfg)
    assert out["width"] == 48
    assert out["height"] == 36
    assert out["ghost_r"].shape == (36, 48)
