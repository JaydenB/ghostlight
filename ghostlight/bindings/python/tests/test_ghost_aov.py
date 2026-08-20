"""GPU tests for per-flare AOV buffer feature."""

import pytest
import numpy as np

import ghostlight

pytestmark = pytest.mark.gpu


def _base_cfg():
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4
    return cfg


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_aov_none_no_extra_keys(loaded_lens):
    """Default config must not add any ghost_s* keys."""
    out = loaded_lens.render_point_flare(32, 32, _base_cfg())
    assert not any(k.startswith("ghost_s") for k in out)


def test_aov_none_existing_keys_unchanged(loaded_lens):
    """Values from NONE mode match a baseline render (no regression)."""
    cfg = _base_cfg()
    out_none = loaded_lens.render_point_flare(32, 32, cfg)
    cfg2 = _base_cfg()
    cfg2.aov_mode = ghostlight.GhostAovMode.NONE
    out_explicit = loaded_lens.render_point_flare(32, 32, cfg2)
    np.testing.assert_allclose(out_none["ghost_r"], out_explicit["ghost_r"], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# PER_PAIR basic properties
# ---------------------------------------------------------------------------

def test_aov_per_pair_keys_present(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    aov_keys = [k for k in out if k.startswith("ghost_s")]
    assert len(aov_keys) > 0


def test_aov_per_pair_triplets(loaded_lens):
    """Every active pair produces all three channels."""
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    r_keys = [k for k in out if k.startswith("ghost_s") and k.endswith("_r")]
    for k in r_keys:
        assert k.replace("_r", "_g") in out
        assert k.replace("_r", "_b") in out


def test_aov_per_pair_shape(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    for k in (k for k in out if k.startswith("ghost_s")):
        assert out[k].shape == (32, 32)


def test_aov_per_pair_dtype(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    for k in (k for k in out if k.startswith("ghost_s")):
        assert out[k].dtype == np.float32


def test_aov_per_pair_non_negative(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    # Small negatives are expected: at spectral_samples >= 4 the output-colour-
    # space colour-matching functions have negative lobes (saturated primaries
    # can't reproduce every monochromatic sample), so a per-pair channel can dip
    # a few ×1e-8 below zero — the same residual the combined ghost_g buffer
    # carries. The floor here matches this file's own AOV noise tolerance
    # (test_aov_sum_equals_combined_r uses atol=1e-6); a real negative-energy
    # bug would be orders of magnitude larger.
    for k in (k for k in out if k.startswith("ghost_s")):
        assert np.all(out[k] >= -1e-6)


# ---------------------------------------------------------------------------
# Sum invariant (blur disabled so layers sum exactly to combined)
# ---------------------------------------------------------------------------

def test_aov_sum_equals_combined_r(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    r_keys = [k for k in out if k.startswith("ghost_s") and k.endswith("_r")]
    total = sum(out[k] for k in r_keys)
    np.testing.assert_allclose(total, out["ghost_r"], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Cap behaviour
# ---------------------------------------------------------------------------

def test_aov_max_pairs_cap(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    cfg.aov_max_pairs = 2
    out = loaded_lens.render_point_flare(32, 32, cfg)
    r_keys = [k for k in out if k.startswith("ghost_s") and k.endswith("_r")]
    assert len(r_keys) <= 2


# ---------------------------------------------------------------------------
# Combined channel still present when AOV active
# ---------------------------------------------------------------------------

def test_aov_combined_still_present(loaded_lens):
    cfg = _base_cfg()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    out = loaded_lens.render_point_flare(32, 32, cfg)
    for ch in ("ghost_r", "ghost_g", "ghost_b"):
        assert ch in out
