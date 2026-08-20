"""Tests for FlareConfig.ghost_filter — designer-level pair selection.

The filter applies AFTER filter_ghost_pairs (the intensity / IOR-contrast
filter). Default mode ALL is a no-op so existing render code is unchanged.
Most tests stay on the CPU side: GhostFilter is a render-time selector,
and verifying which pairs survive the filter doesn't need the GPU. The
AOV integration test is GPU-marked because the per-pair AOV path runs
through the CUDA kernel.
"""
from __future__ import annotations

import numpy as np
import pytest

import ghostlight


# ---------------------------------------------------------------------------
# GhostFilter construction / round-trip
# ---------------------------------------------------------------------------

def test_default_mode_is_all():
    f = ghostlight.GhostFilter()
    assert f.mode == ghostlight.GhostFilter.Mode.ALL
    assert list(f.pairs) == []


def test_set_mode_and_pairs():
    f = ghostlight.GhostFilter()
    f.mode = ghostlight.GhostFilter.Mode.INCLUDE
    f.pairs = [(0, 2), (1, 3)]
    assert f.mode == ghostlight.GhostFilter.Mode.INCLUDE
    assert list(f.pairs) == [(0, 2), (1, 3)]


def test_pairs_property_is_a_copy():
    """Mutating the returned list shouldn't affect the C++ vector — the
    binding hands back a fresh list each time, so callers can append to
    it without seeing those changes reflected in the underlying config."""
    f = ghostlight.GhostFilter()
    f.pairs = [(0, 1)]
    handle = f.pairs
    handle.append((4, 5))
    assert list(f.pairs) == [(0, 1)]


def test_flareconfig_has_ghost_filter_field():
    cfg = ghostlight.PointFlareConfig()
    assert isinstance(cfg.ghost_filter, ghostlight.GhostFilter)
    assert cfg.ghost_filter.mode == ghostlight.GhostFilter.Mode.ALL


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

def _base_cfg() -> ghostlight.PointFlareConfig:
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 12
    cfg.spectral_samples = 3
    cfg.min_ghost_intensity = 0.0  # keep all pre-intensity pairs so the
                                    # filter logic, not intensity, decides.
    cfg.ghost_normalize = False
    return cfg


def _baseline_active_pairs(lens, cfg) -> list[tuple[int, int]]:
    cal = lens.calibration()
    pairs, _boosts = ghostlight.filter_ghost_pairs(
        lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    return [(p.surf_a, p.surf_b) for p in pairs]


@pytest.mark.gpu
def test_filter_all_mode_matches_unfiltered(loaded_lens):
    """ALL mode is a no-op: combined ghost output equals the baseline."""
    cfg = _base_cfg()
    baseline = loaded_lens.render_point_flare(32, 32, cfg)

    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.ALL
    cfg.ghost_filter.pairs = [(0, 1)]  # ignored because mode is ALL
    out = loaded_lens.render_point_flare(32, 32, cfg)
    np.testing.assert_allclose(
        out["ghost_r"], baseline["ghost_r"], rtol=1e-5, atol=1e-7
    )


@pytest.mark.gpu
def test_filter_include_with_no_pairs_renders_nothing(loaded_lens):
    """INCLUDE mode with an empty list AND mode=ALL is a no-op (we
    document that ALL ignores the list). But an empty INCLUDE list with
    non-empty active pairs upstream means "include nothing" — except
    we treat empty filter.pairs as no-op so toggling the filter off
    doesn't require clearing the list. So INCLUDE+empty == ALL."""
    cfg = _base_cfg()
    baseline = loaded_lens.render_point_flare(32, 32, cfg)
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
    cfg.ghost_filter.pairs = []  # empty list — treated as no-op
    out = loaded_lens.render_point_flare(32, 32, cfg)
    np.testing.assert_allclose(
        out["ghost_r"], baseline["ghost_r"], rtol=1e-5, atol=1e-7
    )


@pytest.mark.gpu
def test_filter_include_subset_subset_of_baseline(loaded_lens):
    """Picking a single pair via INCLUDE produces ghost output that's
    <= the baseline everywhere (since we removed pairs that contributed
    positively to the combined image)."""
    cfg = _base_cfg()
    baseline_pairs = _baseline_active_pairs(loaded_lens, cfg)
    assert len(baseline_pairs) >= 2, "lens needs >=2 ghost pairs for this test"
    baseline_out = loaded_lens.render_point_flare(32, 32, cfg)

    pick = baseline_pairs[0]
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
    cfg.ghost_filter.pairs = [pick]
    out = loaded_lens.render_point_flare(32, 32, cfg)
    # Pointwise <= baseline (ghost contributions are non-negative additions).
    assert np.all(out["ghost_r"] <= baseline_out["ghost_r"] + 1e-5)
    # Not just zeros — the included pair should produce some signal.
    assert float(out["ghost_r"].sum()) > 0.0


@pytest.mark.gpu
def test_filter_exclude_subset_of_baseline(loaded_lens):
    """EXCLUDE drops the listed pairs; remaining output is <= baseline."""
    cfg = _base_cfg()
    baseline_pairs = _baseline_active_pairs(loaded_lens, cfg)
    assert len(baseline_pairs) >= 2
    baseline_out = loaded_lens.render_point_flare(32, 32, cfg)

    drop = baseline_pairs[0]
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.EXCLUDE
    cfg.ghost_filter.pairs = [drop]
    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert np.all(out["ghost_r"] <= baseline_out["ghost_r"] + 1e-5)


@pytest.mark.gpu
def test_filter_include_plus_exclude_partition(loaded_lens):
    """INCLUDE one pair + EXCLUDE that same pair (across two renders)
    should sum back to the baseline within numerical tolerance — every
    pair is accounted for in exactly one of the two renders."""
    cfg = _base_cfg()
    baseline_pairs = _baseline_active_pairs(loaded_lens, cfg)
    assert len(baseline_pairs) >= 2
    baseline_out = loaded_lens.render_point_flare(32, 32, cfg)

    pick = baseline_pairs[0]

    cfg_in = _base_cfg()
    cfg_in.ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
    cfg_in.ghost_filter.pairs = [pick]
    out_in = loaded_lens.render_point_flare(32, 32, cfg_in)

    cfg_ex = _base_cfg()
    cfg_ex.ghost_filter.mode = ghostlight.GhostFilter.Mode.EXCLUDE
    cfg_ex.ghost_filter.pairs = [pick]
    out_ex = loaded_lens.render_point_flare(32, 32, cfg_ex)

    combined = out_in["ghost_r"] + out_ex["ghost_r"]
    np.testing.assert_allclose(combined, baseline_out["ghost_r"], rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_filter_aov_layer_count_matches_filter(loaded_lens):
    """AOV mode honours the filter: only filtered pairs get AOV layers."""
    cfg = _base_cfg()
    baseline_pairs = _baseline_active_pairs(loaded_lens, cfg)
    assert len(baseline_pairs) >= 2

    pick = baseline_pairs[0]
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
    cfg.ghost_filter.pairs = [pick]

    out = loaded_lens.render_point_flare(32, 32, cfg)
    # AOV layers are emitted as ghost_s<a>_s<b>_{r,g,b} — count R-channel
    # entries; one survives the include filter.
    aov_r_keys = [k for k in out if k.startswith("ghost_s") and k.endswith("_r")]
    assert len(aov_r_keys) == 1
    surf_a, surf_b = pick
    assert f"ghost_s{surf_a}_s{surf_b}_r" in out


@pytest.mark.gpu
def test_filter_include_nonexistent_pair_renders_empty(loaded_lens):
    """INCLUDE-ing a pair that doesn't survive the intensity filter
    produces an empty ghost buffer — there's no upstream match to keep."""
    cfg = _base_cfg()
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
    # A pair that doesn't exist in any reasonable enumeration:
    cfg.ghost_filter.pairs = [(999, 9999)]
    out = loaded_lens.render_point_flare(32, 32, cfg)
    assert float(out["ghost_r"].max()) == 0.0
    assert float(out["ghost_g"].max()) == 0.0
    assert float(out["ghost_b"].max()) == 0.0


@pytest.mark.gpu
def test_filter_aov_exclude_leaves_remaining_layers(loaded_lens):
    """EXCLUDE mode with AOV: dropped pair has no layer; the rest keep
    theirs. Mirrors the INCLUDE-side AOV test to catch mode-specific
    regressions in the filter branch."""
    cfg = _base_cfg()
    baseline_pairs = _baseline_active_pairs(loaded_lens, cfg)
    assert len(baseline_pairs) >= 2

    drop = baseline_pairs[0]
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    cfg.ghost_filter.mode = ghostlight.GhostFilter.Mode.EXCLUDE
    cfg.ghost_filter.pairs = [drop]

    out = loaded_lens.render_point_flare(32, 32, cfg)
    aov_r_keys = [k for k in out if k.startswith("ghost_s") and k.endswith("_r")]
    assert len(aov_r_keys) == len(baseline_pairs) - 1
    surf_a, surf_b = drop
    assert f"ghost_s{surf_a}_s{surf_b}_r" not in out
