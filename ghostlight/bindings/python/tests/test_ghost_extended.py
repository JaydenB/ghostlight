"""Extended tests for ghost pair enumeration and filtering."""

import pytest
import ghostlight


def _expected_pair_count(n_surfaces):
    """N*(N-1)/2 pairs for N surfaces."""
    return n_surfaces * (n_surfaces - 1) // 2


# ---------------------------------------------------------------------------
# enumerate_ghost_pairs
# ---------------------------------------------------------------------------

def test_enumerate_count_simple(simple_system):
    """N*(N-1)/2 pairs for N-surface system."""
    n = simple_system.num_surfaces()
    pairs = ghostlight.enumerate_ghost_pairs(simple_system)
    assert len(pairs) == _expected_pair_count(n)


def test_enumerate_count_doublegauss(doublegauss_lens):
    n = doublegauss_lens.num_surfaces()
    pairs = ghostlight.enumerate_ghost_pairs(doublegauss_lens)
    assert len(pairs) == _expected_pair_count(n)


def test_enumerate_a_lt_b(simple_system):
    """All pairs must have surf_a < surf_b."""
    for p in ghostlight.enumerate_ghost_pairs(simple_system):
        assert p.surf_a < p.surf_b


def test_enumerate_a_lt_b_complex(doublegauss_lens):
    for p in ghostlight.enumerate_ghost_pairs(doublegauss_lens):
        assert p.surf_a < p.surf_b


def test_enumerate_no_duplicates(simple_system):
    """No (a, b) pair may appear twice."""
    pairs = ghostlight.enumerate_ghost_pairs(simple_system)
    seen = set()
    for p in pairs:
        key = (p.surf_a, p.surf_b)
        assert key not in seen, f"Duplicate pair {key}"
        seen.add(key)


def test_enumerate_indices_in_range(simple_system):
    """All indices must be valid surface indices."""
    n = simple_system.num_surfaces()
    for p in ghostlight.enumerate_ghost_pairs(simple_system):
        assert 0 <= p.surf_a < n
        assert 0 <= p.surf_b < n


def test_ghost_pair_constructor():
    """GhostPair can be constructed directly."""
    p = ghostlight.GhostPair(2, 5)
    assert p.surf_a == 2
    assert p.surf_b == 5


# ---------------------------------------------------------------------------
# filter_ghost_pairs
# ---------------------------------------------------------------------------

def _default_flare_config():
    cfg = ghostlight.FlareConfig()
    cfg.min_ghost_intensity = 1e-7
    cfg.ghost_normalize = True
    cfg.max_area_boost = 100.0
    return cfg


def test_filter_returns_subset(loaded_lens):
    """Filtered pairs must be a subset of all pairs."""
    all_pairs = ghostlight.enumerate_ghost_pairs(loaded_lens)
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    filtered, _ = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    assert len(filtered) <= len(all_pairs)


def test_filter_a_lt_b(loaded_lens):
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    filtered, _ = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    for p in filtered:
        assert p.surf_a < p.surf_b


def test_area_boosts_positive(loaded_lens):
    """All area boost factors must be positive."""
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    _, boosts = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    for b in boosts:
        assert b > 0.0


def test_area_boosts_clamped(loaded_lens):
    """Boosts must not exceed max_area_boost."""
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    cfg.max_area_boost = 10.0
    _, boosts = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    for b in boosts:
        assert b <= 10.0 + 1e-6


def test_area_boosts_length_matches_pairs(loaded_lens):
    """pairs and boosts must have the same length."""
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    filtered, boosts = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    assert len(filtered) == len(boosts)


def test_stricter_threshold_fewer_pairs(loaded_lens):
    """Raising min_ghost_intensity must reduce or equal filtered pair count."""
    cal = loaded_lens.calibration()

    cfg_loose = _default_flare_config()
    cfg_loose.min_ghost_intensity = 1e-9
    filtered_loose, _ = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg_loose
    )

    cfg_strict = _default_flare_config()
    cfg_strict.min_ghost_intensity = 1e-3
    filtered_strict, _ = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg_strict
    )

    assert len(filtered_strict) <= len(filtered_loose)


def test_filter_no_duplicates(loaded_lens):
    """Filtered pairs must contain no duplicates."""
    cal = loaded_lens.calibration()
    cfg = _default_flare_config()
    filtered, _ = ghostlight.filter_ghost_pairs(
        loaded_lens, cal.sensor_half_w, cal.sensor_half_h, cfg
    )
    seen = set()
    for p in filtered:
        key = (p.surf_a, p.surf_b)
        assert key not in seen, f"Duplicate pair {key} in filtered list"
        seen.add(key)


# ---------------------------------------------------------------------------
# Lens wrapper caching
# ---------------------------------------------------------------------------

def test_ghost_pairs_cached_same_object(loaded_lens):
    """Second call to lens.ghost_pairs() must return the same Python object."""
    pairs1 = loaded_lens.ghost_pairs()
    pairs2 = loaded_lens.ghost_pairs()
    assert pairs1 is pairs2


def test_ghost_pairs_invalidated_on_finalize(loaded_lens):
    """Mutating a surface and calling finalize() must invalidate the ghost pairs cache."""
    pairs1 = loaded_lens.ghost_pairs()
    # Mutate
    loaded_lens.surfaces[0].radius += 1.0
    loaded_lens.finalize()
    pairs2 = loaded_lens.ghost_pairs()
    assert pairs1 is not pairs2
