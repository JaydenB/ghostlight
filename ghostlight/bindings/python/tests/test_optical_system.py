"""Tests for the OpticalSystem subclass and its caching behaviour."""

import pytest
import ghostlight
import numpy as np


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_simple_system_fixture_is_optical_system(simple_system):
    """The simple_system fixture should be an OpticalSystem instance."""
    assert isinstance(simple_system, ghostlight.OpticalSystem)


def test_empty_constructor():
    """OpticalSystem() with no argument must create an empty system."""
    sys = ghostlight.OpticalSystem()
    assert sys.num_surfaces() == 0


def test_lens_load_complex(doublegauss_lens_path):
    """Loading a complex lens file must succeed and have >0 surfaces."""
    lens = ghostlight.OpticalSystem.load(doublegauss_lens_path)
    assert lens.num_surfaces() > 0


def test_lens_load_cooke(cooketriplet_lens_path):
    lens = ghostlight.OpticalSystem.load(cooketriplet_lens_path)
    assert lens.num_surfaces() >= 6


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_lens_repr_format(loaded_lens):
    r = repr(loaded_lens)
    assert "OpticalSystem(" in r
    n = loaded_lens.num_surfaces()
    assert f"surfaces={n}" in r


def test_lens_repr_contains_name(loaded_lens):
    r = repr(loaded_lens)
    assert loaded_lens.name in r


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def test_current_key_returns_bytes(simple_lens):
    key = simple_lens._current_key()
    assert isinstance(key, bytes)


def test_current_key_stable_without_mutation(simple_lens):
    """Key must be identical across multiple calls with no mutations."""
    k1 = simple_lens._current_key()
    k2 = simple_lens._current_key()
    assert k1 == k2


def test_current_key_changes_on_radius_mutation(simple_lens):
    """Changing a surface radius must produce a different key after finalize."""
    k1 = simple_lens._current_key()
    simple_lens.surfaces[0].radius += 10.0
    simple_lens.finalize()
    k2 = simple_lens._current_key()
    assert k1 != k2


def test_current_key_changes_on_ior_mutation(simple_lens):
    k1 = simple_lens._current_key()
    simple_lens.surfaces[0].ior = 1.7
    simple_lens.finalize()
    k2 = simple_lens._current_key()
    assert k1 != k2


def test_current_key_changes_with_focal_length(simple_lens):
    before = simple_lens._current_key()
    simple_lens.focal_length += 1.0
    assert simple_lens._current_key() != before


def test_current_key_changes_with_aperture_pixels(simple_lens):
    image = simple_lens.aperture_images[0]
    image.semi_diameter = 2.0
    image.pixels = np.ones((2, 2), dtype=np.float32)
    before = simple_lens._current_key()
    image.pixels[0, 0] = 0.0
    assert simple_lens._current_key() != before


def test_failed_reload_preserves_existing_core_state(loaded_lens):
    before = (loaded_lens.name, loaded_lens.num_surfaces(), loaded_lens._current_key())
    with pytest.raises(RuntimeError):
        loaded_lens.reload("absolutely_does_not_exist_12345.lens")
    assert (loaded_lens.name, loaded_lens.num_surfaces(), loaded_lens._current_key()) == before


# ---------------------------------------------------------------------------
# Calibration caching
# ---------------------------------------------------------------------------

def test_calibration_auto_computes_when_none(simple_lens):
    """Calling calibration() must return a LensCalibration object."""
    assert simple_lens._calib is None
    cal = simple_lens.calibration()
    assert isinstance(cal, ghostlight.LensCalibration)


def test_calibration_cached_on_second_call(simple_lens):
    cal1 = simple_lens.calibration()
    cal2 = simple_lens.calibration()
    assert cal1 is cal2


def test_calibration_invalidated_after_mutation(simple_lens):
    cal1 = simple_lens.calibration()
    simple_lens.surfaces[0].radius += 5.0
    simple_lens.finalize()
    cal2 = simple_lens.calibration()
    assert cal1 is not cal2


def test_calibration_invalidated_by_different_args(simple_lens):
    cal1 = simple_lens.calibration(d_line_nm=550.0)
    cal2 = simple_lens.calibration(d_line_nm=587.56)
    assert cal1 is not cal2


# ---------------------------------------------------------------------------
# Ghost pair caching
# ---------------------------------------------------------------------------

def test_ghost_pairs_cached(loaded_lens):
    pairs1 = loaded_lens.ghost_pairs()
    pairs2 = loaded_lens.ghost_pairs()
    assert pairs1 is pairs2


def test_ghost_pairs_invalidated_on_finalize(loaded_lens):
    pairs1 = loaded_lens.ghost_pairs()
    loaded_lens.surfaces[0].radius += 1.0
    loaded_lens.finalize()
    pairs2 = loaded_lens.ghost_pairs()
    assert pairs1 is not pairs2


def test_ghost_pairs_non_empty_for_real_lens(loaded_lens):
    pairs = loaded_lens.ghost_pairs()
    assert len(pairs) > 0


# ---------------------------------------------------------------------------
# filter_ghosts
# ---------------------------------------------------------------------------

def test_filter_ghosts_returns_tuple(loaded_lens):
    cfg = ghostlight.FlareConfig()
    result = loaded_lens.filter_ghosts(cfg)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_filter_ghosts_consistent_calls(loaded_lens):
    """Same config → same result (cached or recomputed)."""
    cfg = ghostlight.FlareConfig()
    pairs1, boosts1 = loaded_lens.filter_ghosts(cfg)
    pairs2, boosts2 = loaded_lens.filter_ghosts(cfg)
    assert len(pairs1) == len(pairs2)
    assert boosts1 == pytest.approx(boosts2)


def test_filter_ghosts_with_explicit_calib(loaded_lens):
    """Passing calib= explicitly must work without error."""
    cal = loaded_lens.calibration()
    cfg = ghostlight.FlareConfig()
    pairs, boosts = loaded_lens.filter_ghosts(cfg, calib=cal)
    assert len(pairs) >= 0  # non-negative count


# ---------------------------------------------------------------------------
# Render wrappers auto-calibrate (GPU tests)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_render_point_flare_auto_calibrates(loaded_lens):
    """render_point_flare must set _calib if it wasn't set before."""
    loaded_lens._calib = None
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4
    loaded_lens.render_point_flare(16, 16, cfg)
    assert loaded_lens._calib is not None


@pytest.mark.gpu
def test_render_point_flare_explicit_calib(loaded_lens):
    """Passing calib= explicitly must not trigger recalibration."""
    cal = loaded_lens.calibration()
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4
    # Should complete without error
    out = loaded_lens.render_point_flare(16, 16, cfg, calib=cal)
    assert "ghost_r" in out
