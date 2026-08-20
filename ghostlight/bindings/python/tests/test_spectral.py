"""Tests for the spectral sampling system."""

import math
import pytest
import ghostlight


# ---------------------------------------------------------------------------
# build_spectral_lambdas
# ---------------------------------------------------------------------------

def test_build_spectral_lambdas_count():
    lambdas = ghostlight.build_spectral_lambdas(16)
    assert len(lambdas) == 16


def test_build_spectral_lambdas_default_range():
    """Default range is 400–700 nm; all samples must be within it."""
    lambdas = ghostlight.build_spectral_lambdas(16)
    assert all(400.0 <= lam <= 700.0 for lam in lambdas)


def test_build_spectral_lambdas_cell_centered_first():
    """Cell-centred spacing: first sample at min + half-step."""
    lambdas = ghostlight.build_spectral_lambdas(4, 400.0, 700.0)
    step = (700.0 - 400.0) / 4
    expected_first = 400.0 + step / 2
    assert lambdas[0] == pytest.approx(expected_first)


def test_build_spectral_lambdas_cell_centered_last():
    """Last sample at max - half-step."""
    lambdas = ghostlight.build_spectral_lambdas(4, 400.0, 700.0)
    step = (700.0 - 400.0) / 4
    expected_last = 700.0 - step / 2
    assert lambdas[-1] == pytest.approx(expected_last)


def test_build_spectral_lambdas_monotone():
    """Wavelength samples must be strictly increasing."""
    lambdas = ghostlight.build_spectral_lambdas(16)
    for i in range(1, len(lambdas)):
        assert lambdas[i] > lambdas[i - 1]


def test_build_spectral_lambdas_uniform_spacing():
    """All gaps between consecutive samples must be equal."""
    lambdas = ghostlight.build_spectral_lambdas(10, 400.0, 700.0)
    gaps = [lambdas[i + 1] - lambdas[i] for i in range(len(lambdas) - 1)]
    expected_gap = gaps[0]
    for g in gaps:
        assert g == pytest.approx(expected_gap, rel=1e-5)


def test_build_spectral_lambdas_custom_range():
    """Custom min/max range must be respected."""
    lambdas = ghostlight.build_spectral_lambdas(8, 450.0, 650.0)
    assert all(450.0 <= lam <= 650.0 for lam in lambdas)


def test_build_spectral_lambdas_count_one():
    """A single sample must return exactly one wavelength in range."""
    lambdas = ghostlight.build_spectral_lambdas(1, 400.0, 700.0)
    assert len(lambdas) == 1
    assert 400.0 <= lambdas[0] <= 700.0


def test_build_spectral_lambdas_count_two():
    lambdas = ghostlight.build_spectral_lambdas(2, 400.0, 700.0)
    assert len(lambdas) == 2
    assert lambdas[0] < lambdas[1]


# ---------------------------------------------------------------------------
# GPU: different spectral_samples produce valid (non-zero) output
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_spectral_samples_low_vs_high(loaded_lens):
    """Renders with different spectral_samples must both produce non-zero output."""
    cfg4 = ghostlight.PointFlareConfig()
    cfg4.ray_grid = 16
    cfg4.spectral_samples = 4
    cfg4.source_r = 5.0
    cfg4.source_g = 5.0
    cfg4.source_b = 5.0

    cfg16 = ghostlight.PointFlareConfig()
    cfg16.ray_grid = 16
    cfg16.spectral_samples = 16
    cfg16.source_r = 5.0
    cfg16.source_g = 5.0
    cfg16.source_b = 5.0

    out4 = loaded_lens.render_point_flare(32, 32, cfg4)
    out16 = loaded_lens.render_point_flare(32, 32, cfg16)

    import numpy as np
    assert np.any(out4["ghost_r"] > 0) or np.any(out4["ghost_g"] > 0)
    assert np.any(out16["ghost_r"] > 0) or np.any(out16["ghost_g"] > 0)
