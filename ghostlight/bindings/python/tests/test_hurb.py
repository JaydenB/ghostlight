"""HURB edge-diffraction sampler tests.

HURB (Heisenberg Uncertainty Ray Bending) kicks each ray that passes an edge at
perpendicular distance d by a random angle of scale sigma = lambda * K / d. These
are pure host-math checks of the sampler in hurb.h (via the _hurb_sample_debug
binding) — no GPU — covering:

  * the sigma = lambda*K/d formula and the K constants,
  * that the samples are the intended distribution (Gaussian std == sigma;
    Lorentzian is Cauchy, median-abs and IQR/2 == gamma) and unbiased,
  * chromatic scaling (width proportional to lambda) and the max-kick clamp,
  * the calibration: integrated over a slit, the kick envelope sits at the
    diffraction scale theta0 = lambda/w, the Gaussian concentrates in the sinc^2
    central lobe while the Lorentzian carries the heavy 1/theta^2 wings, and the
    envelope width scales as 1/w — i.e. the K constants put HURB at the true
    diffraction scale.
"""
import numpy as np
import pytest

import ghostlight

_dbg = ghostlight._ghostlight._hurb_sample_debug
G = int(ghostlight.HurbKickDistribution.GAUSSIAN)
L = int(ghostlight.HurbKickDistribution.LORENTZIAN)
BIG = 1.0e3   # effectively-unclamped max_kick for shape tests


def _kicks(lam_nm, d, dist, seed=1, max_kick=BIG):
    return np.asarray(_dbg(float(lam_nm), np.asarray(d, np.float32), dist,
                           float(max_kick), int(seed))["kicks"])


# ---------------------------------------------------------------------------
# Constants + sigma formula
# ---------------------------------------------------------------------------

def test_hurb_constants():
    r = _dbg(550.0, np.array([0.5], np.float32), G, 1.0, 1)
    assert float(r["gauss_k"]) == pytest.approx(1.0 / (2.0 * np.sqrt(2.0) * np.pi), rel=1e-5)
    assert float(r["lorentz_k"]) == pytest.approx(1.0 / (2.0 * np.pi), rel=1e-5)


def test_hurb_sigma_formula():
    r = _dbg(550.0, np.array([0.5], np.float32), G, 1.0, 1)
    gk = float(r["gauss_k"])
    lam_mm = 550.0e-6
    for d in (0.05, 0.2, 1.0, 3.0):
        s = float(_dbg(550.0, np.array([d], np.float32), G, 1.0, 1)["sigma"][0])
        assert s == pytest.approx(lam_mm * gk / d, rel=1e-4)


def test_hurb_sigma_zero_on_or_past_edge():
    """d <= 0 yields no kick (the caller guards on-edge rays)."""
    r = _dbg(550.0, np.array([0.0, -1.0], np.float32), G, 1.0, 1)
    assert float(r["sigma"][0]) == 0.0
    assert float(r["kicks"][0]) == 0.0
    assert float(r["kicks"][1]) == 0.0


# ---------------------------------------------------------------------------
# Distribution shape + statistics
# ---------------------------------------------------------------------------

def _fixed(d0, n):
    return np.full(n, d0, np.float32)


def test_hurb_gaussian_is_normal_sigma():
    d0, N = 0.5, 400_000
    sigma = 550.0e-6 * (1.0 / (2 * np.sqrt(2) * np.pi)) / d0
    k = _kicks(550.0, _fixed(d0, N), G, seed=11)
    assert np.std(k) == pytest.approx(sigma, rel=0.02)
    assert abs(np.mean(k)) < 0.02 * sigma          # unbiased


def test_hurb_lorentzian_is_cauchy_gamma():
    d0, N = 0.5, 400_000
    gamma = 550.0e-6 * (1.0 / (2 * np.pi)) / d0
    k = _kicks(550.0, _fixed(d0, N), L, seed=13)
    # Cauchy: median(|x|) == gamma and IQR/2 == gamma (variance is undefined).
    assert np.median(np.abs(k)) == pytest.approx(gamma, rel=0.03)
    iqr = np.percentile(k, 75) - np.percentile(k, 25)
    assert iqr / 2.0 == pytest.approx(gamma, rel=0.03)


def test_hurb_chromatic_width_scales_with_lambda():
    d0, N = 0.5, 300_000
    base = np.std(_kicks(550.0, _fixed(d0, N), G, seed=5))
    for lam in (450.0, 650.0):
        w = np.std(_kicks(lam, _fixed(d0, N), G, seed=5))
        assert w / base == pytest.approx(lam / 550.0, rel=0.04)


def test_hurb_max_kick_clamps():
    # Tiny d -> enormous sigma; every sample must land inside the clamp.
    k = _kicks(550.0, _fixed(1.0e-4, 20_000), L, seed=7, max_kick=0.35)
    assert np.all(np.abs(k) <= 0.35 + 1e-6)
    assert np.max(np.abs(k)) == pytest.approx(0.35, rel=0.05)   # the tail reaches it


def test_hurb_deterministic():
    d = _fixed(0.4, 5000)
    a = _kicks(550.0, d, L, seed=99)
    b = _kicks(550.0, d, L, seed=99)
    assert np.array_equal(a, b)
    c = _kicks(550.0, d, L, seed=100)
    assert not np.array_equal(a, c)               # a different seed differs


# ---------------------------------------------------------------------------
# Slit calibration: the envelope vs the sinc^2 diffraction pattern
# ---------------------------------------------------------------------------

def _slit_kicks(lam_nm, a, dist, N=2_000_000, seed=3, max_kick=0.35):
    """HURB kicks for rays uniformly filling a slit of half-width a; d is the
    distance to the nearer edge."""
    x = (np.random.default_rng(0).random(N).astype(np.float32) * 2 - 1) * a
    d = (a - np.abs(x)).astype(np.float32)
    return _kicks(lam_nm, d, dist, seed=seed, max_kick=max_kick)


def _sinc2_core_fraction(lam_mm, w):
    """Fraction of sinc^2(pi*w*theta/lambda) energy inside the central lobe.

    np.sinc(x) = sin(pi x)/(pi x), so sinc^2(pi*w*theta/lambda) = np.sinc(w*theta/
    lambda)**2 (finite at theta=0, no divide warning)."""
    th = np.linspace(-60.0, 60.0, 600001) * (lam_mm / w)
    s2 = np.sinc(w * th / lam_mm) ** 2
    return float(s2[np.abs(th) < lam_mm / w].sum() / s2.sum())


def test_hurb_slit_envelope_at_diffraction_scale():
    """The kick envelope's characteristic width sits at theta0 = lambda/w, so the
    K constants put HURB at the true diffraction scale (not an arbitrary blur)."""
    lam_mm, a = 550.0e-6, 0.5
    theta0 = lam_mm / (2 * a)
    for dist in (G, L):
        k = _slit_kicks(550.0, a, dist)
        med = np.median(np.abs(k))
        assert 0.2 * theta0 <= med <= 1.5 * theta0


def test_hurb_gaussian_core_matches_sinc2():
    """The Gaussian kick concentrates in the sinc^2 central lobe (core match)."""
    lam_mm, a = 550.0e-6, 0.5
    theta0 = lam_mm / (2 * a)
    k = _slit_kicks(550.0, a, G)
    frac = float(np.mean(np.abs(k) < theta0))
    sinc_core = _sinc2_core_fraction(lam_mm, 2 * a)          # ~0.90
    assert 0.70 <= frac <= 0.98
    assert abs(frac - sinc_core) < 0.20                      # near the sinc^2 core


def test_hurb_lorentzian_has_heavier_tails_than_gaussian():
    """The Lorentzian kick carries the heavy 1/theta^2 wings (the glare tail): far
    more energy beyond the central lobe than the Gaussian."""
    lam_mm, a = 550.0e-6, 0.5
    theta0 = lam_mm / (2 * a)
    kg = _slit_kicks(550.0, a, G, seed=3)
    kl = _slit_kicks(550.0, a, L, seed=4)
    tail_g = float(np.mean(np.abs(kg) > 3 * theta0))
    tail_l = float(np.mean(np.abs(kl) > 3 * theta0))
    assert tail_l > 2.0 * tail_g                             # markedly heavier tail


def test_hurb_slit_width_scales_inversely_with_slit():
    """theta0 ~ 1/w: a wider slit gives a proportionally tighter envelope."""
    lam = 550.0
    med_narrow = np.median(np.abs(_slit_kicks(lam, 0.25, G)))  # w = 0.5 mm
    med_wide = np.median(np.abs(_slit_kicks(lam, 0.50, G)))    # w = 1.0 mm
    # Halving the slit doubles theta0, so the narrow slit's envelope is ~2x wider.
    assert med_narrow / med_wide == pytest.approx(2.0, rel=0.25)
