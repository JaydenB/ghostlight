"""Veiling-glare ("veil") tests.

The veil is the physically-based glare-spread-function halo: a broad, soft,
spectral, energy-conserving glow around each bright source, rendered as a
separate additive layer (render_veil -> veil_r/g/b) that the caller folds into
the metered flare layer.

Config round-trips are pure binding checks; the render tests need CUDA (marked
``gpu``, auto-skipped without a device by conftest). The GPU gate covers: off =
no layer, energy conservation (total ~ source_flux * veil_gain), gain/source
linearity, spectral neutrality (white -> neutral), determinism, and the analytic
radial falloff via the _render_veil_debug reference.
"""
import numpy as np
import pytest

import ghostlight


# ---------------------------------------------------------------------------
# Config / binding round-trips (no GPU)
# ---------------------------------------------------------------------------

def test_veil_config_defaults():
    """A default config has the veil OFF (byte-identical guard) with the
    documented shape defaults."""
    cfg = ghostlight.PointFlareConfig()
    assert cfg.diffraction.veil is False
    assert cfg.diffraction.veil_gain == pytest.approx(0.03)
    assert cfg.diffraction.veil_spread == pytest.approx(0.12)
    assert cfg.diffraction.veil_falloff == pytest.approx(1.5)


def test_veil_config_roundtrip():
    cfg = ghostlight.PointFlareConfig()
    cfg.diffraction.veil = True
    cfg.diffraction.veil_gain = 0.2
    cfg.diffraction.veil_spread = 0.3
    cfg.diffraction.veil_falloff = 2.0
    assert cfg.diffraction.veil is True
    assert cfg.diffraction.veil_gain == pytest.approx(0.2)
    assert cfg.diffraction.veil_spread == pytest.approx(0.3)
    assert cfg.diffraction.veil_falloff == pytest.approx(2.0)


def test_veil_to_hwc_absent_returns_none():
    """veil_to_hwc returns None when the veil layer is absent so the additive
    composite simply skips the term."""
    assert ghostlight._arrays.veil_to_hwc({"ghost_r": np.zeros((4, 4))}) is None


def test_veil_to_hwc_stacks_when_present():
    r = np.ones((3, 5), dtype=np.float32)
    hwc = ghostlight._arrays.veil_to_hwc({"veil_r": r, "veil_g": 2 * r, "veil_b": 3 * r})
    assert hwc.shape == (3, 5, 3)
    assert hwc[..., 0].max() == pytest.approx(1.0)
    assert hwc[..., 2].max() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Render helpers (GPU)
# ---------------------------------------------------------------------------

def _cfg(veil=True, gain=0.05, spread=0.12, falloff=1.5, ns=6,
         sx=0.5, sy=0.5, flux=10.0):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = flux
    c.spectral_samples = ns
    c.diffraction.veil = veil
    c.diffraction.veil_gain = gain
    c.diffraction.veil_spread = spread
    c.diffraction.veil_falloff = falloff
    return c


def _veil_total(out):
    return float(out["veil_r"].sum() + out["veil_g"].sum() + out["veil_b"].sum())


# ---------------------------------------------------------------------------
# Presence / off-parity (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_veil_off_absent(loaded_lens):
    """Off (the default) -> no veil layer in the output dict."""
    out = loaded_lens.render_point_flare(96, 96, _cfg(veil=False))
    assert "veil_r" not in out
    assert "veil_g" not in out
    assert "veil_b" not in out


@pytest.mark.gpu
def test_veil_on_present_finite_positive(loaded_lens):
    out = loaded_lens.render_point_flare(96, 96, _cfg(veil=True))
    for k in ("veil_r", "veil_g", "veil_b"):
        assert k in out
        assert out[k].shape == (96, 96)
        assert np.all(np.isfinite(out[k]))
        assert np.all(out[k] >= 0.0)          # the GSF is non-negative everywhere
    assert _veil_total(out) > 0.0


@pytest.mark.gpu
def test_veil_does_not_disturb_ghost(loaded_lens):
    """The veil is a separate layer: turning it on leaves the ghost buffers
    unchanged (within the GPU atomic-add floor)."""
    out_off = loaded_lens.render_point_flare(96, 96, _cfg(veil=False))
    out_on = loaded_lens.render_point_flare(96, 96, _cfg(veil=True))
    peak = max(float(out_off["ghost_r"].max()), 1e-20)
    assert np.abs(out_on["ghost_r"] - out_off["ghost_r"]).max() <= 1e-4 * peak


# ---------------------------------------------------------------------------
# Energy conservation (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_veil_energy_matches_gain(loaded_lens):
    """The energy-normalised GSF integrates to the source flux, so the total
    veil ~ source_flux * veil_gain (a centred source with a small spread lands
    almost entirely on-frame). ACEScg in == ACEScg out, so the output-space flux
    equals source_r. Bounded above by source*gain (no energy is created)."""
    flux, gain = 10.0, 0.05
    out = loaded_lens.render_point_flare(
        128, 128, _cfg(veil=True, gain=gain, spread=0.02, flux=flux))
    total_r = float(out["veil_r"].sum())
    expected = flux * gain
    # On-frame fraction is high (small spread, centred) but < 1 (16-radius sprite
    # edge + discretisation); assert the total is O(1)x expected and never over.
    assert 0.6 * expected <= total_r <= 1.05 * expected


@pytest.mark.gpu
def test_veil_gain_is_linear(loaded_lens):
    """Doubling veil_gain doubles the veil energy."""
    out1 = loaded_lens.render_point_flare(96, 96, _cfg(gain=0.05))
    out2 = loaded_lens.render_point_flare(96, 96, _cfg(gain=0.10))
    t1, t2 = _veil_total(out1), _veil_total(out2)
    assert t1 > 0.0
    assert t2 / t1 == pytest.approx(2.0, rel=0.02)


@pytest.mark.gpu
def test_veil_source_flux_is_linear(loaded_lens):
    """Doubling the source flux doubles the veil energy."""
    out1 = loaded_lens.render_point_flare(96, 96, _cfg(flux=5.0))
    out2 = loaded_lens.render_point_flare(96, 96, _cfg(flux=10.0))
    t1, t2 = _veil_total(out1), _veil_total(out2)
    assert t1 > 0.0
    assert t2 / t1 == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Spectral neutrality + determinism (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_veil_sprite_is_achromatic(loaded_lens):
    """The halo itself carries no chromatic bias: the veil sprite is neutral
    (R == G == B per texel). Any colour in the final veil comes purely from the
    source tint, not the renderer (so white-point handling is the shared colour
    pipeline's job, exactly as for the ghost and starburst layers)."""
    d = ghostlight._ghostlight._render_veil_debug(
        96, 96, loaded_lens, loaded_lens.calibration(), _cfg())
    sprite = np.asarray(d["sprite_rgb"])          # (grid, grid, 3)
    assert np.array_equal(sprite[..., 0], sprite[..., 1])
    assert np.array_equal(sprite[..., 0], sprite[..., 2])


@pytest.mark.gpu
def test_veil_takes_source_colour(loaded_lens):
    """A red source yields a red-dominant veil (the achromatic halo is tinted by
    the source's own colour)."""
    cfg = _cfg(spread=0.02)
    cfg.source_r, cfg.source_g, cfg.source_b = 10.0, 1.0, 1.0
    out = loaded_lens.render_point_flare(128, 128, cfg)
    tr = float(out["veil_r"].sum())
    tg = float(out["veil_g"].sum())
    tb = float(out["veil_b"].sum())
    assert tr > 3.0 * tg
    assert tr > 3.0 * tb


@pytest.mark.gpu
def test_veil_is_deterministic(loaded_lens):
    """The veil is an analytic splat with no RNG -> two renders match closely."""
    out1 = loaded_lens.render_point_flare(96, 96, _cfg())
    out2 = loaded_lens.render_point_flare(96, 96, _cfg())
    peak = max(float(out1["veil_r"].max()), 1e-20)
    assert np.abs(out1["veil_r"] - out2["veil_r"]).max() <= 1e-5 * peak


@pytest.mark.gpu
def test_veil_larger_spread_spreads_more(loaded_lens):
    """A wider spread pushes more energy off a centred small frame, so the
    on-frame veil total drops (physical: the halo washes wider)."""
    small = loaded_lens.render_point_flare(96, 96, _cfg(spread=0.10))
    wide = loaded_lens.render_point_flare(96, 96, _cfg(spread=0.60))
    assert _veil_total(wide) < _veil_total(small)


# ---------------------------------------------------------------------------
# Analytic radial falloff via the debug reference (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
def test_veil_reference_matches_generalized_lorentzian(loaded_lens, p):
    """The mono GSF reference is g(r) = (a^2/(r^2+a^2))^p exactly (evaluated per
    texel, so the ratio to the centre is analytic to float precision)."""
    d = ghostlight._ghostlight._render_veil_debug(
        96, 96, loaded_lens, loaded_lens.calibration(),
        _cfg(veil=True, falloff=p))
    ref = np.asarray(d["reference"])
    N = d["grid"]
    a = d["core_texels"]
    assert d["falloff"] == pytest.approx(p)
    c = N // 2
    centre = ref[c, c]
    assert centre > 0.0
    for dist in (16, 32, 64, 128):
        base = (a * a) / (dist * dist + a * a)
        expected = base ** p
        measured = ref[c, c + dist] / centre
        assert measured == pytest.approx(expected, rel=1e-3, abs=1e-6)


@pytest.mark.gpu
def test_veil_core_scales_with_spread(loaded_lens):
    """The GSF core radius on the sensor scales linearly with veil_spread."""
    d1 = ghostlight._ghostlight._render_veil_debug(
        96, 96, loaded_lens, loaded_lens.calibration(), _cfg(spread=0.10))
    d2 = ghostlight._ghostlight._render_veil_debug(
        96, 96, loaded_lens, loaded_lens.calibration(), _cfg(spread=0.20))
    assert d1["core_mm"] > 0.0
    assert d2["core_mm"] / d1["core_mm"] == pytest.approx(2.0, rel=1e-3)
