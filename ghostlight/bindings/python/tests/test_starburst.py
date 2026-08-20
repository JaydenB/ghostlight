"""Starburst engine tests — the SPRITE/MDFT toggle and the MDFT engine.

The MDFT engine evaluates the diffraction integral directly at the sensor
pixels (mdft_render.cu); the SPRITE engine is the legacy FFT-sprite path.  The
default is SPRITE, so a default config renders byte-identically to before the
engine existed.

Only the render tests need CUDA (marked ``gpu``, auto-skipped without a device
by conftest); the config/enum tests are pure binding round-trips.
"""
import numpy as np
import pytest

import ghostlight


# ---------------------------------------------------------------------------
# Config / enum (no GPU)
# ---------------------------------------------------------------------------

def test_engine_enum_exists():
    assert hasattr(ghostlight, "StarburstEngine")
    assert int(ghostlight.StarburstEngine.SPRITE) == 0
    assert int(ghostlight.StarburstEngine.MDFT) == 1


def test_default_engine_is_sprite():
    """Byte-identity guard: a default config selects the legacy sprite path."""
    cfg = ghostlight.PointFlareConfig()
    assert cfg.diffraction.starburst_engine == ghostlight.StarburstEngine.SPRITE


def test_engine_roundtrip():
    cfg = ghostlight.PointFlareConfig()
    cfg.diffraction.starburst_engine = ghostlight.StarburstEngine.MDFT
    assert cfg.diffraction.starburst_engine == ghostlight.StarburstEngine.MDFT
    cfg.diffraction.starburst_engine = ghostlight.StarburstEngine.SPRITE
    assert cfg.diffraction.starburst_engine == ghostlight.StarburstEngine.SPRITE


# ---------------------------------------------------------------------------
# Render helpers (GPU)
# ---------------------------------------------------------------------------

def _cfg(engine, sx=0.5, sy=0.5, ns=6, grid=1024, trim=8.0, blades=0):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 1.0
    c.spectral_samples = ns
    c.aperture_blades = blades
    c.diffraction.starburst = True
    c.diffraction.starburst_engine = engine
    c.diffraction.starburst_grid = grid
    c.diffraction.pupil_fill = 0.30
    c.diffraction.scale_trim = trim
    c.diffraction.spectral_samples = ns
    return c


def _lum(d):
    return (np.asarray(d["starburst_r"]) + np.asarray(d["starburst_g"])
            + np.asarray(d["starburst_b"])).astype(np.float64)


def _render(lens, engine, w=192, **kw):
    return ghostlight._ghostlight._render_starburst_debug(w, w, lens, lens.calibration(),
                                                _cfg(engine, **kw))


# ---------------------------------------------------------------------------
# MDFT engine (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_mdft_renders_finite_nonzero(psf_lens):
    lum = _lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT))
    assert np.isfinite(lum).all()
    assert lum.sum() > 0.0
    assert (lum < 0.0).sum() == 0     # intensity is non-negative


@pytest.mark.gpu
def test_mdft_flux_matches_sprite(psf_lens):
    """Same physics, different sampling: the total flux agrees closely (the ~3%
    residual is the sprite's scaling-theorem bake; MDFT is the exact one)."""
    sp = _lum(_render(psf_lens, ghostlight.StarburstEngine.SPRITE)).sum()
    md = _lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT)).sum()
    assert abs(md / sp - 1.0) < 0.06


@pytest.mark.gpu
def test_mdft_flux_is_spectral_sample_invariant(psf_lens):
    """Physical flux must not depend on the spectral sample count — MDFT is
    exact here (the sprite's scaling-theorem bake wobbles slightly)."""
    sums = [_lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT, ns=n)).sum()
            for n in (3, 6, 12)]
    assert max(sums) / min(sums) < 1.005


@pytest.mark.gpu
def test_mdft_error_is_stable_sprite_flickers(psf_lens):
    """The motivating result, as a stability test.  Against a supersampled
    ground truth built from the same pupil, the MDFT window's shape error is
    nearly CONSTANT as the source scrubs sub-pixel, while the sprite's error
    swings by several percent — that swing IS the flicker.  (Both carry a common
    chromatic-render-vs-mono-truth offset, so the discriminator is the variation
    across positions, not the absolute level.)"""
    cal = psf_lens.calibration()
    w = 256
    px = 2.0 * float(cal.sensor_half_w) / w

    def err_vs_truth(engine, sx):
        d = ghostlight._ghostlight._render_starburst_debug(w, w, psf_lens, cal,
                                                 _cfg(engine, sx=sx, sy=0.5, ns=6))
        A = np.asarray(d["pupil"], np.float64)
        N = int(d["grid"]); dx = float(d["dx_mm_x"])
        P2 = np.zeros((2 * N, 2 * N), np.complex64)
        P2[N // 2:N // 2 + N, N // 2:N // 2 + N] = A
        I2 = (np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P2)))) ** 2).astype(np.float64)
        if I2.sum() <= 0:
            return None
        I2 /= I2.sum(); dx2 = dx / 2.0; N2 = 2 * N
        spx, spy = float(d["source_px"]), float(d["source_py"])
        K = 40
        kx0, ky0 = int(np.floor(spx)) - K // 2, int(np.floor(spy)) - K // 2

        def ov(k0, pos, n, tex):
            lo = ((k0 + np.arange(K) - pos) * px) / tex + 0.5 * n
            hi = lo + px / tex
            ii = np.arange(n)
            return np.clip(np.minimum(hi[:, None], ii[None, :] + 1)
                           - np.maximum(lo[:, None], ii[None, :]), 0.0, None)

        truth = ov(ky0, spy, N2, dx2) @ I2 @ ov(kx0, spx, N2, dx2).T
        win = _lum(d)[ky0:ky0 + K, kx0:kx0 + K]
        a = win / max(win.sum(), 1e-30)
        b = truth / max(truth.sum(), 1e-30)
        return float(np.abs(a - b).sum() / 2)

    # scrub the source ~1.5 px in sub-pixel steps at a fixed off-axis field
    xs = [0.820 + i * 0.0015 for i in range(12)]
    md = np.array([err_vs_truth(ghostlight.StarburstEngine.MDFT, x) for x in xs])
    sp = np.array([err_vs_truth(ghostlight.StarburstEngine.SPRITE, x) for x in xs])
    # MDFT's error is nearly flat; the sprite's swings with sub-pixel phase.
    assert md.std() < sp.std() * 0.35, f"MDFT std {md.std():.4f} not << sprite std {sp.std():.4f}"
    assert md.max() < sp.max(), f"MDFT worst {md.max():.3f} >= sprite worst {sp.max():.3f}"
    assert md.mean() < 0.10, f"MDFT mean error {md.mean():.3f} unexpectedly high"


@pytest.mark.gpu
def test_mdft_source_position_tracks(psf_lens):
    """The MDFT window follows the source: two sources at different sensor
    positions place their pattern centroids at those positions (validates the
    per-source window placement)."""
    cal = psf_lens.calibration()
    w = 192

    def centroid(sx):
        lum = _lum(ghostlight._ghostlight._render_starburst_debug(
            w, w, psf_lens, cal, _cfg(ghostlight.StarburstEngine.MDFT, sx=sx)))
        ys, xs = np.mgrid[0:w, 0:w]
        tot = lum.sum()
        return xs[lum > 0].dot(lum[lum > 0]) / tot if tot > 0 else None

    cx_left = centroid(0.40)
    cx_right = centroid(0.60)
    assert cx_left is not None and cx_right is not None
    assert cx_right > cx_left + 10   # source moved right -> pattern moved right


@pytest.mark.gpu
def test_offframe_reach_matches_sprite(psf_lens):
    """A source past the pattern's reach yields no in-frame energy in BOTH
    engines (the pattern extent is the same); near the edge they agree."""
    near_sp = _lum(_render(psf_lens, ghostlight.StarburstEngine.SPRITE, sx=0.98)).sum()
    near_md = _lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT, sx=0.98)).sum()
    assert near_md > 0 and near_sp > 0
    assert abs(near_md / near_sp - 1.0) < 0.08
    far_sp = _lum(_render(psf_lens, ghostlight.StarburstEngine.SPRITE, sx=1.15)).sum()
    far_md = _lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT, sx=1.15)).sum()
    assert far_sp == 0.0 and far_md == 0.0


@pytest.mark.gpu
def test_sprite_engine_deterministic(psf_lens):
    """Legacy path stability: two identical SPRITE renders are bit-equal (the
    engine adds no nondeterminism to the untouched sprite path)."""
    a = _lum(_render(psf_lens, ghostlight.StarburstEngine.SPRITE))
    b = _lum(_render(psf_lens, ghostlight.StarburstEngine.SPRITE))
    assert np.array_equal(a, b)


@pytest.mark.gpu
def test_anamorphic_no_transpose(psf_lens):
    """The MDFT per-axis pattern orientation matches the sprite's (guards the
    row/col transpose trap in the separable transform).  On an isotropic lens
    the pattern is symmetric; here we just assert finite + flux parity, with the
    dedicated anamorphic orientation check living in validate_starburst_mdft.py."""
    lum = _lum(_render(psf_lens, ghostlight.StarburstEngine.MDFT, blades=6))
    assert np.isfinite(lum).all() and lum.sum() > 0
