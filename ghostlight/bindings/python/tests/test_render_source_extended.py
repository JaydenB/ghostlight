"""Extended tests for render_source_flare — all require GPU.

render_source_flare renders an extended source as N weighted angular offsets
around the screen-space center; render_point_flare is now the degenerate
one-offset case.  Comparisons are tolerance-based (never byte-exact): GPU
atomicAdd ordering makes repeated renders differ at the ulp level.
"""

import numpy as np
import pytest

import ghostlight
from ghostlight import source_sampling

pytestmark = pytest.mark.gpu


def _cfg(ray_grid=16, spectral_samples=4, **kwargs):
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = ray_grid
    cfg.spectral_samples = spectral_samples
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _total(out):
    return float(out["ghost_r"].sum() + out["ghost_g"].sum() + out["ghost_b"].sum())


def _assert_close_images(a, b, rel_of_peak=1e-3):
    """Max abs difference bounded by a fraction of the brighter image's peak."""
    peak = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-20)
    assert float(np.abs(a - b).max()) <= rel_of_peak * peak


# ---------------------------------------------------------------------------
# Parity with render_point_flare
# ---------------------------------------------------------------------------

def test_single_center_offset_matches_point_flare(loaded_lens):
    """One zero-offset full-weight row is exactly the point-flare code path."""
    cfg = _cfg(source_x=0.3, source_y=0.6,
               source_r=10.0, source_g=10.0, source_b=10.0)
    out_pt = loaded_lens.render_point_flare(64, 64, cfg)
    out_src = loaded_lens.render_source_flare(source_sampling.sample_point(), 64, 64, cfg)
    for key in ("ghost_r", "ghost_g", "ghost_b"):
        _assert_close_images(out_pt[key], out_src[key])


def test_degenerate_shape_matches_point_flare(loaded_lens):
    """N coincident samples with 1/N weights must average back to the point."""
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    offsets = np.zeros((8, 3), dtype=np.float32)
    offsets[:, 2] = 1.0 / 8.0
    out_pt = loaded_lens.render_point_flare(64, 64, cfg)
    out_src = loaded_lens.render_source_flare(offsets, 64, 64, cfg)
    for key in ("ghost_r", "ghost_g", "ghost_b"):
        _assert_close_images(out_pt[key], out_src[key])


# ---------------------------------------------------------------------------
# Linearity: chunked accumulation == single call
# ---------------------------------------------------------------------------

def test_chunked_sum_matches_single_call(loaded_lens):
    """Summing per-chunk renders must reproduce the full-offsets render.

    cull_dead_pairs is disabled so pair activation cannot differ between the
    coarse probe of a chunk and that of the full set.
    """
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0,
               cull_dead_pairs=False)
    offsets = source_sampling.sample_disk(0.02, n=16)
    out_full = loaded_lens.render_source_flare(offsets, 64, 64, cfg)
    acc = {k: np.zeros((64, 64), dtype=np.float64)
           for k in ("ghost_r", "ghost_g", "ghost_b")}
    for chunk in source_sampling.chunk_offsets(offsets, 5):
        out = loaded_lens.render_source_flare(chunk, 64, 64, cfg)
        for k in acc:
            acc[k] += out[k]
    for k in acc:
        _assert_close_images(out_full[k], acc[k].astype(np.float32),
                             rel_of_peak=5e-3)


def test_weight_scaling_is_linear(loaded_lens):
    """Doubling every sample weight must double the output."""
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0,
               cull_dead_pairs=False)
    offsets = source_sampling.sample_disk(0.01, n=8)
    doubled = offsets.copy()
    doubled[:, 2] *= 2.0
    out1 = loaded_lens.render_source_flare(offsets, 64, 64, cfg)
    out2 = loaded_lens.render_source_flare(doubled, 64, 64, cfg)
    s1, s2 = _total(out1), _total(out2)
    if s1 > 0:
        assert s2 / s1 == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Extended shapes change the picture the right way
# ---------------------------------------------------------------------------

def test_disk_source_differs_from_point(loaded_lens):
    """A wide disk source must produce a visibly different ghost pattern."""
    cfg = _cfg(source_x=0.3, source_y=0.5,
               source_r=10.0, source_g=10.0, source_b=10.0)
    out_pt = loaded_lens.render_point_flare(64, 64, cfg)
    out_disk = loaded_lens.render_source_flare(
        source_sampling.sample_disk(np.deg2rad(3.0), n=32), 64, 64, cfg)
    peak = max(out_pt["ghost_r"].max(), 1e-20)
    assert np.abs(out_pt["ghost_r"] - out_disk["ghost_r"]).max() > 0.01 * peak


def test_tiny_disk_approaches_point(loaded_lens):
    """A sun-sized disk (0.265 deg radius) barely differs from the point."""
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    out_pt = loaded_lens.render_point_flare(64, 64, cfg)
    out_sun = loaded_lens.render_source_flare(
        source_sampling.sample_disk(np.deg2rad(0.265), n=16), 64, 64, cfg)
    total_pt, total_sun = _total(out_pt), _total(out_sun)
    if total_pt > 0:
        assert total_sun == pytest.approx(total_pt, rel=0.25)


def test_disk_softens_peak(loaded_lens):
    """Averaging over a wide disk must not brighten the hottest pixel."""
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    out_pt = loaded_lens.render_point_flare(64, 64, cfg)
    out_disk = loaded_lens.render_source_flare(
        source_sampling.sample_disk(np.deg2rad(4.0), n=64), 64, 64, cfg)
    assert out_disk["ghost_r"].max() <= out_pt["ghost_r"].max() * 1.10


def test_rect_source_renders_finite(loaded_lens):
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    out = loaded_lens.render_source_flare(
        source_sampling.sample_rect(np.deg2rad(2.0), np.deg2rad(1.0), n=24), 48, 36, cfg)
    assert out["ghost_r"].shape == (36, 48)
    assert out["ghost_r"].dtype == np.float32
    for k in ("ghost_r", "ghost_g", "ghost_b"):
        assert np.all(np.isfinite(out[k]))
        # CMF-lobe negatives (a few % of peak); no blur pass smooths them.
        peak = max(float(out[k].max()), 1e-30)
        assert out[k].min() >= -0.1 * peak


def test_polygon_source_renders_finite(loaded_lens):
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    offsets = source_sampling.rotate_offsets(
        source_sampling.sample_polygon(np.deg2rad(1.5), 6, n=48), np.deg2rad(15.0))
    out = loaded_lens.render_source_flare(offsets, 64, 64, cfg)
    for k in ("ghost_r", "ghost_g", "ghost_b"):
        assert np.all(np.isfinite(out[k]))
        # CMF-lobe negatives (a few % of peak); no blur pass smooths them.
        peak = max(float(out[k].max()), 1e-30)
        assert out[k].min() >= -0.1 * peak
    assert _total(out) > 0.0


def test_rect_rotation_changes_pattern_but_not_energy(loaded_lens):
    """Rotating an anisotropic (rect) source rotates the flare (pattern
    differs) while conserving total energy."""
    cfg = _cfg(source_x=0.32, source_y=0.5,
               source_r=10.0, source_g=10.0, source_b=10.0,
               cull_dead_pairs=False)
    base = source_sampling.sample_rect(np.deg2rad(5.0), np.deg2rad(1.0), n=64)
    out0 = loaded_lens.render_source_flare(base, 64, 64, cfg)
    out90 = loaded_lens.render_source_flare(
        source_sampling.rotate_offsets(base, np.deg2rad(90.0)), 64, 64, cfg)
    t0, t90 = _total(out0), _total(out90)
    if t0 > 0:
        assert t90 == pytest.approx(t0, rel=0.05)  # energy conserved
        peak = max(out0["ghost_r"].max(), 1e-20)
        assert np.abs(out0["ghost_r"] - out90["ghost_r"]).max() > 0.01 * peak


# ---------------------------------------------------------------------------
# Interop with existing FlareConfig machinery
# ---------------------------------------------------------------------------

def test_ghost_filter_applies_to_source_render(loaded_lens):
    """An INCLUDE filter matching no enumerated pair must zero the ghost output.

    (An *empty* pair list is deliberately a no-op in the core, so the filter
    here names a pair that cannot exist.)
    """
    cfg = _cfg(source_r=10.0, source_g=10.0, source_b=10.0)
    flt = ghostlight.GhostFilter()
    flt.mode = ghostlight.GhostFilter.Mode.INCLUDE
    flt.pairs = [(97, 98)]
    cfg.ghost_filter = flt
    out = loaded_lens.render_source_flare(
        source_sampling.sample_disk(np.deg2rad(1.0), n=8), 32, 32, cfg)
    assert out["ghost_r"].sum() == pytest.approx(0.0, abs=1e-6)


def test_no_source_keys_in_output(loaded_lens):
    cfg = _cfg()
    out = loaded_lens.render_source_flare(source_sampling.sample_point(), 32, 32, cfg)
    assert "source_r" not in out
    assert "source_g" not in out
    assert "source_b" not in out
