"""Tests for ghostlight.source_sampling — CPU only, no GPU required."""

import numpy as np
import pytest

import ghostlight
from ghostlight import source_sampling


# ---------------------------------------------------------------------------
# Shared sampler properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 64, 257])
def test_weights_sum_to_one_disk(n):
    pts = source_sampling.sample_disk(0.01, n=n)
    assert pts.shape == (n, 3)
    assert pts.dtype == np.float32
    assert pts[:, 2].sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 257])
def test_weights_sum_to_one_rect(n):
    pts = source_sampling.sample_rect(0.02, 0.01, n=n)
    assert pts.shape == (n, 3)
    assert pts[:, 2].sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 257])
@pytest.mark.parametrize("sides", [3, 5, 6, 8])
def test_weights_sum_to_one_polygon(n, sides):
    pts = source_sampling.sample_polygon(0.02, sides, n=n)
    assert pts.shape == (n, 3)
    assert pts[:, 2].sum() == pytest.approx(1.0, abs=1e-5)


def test_single_sample_is_center():
    for pts in (source_sampling.sample_point(), source_sampling.sample_disk(0.05, n=1),
                source_sampling.sample_rect(0.05, 0.02, n=1), source_sampling.sample_square(0.05, n=1)):
        assert pts.shape == (1, 3)
        assert pts[0, 0] == 0.0 and pts[0, 1] == 0.0 and pts[0, 2] == 1.0


def test_samplers_deterministic():
    a = source_sampling.sample_disk(0.03, n=50)
    b = source_sampling.sample_disk(0.03, n=50)
    np.testing.assert_array_equal(a, b)
    a = source_sampling.sample_rect(0.03, 0.01, n=50)
    b = source_sampling.sample_rect(0.03, 0.01, n=50)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Points lie inside the shape and cover it
# ---------------------------------------------------------------------------

def test_disk_points_inside_radius():
    r = 0.02
    pts = source_sampling.sample_disk(r, n=200)
    dist = np.hypot(pts[:, 0], pts[:, 1])
    assert dist.max() <= r + 1e-7


def test_ellipse_points_inside():
    rx, ry = 0.03, 0.01
    pts = source_sampling.sample_disk(rx, ry, n=200)
    assert np.all((pts[:, 0] / rx) ** 2 + (pts[:, 1] / ry) ** 2 <= 1.0 + 1e-6)


def test_rect_points_inside_and_cover_quadrants():
    hw, hh = 0.04, 0.015
    pts = source_sampling.sample_rect(hw, hh, n=200)
    assert np.abs(pts[:, 0]).max() <= hw + 1e-7
    assert np.abs(pts[:, 1]).max() <= hh + 1e-7
    # Area-uniform sampling must land in all four quadrants
    assert (pts[:, 0] > 0).any() and (pts[:, 0] < 0).any()
    assert (pts[:, 1] > 0).any() and (pts[:, 1] < 0).any()


def test_disk_mean_near_center():
    pts = source_sampling.sample_disk(0.02, n=512)
    # Uniform-by-area disk sampling: centroid tends to the center
    assert abs(pts[:, 0].mean()) < 0.002
    assert abs(pts[:, 1].mean()) < 0.002


def test_square_matches_rect():
    np.testing.assert_array_equal(source_sampling.sample_square(0.02, n=32),
                                  source_sampling.sample_rect(0.02, 0.02, n=32))


# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------

def test_polygon_points_inside_circumradius():
    r = 0.03
    pts = source_sampling.sample_polygon(r, 6, n=300)
    assert np.hypot(pts[:, 0], pts[:, 1]).max() <= r + 1e-7


def test_polygon_inside_all_edge_halfplanes():
    """Every sample must sit inside every edge of the regular polygon."""
    r, sides = 0.03, 5
    verts = source_sampling.polygon_vertices(r, sides)
    pts = source_sampling.sample_polygon(r, sides, n=400)[:, :2]
    for i in range(sides):
        a = verts[i]
        b = verts[(i + 1) % sides]
        edge = b - a
        # Inward normal points toward the center (origin); cross(edge, p-a)
        # must share sign with cross(edge, center-a) for every sample.
        center_side = edge[0] * (0.0 - a[1]) - edge[1] * (0.0 - a[0])
        cross = edge[0] * (pts[:, 1] - a[1]) - edge[1] * (pts[:, 0] - a[0])
        assert np.all(cross * center_side >= -1e-9), f"edge {i}"


def test_polygon_deterministic_and_sides_clamped():
    np.testing.assert_array_equal(source_sampling.sample_polygon(0.02, 6, n=50),
                                  source_sampling.sample_polygon(0.02, 6, n=50))
    # < 3 sides clamps to a triangle rather than erroring.
    assert source_sampling.sample_polygon(0.02, 2, n=30).shape == (30, 3)


def test_polygon_vertices_shape_and_radius():
    v = source_sampling.polygon_vertices(0.05, 7)
    assert v.shape == (7, 2)
    assert np.hypot(v[:, 0], v[:, 1]).max() == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_rotate_offsets_preserves_radius_and_weights():
    pts = source_sampling.sample_rect(0.03, 0.01, n=64)
    rot = source_sampling.rotate_offsets(pts, np.deg2rad(37.0))
    assert rot.shape == pts.shape
    np.testing.assert_allclose(np.hypot(rot[:, 0], rot[:, 1]),
                               np.hypot(pts[:, 0], pts[:, 1]), atol=1e-7)
    np.testing.assert_array_equal(rot[:, 2], pts[:, 2])


def test_rotate_offsets_zero_is_noop():
    pts = source_sampling.sample_rect(0.03, 0.01, n=16)
    assert source_sampling.rotate_offsets(pts, 0.0) is pts


def test_rotate_90_swaps_axes():
    pts = source_sampling.sample_rect(0.03, 0.01, n=32)
    rot = source_sampling.rotate_offsets(pts, np.deg2rad(90.0))
    # (x, y) -> (-y, x)
    np.testing.assert_allclose(rot[:, 0], -pts[:, 1], atol=1e-7)
    np.testing.assert_allclose(rot[:, 1], pts[:, 0], atol=1e-7)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunk_offsets_reassembles():
    pts = source_sampling.sample_disk(0.02, n=50)
    chunks = source_sampling.chunk_offsets(pts, 8)
    assert len(chunks) == 7  # 6 full chunks of 8 + one of 2
    np.testing.assert_array_equal(np.concatenate(chunks, axis=0), pts)
    # Chunk rows keep their global weights
    assert sum(c[:, 2].sum() for c in chunks) == pytest.approx(1.0, abs=1e-5)


def test_chunk_offsets_rejects_bad_size():
    pts = source_sampling.sample_disk(0.02, n=8)
    with pytest.raises(ValueError):
        source_sampling.chunk_offsets(pts, 0)


def test_samplers_reject_bad_n():
    with pytest.raises(ValueError):
        source_sampling.sample_disk(0.02, n=0)
    with pytest.raises(ValueError):
        source_sampling.sample_rect(0.02, 0.02, n=-1)


# ---------------------------------------------------------------------------
# Wrapper validation (raises before any GPU work)
# ---------------------------------------------------------------------------

def test_render_source_flare_rejects_bad_offsets(loaded_lens):
    cfg = ghostlight.PointFlareConfig()
    with pytest.raises(ValueError):
        loaded_lens.render_source_flare(np.zeros((0, 3), np.float32), 32, 32, cfg)
    with pytest.raises(ValueError):
        loaded_lens.render_source_flare(np.zeros((4, 2), np.float32), 32, 32, cfg)
    with pytest.raises(ValueError):
        loaded_lens.render_source_flare(np.zeros(3, np.float32), 32, 32, cfg)
