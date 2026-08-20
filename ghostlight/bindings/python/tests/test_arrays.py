"""Tests for the ghostlight._arrays helper utilities."""

import numpy as np
import pytest
import ghostlight
from ghostlight import _arrays


def _make_ghost_dict(h=8, w=8):
    r = np.random.rand(h, w).astype(np.float32)
    g = np.random.rand(h, w).astype(np.float32)
    b = np.random.rand(h, w).astype(np.float32)
    return {"ghost_r": r, "ghost_g": g, "ghost_b": b,
            "width": w, "height": h}


# ---------------------------------------------------------------------------
# ghost_to_hwc
# ---------------------------------------------------------------------------

def test_ghost_to_hwc_shape():
    d = _make_ghost_dict(16, 32)
    hwc = _arrays.ghost_to_hwc(d)
    assert hwc.shape == (16, 32, 3)


def test_ghost_to_hwc_channel_order():
    d = _make_ghost_dict()
    hwc = _arrays.ghost_to_hwc(d)
    np.testing.assert_array_equal(hwc[..., 0], d["ghost_r"])
    np.testing.assert_array_equal(hwc[..., 1], d["ghost_g"])
    np.testing.assert_array_equal(hwc[..., 2], d["ghost_b"])


def test_ghost_to_hwc_dtype_float32():
    d = _make_ghost_dict()
    hwc = _arrays.ghost_to_hwc(d)
    assert hwc.dtype == np.float32


# ---------------------------------------------------------------------------
# planar_from_hwc
# ---------------------------------------------------------------------------

def test_planar_from_hwc_roundtrip():
    h, w = 8, 8
    original_r = np.random.rand(h, w).astype(np.float32)
    original_g = np.random.rand(h, w).astype(np.float32)
    original_b = np.random.rand(h, w).astype(np.float32)
    hwc = np.stack([original_r, original_g, original_b], axis=-1)
    r, g, b = _arrays.planar_from_hwc(hwc)
    np.testing.assert_array_almost_equal(r, original_r)
    np.testing.assert_array_almost_equal(g, original_g)
    np.testing.assert_array_almost_equal(b, original_b)


def test_planar_from_hwc_shapes():
    h, w = 6, 10
    hwc = np.zeros((h, w, 3), dtype=np.float32)
    r, g, b = _arrays.planar_from_hwc(hwc)
    assert r.shape == (h, w)
    assert g.shape == (h, w)
    assert b.shape == (h, w)


def test_planar_from_hwc_returns_float32():
    hwc = np.zeros((4, 4, 3), dtype=np.float64)
    r, g, b = _arrays.planar_from_hwc(hwc)
    assert r.dtype == np.float32
    assert g.dtype == np.float32
    assert b.dtype == np.float32


def test_ghost_planar_roundtrip_via_hwc():
    """ghost_to_hwc followed by planar_from_hwc must recover the originals."""
    d = _make_ghost_dict()
    hwc = _arrays.ghost_to_hwc(d)
    r, g, b = _arrays.planar_from_hwc(hwc)
    np.testing.assert_array_almost_equal(r, d["ghost_r"])
    np.testing.assert_array_almost_equal(g, d["ghost_g"])
    np.testing.assert_array_almost_equal(b, d["ghost_b"])
