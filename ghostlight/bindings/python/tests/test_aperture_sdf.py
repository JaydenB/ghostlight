"""Tests for the image-aperture-stop signed distance field (HURB edge kicks).

Host gates (no GPU) exercise the bake via ghostlight._ghostlight._aperture_sdf_debug:
  - distance accuracy vs an analytic disc / half-plane
  - edge-normal direction
  - anisotropic (aspect != 1, non-square) mm-per-texel scale
  - half-texel boundary correction
  - stop-only scoping (front / non-stop image mattes get no SDF)
  - degenerate images

GPU gates drive a full ghost render to prove the kick actually fires only when
HURB is on, redistributes (not creates) energy, and leaves the off path alone.
"""

import numpy as np
import pytest
import ghostlight

_impl = ghostlight._ghostlight


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_image_stop_system(pixels, semi_diameter=5.0, aspect=1.0,
                            semi_aperture=10.0):
    """Single flat image-aperture stop (matches test_aperture_images helper)."""
    sys = ghostlight.OpticalSystem()
    sys.name = "sdf_image_stop"
    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = semi_aperture
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    stop.aperture_aspect = aspect
    stop.aperture_semi_diameter = semi_diameter
    sys.surfaces.append(stop)
    sys.finalize()
    img = sys.aperture_images[0]
    img.semi_diameter = semi_diameter
    img.pixels = pixels.astype(np.float32, copy=False)
    return sys


def _pixel_world_xy(W, H, sd, aspect):
    """World (x_mm, y_mm) of each texel center, matching the trace UV mapping."""
    i = np.arange(W)
    j = np.arange(H)
    x = aspect * sd * ((2.0 * i + 1.0) / W - 1.0)   # columns -> x
    y = sd * ((2.0 * j + 1.0) / H - 1.0)            # rows    -> y
    return np.meshgrid(x, y)   # (H,W) each


def _disc_mask(W, H, sd, radius_mm, aspect=1.0):
    xx, yy = _pixel_world_xy(W, H, sd, aspect)
    return (np.sqrt(xx * xx + yy * yy) < radius_mm).astype(np.float32)


# ---------------------------------------------------------------------------
# Host gates
# ---------------------------------------------------------------------------

def test_disc_distance_accuracy():
    W = H = 256
    sd = 8.0
    R = 5.0
    sys = _make_image_stop_system(_disc_mask(W, H, sd, R), semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    assert d["stop_index"] == 0
    sdf = np.asarray(d["sdf"])
    assert sdf.shape == (H, W)

    xx, yy = _pixel_world_xy(W, H, sd, 1.0)
    r = np.sqrt(xx * xx + yy * yy)
    analytic = R - r                      # +inside, -outside
    tol = 1.2 * max(d["sx"], d["sy"])     # ~one texel

    # Compare in an annulus around the edge (the field is exact there; far
    # corners clamp against the image border so skip |analytic| large).
    band = np.abs(analytic) < (R * 0.6)
    err = np.abs(sdf[band] - analytic[band])
    assert np.percentile(err, 95) < tol, f"95th-pct err {np.percentile(err,95):.4f} vs tol {tol:.4f}"


def test_disc_normal_is_radial():
    W = H = 256
    sd = 8.0
    R = 5.0
    sys = _make_image_stop_system(_disc_mask(W, H, sd, R), semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    nx = np.asarray(d["nx"])
    ny = np.asarray(d["ny"])
    xx, yy = _pixel_world_xy(W, H, sd, 1.0)
    r = np.sqrt(xx * xx + yy * yy)
    # Evaluate on a ring near the edge where the normal is well defined.
    ring = (np.abs(R - r) < 1.0) & (r > 0.5)
    rhx = xx[ring] / r[ring]
    rhy = yy[ring] / r[ring]
    dotp = np.abs(nx[ring] * rhx + ny[ring] * rhy)   # sign irrelevant
    assert np.median(dotp) > 0.98


def test_sign_matches_matte_boundary():
    W = H = 128
    sd = 6.0
    R = 4.0
    mask = _disc_mask(W, H, sd, R)
    sys = _make_image_stop_system(mask, semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    sdf = np.asarray(d["sdf"])
    inside = mask > 0.5
    # Away from the boundary (|sdf| > one texel) the sign must match the matte.
    far = np.abs(sdf) > 1.5 * max(d["sx"], d["sy"])
    assert np.all((sdf[far] > 0) == inside[far])


def test_anisotropic_scale_vertical_edge():
    # Vertical edge at x = x_edge: distance exercises sx = 2*aspect*sd/W.
    W, H = 200, 120
    sd = 5.0
    aspect = 2.0
    x_edge = 1.3
    xx, yy = _pixel_world_xy(W, H, sd, aspect)
    mask = (xx < x_edge).astype(np.float32)
    sys = _make_image_stop_system(mask, semi_diameter=sd, aspect=aspect)
    d = _impl._aperture_sdf_debug(sys)
    assert d["sx"] == pytest.approx(2.0 * aspect * sd / W, rel=1e-5)
    assert d["sy"] == pytest.approx(2.0 * sd / H, rel=1e-5)
    sdf = np.asarray(d["sdf"])
    analytic = x_edge - xx
    band = np.abs(analytic) < 2.0
    err = np.abs(sdf[band] - analytic[band])
    assert np.percentile(err, 95) < 1.2 * d["sx"]


def test_anisotropic_scale_horizontal_edge():
    W, H = 120, 200
    sd = 5.0
    y_edge = -0.7
    xx, yy = _pixel_world_xy(W, H, sd, 1.0)
    mask = (yy < y_edge).astype(np.float32)
    sys = _make_image_stop_system(mask, semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    sdf = np.asarray(d["sdf"])
    analytic = y_edge - yy
    band = np.abs(analytic) < 2.0
    err = np.abs(sdf[band] - analytic[band])
    assert np.percentile(err, 95) < 1.2 * d["sy"]


def test_half_texel_correction():
    # Straight edge falling exactly between two columns: the inside column
    # adjacent to the boundary should read ~half a texel, not a full one.
    W = H = 64
    sd = 4.0
    # Edge between col 31 and 32 -> x boundary at texel-gap center.
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, :32] = 1.0
    sys = _make_image_stop_system(mask, semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    sdf = np.asarray(d["sdf"])
    sx = d["sx"]
    near = sdf[H // 2, 31]     # inside column touching the edge
    assert near == pytest.approx(0.5 * sx, abs=0.35 * sx)


def test_stop_only_scoping_non_stop_image_gets_no_sdf():
    # Image aperture on a NON-stop surface + an analytic stop -> no SDF target.
    sys = ghostlight.OpticalSystem()
    s0 = ghostlight.Surface()
    s0.radius = 0.0
    s0.thickness = 5.0
    s0.ior = 1.0
    s0.abbe_v = 0.0
    s0.semi_aperture = 10.0
    s0.disp_model = ghostlight.DispersionModel.AIR
    s0.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    s0.aperture_semi_diameter = 5.0
    sys.surfaces.append(s0)

    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = 8.0
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(stop)
    sys.finalize()
    sys.aperture_images[0].semi_diameter = 5.0
    sys.aperture_images[0].pixels = _disc_mask(64, 64, 5.0, 3.0)

    d = _impl._aperture_sdf_debug(sys)
    assert d["stop_index"] == -1
    assert "sdf" not in d


def test_image_stop_is_scoped():
    sys = _make_image_stop_system(_disc_mask(64, 64, 5.0, 3.0), semi_diameter=5.0)
    d = _impl._aperture_sdf_debug(sys)
    assert d["stop_index"] == 0
    assert np.asarray(d["sdf"]).shape == (64, 64)


def test_empty_image_no_sdf():
    sys = ghostlight.OpticalSystem()
    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = 8.0
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    stop.aperture_semi_diameter = 5.0
    sys.surfaces.append(stop)
    sys.finalize()   # no pixels loaded
    d = _impl._aperture_sdf_debug(sys)
    assert d["stop_index"] == -1


def test_all_white_mask_has_no_near_edge_kicks():
    # A fully-open image has no edge -> distances are all "far" -> no kicks.
    W = H = 64
    sd = 5.0
    sys = _make_image_stop_system(np.ones((H, W), np.float32), semi_diameter=sd)
    d = _impl._aperture_sdf_debug(sys)
    sdf = np.asarray(d["sdf"])
    assert np.all(np.isfinite(sdf))
    assert sdf.min() > 100.0   # everything reads as very far inside


# ---------------------------------------------------------------------------
# GPU gates
# ---------------------------------------------------------------------------

def _requires_cuda():
    if not _impl._cuda_available():
        pytest.skip("no CUDA device")


def _ghost_lens_with_image_stop(sd=8.0, R=6.0, W=512):
    """Singlet + a filled-disc image-aperture stop (ghost pair on the glass)."""
    sys = ghostlight.OpticalSystem()
    sys.name = "sdf_ghost"

    s0 = ghostlight.Surface()
    s0.radius = 47.0
    s0.thickness = 5.0
    s0.ior = 1.5168
    s0.abbe_v = 64.17
    s0.semi_aperture = 15.0
    s0.disp_model = ghostlight.DispersionModel.ABBE
    sys.surfaces.append(s0)

    s1 = ghostlight.Surface()
    s1.radius = -47.0
    s1.thickness = 10.0
    s1.ior = 1.0
    s1.abbe_v = 0.0
    s1.semi_aperture = 15.0
    s1.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(s1)

    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = sd
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.IMAGE)
    stop.aperture_semi_diameter = sd
    sys.surfaces.append(stop)

    sys.finalize()
    sys.aperture_images[-1].semi_diameter = sd
    sys.aperture_images[-1].pixels = _disc_mask(W, W, sd, R)
    return sys


def _point_cfg(hurb):
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = 0.35
    cfg.source_y = 0.35
    cfg.source_r = 6.0
    cfg.source_g = 5.0
    cfg.source_b = 4.0
    cfg.ray_grid = 64
    cfg.spectral_samples = 6
    cfg.diffraction.hurb = hurb
    return cfg


def test_hurb_off_is_deterministic_on_image_stop():
    _requires_cuda()
    lens = _ghost_lens_with_image_stop()
    a = lens.render_point_flare(96, 96, _point_cfg(False))
    b = lens.render_point_flare(96, 96, _point_cfg(False))
    for k in ("ghost_r", "ghost_g", "ghost_b"):
        assert np.array_equal(a[k], b[k])


def test_hurb_changes_the_image_stop_render():
    _requires_cuda()
    lens = _ghost_lens_with_image_stop()
    off = lens.render_point_flare(96, 96, _point_cfg(False))
    on = lens.render_point_flare(96, 96, _point_cfg(True))

    off_s = np.stack([off["ghost_r"], off["ghost_g"], off["ghost_b"]])
    on_s = np.stack([on["ghost_r"], on["ghost_g"], on["ghost_b"]])

    # The kick must actually change the render (pre-feature these were identical).
    assert not np.array_equal(off_s, on_s)
    assert np.abs(on_s - off_s).max() > 0

    # Energy is redistributed, not created: totals stay the same order (kicks can
    # push a little light off-sensor, so allow a modest loss, no large gain).
    off_sum = float(off_s.sum())
    on_sum = float(on_s.sum())
    assert on_sum <= off_sum * 1.05
    assert on_sum >= off_sum * 0.5
