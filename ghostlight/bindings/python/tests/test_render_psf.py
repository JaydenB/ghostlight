"""Tests for render_psf — geometric PSF grid renderer."""

import numpy as np
import pytest

import ghostlight

from _corpus import lens_path

pytestmark = pytest.mark.gpu


def _grid_sources(grid_nx: int, grid_ny: int, max_angle_h: float, max_angle_v: float):
    """Build an (N, 2) array of field-point angles spanning [-max, +max]."""
    if grid_nx == 1:
        xs = np.array([0.0], dtype=np.float32)
    else:
        xs = np.linspace(-max_angle_h, max_angle_h, grid_nx, dtype=np.float32)
    if grid_ny == 1:
        ys = np.array([0.0], dtype=np.float32)
    else:
        ys = np.linspace(-max_angle_v, max_angle_v, grid_ny, dtype=np.float32)
    angles = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    return angles.astype(np.float32)


def test_psf_output_shape_and_keys(loaded_lens):
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 3
    cfg.grid_ny = 3
    cfg.tile_w = 32
    cfg.tile_h = 32
    cfg.tile_extent_mm = 0.1
    cfg.spectral_samples = 4
    cfg.ray_grid = 16

    calib = loaded_lens.calibration()
    sources = _grid_sources(3, 3, calib.max_half_angle_h * 0.5, calib.max_half_angle_v * 0.5)

    out = loaded_lens.render_psf(sources, cfg)

    assert out["composite_w"] == 3 * 32
    assert out["composite_h"] == 3 * 32
    assert out["tile_w"] == 32
    assert out["tile_h"] == 32
    assert out["grid_nx"] == 3
    assert out["grid_ny"] == 3
    assert out["r"].shape == (96, 96)
    assert out["g"].shape == (96, 96)
    assert out["b"].shape == (96, 96)
    assert out["chief_x_mm"].shape == (9,)
    assert out["chief_y_mm"].shape == (9,)
    assert out["r"].dtype == np.float32


def test_psf_on_axis_centred_in_tile(loaded_lens):
    """A single on-axis source should produce a tight PSF centred on the tile."""
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 1
    cfg.grid_ny = 1
    cfg.tile_w = 64
    cfg.tile_h = 64
    cfg.tile_extent_mm = 0.05
    cfg.spectral_samples = 8
    cfg.ray_grid = 32

    sources = np.array([[0.0, 0.0]], dtype=np.float32)
    out = loaded_lens.render_psf(sources, cfg)

    # On-axis chief ray should land at (0, 0) on the sensor.
    assert abs(out["chief_x_mm"][0]) < 1e-3
    assert abs(out["chief_y_mm"][0]) < 1e-3

    # Find the centroid of the rendered PSF in the single tile (r + g + b).
    img = out["r"] + out["g"] + out["b"]
    total = img.sum()
    assert total > 0.0, "no energy hit the tile — chief-ray centering may be broken"

    ys, xs = np.indices(img.shape)
    cx = (xs * img).sum() / total
    cy = (ys * img).sum() / total

    # On-axis PSF should sit near the centre of the (64, 64) tile.
    assert abs(cx - 32.0) < 2.0
    assert abs(cy - 32.0) < 2.0


def test_psf_off_axis_chief_ray_lands_offset(loaded_lens):
    """An off-axis source should have a chief-ray landing offset from origin."""
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 1
    cfg.grid_ny = 1
    cfg.tile_w = 64
    cfg.tile_h = 64
    cfg.tile_extent_mm = 0.05
    cfg.spectral_samples = 4
    cfg.ray_grid = 16

    calib = loaded_lens.calibration()
    angle = calib.max_half_angle_h * 0.5
    sources = np.array([[angle, 0.0]], dtype=np.float32)
    out = loaded_lens.render_psf(sources, cfg)

    # Chief ray should land away from the optical axis.
    assert abs(out["chief_x_mm"][0]) > 0.1
    assert abs(out["chief_y_mm"][0]) < 0.05  # ~symmetric


def test_psf_monochromatic_zeros_gb(loaded_lens):
    """In monochromatic mode, g and b channels must be all zero."""
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 1
    cfg.grid_ny = 1
    cfg.tile_w = 32
    cfg.tile_h = 32
    cfg.tile_extent_mm = 0.05
    cfg.spectral_samples = 4
    cfg.ray_grid = 16
    cfg.monochromatic = True

    sources = np.array([[0.0, 0.0]], dtype=np.float32)
    out = loaded_lens.render_psf(sources, cfg)

    assert out["r"].sum() > 0.0
    assert (out["g"] == 0.0).all()
    assert (out["b"] == 0.0).all()


def test_psf_too_many_sources_raises(loaded_lens):
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 2
    cfg.grid_ny = 2
    cfg.tile_w = 16
    cfg.tile_h = 16
    cfg.tile_extent_mm = 0.05
    cfg.spectral_samples = 4
    cfg.ray_grid = 8

    # 5 sources, only 4 tiles
    sources = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError):
        loaded_lens.render_psf(sources, cfg)


def test_psf_partial_grid_leaves_unused_tiles_zero(loaded_lens):
    """Fewer sources than grid capacity → unused tiles remain zero."""
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = 3
    cfg.grid_ny = 3
    cfg.tile_w = 16
    cfg.tile_h = 16
    cfg.tile_extent_mm = 0.05
    cfg.spectral_samples = 4
    cfg.ray_grid = 16

    # Only one source — tile (0, 0).  Tiles 1..8 must be entirely zero.
    sources = np.array([[0.0, 0.0]], dtype=np.float32)
    out = loaded_lens.render_psf(sources, cfg)

    tw, th = cfg.tile_w, cfg.tile_h
    tile00 = out["r"][0:th, 0:tw]
    assert tile00.sum() > 0.0

    for i in range(1, 9):
        gx, gy = i % 3, i // 3
        tile = out["r"][gy * th:(gy + 1) * th, gx * tw:(gx + 1) * tw]
        assert (tile == 0.0).all(), f"tile (gx={gx}, gy={gy}) should be empty"


# ---------------------------------------------------------------------------
# FIXED_TARGET mode — aim the chief ray at each sensor cell.
# ---------------------------------------------------------------------------

def _sensor_cells(grid_n, calib, half_w, half_h):
    """Replicate the panel's cell builder: raster order (row 0 = -y_mm),
    tan-linear seeds, cell-centre targets. Returns (seeds, targets)."""
    import math
    cal_hw = max(1e-6, float(calib.sensor_half_w))
    cal_hh = max(1e-6, float(calib.sensor_half_h))
    tan_h = math.tan(float(calib.max_half_angle_h))
    tan_v = math.tan(float(calib.max_half_angle_v))
    seeds = np.empty((grid_n * grid_n, 2), np.float32)
    targets = np.empty((grid_n * grid_n, 2), np.float32)
    k = 0
    for row in range(grid_n):
        y = 0.0 if grid_n == 1 else (-half_h + (row + 0.5) * (2 * half_h / grid_n))
        for col in range(grid_n):
            x = 0.0 if grid_n == 1 else (-half_w + (col + 0.5) * (2 * half_w / grid_n))
            targets[k] = (x, y)
            rx = max(-1.0, min(1.0, x / cal_hw))
            ry = max(-1.0, min(1.0, y / cal_hh))
            seeds[k] = (math.atan(rx * tan_h), math.atan(ry * tan_v))
            k += 1
    return seeds, targets


def _target_cfg(grid_n, tile=96, extent=0.5):
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = grid_n
    cfg.tile_w = cfg.tile_h = tile
    cfg.tile_extent_mm = extent
    cfg.ray_grid = 128
    cfg.spectral_samples = 8
    cfg.splat_sigma_um = 4.0
    cfg.pupil_jitter = 2
    cfg.center_mode = ghostlight.PSFCenterMode.FIXED_TARGET
    return cfg


def test_psf_target_mode_output_keys(psf_lens):
    """FIXED_TARGET output carries the per-cell report arrays."""
    calib = psf_lens.calibration()
    seeds, targets = _sensor_cells(3, calib, 12.4, 9.3)
    out = psf_lens.render_psf(seeds, _target_cfg(3), targets_mm=targets)
    for key in ("status", "pupil_fraction", "aim_residual_mm"):
        assert key in out, f"missing {key}"
        assert out[key].shape == (9,)
    assert out["status"].dtype == np.uint8


def test_psf_target_mode_aims_chief_onto_cell(psf_lens):
    """In a 7x7 Super-35 grid every cell's chief ray lands on its
    intended cell centre (FIXED_TARGET aim; within 1 um)."""
    calib = psf_lens.calibration()
    grid_n = 7
    half_w, half_h = 24.89 / 2, 18.66 / 2
    seeds, targets = _sensor_cells(grid_n, calib, half_w, half_h)
    out = psf_lens.render_psf(seeds, _target_cfg(grid_n), targets_mm=targets)

    chief = np.stack([out["chief_x_mm"], out["chief_y_mm"]], axis=-1)
    err_um = np.hypot(chief[:, 0] - targets[:, 0], chief[:, 1] - targets[:, 1]) * 1e3
    assert err_um.max() < 1.0, f"chief missed cell centre by {err_um.max():.3f} um"
    assert np.asarray(out["aim_residual_mm"]).max() * 1e3 < 1.0
    # Every cell within Super-35 is imageable -> all OK.
    assert (np.asarray(out["status"]) == int(ghostlight.PSFCellStatus.OK)).all()


def test_psf_target_mode_peak_centred(psf_lens):
    """The visible peak sits within a couple of pixels of every cell centre and
    does not grow radially (a regression guard on radial peak-offset drift)."""
    calib = psf_lens.calibration()
    grid_n, tile, extent = 7, 96, 0.5
    seeds, targets = _sensor_cells(grid_n, calib, 24.89 / 2, 18.66 / 2)
    out = psf_lens.render_psf(seeds, _target_cfg(grid_n, tile, extent),
                                      targets_mm=targets)
    lum = out["r"] + out["g"] + out["b"]
    mmpp = extent / tile
    offs = []
    for i in range(grid_n * grid_n):
        gx, gy = i % grid_n, i // grid_n
        t = lum[gy * tile:(gy + 1) * tile, gx * tile:(gx + 1) * tile]
        iy, ix = np.unravel_index(np.argmax(t), t.shape)
        offs.append(np.hypot(ix + 0.5 - tile / 2, iy + 0.5 - tile / 2) * mmpp * 1e3)
    offs = np.array(offs)
    # ~2 px of peak-vs-chief coma separation + quantisation, uniform across field.
    assert offs.max() < 16.0, f"peak off by {offs.max():.1f} um"
    corners = offs[[0, grid_n - 1, -grid_n, -1]]
    assert corners.max() < 13.0, "corner peak offset grew radially"


def test_psf_big_sensor_flags_vignetting(psf_lens):
    """A sensor larger than the image circle flags corner cells vignetted, and a
    vignetted cell still renders its partial PSF (the deleted-light regime)."""
    calib = psf_lens.calibration()
    grid_n = 7
    seeds, targets = _sensor_cells(grid_n, calib, 70.0 / 2, 48.5 / 2)  # IMAX-70
    cfg = _target_cfg(grid_n)
    out = psf_lens.render_psf(seeds, cfg, targets_mm=targets)
    status = np.asarray(out["status"])
    n_flagged = int((status != int(ghostlight.PSFCellStatus.OK)).sum())
    assert n_flagged > 0, "big sensor produced no vignetted cells"

    # At least one CHIEF_VIGNETTED cell must still have rendered light.
    lum = out["r"] + out["g"] + out["b"]
    tile = cfg.tile_w
    vig_with_light = 0
    for i in range(grid_n * grid_n):
        if status[i] == int(ghostlight.PSFCellStatus.CHIEF_VIGNETTED):
            gx, gy = i % grid_n, i // grid_n
            if lum[gy * tile:(gy + 1) * tile, gx * tile:(gx + 1) * tile].sum() > 0:
                vig_with_light += 1
    assert vig_with_light > 0, "vignetted cells rendered no light (deleted-light bug)"


def test_psf_smaller_sensor_fewer_vignetted(psf_lens):
    """Spec: bigger sensor -> more vignetted cells, smaller -> fewer."""
    calib = psf_lens.calibration()
    grid_n = 7
    cfg = _target_cfg(grid_n)

    def n_vig(half_w, half_h):
        s, t = _sensor_cells(grid_n, calib, half_w, half_h)
        out = psf_lens.render_psf(s, cfg, targets_mm=t)
        return int((np.asarray(out["status"]) != int(ghostlight.PSFCellStatus.OK)).sum())

    big = n_vig(70.0 / 2, 48.5 / 2)     # IMAX-70
    small = n_vig(24.89 / 2, 18.66 / 2)  # Super-35
    assert small < big
    assert small == 0  # Super-35 is well within the double-gauss image circle


# ---------------------------------------------------------------------------
# Orientation — the rendered tile must match the +y-down raster view of the
# traced ray cloud, on BOTH axes at once (diagonal field).  See the evaluation
# figures J / K.
# ---------------------------------------------------------------------------

def _trace_cloud(lens, ax, ay, chief_x, chief_y, tile, extent, lam=587.56, n=31):
    """Bin a monochromatic pupil cloud around (chief_x, chief_y); return the
    physical +y-up view (row 0 = +y) normalised to [0, 1]."""
    front_R = lens.surfaces[0].semi_aperture
    start_z = lens.surfaces[0].z - 20.0
    import math
    bx, by = math.tan(ax), math.tan(ay)
    nb = math.sqrt(bx * bx + by * by + 1.0)
    d = ghostlight.Vec3f(bx / nb, by / nb, 1.0 / nb)
    xs = []
    ys = []
    ws = []
    for v in np.linspace(-1, 1, n):
        for u in np.linspace(-1, 1, n):
            if u * u + v * v > 1.0:
                continue
            o = ghostlight.Vec3f(u * front_R, v * front_R, start_z)
            res = ghostlight.trace_primary_ray(ghostlight.Ray(o, d, lam), lens)
            if res.status == ghostlight.TraceStatus.OK and np.isfinite(res.position.x):
                xs.append(res.position.x)
                ys.append(res.position.y)
                ws.append(res.weight)
    xs = np.array(xs)
    ys = np.array(ys)
    ws = np.array(ws)
    xe = np.linspace(chief_x - extent / 2, chief_x + extent / 2, tile + 1)
    ye = np.linspace(chief_y - extent / 2, chief_y + extent / 2, tile + 1)
    H, _, _ = np.histogram2d(xs, ys, bins=[xe, ye], weights=ws)
    phys = H.T[::-1, :]  # row 0 = +y (physical up)
    m = phys.max()
    return phys / m if m > 0 else phys


def _orientation_corr(lens, frac_x, frac_y):
    """Render a monochromatic 1x1 PSF at a field point and correlate the tile
    against the four flips of the traced cloud.  Returns
    {physical, flip_x, raster(=flip_y), flip_xy}."""
    calib = lens.calibration()
    ax = frac_x * float(calib.max_half_angle_h)
    ay = frac_y * float(calib.max_half_angle_v)
    tile, extent = 64, 0.5
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = 1
    cfg.tile_w = cfg.tile_h = tile
    cfg.tile_extent_mm = extent
    cfg.ray_grid = 192
    cfg.spectral_samples = 1
    cfg.monochromatic = True
    cfg.pupil_jitter = 2
    out = lens.render_psf(np.array([[ax, ay]], dtype=np.float32), cfg)
    rendered = out["r"]
    rmax = rendered.max()
    assert rmax > 0, "no light rendered"
    rendered = rendered / rmax
    phys = _trace_cloud(lens, ax, ay,
                        float(out["chief_x_mm"][0]), float(out["chief_y_mm"][0]),
                        tile, extent)
    variants = {
        "physical": phys,
        "flip_x": phys[:, ::-1],
        "raster": phys[::-1, :],
        "flip_xy": phys[::-1, ::-1],
    }
    return {k: float(np.corrcoef(rendered.ravel(), v.ravel())[0, 1])
            for k, v in variants.items()}


def test_psf_orientation_diagonal(psf_lens):
    """A diagonal field is asymmetric on both axes: the rendered tile must match
    the +y-down raster cloud and reject flip_x (x correct) and physical (y is
    raster) — verifying both axes at once with no symmetry assumption."""
    c = _orientation_corr(psf_lens, 0.62, 0.62)
    # The margins are the real proof (x correct, y raster); raster wins each flip
    # by a wide margin.  The absolute value is modest here only because the test
    # uses coarse sampling and doesn't blur-match the cloud to the splat.
    assert c["raster"] == max(c.values()), f"raster did not win: {c}"
    assert c["raster"] > 0.5, f"raster corr too low: {c}"
    assert c["raster"] - c["flip_x"] > 0.3, f"x axis ambiguous: {c}"
    assert c["raster"] - c["physical"] > 0.3, f"y axis ambiguous: {c}"


def test_psf_orientation_per_axis(psf_lens):
    """Isolate each axis: a horizontal field pins x (identity beats flip_x),
    a vertical field pins y (raster beats physical)."""
    ch = _orientation_corr(psf_lens, 0.62, 0.0)
    # x-asymmetric, y-symmetric: raster (== physical here) must beat flip_x.
    assert max(ch["raster"], ch["physical"]) - max(ch["flip_x"], ch["flip_xy"]) > 0.3, ch
    cv = _orientation_corr(psf_lens, 0.0, 0.62)
    # y-asymmetric, x-symmetric: raster must beat physical.
    assert ch["raster"] > 0.0
    assert cv["raster"] - cv["physical"] > 0.3, cv


def test_psf_spectral_energy_sample_count_independent(psf_lens):
    """Brightness must not depend on spectral_samples (a quality knob).  The
    table is dλ-weighted so a white source integrates the same at any count."""
    calib = psf_lens.calibration()
    seeds, targets = _sensor_cells(3, calib, 8.0, 6.0)
    energies = []
    for spec in (8, 16, 32):
        cfg = _target_cfg(3)
        cfg.spectral_samples = spec
        out = psf_lens.render_psf(seeds, cfg, targets_mm=targets)
        energies.append(float((out["r"] + out["g"] + out["b"]).sum()))
    lo, hi = min(energies), max(energies)
    assert (hi - lo) / hi < 0.08, f"energy varies with sample count: {energies}"


def test_psf_splat_physical_resolution_independent(psf_lens):
    """splat_sigma_um is physical: the PSF's rendered size (mm) must be the same
    at LOW and HIGH tile resolution, so presets differ in fidelity not picture."""
    import math
    calib = psf_lens.calibration()
    seeds, targets = _sensor_cells(3, calib, 8.0, 6.0)

    def psf_rms_um(tile):
        cfg = _target_cfg(3, tile=tile, extent=0.5)
        cfg.splat_sigma_um = 4.0
        out = psf_lens.render_psf(seeds, cfg, targets_mm=targets)
        lum = out["r"] + out["g"] + out["b"]
        # centre tile
        t = lum[tile:2*tile, tile:2*tile]
        tot = t.sum()
        yy, xx = np.mgrid[0:tile, 0:tile]
        cx = (xx * t).sum() / tot; cy = (yy * t).sum() / tot
        rms_px = math.sqrt((((xx-cx)**2 + (yy-cy)**2) * t).sum() / tot)
        return rms_px * (0.5 / tile) * 1000

    lo = psf_rms_um(128)
    hi = psf_rms_um(512)
    assert abs(hi - lo) / lo < 0.1, f"PSF size differs with resolution: {lo:.2f} vs {hi:.2f} um"


def test_psf_legacy_mode_still_default(psf_lens):
    """CHIEF_CENTROID is the default mode and produces a well-formed report."""
    calib = psf_lens.calibration()
    sources = _grid_sources(3, 3, calib.max_half_angle_h * 0.5,
                            calib.max_half_angle_v * 0.5)
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = 3
    cfg.tile_w = cfg.tile_h = 32
    cfg.tile_extent_mm = 0.1
    cfg.spectral_samples = 4
    cfg.ray_grid = 16
    assert cfg.center_mode == int(ghostlight.PSFCenterMode.CHIEF_CENTROID)
    out = psf_lens.render_psf(sources, cfg)
    assert out["status"].shape == (9,)
    assert (np.asarray(out["status"]) == int(ghostlight.PSFCellStatus.OK)).all()


# ---------------------------------------------------------------------------
# Deep-stop lenses: the aim's whole-disc probes starve, not the lens.
# ---------------------------------------------------------------------------

def _deep_stop_cells(lens, grid_n, half_w, half_h):
    calib = lens.calibration()
    fx = calib.sensor_half_w / np.tan(calib.max_half_angle_h)
    fy = calib.sensor_half_h / np.tan(calib.max_half_angle_v)
    seeds, targets = [], []
    for j in range(grid_n):
        for i in range(grid_n):
            tx = (-1.0 + 2.0 * (i + 0.5) / grid_n) * half_w
            ty = (-1.0 + 2.0 * (j + 0.5) / grid_n) * half_h
            seeds.append([np.arctan(tx / fx), np.arctan(ty / fy), 1.0, 1.0, 1.0])
            targets.append([tx, ty])
    return np.asarray(seeds, np.float32), np.asarray(targets, np.float32)


@pytest.mark.parametrize("lens_file", [
    "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens",
    "zodiac1.lens",
])
def test_deep_stop_cells_are_not_reported_dark(lens_file):
    """A lens whose stop is far smaller than its front element must still
    resolve its lit field.

    Both the clean-chief and the 9x9 survivor-mean tiers sample the WHOLE front
    disc.  The Atlas anamorphic admits through 16% of its front radius and
    zodiac1 through 4%, so that probe lands about two useful samples on the
    first and a fraction of one on the second -- under the minimum-hits bar, so
    every cell came back DARK and the panel painted a perfectly imageable frame
    as unreachable.  It was unsampled, not unreachable: the deep-stop tier
    probes the window that actually passes light (spawn_probe.h).
    """
    lens = ghostlight.OpticalSystem.load(str(lens_path(lens_file)))
    grid_n = 7
    seeds, targets = _deep_stop_cells(lens, grid_n, 24.89 / 2, 18.66 / 2)
    cfg = _target_cfg(grid_n, tile=32, extent=0.5)
    out = lens.render_psf(seeds, cfg, targets_mm=targets)

    status = np.asarray(out["status"])
    dark = int((status == int(ghostlight.PSFCellStatus.DARK)).sum())
    assert dark <= 3, (
        f"{lens_file}: {dark}/{status.size} cells report DARK on a lit frame")

    lit = status != int(ghostlight.PSFCellStatus.DARK)
    resid = np.asarray(out["aim_residual_mm"])[lit]
    assert np.isfinite(resid).all()
    # Aimed through a partial beam, so the soft 5 um tolerance governs; the bar
    # here is only that the aim is real, not that it matched the clean chief.
    assert resid.max() < 0.5, f"aim residual {resid.max():.4f} mm on lit cells"


def test_well_behaved_lens_is_untouched_by_the_deep_stop_fallback(psf_lens):
    """The deep-stop tier must fire only where the tiers above it fail.

    On a lens that fills its front element every cell resolves through the
    clean chief, so nothing about this render may change -- the fix is allowed
    to reach cells that were DARK and nothing else.
    """
    grid_n = 7
    seeds, targets = _deep_stop_cells(psf_lens, grid_n, 24.89 / 2, 18.66 / 2)
    out = psf_lens.render_psf(seeds, _target_cfg(grid_n, tile=32, extent=0.5),
                              targets_mm=targets)
    status = np.asarray(out["status"])
    assert (status == int(ghostlight.PSFCellStatus.OK)).all()
    err_um = np.hypot(np.asarray(out["chief_x_mm"]) - targets[:, 0],
                      np.asarray(out["chief_y_mm"]) - targets[:, 1]) * 1e3
    assert err_um.max() < 1.0
