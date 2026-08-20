"""Visual validation for the PSF sensor-cell rework.

Renders the double gauss PSF grid two ways and writes a before/after montage:

  * BEFORE — legacy CHIEF_CENTROID mode with the old angle-linspace grid: the
    dot (peak) drifts off the tile centre, worse toward the field edge.
  * AFTER  — FIXED_TARGET mode with the sensor-cell grid: the chief ray is aimed
    at each cell centre, so the dot is centred by construction.
  * VIGNETTE — an oversized (IMAX-70) sensor: corner cells whose primary ray
    can't reach get the panel's 0.5-alpha red overlay, with the partial PSF
    still visible underneath.

    python validate_psf_cells.py [out.png]

Gate: prints the max peak-vs-cell-centre offset for both modes and the vignette
cell counts.  Not byte-exact (float32 / MC); compares peak offset in microns.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight

LENS = str(lens_file("DoubleGauss.lens"))
GRID_N, TILE, EXT = 7, 96, 0.5
SUPER35 = (24.89, 18.66)
IMAX70 = (70.0, 48.5)


def _tone(comp, gain=50.0):
    pos = np.maximum(comp, 0.0)
    out = np.zeros_like(pos)
    lum = pos.sum(-1)
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            sl = (slice(gy*TILE, (gy+1)*TILE), slice(gx*TILE, (gx+1)*TILE))
            pk = max(float(lum[sl].max()), 1e-30)
            out[sl] = pos[sl] / pk
    return np.clip(np.log1p(out*gain)/np.log1p(gain), 0, 1)


def _cells(calib, half_w, half_h):
    cal_hw, cal_hh = float(calib.sensor_half_w), float(calib.sensor_half_h)
    th, tv = math.tan(float(calib.max_half_angle_h)), math.tan(float(calib.max_half_angle_v))
    seeds = np.empty((GRID_N*GRID_N, 2), np.float32)
    targets = np.empty((GRID_N*GRID_N, 2), np.float32)
    k = 0
    for row in range(GRID_N):
        y = -half_h + (row+0.5)*(2*half_h/GRID_N)
        for col in range(GRID_N):
            x = -half_w + (col+0.5)*(2*half_w/GRID_N)
            targets[k] = (x, y)
            rx, ry = max(-1, min(1, x/cal_hw)), max(-1, min(1, y/cal_hh))
            seeds[k] = (math.atan(rx*th), math.atan(ry*tv))
            k += 1
    return seeds, targets


def _legacy_angles(calib, half_w, half_h):
    # Old panel path: linear-in-angle, y flipped +y-up (the buggy arrangement).
    ax = math.atan(min(1.0, (half_w/calib.sensor_half_w)) * math.tan(calib.max_half_angle_h))
    ay = math.atan(min(1.0, (half_h/calib.sensor_half_h)) * math.tan(calib.max_half_angle_v))
    xs = np.linspace(-ax, ax, GRID_N, dtype=np.float32)
    ys = np.linspace(ay, -ay, GRID_N, dtype=np.float32)  # +y-up (old)
    return np.array([[x, y] for y in ys for x in xs], np.float32)


def _base_cfg(mode):
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = GRID_N
    cfg.tile_w = cfg.tile_h = TILE
    cfg.tile_extent_mm = EXT
    cfg.ray_grid = 256
    cfg.spectral_samples = 16
    cfg.splat_sigma_um = 4.0
    cfg.pupil_jitter = 2
    cfg.center_mode = mode
    return cfg


def _peak_offsets(comp):
    lum = comp.sum(-1)
    mmpp = EXT / TILE
    offs = np.full(GRID_N*GRID_N, np.nan)
    for i in range(GRID_N*GRID_N):
        gx, gy = i % GRID_N, i // GRID_N
        t = lum[gy*TILE:(gy+1)*TILE, gx*TILE:(gx+1)*TILE]
        if t.max() <= 0:
            continue
        iy, ix = np.unravel_index(np.argmax(t), t.shape)
        offs[i] = math.hypot(ix+0.5-TILE/2, iy+0.5-TILE/2) * mmpp * 1e3
    return offs


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(
        pathlib.Path(__file__).resolve().parent / "psf_cells_validation.png")
    lens = ghostlight.OpticalSystem.load(LENS)
    calib = lens.calibration()
    hw, hh = SUPER35[0]/2, SUPER35[1]/2

    # BEFORE: legacy angle grid + centroid centring
    ob = lens.render_psf(_legacy_angles(calib, hw, hh),
                         _base_cfg(ghostlight.PSFCenterMode.CHIEF_CENTROID))
    before = np.stack([ob["r"], ob["g"], ob["b"]], -1)

    # AFTER: sensor cells + aimed chief
    seeds, targets = _cells(calib, hw, hh)
    oa = lens.render_psf(seeds, _base_cfg(ghostlight.PSFCenterMode.FIXED_TARGET), targets_mm=targets)
    after = np.stack([oa["r"], oa["g"], oa["b"]], -1)

    # VIGNETTE: IMAX-70 target mode
    bseeds, btargets = _cells(calib, IMAX70[0]/2, IMAX70[1]/2)
    ov = lens.render_psf(bseeds, _base_cfg(ghostlight.PSFCenterMode.FIXED_TARGET), targets_mm=btargets)
    vig = np.stack([ov["r"], ov["g"], ov["b"]], -1)
    vstatus = np.asarray(ov["status"])

    off_b, off_a = _peak_offsets(before), _peak_offsets(after)
    n_vig = int((vstatus != int(ghostlight.PSFCellStatus.OK)).sum())
    print(f"BEFORE peak offset (um): max {np.nanmax(off_b):.1f} mean {np.nanmean(off_b):.1f}")
    print(f"AFTER  peak offset (um): max {np.nanmax(off_a):.1f} mean {np.nanmean(off_a):.1f}")
    print(f"AFTER  aim residual (um): max {np.asarray(oa['aim_residual_mm']).max()*1e3:.4f}")
    print(f"VIGNETTE cells on IMAX-70: {n_vig}/{GRID_N*GRID_N}")
    assert np.nanmax(off_a) < np.nanmax(off_b), "target mode did not improve centring"

    # ---- montage ----
    BG, PANEL, GRIDC = "#101014", "#16161c", "#2c2c36"
    AZURE, RED, MUTED, INK = "#5aa2ff", "#ff6b6b", "#9a9aa6", "#e6e6ea"
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.06)

    def draw(ax, comp, offs, title, status=None):
        ax.imshow(_tone(comp), origin="upper", interpolation="nearest")
        for g in range(1, GRID_N):
            ax.axvline(g*TILE-0.5, color=GRIDC, lw=0.7)
            ax.axhline(g*TILE-0.5, color=GRIDC, lw=0.7)
        for i in range(GRID_N*GRID_N):
            gx, gy = i % GRID_N, i // GRID_N
            cx, cy = gx*TILE+TILE/2-0.5, gy*TILE+TILE/2-0.5
            if status is not None and status[i] != int(ghostlight.PSFCellStatus.OK):
                ax.add_patch(Rectangle((gx*TILE-0.5, gy*TILE-0.5), TILE, TILE,
                                       color="#ff2020", alpha=0.5, zorder=5))
            ax.plot(cx, cy, "+", color=AZURE, ms=9, mew=1.2, zorder=6)
            lum = comp[gy*TILE:(gy+1)*TILE, gx*TILE:(gx+1)*TILE].sum(-1)
            if lum.max() > 0 and (status is None or status[i] == int(ghostlight.PSFCellStatus.OK)):
                iy, ix = np.unravel_index(np.argmax(lum), lum.shape)
                ax.add_patch(Circle((gx*TILE+ix, gy*TILE+iy), 3.0, fill=False,
                                    color=RED, lw=1.2, zorder=6))
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xticks([]); ax.set_yticks([])

    draw(axes[0], before, off_b,
         f"BEFORE — angle grid + centroid\npeak drifts up to {np.nanmax(off_b):.0f} µm (worse toward edge)")
    draw(axes[1], after, off_a,
         f"AFTER — sensor cells + aimed chief\ndot centred: chief on cell to {np.asarray(oa['aim_residual_mm']).max()*1e3:.2f} µm")
    draw(axes[2], vig, _peak_offsets(vig),
         f"VIGNETTE — IMAX-70 sensor\n{n_vig}/{GRID_N*GRID_N} cells flagged; partial PSF under the tint",
         status=vstatus)
    fig.suptitle("PSF sensor-cell rework — blue + = cell centre, red ○ = peak (the dot)",
                 fontsize=12, x=0.01, ha="left", color=INK)
    fig.savefig(out_path, dpi=150)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
