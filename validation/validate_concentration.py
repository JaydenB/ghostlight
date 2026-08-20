"""Concentrated-sampling validation.

Concentration relays each (pair, source)'s sample budget into its probe-measured
pupil survivor box. It is STATISTICALLY equivalent by construction (weights scale
by |R|/A_ref; mask-rejected samples count as zeros), so the gates are the ones
used here:

  SHAPE: energy-weighted centroid + covariance spread,
      measured as ERROR AGAINST THE DENSE TRUTH (ray_grid 768, classic
      sampling = the converged reference) for BOTH classic-256 and
      concentrated-256. Concentration passes iff its truth-error is <=
      classic's (with slack) AND far below the real-silhouette-change signal.
  COLOR / ENERGY: chroma + integrated energy vs OFF, within the resample
      floor's own wobble.
  NOISE WIN (the payoff): RMSE against the dense truth — concentrated 256
      must beat classic 256. That is the entire point.

Cases: both lenses x 3 off-axis sources x {circle, hex override}.

    python validate_concentration.py
"""
from __future__ import annotations
import math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import ARTIFACTS, lens_file  # noqa: E402
import ghostlight  # noqa

OUT = ARTIFACTS / "concentration"
OUT.mkdir(parents=True, exist_ok=True)
W = H = 320
SH = 14.0
DENSE_GRID = 768                     # 9x the sample count of 256 -> the truth
CASES = [("double_gauss", lens_file("DoubleGauss.lens")),
         ("sirui", lens_file("Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens"))]
POSITIONS = [(0.9, 0.5), (1.3, 0.5), (1.6, 0.9)]
BLADES = [0, 6]                      # circle + the hex override that bit culling


def cfg(sx, sy, conc, grid=256, blades=0, jitter=2, seed=12345):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = grid
    c.spectral_samples = 16
    c.pupil_jitter = jitter
    c.jitter_seed = seed
    c.sensor_half_w = SH
    c.sensor_half_h = SH
    c.min_ghost_intensity = 0.0
    if blades:
        c.aperture_blades = blades
    c.cull_dead_pairs = True
    c.concentrate_samples = conc
    c.adaptive_sample_budgets = False  # pin: this gate tests PURE concentration,
    #                                    not the now-default adaptive path.
    return c


def render(lens, calib, sx, sy, conc, grid=256, blades=0, jitter=2, seed=12345):
    o = lens.render_point_flare(W, H, cfg(sx, sy, conc, grid, blades, jitter, seed),
                                calib=calib)
    return np.stack([np.asarray(o["ghost_r"]), np.asarray(o["ghost_g"]),
                     np.asarray(o["ghost_b"])], -1)


def timed(lens, calib, *a, n=3, **kw):
    best, img = 1e9, None
    for _ in range(n):
        t = time.perf_counter()
        img = render(lens, calib, *a, **kw)
        best = min(best, time.perf_counter() - t)
    return img, best * 1e3


def moments(img):
    d = img.sum(-1).astype(np.float64); tot = d.sum()
    if tot < 1e-20:
        return None
    ys, xs = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    cx = (d * xs).sum() / tot; cy = (d * ys).sum() / tot
    mxx = (d * (xs - cx) ** 2).sum() / tot; myy = (d * (ys - cy) ** 2).sum() / tot
    mxy = (d * (xs - cx) * (ys - cy)).sum() / tot
    ev = np.linalg.eigvalsh(np.array([[mxx, mxy], [mxy, myy]]))
    return cx, cy, math.sqrt(max(ev[0], 0)) + math.sqrt(max(ev[1], 0))


def shape_delta(a, b):
    ma, mb = moments(a), moments(b)
    if ma is None or mb is None:
        return 0.0, 1.0
    return math.hypot(ma[0] - mb[0], ma[1] - mb[1]), (mb[2] / ma[2] if ma[2] > 0 else 1.0)


def chroma(img):
    s = img.reshape(-1, 3).sum(0)
    return s / s.sum() if s.sum() > 0 else np.zeros(3)


def rmse(a, ref):
    return float(np.sqrt(np.mean((a - ref) ** 2)))


def tonemap(img, exp):
    x = np.clip(img * exp, 0, None); x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    if not ghostlight._cuda_available():
        print("No CUDA."); return
    all_pass = True
    for name, path in CASES:
        if not path.exists():
            print(f"missing {path}"); all_pass = False; continue
        lens = ghostlight.OpticalSystem.load(str(path)); calib = lens.calibration()
        render(lens, calib, 0.5, 0.5, True)                     # warmup
        print(f"\n=== {name} ===  (cenERR/sprERR = |render - dense truth|; ON must be <= OFF)")
        print(f"  {'case':>16} {'cenERRoff':>9} {'cenERRon':>9} {'sprERRoff':>9} {'sprERRon':>9} "
              f"{'chroma':>7} {'energy':>7} {'floorE':>7} {'RMSEoff':>9} {'RMSEon':>9} "
              f"{'noise/':>6} {'speed':>6} {'verdict':>8}")
        n_rows = len(POSITIONS) * len(BLADES)
        fig, axes = plt.subplots(n_rows, 3, figsize=(10.5, 3.2 * n_rows))
        row = 0
        for blades in BLADES:
            for sx, sy in POSITIONS:
                # classic (OFF) / concentrated (ON) at the SAME budget
                off, t_off = timed(lens, calib, sx, sy, False, blades=blades)
                on,  t_on  = timed(lens, calib, sx, sy, True,  blades=blades)
                # dense truth (classic sampling at 9x samples): the reference
                # every shape/noise error is measured against.
                dense = render(lens, calib, sx, sy, False, grid=DENSE_GRID, blades=blades)
                # resample floor for the energy wobble only
                fA = render(lens, calib, sx, sy, False, blades=blades, jitter=1, seed=311)
                fB = render(lens, calib, sx, sy, False, blades=blades, jitter=1, seed=622)
                floor_e = abs(float(fA.sum() / (fB.sum() + 1e-20)) - 1.0)
                # truth-relative shape errors for BOTH estimators
                cen_off, spr_off = shape_delta(dense, off)
                cen_on,  spr_on  = shape_delta(dense, on)
                spr_off_e, spr_on_e = abs(spr_off - 1.0), abs(spr_on - 1.0)
                chr_d = float(np.linalg.norm(chroma(off) - chroma(on))) * 1e3
                energy = float(on.sum() / (off.sum() + 1e-20))
                r_off, r_on = rmse(off, dense), rmse(on, dense)
                # PASS: concentration at least as CLOSE TO TRUTH as classic
                # (25% slack + a 0.05px absolute floor for near-zero errors),
                # absolute shape error far below the 2.2px real-change signal,
                # energy within the resample wobble, and no noise regression.
                ok = (cen_on <= max(1.25 * cen_off, 0.05) and cen_on < 0.5
                      and spr_on_e <= max(1.25 * spr_off_e, 0.005)
                      and abs(energy - 1.0) <= max(3.0 * floor_e, 0.01)
                      and r_on <= r_off * 1.05)
                all_pass = all_pass and ok
                tag = f"({sx},{sy})" + (" hex" if blades else "")
                print(f"  {tag:>16} {cen_off:9.3f} {cen_on:9.3f} {spr_off_e:9.4f} {spr_on_e:9.4f} "
                      f"{chr_d:7.3f} {energy:7.4f} {floor_e:7.4f} {r_off:9.3e} {r_on:9.3e} "
                      f"{r_off/max(r_on,1e-30):5.1f}x {t_off/t_on:5.2f}x "
                      f"{'PASS' if ok else 'FAIL':>8}")
                exp = 0.9 / (np.percentile(dense.sum(-1), 99.5) + 1e-9)
                for col, (lab, im) in enumerate([
                        ("classic 256 (OFF)", off),
                        ("concentrated 256 (ON)", on),
                        (f"dense truth {DENSE_GRID} (OFF)", dense)]):
                    ax = axes[row, col] if n_rows > 1 else axes[col]
                    ax.imshow(tonemap(im, exp)); ax.axis("off")
                    if row == 0:
                        ax.set_title(lab, fontsize=9)
                    if col == 0:
                        ax.text(-0.04, 0.5, tag, transform=ax.transAxes, rotation=90,
                                va="center", ha="right", fontsize=8)
                row += 1
        fig.suptitle(f"{name} — concentration: same 256-grid budget, laid into the survivor box\n"
                     f"(middle must match left's SHAPE but approach right's SMOOTHNESS)",
                     fontsize=10)
        fig.tight_layout()
        p = OUT / f"concentration_{name}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"  montage -> {p}")
    print("\ngates: shape/energy at the resample floor; RMSE(on) <= RMSE(off) vs dense truth")
    print("RESULT:", "ALL PASS" if all_pass else "FAIL — investigate before shipping")


if __name__ == "__main__":
    main()
