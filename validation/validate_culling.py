"""Ghost-pair culling losslessness validation.

For each lens + off-axis source it renders cull-OFF (reference: every pair) vs
cull-ON (optimized), and:
  * saves a  [reference | culled | 25x |difference|]  montage per lens, and
  * prints, per position:
      - pairs culled + wall-clock speedup            (the SAVING), and
      - shape delta (centroid px, spread ratio),
        chroma shift (r/g/b fractions) and energy ratio, and the worst
        per-pixel difference                         (proof of NO LOSS).
Correct culling only removes pairs that put zero rays on the sensor, so the two
images must be identical to the deterministic floor: centroid ~0px, energy ~1.0,
diff ~0. Any real ghost wrongly culled would light up the difference panel.

    python validate_culling.py
"""
from __future__ import annotations
import math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import ARTIFACTS, lens_file  # noqa: E402
import ghostlight  # noqa

OUT = ARTIFACTS / "culling"
OUT.mkdir(parents=True, exist_ok=True)
W = H = 320
SH = 14.0
CASES = [("double_gauss", lens_file("DoubleGauss.lens")),
         ("sirui", lens_file("Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens"))]
POSITIONS = [(0.9, 0.5), (1.3, 0.5), (1.6, 0.9)]   # off-axis: culling active


def _cfg(sx, sy, cull, stats=False):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = 256; c.spectral_samples = 16
    c.pupil_jitter = 2; c.jitter_seed = 12345      # Halton: deterministic
    c.sensor_half_w = SH; c.sensor_half_h = SH
    c.min_ghost_intensity = 0.0
    c.cull_dead_pairs = cull
    c.collect_stats = stats
    return c


def render(lens, calib, sx, sy, cull, stats=False):
    o = lens.render_point_flare(W, H, _cfg(sx, sy, cull, stats), calib=calib)
    img = np.stack([np.asarray(o["ghost_r"]), np.asarray(o["ghost_g"]),
                    np.asarray(o["ghost_b"])], -1)
    return img, o.get("stats")


def timed(lens, calib, sx, sy, cull, n=5):
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        img, _ = render(lens, calib, sx, sy, cull)
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


def chroma(img):
    s = img.reshape(-1, 3).sum(0)
    return s / s.sum() if s.sum() > 0 else np.zeros(3)


def tonemap(img, exp):
    x = np.clip(img * exp, 0, None); x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    if not ghostlight._cuda_available():
        print("No CUDA."); return
    for name, path in CASES:
        if not path.exists():
            print(f"missing {path}"); continue
        lens = ghostlight.OpticalSystem.load(str(path)); calib = lens.calibration()
        render(lens, calib, 0.5, 0.5, True)  # warmup
        print(f"\n=== {name} ===")
        print(f"    {'src':>10} {'pairs':>6} {'culled':>6} {'speedup':>8}   "
              f"{'centroid_px':>11} {'spread':>7} {'chroma_d':>9} {'energy':>7} {'maxpix_ppm':>10}")
        fig, axes = plt.subplots(len(POSITIONS), 3, figsize=(11, 3.4 * len(POSITIONS)))
        for row, (sx, sy) in enumerate(POSITIONS):
            _, st = render(lens, calib, sx, sy, True, stats=True)
            traces = np.array(st["pair_traces"], float); ons = np.array(st["pair_on_sensor"], float)
            n_pairs = int(st["n_pairs"]); n_culled = int(((ons == 0) & (traces > 0)).sum())
            full, t_full = timed(lens, calib, sx, sy, False)
            cull, t_cull = timed(lens, calib, sx, sy, True)
            mf, mc = moments(full), moments(cull)
            centroid = math.hypot(mf[0] - mc[0], mf[1] - mc[1]) if mf and mc else 0.0
            spread = (mc[2] / mf[2]) if (mf and mc and mf[2] > 0) else 1.0
            chroma_d = float(np.linalg.norm(chroma(full) - chroma(cull))) * 1e3
            energy = float(cull.sum() / (full.sum() + 1e-20))
            peak = float(full.sum(-1).max())
            maxpix = float(np.abs(full - cull).sum(-1).max()) / (peak + 1e-20) * 1e6
            print(f"    ({sx:.2f},{sy:.2f}) {n_pairs:6d} {n_culled:6d} {t_full/t_cull:7.2f}x   "
                  f"{centroid:11.3f} {spread:7.3f} {chroma_d:9.3f} {energy:7.4f} {maxpix:10.1f}")
            exp = 0.9 / (np.percentile(full.sum(-1), 99.5) + 1e-6)
            for col, (lab, im) in enumerate([("reference (all pairs)", full),
                                             (f"culled ({n_culled}/{n_pairs}, {t_full/t_cull:.1f}x)", cull),
                                             ("25x |difference|", np.abs(full - cull) * 25)]):
                ax = axes[row, col] if len(POSITIONS) > 1 else axes[col]
                ax.imshow(tonemap(im, exp)); ax.axis("off")
                if row == 0:
                    ax.set_title(lab, fontsize=9)
                if col == 0:
                    ax.text(-0.04, 0.5, f"({sx},{sy})", transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=9)
        fig.suptitle(f"{name} — culling: reference vs culled (difference must stay black)",
                     fontsize=11)
        fig.tight_layout()
        p = OUT / f"cull_validation_{name}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        print(f"    montage -> {p}")


if __name__ == "__main__":
    main()
