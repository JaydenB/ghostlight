"""Adjacent-source-position stability under adaptive sampling.

This checks whether variable per-(pair, source) budgets introduce frame-to-frame
popping as the source scrubs across the viewport.

Method: sweep a point source in fine steps along an off-axis path. At each step render
both adaptive and fixed-budget concentration, using Halton so each
frame is deterministic (no reseed noise — any adjacent-frame change is the genuine
geometric motion plus whatever instability the sampler adds). For each adjacent step
pair compute the largest single-pixel change (ppm of peak). Because both modes see the
IDENTICAL geometry, their adjacent-delta curves must track; an adaptive-only spike =
a pop. We gate adaptive's curve against concentration's, and print the wang RESAMPLE
FLOOR (reseed at a fixed position) as the irreducible-MC-noise yardstick.

    python validate_adaptive_scrub.py

Corroborated by diag_conc_cliff.py (the discrete smooth<->speckle / cull-flicker
detector), which now also runs under adaptive since the default flipped.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import ARTIFACTS, lens_file  # noqa: E402
import ghostlight  # noqa

OUT = ARTIFACTS / "adaptive_scrub"
OUT.mkdir(parents=True, exist_ok=True)
W = H = 320
SH = 14.0
# fine horizontal scrub from mildly to heavily off-axis (where budgets shrink most).
SWEEP_X = np.round(np.arange(0.60, 1.65, 0.025), 3)     # 42 steps
SWEEP_Y = 0.5
# lens, blades, label. sirui = the grainiest (anamorphic) = worst case for popping.
CASES = [
    ("double_gauss", lens_file("DoubleGauss.lens"), 0, "double_gauss"),
    ("sirui", lens_file("Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens"), 0, "sirui"),
]


def cfg(sx, sy, mode, blades, jitter=2, seed=12345):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = float(sx), float(sy)
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = 256; c.spectral_samples = 16
    c.pupil_jitter = jitter; c.jitter_seed = seed
    c.sensor_half_w = SH; c.sensor_half_h = SH
    c.min_ghost_intensity = 0.0
    if blades:
        c.aperture_blades = blades
    c.cull_dead_pairs = True
    c.concentrate_samples = (mode != "classic")
    c.adaptive_sample_budgets = (mode == "adaptive")
    return c


def render(lens, calib, sx, sy, mode, blades, jitter=2, seed=12345):
    o = lens.render_point_flare(W, H, cfg(sx, sy, mode, blades, jitter, seed), calib=calib)
    return np.stack([np.asarray(o["ghost_r"]), np.asarray(o["ghost_g"]),
                     np.asarray(o["ghost_b"])], -1)


def maxpix_ppm(a, b, peak):
    return float(np.abs(a - b).sum(-1).max()) / (peak + 1e-20) * 1e6


def main():
    if not ghostlight._cuda_available():
        print("No CUDA."); return
    fig, axes = plt.subplots(len(CASES), 1, figsize=(11, 3.4 * len(CASES)), squeeze=False)
    all_pass = True
    print(f"{'lens':>12} | {'adj-delta ppm (of peak)':^30} | {'resample':>9} | {'verdict':>7}")
    print(f"{'':>12} | {'con med':>9} {'ada med':>9} {'ada/con':>8} | {'floor':>9} |")
    for row, (name, path, blades, lab) in enumerate(CASES):
        lens = ghostlight.OpticalSystem.load(str(path)); calib = lens.calibration()
        render(lens, calib, 1.0, SWEEP_Y, "adaptive", blades)              # warmup

        # sweep: keep frames per mode, track the running peak for normalization.
        frames = {"conc": [], "adaptive": []}
        peak = 1e-20
        for sx in SWEEP_X:
            for mode in ("conc", "adaptive"):
                im = render(lens, calib, sx, SWEEP_Y, mode, blades)
                frames[mode].append(im)
                peak = max(peak, float(im.sum(-1).max()))

        d_con = np.array([maxpix_ppm(frames["conc"][i], frames["conc"][i + 1], peak)
                          for i in range(len(SWEEP_X) - 1)])
        d_ada = np.array([maxpix_ppm(frames["adaptive"][i], frames["adaptive"][i + 1], peak)
                          for i in range(len(SWEEP_X) - 1)])

        # wang resample floor: reseed at 3 fixed positions, measure adaptive's frame
        # spread from pure MC noise (no geometry change) — the "how big is a delta"
        # yardstick. Use jitter=1 (wang) so seeds actually vary.
        floor = 0.0
        for sx in (0.8, 1.1, 1.4):
            a0 = render(lens, calib, sx, SWEEP_Y, "adaptive", blades, jitter=1, seed=1)
            a1 = render(lens, calib, sx, SWEEP_Y, "adaptive", blades, jitter=1, seed=2)
            floor = max(floor, maxpix_ppm(a0, a1, peak))

        # both modes see identical geometry, so the ratio isolates adaptive's added
        # instability. Guard against dividing by near-zero (source barely moved the
        # ghost) by only ratioing steps where concentration itself moved > the floor.
        moved = d_con > floor
        ratio = float(np.median(d_ada[moved] / d_con[moved])) if moved.any() else 1.0
        max_ratio = float(np.max(d_ada[moved] / d_con[moved])) if moved.any() else 1.0
        # PASS: adaptive tracks concentration (median ~1) with no isolated pop
        # (worst adjacent step < 2.5x concentration's, i.e. no smooth->speckle jump).
        ok = ratio < 1.5 and max_ratio < 2.5
        all_pass = all_pass and ok
        print(f"{lab:>12} | {np.median(d_con):9.0f} {np.median(d_ada):9.0f} {ratio:8.2f} "
              f"| {floor:9.0f} | {'PASS' if ok else 'FAIL':>7}   (max ada/con {max_ratio:.2f})")

        ax = axes[row][0]
        xs = SWEEP_X[:-1]
        ax.plot(xs, d_con, "-o", ms=3, lw=1, label="fixed-budget concentration", color="#1f77b4")
        ax.plot(xs, d_ada, "-o", ms=3, lw=1, label="adaptive", color="#d62728")
        ax.axhline(floor, ls="--", lw=1, color="gray", label=f"wang resample floor ({floor:.0f} ppm)")
        ax.set_title(f"{lab} — adjacent-frame max-pixel change as the source scrubs "
                     f"(overlap = no added instability; a spike = a pop)", fontsize=9)
        ax.set_xlabel("source_x"); ax.set_ylabel("adj-delta (ppm of peak)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Source-scrub temporal stability — adaptive must track concentration "
                 "(no adaptive-only spikes as the source moves)", fontsize=11)
    fig.tight_layout()
    p = OUT / "adaptive_scrub.png"; fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\nmontage -> {p}")
    print("gate: median(adaptive/concentration adjacent-delta) < 1.5 AND max ratio < 2.5 "
          "(no isolated smooth->speckle pop)")
    print("RESULT:", "ALL PASS" if all_pass else "FAIL — investigate")


if __name__ == "__main__":
    main()
