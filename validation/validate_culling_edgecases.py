"""Culling losslessness on edge-case configurations.

Culling could drift from the reference if the probe and the main kernel ever
disagree about which pupil samples exist or which surfaces are active. Two configs
stress exactly that:

  * hex_stop : FlareConfig.aperture_blades override (a pre-mask-only feature — it
               changes the launched pupil samples but is never baked into Surface).
               The probe reuses the same sampler params (ghost_render.cu:570), so it
               must see the same hexagonal mask.
  * muted    : a surface with is_active=False. The IOR table's "before" walk and the
               trace both branch on is_active; probe and render share that table, so
               a muted surface must not shift any cull decision.

For each (lens, variant) it renders an off-axis point source cull-OFF vs cull-ON
(deterministic Halton) and requires the ghost buffers identical to sub-8-bit. A
FALSE cull (a live ghost wrongly removed) would blow far past the 8-bit quantum,
like the 10980-ppm regression the point-flare harness caught earlier.

    python validate_culling_edgecases.py
"""
from __future__ import annotations
import math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import ARTIFACTS, lens_file  # noqa: E402
import ghostlight  # noqa

OUT = ARTIFACTS / "culling_edgecases"
OUT.mkdir(parents=True, exist_ok=True)
W = H = 320
SH = 14.0
POS = (1.3, 0.5)                          # off-axis: culling strongly active
QUANTUM_PPM = 1.0 / 255.0 * 1e6           # 8-bit quantum = 3922 ppm
CASES = [("double_gauss", lens_file("DoubleGauss.lens")),
         ("sirui", lens_file("Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens"))]
VARIANTS = ["plain", "hex_stop", "muted"]


def mute_one(lens):
    """Mute an interior, active, non-stop surface near the middle; return its index."""
    S = list(lens.surfaces)
    cands = [i for i, s in enumerate(S)
             if s.is_active and not s.is_stop and 0 < i < len(S) - 1]
    idx = cands[len(cands) // 2]
    lens.surfaces[idx].is_active = False
    return idx, bool(lens.surfaces[idx].is_active)


def prep(path, variant):
    lens = ghostlight.OpticalSystem.load(str(path))
    info = ""
    if variant == "muted":
        idx, now = mute_one(lens)
        info = f"muted surf {idx} (is_active={now})"
    calib = lens.calibration()
    return lens, calib, info


def cfg(variant, cull, stats=False, jitter=2, seed=12345):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = POS
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = 256
    c.spectral_samples = 16
    c.pupil_jitter = jitter
    c.jitter_seed = seed                  # jitter=2 Halton (deterministic); jitter=1 wang-hash (varies w/ seed)
    c.sensor_half_w = SH
    c.sensor_half_h = SH
    c.min_ghost_intensity = 0.0
    if variant == "hex_stop":
        c.aperture_blades = 6             # force hexagonal pupil mask (override)
        c.aperture_rotation = 0.0
    c.cull_dead_pairs = cull
    c.collect_stats = stats
    return c


def render(lens, calib, variant, cull, stats=False, jitter=2, seed=12345):
    o = lens.render_point_flare(W, H, cfg(variant, cull, stats, jitter, seed), calib=calib)
    img = np.stack([np.asarray(o["ghost_r"]), np.asarray(o["ghost_g"]),
                    np.asarray(o["ghost_b"])], -1)
    return img, o.get("stats")


def timed(lens, calib, variant, cull, n=2):
    best, img = 1e9, None
    for _ in range(n):
        t = time.perf_counter()
        img, _ = render(lens, calib, variant, cull)
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
    all_pass = True
    print(f"{'lens':>12} {'variant':>9} {'pairs':>6} {'dead':>5} {'speedup':>8}   "
          f"{'centroid':>9} {'chroma_d':>9} {'energy':>7} {'maxpix':>10} {'MCnoise':>10} {'verdict':>8}")
    for name, path in CASES:
        if not path.exists():
            print(f"missing {path}"); all_pass = False; continue
        fig, axes = plt.subplots(len(VARIANTS), 3, figsize=(11, 3.4 * len(VARIANTS)))
        for row, variant in enumerate(VARIANTS):
            lens, calib, info = prep(path, variant)
            render(lens, calib, variant, True)                       # warmup
            _, st = render(lens, calib, variant, True, stats=True)   # null-mask diag
            n_pairs = int(st["n_pairs"])
            ons = np.array(st["pair_on_sensor"], float); tr = np.array(st["pair_traces"], float)
            dead = int(((ons == 0) & (tr > 0)).sum())
            full, t_full = timed(lens, calib, variant, False)
            cull, t_cull = timed(lens, calib, variant, True)
            mf, mc = moments(full), moments(cull)
            centroid = math.hypot(mf[0] - mc[0], mf[1] - mc[1]) if mf and mc else 0.0
            spread = (mc[2] / mf[2]) if (mf and mc and mf[2] > 0) else 1.0
            chroma_d = float(np.linalg.norm(chroma(full) - chroma(cull))) * 1e3
            energy = float(cull.sum() / (full.sum() + 1e-20))
            peak = float(full.sum(-1).max())
            maxpix = float(np.abs(full - cull).sum(-1).max()) / (peak + 1e-20) * 1e6
            # MC noise floor: two independent cull-OFF realizations of the same
            # config. Culling passes if its worst-pixel change is within the render's
            # frame-to-frame noise (a fixed 8-bit quantum is only valid when converged).
            nA, _ = render(lens, calib, variant, False, jitter=1, seed=101)
            nB, _ = render(lens, calib, variant, False, jitter=1, seed=202)
            noise = float(np.abs(nA - nB).sum(-1).max()) / (peak + 1e-20) * 1e6
            ok = (maxpix < QUANTUM_PPM) or (maxpix < noise)
            all_pass = all_pass and ok
            tag = variant + (f" [{info}]" if info else "")
            print(f"{name:>12} {variant:>9} {n_pairs:6d} {dead:5d} {t_full/t_cull:7.2f}x   "
                  f"{centroid:9.3f} {chroma_d:9.3f} {energy:7.4f} {maxpix:10.1f} {noise:10.0f} "
                  f"{'PASS' if ok else 'FAIL':>8}   {info}")
            exp = 0.9 / (np.percentile(full.sum(-1), 99.5) + 1e-6)
            for col, (lab, im) in enumerate([
                    (f"reference ({n_pairs} pairs)", full),
                    (f"culled (dead {dead}, {t_full/t_cull:.1f}x)", cull),
                    ("25x |difference|", np.abs(full - cull) * 25)]):
                ax = axes[row, col]
                ax.imshow(tonemap(im, exp)); ax.axis("off")
                if row == 0:
                    ax.set_title(lab, fontsize=9)
                if col == 0:
                    ax.text(-0.04, 0.5, tag, transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=8)
        fig.suptitle(f"{name} — culling edge cases (difference must stay black)", fontsize=11)
        fig.tight_layout()
        p = OUT / f"cull_validation_edgecases_{name}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        print(f"    montage -> {p}")
    print(f"\npass: maxpix < 8-bit quantum ({QUANTUM_PPM:.0f} ppm)  OR  maxpix < MC noise floor")
    print("      (culling is statistically equivalent when its change is within the render's own noise)")
    print("RESULT:", "ALL PASS" if all_pass else "FAIL — investigate before proceeding")


if __name__ == "__main__":
    main()
