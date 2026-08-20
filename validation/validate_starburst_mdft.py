"""Validation harness for the MDFT starburst engine (mdft_render.cu).

Gates the exact resample-free engine against the legacy sprite path and a
supersampled ground truth built from the same effective pupil.  Prints one
GATE line per check (tier-2 convention) and writes fig_mdft.png; exits 0 iff
every gate passes, 2 otherwise.

Run:  python validate_starburst_mdft.py [--out DIR]
"""
import argparse
import sys
import time

import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight

DG = str(lens_file("DoubleGauss.lens"))
SIRUI = str(lens_file("sirui_75mm_133xANA_f1_8_v01.lens"))

_gates = []


def gate(name, ok, detail=""):
    _gates.append(bool(ok))
    print(f"GATE {name:44s} {'PASS' if ok else 'FAIL'}   {detail}")


def cfg(engine, sx=0.5, sy=0.5, ns=12, grid=1024, trim=8.0, blades=0):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 1.0
    c.spectral_samples = ns
    c.aperture_blades = blades
    c.diffraction.starburst = True
    c.diffraction.starburst_engine = engine
    c.diffraction.starburst_grid = grid
    c.diffraction.pupil_fill = 0.30
    c.diffraction.scale_trim = trim
    c.diffraction.spectral_samples = ns
    return c


def render(lens, cal, engine, w=256, **kw):
    d = ghostlight._ghostlight._render_starburst_debug(w, w, lens, cal, cfg(engine, **kw))
    lum = (np.asarray(d["starburst_r"]) + np.asarray(d["starburst_g"])
           + np.asarray(d["starburst_b"])).astype(np.float64)
    return lum, d


def truth_window(A, N, dx, px, spx, spy, K):
    """Supersampled (2x-pad) exact pixel box-integral of |FFT(A)|^2 (mono)."""
    P2 = np.zeros((2 * N, 2 * N), np.complex64)
    P2[N // 2:N // 2 + N, N // 2:N // 2 + N] = A
    I2 = (np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P2)))) ** 2).astype(np.float64)
    if I2.sum() <= 0:
        return None, 0, 0
    I2 /= I2.sum(); dx2, N2 = dx / 2.0, 2 * N
    kx0, ky0 = int(np.floor(spx)) - K // 2, int(np.floor(spy)) - K // 2

    def ov(k0, pos, n, tex):
        lo = ((k0 + np.arange(K) - pos) * px) / tex + 0.5 * n
        hi = lo + px / tex
        ii = np.arange(n)
        return np.clip(np.minimum(hi[:, None], ii[None, :] + 1)
                       - np.maximum(lo[:, None], ii[None, :]), 0.0, None)

    return ov(ky0, spy, N2, dx2) @ I2 @ ov(kx0, spx, N2, dx2).T, kx0, ky0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    if not ghostlight._cuda_available():
        print("no CUDA device — cannot validate the GPU engine")
        return 1

    lens = ghostlight.OpticalSystem.load(DG)
    cal = lens.calibration()
    print(f"double_gauss  f/{cal.f_number_x:.2f}  EFL {cal.focal_length_x:.1f}mm")
    print(f"default engine = {ghostlight.PointFlareConfig().diffraction.starburst_engine}\n")

    SP, MD = ghostlight.StarburstEngine.SPRITE, ghostlight.StarburstEngine.MDFT
    W = 256
    px = 2.0 * float(cal.sensor_half_w) / W

    # G-default: a default config selects the legacy sprite path (byte-identity).
    gate("default engine is SPRITE",
         ghostlight.PointFlareConfig().diffraction.starburst_engine == SP)

    # G-legacy: two SPRITE renders are bit-equal (untouched deterministic path).
    a, _ = render(lens, cal, SP, sx=0.7)
    b, _ = render(lens, cal, SP, sx=0.7)
    gate("SPRITE path deterministic (bit-equal)", np.array_equal(a, b))

    # G-flux: engine parity + MDFT spectral-sample invariance.
    sp_sum = render(lens, cal, SP)[0].sum()
    md_sums = [render(lens, cal, MD, ns=n)[0].sum() for n in (3, 6, 12, 24)]
    ratio = md_sums[2] / sp_sum
    ns_spread = max(md_sums) / min(md_sums)
    gate("flux parity mdft/sprite within 5%", abs(ratio - 1.0) < 0.05,
         f"ratio={ratio:.4f}")
    gate("MDFT flux is spectral-sample invariant", ns_spread < 1.005,
         f"ns spread={ns_spread:.4f} (sprite scaling-theorem bake wobbles)")

    # G-stability: MDFT window error vs truth is flat; sprite flickers.
    xs = [0.820 + i * 0.0015 for i in range(14)]
    def errcurve(engine):
        out = []
        for x in xs:
            lum, d = render(lens, cal, engine, sx=x, ns=6)
            A = np.asarray(d["pupil"], np.float64)
            N = int(d["grid"]); dx = float(d["dx_mm_x"])
            tw, kx0, ky0 = truth_window(A, N, dx, px, float(d["source_px"]),
                                        float(d["source_py"]), 40)
            if tw is None:
                continue
            win = lum[ky0:ky0 + 40, kx0:kx0 + 40]
            a_ = win / max(win.sum(), 1e-30); b_ = tw / max(tw.sum(), 1e-30)
            out.append(np.abs(a_ - b_).sum() / 2 * 100)
        return np.array(out)
    md_e, sp_e = errcurve(MD), errcurve(SP)
    gate("MDFT error stable vs sprite flicker", md_e.std() < sp_e.std() * 0.35,
         f"std MDFT={md_e.std():.2f}% sprite={sp_e.std():.2f}%  "
         f"(worst MDFT={md_e.max():.1f}% sprite={sp_e.max():.1f}%)")

    # G-noalias: window intensity is non-negative & finite (no aliasing lattice).
    lum, _ = render(lens, cal, MD, sx=0.845)
    gate("MDFT output finite & non-negative",
         np.isfinite(lum).all() and (lum < 0).sum() == 0)

    # G-offframe: reach matches sprite (near-edge agrees, far is zero in both).
    near = abs(render(lens, cal, MD, sx=0.98)[0].sum()
               / max(render(lens, cal, SP, sx=0.98)[0].sum(), 1e-30) - 1.0)
    far_md = render(lens, cal, MD, sx=1.15)[0].sum()
    far_sp = render(lens, cal, SP, sx=1.15)[0].sum()
    gate("off-frame reach matches sprite", near < 0.08 and far_md == 0 and far_sp == 0,
         f"near ratio err={near:.3f}, far both zero={far_md == 0 and far_sp == 0}")

    # G-route: an overflow starburst (frac>1 — a big size on the small anamorphic
    # sensor) auto-routes MDFT->sprite, so requesting either engine yields the
    # identical render.  This is the fix for MDFT's magnified per-wavelength
    # "beading" in that regime (the sprite's bilinear spectral bake stays smooth).
    alens = ghostlight.OpticalSystem.load(SIRUI)
    acal = alens.calibration()
    lum_md, dmd = render(alens, acal, MD, ns=9, blades=6, trim=40.0, w=384)
    lum_sp, dsp = render(alens, acal, SP, ns=9, blades=6, trim=40.0, w=384)
    gate("overflow starburst routes MDFT->sprite (identical)",
         np.array_equal(lum_md, lum_sp) and int(dmd["grid"]) == int(dsp["grid"]),
         f"identical={np.array_equal(lum_md, lum_sp)} grid={int(dmd['grid'])}")

    # G-anamorphic: in the COMPACT regime the MDFT engine runs; its per-axis
    # pattern matches the sprite's shape (row/col transpose guard).  The Sirui is
    # nearly round (f# ratio ~1.04), so the test is same-shape, not a strong aspect.
    def cextent(engine):
        lu, dd = render(alens, acal, engine, ns=9, blades=6, trim=2.0, w=384)
        ys, xxs = np.where(lu > lu.max() * 1e-3)   # low thr captures the spikes, not just the core
        return ((int(xxs.max() - xxs.min()), int(ys.max() - ys.min())), int(dd["grid"])) \
            if lu.max() > 0 else ((0, 0), 0)
    (cm, gm), (cs, _) = cextent(MD), cextent(SP)
    gate("anamorphic MDFT matches sprite shape (no transpose)",
         gm == 1024 and abs(cm[0] - cs[0]) <= 8 and abs(cm[1] - cs[1]) <= 8,
         f"mdft x/y={cm}@{gm} sprite x/y={cs}")

    # G-perf: MDFT point-flare render within budget (and not slower than sprite).
    for e in (SP, MD):
        render(lens, cal, e)  # warm
    t = time.time()
    for _ in range(8):
        render(lens, cal, MD)
    t_md = (time.time() - t) / 8 * 1000
    t = time.time()
    for _ in range(8):
        render(lens, cal, SP)
    t_sp = (time.time() - t) / 8 * 1000
    gate("MDFT perf <= 25ms and <= 1.5x sprite", t_md < 25.0 and t_md < t_sp * 1.5,
         f"mdft={t_md:.1f}ms sprite={t_sp:.1f}ms")

    # G-extent: a prominent starburst grows the grid to reach the frame edge (no
    # floating box); a compact one is left untouched (grid unchanged, bit-identity).
    _, d_comp = render(lens, cal, SP, trim=8.0)            # frac ~0.10 -> no extend
    lum_prom, d_prom = render(lens, cal, SP, trim=40.0)    # frac ~0.49 -> extend
    g_comp, g_prom = int(d_comp["grid"]), int(d_prom["grid"])
    band = max(4, lum_prom.shape[0] // 16)                 # frame-edge ring
    ring = lum_prom.copy(); ring[band:-band, band:-band] = 0.0
    reaches = lum_prom.sum() > 0 and ring.sum() > 0        # a floating box leaves the ring ~0
    gate("auto-extent: compact untouched, prominent reaches frame",
         g_comp == 1024 and g_prom > 1024 and reaches,
         f"grid compact={g_comp} prominent={g_prom}, frame-ring energy={reaches}")

    # ---- figure ----
    try:
        _figure(lens, cal, args.out, px, W)
    except Exception as e:  # pragma: no cover — figure is diagnostic only
        print(f"(figure skipped: {e})")

    n_pass = sum(_gates)
    print(f"\n==== {n_pass}/{len(_gates)} gates passed ====")
    return 0 if n_pass == len(_gates) else 2


def _figure(lens, cal, out, px, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SP, MD = ghostlight.StarburstEngine.SPRITE, ghostlight.StarburstEngine.MDFT
    xs = [0.820 + i * 0.0015 for i in range(14)]
    curves = {}
    for eng, name in [(MD, "mdft"), (SP, "sprite")]:
        e = []
        for x in xs:
            lum, d = render(lens, cal, eng, sx=x, ns=6)
            A = np.asarray(d["pupil"], np.float64); N = int(d["grid"]); dx = float(d["dx_mm_x"])
            tw, kx0, ky0 = truth_window(A, N, dx, px, float(d["source_px"]), float(d["source_py"]), 40)
            win = lum[ky0:ky0 + 40, kx0:kx0 + 40]
            a_ = win / max(win.sum(), 1e-30); b_ = tw / max(tw.sum(), 1e-30)
            e.append(np.abs(a_ - b_).sum() / 2 * 100)
        curves[name] = e
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#10161f")
    ax.set_facecolor("#10161f")
    dxpx = [(x - xs[0]) * 2 * float(cal.sensor_half_w) / (2 * float(cal.sensor_half_w) / W) for x in xs]
    ax.plot(dxpx, curves["sprite"], color="#e0604e", lw=2, label="sprite")
    ax.plot(dxpx, curves["mdft"], color="#21a38d", lw=2, label="mdft")
    ax.set_xlabel("source sub-pixel offset", color="#8a94a6")
    ax.set_ylabel("shape error vs truth (%)", color="#8a94a6")
    ax.set_title("MDFT error is flat; sprite flickers with sub-pixel phase", color="#eef1f6", fontsize=10)
    ax.tick_params(colors="#8a94a6")
    for s in ax.spines.values():
        s.set_color("#28303d")
    ax.legend(facecolor="#1a2130", edgecolor="#28303d", labelcolor="#eef1f6")
    fig.tight_layout()
    path = out.rstrip("/\\") + "/fig_mdft.png"
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
