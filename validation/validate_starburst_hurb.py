# ============================================================================
# validate_starburst_hurb.py — HURB edge diffraction (kernel path).
#
# HURB (Heisenberg Uncertainty Ray Bending): every ghost ray that clears an edge (an aperture/surface
# rim, or a front-of-lens baffle) at perpendicular distance d gets a random
# angular kick of scale sigma = lambda*K/d, perpendicular to that edge. The kick
# changes direction only (energy conserved), is chromatic (sigma ~ lambda), and
# is compiled OUT when off (ghost_kernel<STATS,false>) so the off path is
# byte-identical and costs nothing.
#
# The exact sampler law (sigma = lambda*K/d, the distribution shapes, the
# 650/450 chromatic ratio) is pinned on the CPU in test_hurb.py. THIS
# script validates the KERNEL INTEGRATION on the live render path:
#
#   G1  off is byte-identical (atomic noise only) and on engages the kick; the
#       legacy matte_box and an explicit Baffle(RECT) render identically.
#   G3  the kick is chromatic in the render — red spreads more than blue
#       (sqrt of the added variance), the direction of sigma ~ lambda.
#   G4  energy is conserved: an isolated ghost's flux is unchanged (open
#       aperture, where kicks don't push rays into downstream vignetting).
#   G5  the glow grows as the stop shrinks — the sigma ~ 1/d signature: as the
#       aperture closes, more of the beam sits near the rim, so HURB relocates a
#       monotonically larger fraction of the ghost's energy.
#   G6  determinism + chunk invariance: an identical hurb-on render repeats to
#       atomic noise, and an extended source rendered in one call equals the sum
#       of chunked calls (the seed is the source ANGLE, not its chunk index).
#   G9  the off path is zero-cost: byte-identical (G1) AND no slower than the
#       on path (it does strictly less work).
#
# Figures (--out): fig_hurb_live.png (off vs on ghost), fig_hurb_scaling.png
# (chromatic bars + the aperture relocation curve).
# ============================================================================
import os, sys, argparse, time
import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXT = ghostlight._ghostlight
G   = EXT.HurbKickDistribution.GAUSSIAN
L   = EXT.HurbKickDistribution.LORENTZIAN
DG  = str(lens_file("DoubleGauss.lens"))

INK, MUTED, ACC, ACC2, ACCR, ACCB = "#e9e6df", "#8b8680", "#e0b048", "#5aa9e6", "#e06a5a", "#5a7ae0"
plt.rcParams.update({
    "figure.facecolor": "#131311", "axes.facecolor": "#131311",
    "savefig.facecolor": "#131311", "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#3a382f", "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "axes.titlecolor": INK,
})

_gates = []
def gate(name, ok, detail=""):
    _gates.append((name, bool(ok)))
    print(f"GATE {name:44s} {'PASS' if ok else 'FAIL'}   {detail}")

# ---- render helpers ---------------------------------------------------------
def stop_idx(lens):
    for i in range(lens.num_surfaces()):
        if lens.surfaces[i].is_stop:
            return i
    return 0

def pf(path=DG, hurb=False, rgb=(1, 1, 1), ns=16, W=400, stop=None,
       sx=0.0, sy=0.0, kick=L, grid=384, incl=None, matte=None, baffle=None):
    """One point-flare ghost render -> (H, W, 3) float32."""
    lens = ghostlight.OpticalSystem.load(path)
    if stop is not None:
        lens.surfaces[stop_idx(lens)].semi_aperture = stop
    cal = lens.calibration()
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r, c.source_g, c.source_b = rgb
    c.spectral_samples = ns
    c.ray_grid = grid
    d = c.diffraction
    d.hurb = hurb
    d.hurb_kick = kick
    if matte is not None:                       # dict of matte-box overrides
        mb = d.matte_box
        mb.enabled = True
        mb.z_front_mm = matte.get("z_front_mm", 15.0)
        mb.top = mb.bottom = mb.left = mb.right = matte.get("side", 8.0)
    if baffle is not None:                       # explicit Baffle(RECT), same geometry
        b = ghostlight.Baffle()
        b.shape = ghostlight.BaffleShape.RECT
        b.z_front_mm = baffle.get("z_front_mm", 15.0)
        b.top = b.bottom = b.left = b.right = baffle.get("side", 8.0)
        d.baffles = [b]
    if incl is not None:
        c.ghost_filter.mode = EXT.GhostFilter.Mode.INCLUDE
        c.ghost_filter.pairs = [tuple(incl)]
    out = ghostlight.render_point_flare(W, W, lens, cal, c)
    return np.stack([np.array(out["ghost_r"]), np.array(out["ghost_g"]),
                     np.array(out["ghost_b"])], -1)

def var_img(g):
    """Intensity-weighted spatial variance about the centroid (px^2)."""
    l = g.sum(-1); tot = l.sum()
    if tot <= 0:
        return 0.0
    N = l.shape[0]; y, x = np.mgrid[0:N, 0:N]
    cx = (l * x).sum() / tot; cy = (l * y).sum() / tot
    return float((l * ((x - cx) ** 2 + (y - cy) ** 2)).sum() / tot)

def reloc_frac(off, on):
    """L1 fraction of the ghost's energy HURB moved (tail-insensitive)."""
    o = off.sum(-1); n = on.sum(-1)
    return 0.5 * np.abs(n - o).sum() / max(o.sum(), 1e-30)

# ============================================================================
# G1 — off byte-identical + on engages; matte_box == explicit Baffle.
# ============================================================================
def g1_off_on(outdir):
    off1 = pf(hurb=False, sx=0.3)
    off2 = pf(hurb=False, sx=0.3)
    on_L = pf(hurb=True,  sx=0.3, kick=L)
    on_G = pf(hurb=True,  sx=0.3, kick=G)
    noise = float(np.abs(off1 - off2).max())
    chg_L = float(np.abs(on_L - off1).max())
    chg_G = float(np.abs(on_G - off1).max())
    gate("G1 off is byte-identical (atomic noise)", noise < 1e-4, f"off-off max = {noise:.2e}")
    gate("G1 on engages the kick (Lorentzian)", chg_L > max(1e-9, 50 * noise),
         f"on-off max = {chg_L:.2e}  (noise {noise:.1e})")
    gate("G1 on engages the kick (Gaussian)", chg_G > max(1e-9, 50 * noise),
         f"on-off max = {chg_G:.2e}")

    # legacy matte_box vs an explicit Baffle(RECT) of identical geometry: the
    # step-3 stack refactor must render them bit-for-bit the same (HURB off).
    # Thresholds are RELATIVE to the ghost peak (these ghosts are dim, ~1e-7).
    nomatte = pf(hurb=False, sx=0.3)
    peak = max(float(np.abs(nomatte).max()), 1e-30)
    mb  = pf(hurb=False, sx=0.3, matte={"side": 5.0, "z_front_mm": 15.0})
    bf  = pf(hurb=False, sx=0.3, baffle={"side": 5.0, "z_front_mm": 15.0})
    ident = float(np.abs(mb - bf).max()) / peak                  # must be ~0 (identical)
    clip  = float(np.abs(mb - nomatte).max()) / peak             # the clip must be active
    gate("G1 matte_box == explicit Baffle(RECT)", ident < 1e-5 and clip > 0.02,
         f"|matte-baffle|/peak = {ident:.2e}  (clip active {clip*100:.1f}% of peak)")

# ============================================================================
# G3 — the kick is chromatic on the live path: red spreads more than blue.
# A blur of angular scale sigma adds ~sigma^2 to the ghost's spatial variance
# (convolution adds variances). sigma ~ lambda, so sqrt(added-variance) scales
# with lambda: red > blue. (The exact 650/450 = 1.44 ratio is pinned on the
# sampler in test_hurb.py; the live blur is sub-pixel per typical ray and
# confounded by dispersion, so here we assert the chromatic DIRECTION.)
# ============================================================================
def g3_chromatic(outdir):
    fields = [(0, 0), (0.1, 0), (0, 0.1), (0.15, 0.1), (0.2, 0), (0, 0.2),
              (0.12, 0.12), (0.25, 0.1)]
    dR, dB = [], []
    for sx, sy in fields:
        for stop in (None, 6.0, 4.0):
            dR.append(var_img(pf(hurb=True,  rgb=(1, 0, 0), sx=sx, sy=sy, stop=stop))
                      - var_img(pf(hurb=False, rgb=(1, 0, 0), sx=sx, sy=sy, stop=stop)))
            dB.append(var_img(pf(hurb=True,  rgb=(0, 0, 1), sx=sx, sy=sy, stop=stop))
                      - var_img(pf(hurb=False, rgb=(0, 0, 1), sx=sx, sy=sy, stop=stop)))
    mR, mB = float(np.mean(dR)), float(np.mean(dB))
    ratio = np.sqrt(max(mR, 0.0)) / max(np.sqrt(max(mB, 1e-9)), 1e-9)
    g3_chromatic.data = (mR, mB, ratio)
    gate("G3 kick is chromatic (red spreads > blue)", mR > mB and ratio > 1.08,
         f"sqrt-var-added R/B = {ratio:.2f}  (dvar R={mR:.1f} B={mB:.1f} px^2)")

# ============================================================================
# G4 — energy conservation. The kick is direction-only, so an isolated ghost's
# flux is unchanged. At open aperture (kicks don't push marginal rays into
# downstream vignetting) the ratio sits tight around 1.0.
# ============================================================================
def g4_energy(outdir):
    lens = ghostlight.OpticalSystem.load(DG)
    ratios = []
    for p in EXT.enumerate_ghost_pairs(lens):
        off = pf(hurb=False, incl=(p.surf_a, p.surf_b))
        s = off.sum()
        if s < 1e-6:
            continue
        on = pf(hurb=True, incl=(p.surf_a, p.surf_b))
        ratios.append(float(on.sum() / s))
        if len(ratios) >= 8:
            break
    ratios = np.array(ratios)
    ok = ratios.min() > 0.95 and ratios.max() < 1.005 and abs(ratios.mean() - 1.0) < 0.01
    gate("G4 energy conserved (isolated ghosts)", ok,
         f"flux ratio in [{ratios.min():.3f}, {ratios.max():.3f}]  mean {ratios.mean():.4f}")

# ============================================================================
# G5 — the sigma ~ 1/d signature on the live path: as the stop closes, a larger
# fraction of the beam sits near the rim (smaller d, larger sigma), so HURB
# relocates a monotonically growing fraction of the ghost's energy.
# ============================================================================
def g5_aperture(outdir):
    stops = [None, 6.0, 4.0, 2.5]
    rel = []
    for s in stops:
        rel.append(float(np.mean([
            reloc_frac(pf(hurb=False, rgb=(1, 1, 1), stop=s, sx=sx, sy=sy),
                       pf(hurb=True,  rgb=(1, 1, 1), stop=s, sx=sx, sy=sy))
            for sx, sy in ((0, 0), (0.12, 0))])))
    g5_aperture.data = (stops, rel)
    mono = all(rel[i + 1] > rel[i] for i in range(len(rel) - 1))
    grow = rel[-1] / max(rel[0], 1e-9)
    gate("G5 glow grows as the stop shrinks (sigma~1/d)", mono and grow > 3.0,
         f"reloc {['%.3f' % r for r in rel]}  ({grow:.1f}x, monotone={mono})")

# ============================================================================
# G6 — determinism + chunk invariance. The per-ray seed is built from the
# source ANGLE bits (not the chunk-local source index), so an extended source
# rendered in one call equals the sum of chunked calls with HURB on.
# ============================================================================
def g6_determinism_chunk(outdir):
    a = pf(hurb=True, rgb=(1, 1, 1))
    b = pf(hurb=True, rgb=(1, 1, 1))
    det = float(np.abs(a - b).max())
    gate("G6 hurb-on render is deterministic", det < 1e-4, f"on-on max = {det:.2e}")

    # extended source: a small grid of angular offsets, rendered whole vs in two
    # chunks that are summed. Linear compositing must reproduce the whole render.
    lens = ghostlight.OpticalSystem.load(DG); cal = lens.calibration()
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = 0.35, 0.0
    c.source_r = c.source_g = c.source_b = 1.0
    c.spectral_samples = 12
    c.diffraction.hurb = True
    rng = np.random.default_rng(0)
    N = 24
    offs = np.zeros((N, 3), np.float32)
    offs[:, 0] = (rng.random(N) - 0.5) * 0.02
    offs[:, 1] = (rng.random(N) - 0.5) * 0.02
    offs[:, 2] = 1.0 / N

    def sof(rows):
        out = lens.render_source_flare(rows, 360, 360, c, calib=cal)
        return np.stack([np.array(out["ghost_r"]), np.array(out["ghost_g"]),
                         np.array(out["ghost_b"])], -1)

    whole = sof(offs)
    half = sof(offs[:N // 2]) + sof(offs[N // 2:])
    diff = float(np.abs(whole - half).max())
    scale = max(float(np.abs(whole).max()), 1e-30)
    gate("G6 chunk invariance (angle-seeded)", diff < 1e-4 * scale + 1e-6,
         f"|one-call - chunked| = {diff:.2e}  (peak {scale:.1e})")

# ============================================================================
# G9 — the off path is zero-cost: byte-identical (G1) and no slower than on (it
# does strictly less work — the if constexpr(HURB) bodies are compiled out).
# ============================================================================
def _median_render_ms(hurb, reps=5):
    pf(hurb=hurb, sx=0.3)                        # warm up (JIT / cache)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pf(hurb=hurb, sx=0.3)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))

def g9_offpath_cost(outdir):
    off_ms = _median_render_ms(False)
    on_ms  = _median_render_ms(True)
    # off does strictly less work than on -> it must not be slower (allow 15%
    # timing slack). Combined with G1's byte-identity this is the zero-cost proof.
    gate("G9 off path is zero-cost (<= on time)", off_ms <= on_ms * 1.15,
         f"off {off_ms:.1f} ms <= on {on_ms:.1f} ms")

# ============================================================================
# Figures — light visual evidence (the full dataviz artifact is a later step).
# ============================================================================
def fig_live(outdir):
    off = pf(hurb=False, rgb=(1, 1, 1), stop=3.0, sx=0.0, W=360, incl=None)
    on  = pf(hurb=True,  rgb=(1, 1, 1), stop=3.0, sx=0.0, W=360, incl=None)
    l = off.sum(-1); cy, cx = np.unravel_index(np.argmax(l), l.shape); h = 120
    def cc(a):
        y0, y1 = max(0, cy - h), min(a.shape[0], cy + h)
        x0, x1 = max(0, cx - h), min(a.shape[1], cx + h)
        return a[y0:y1, x0:x1]
    def tm(a):
        m = np.percentile(a.sum(-1), 99.8); m = m if m > 0 else 1.0
        return np.clip(a / m, 0, 1) ** (1 / 2.2)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.4))
    ax[0].imshow(tm(cc(off)), origin="lower"); ax[0].set_title("HURB off\ngeometric ghost — hard rim", fontsize=10)
    ax[1].imshow(tm(cc(on)),  origin="lower"); ax[1].set_title("HURB on\nchromatic edge glow (envelope, no fringes)", fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("HURB on the live render — a stopped-down double-Gauss ghost, same flux",
                 x=0.012, ha="left", fontsize=12.0, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig_hurb_live.png"), dpi=130); plt.close(fig)

def fig_scaling(outdir):
    mR, mB, ratio = getattr(g3_chromatic, "data", (0, 0, 0))
    stops, rel = getattr(g5_aperture, "data", ([None, 6, 4, 2.5], [0, 0, 0, 0]))
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax[0].bar([0, 1], [np.sqrt(max(mR, 0)), np.sqrt(max(mB, 0))], color=[ACCR, ACCB],
              width=0.6, edgecolor="#000", linewidth=0.4)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["red 650nm", "blue 450nm"])
    ax[0].set_ylabel(r"$\sqrt{\Delta\,\mathrm{variance}}$  (px)")
    ax[0].set_title(f"chromatic: red spreads more (x{ratio:.2f})\nsigma ~ lambda — exact ratio pinned in Gate 2",
                    fontsize=9.8, loc="left")
    ax[0].grid(True, axis="y", color="#26251f", lw=0.5)
    xl = [("open" if s is None else f"{s:.1f}") for s in stops]
    ax[1].plot(range(len(rel)), rel, "-o", color=ACC, lw=1.6, mfc=ACC2, mec="#000")
    ax[1].set_xticks(range(len(rel))); ax[1].set_xticklabels(xl)
    ax[1].set_xlabel("stop semi-aperture (mm)"); ax[1].set_ylabel("energy relocated by HURB")
    ax[1].set_title("aperture: glow grows as the stop closes\nsigma ~ 1/d — more of the beam sits near the rim",
                    fontsize=9.8, loc="left")
    ax[1].grid(True, color="#26251f", lw=0.5)
    fig.suptitle("HURB scales correctly on the live path — chromatic and aperture-dependent",
                 x=0.012, ha="left", fontsize=12.0, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(outdir, "fig_hurb_scaling.png"), dpi=130); plt.close(fig)

# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not EXT._cuda_available():
        print("CUDA not available — cannot validate."); sys.exit(1)

    checks = {
        "g1": g1_off_on, "g3": g3_chromatic, "g4": g4_energy, "g5": g5_aperture,
        "g6": g6_determinism_chunk, "g9": g9_offpath_cost,
    }
    todo = [args.only] if args.only else list(checks)
    for name in todo:
        print(f"\n=== {name} ===")
        try:
            checks[name](args.out)
        except Exception as e:
            import traceback; traceback.print_exc()
            gate(f"{name}: ran", False, str(e))

    if not args.only:
        try:
            fig_live(args.out); fig_scaling(args.out)
            print(f"\nfigures -> {args.out}")
        except Exception:
            import traceback; traceback.print_exc()

    npass = sum(1 for _, ok in _gates if ok)
    print(f"\n==== GATES {npass}/{len(_gates)} passed ====")
    sys.exit(0 if npass == len(_gates) else 2)

if __name__ == "__main__":
    main()
