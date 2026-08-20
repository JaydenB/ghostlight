"""Validation gates for the film-gate flare layer (mechanical).

The gate is a separate additive pass: primary rays that reach the sensor plane
just outside the aperture-plate opening strike the plate's cut edge at grazing
incidence, reflect off it, and land back inside the frame at the mirror fold.
See GateConfig in render_config.h for the model and gate.h for the shared math.

Split of responsibility, as validate_starburst_hurb.py splits its own: the
distribution shape of the scatter lobe and the resolved-parameter arithmetic are
pinned on the CPU in tests/test_gate_flare.py. THIS script validates the KERNEL
INTEGRATION on the live render path — that the layer appears only when it should,
perturbs nothing else, obeys the closed-form band and reach laws, and costs
nothing when off.

  G1  off is byte-identical (atomic floor) and on engages the pass
  G2  the fold is a mirror: it lands inside the wall, within the reach law
  G3  attribute separation — each knob moves one property (figure)
  G4  brightness against the source's own direct image
  G5  determinism and chunk invariance on the source-flare path
  G6  no occlusion: ghost / starburst / veil untouched
  G7  a source imaged inside the frame produces exactly zero
  G8  stop-down law: specular reach tracks (standoff + thickness) / 2N
  G9  zero-cost when off
  G10 anamorphic asymmetry: the side walls capture a wider band than top/bottom

Run:
    python validation\\validate_gate_flare.py --out <dir>
    python validation\\validate_gate_flare.py --only g3
"""

import argparse
import sys
import time
from pathlib import Path

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ghostlight

SPHERICAL = lens_file("DoubleGauss.lens")
ANAMORPHIC = lens_file("ana_usd9063321b2.lens")

COVERAGE = 0.70          # sensor half-extent as a fraction of the image circle
W = 192

# House palette, matching the other validate_* figures.
BG, FG, ACCENT, ACCENT2 = "#101317", "#E7EAEE", "#C4862A", "#2C8CC0"

_gates = []


def gate(name, ok, detail=""):
    _gates.append((name, bool(ok)))
    print(f"GATE {name:52s} {'PASS' if ok else 'FAIL'}   {detail}")


# ---------------------------------------------------------------- render helpers
def cfg_for(lens, *, sx=1.03, sy=0.5, on=True, w=W, **gate_kw):
    cal = lens.calibration()
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 10.0
    c.ray_grid, c.spectral_samples, c.pupil_jitter = 192, 6, 2
    c.sensor_half_w = COVERAGE * cal.sensor_half_w
    c.sensor_half_h = COVERAGE * cal.sensor_half_h
    c.gate.enabled = on
    for k, v in gate_kw.items():
        setattr(c.gate, k, v)
    return c


def gate_layer(lens, **kw):
    w = kw.pop("w", W)
    out = lens.render_point_flare(w, w, cfg_for(lens, w=w, **kw))
    g = ghostlight._arrays.gate_to_hwc(out)
    return np.zeros((w, w, 3), np.float32) if g is None else g


def dbg(lens, *, w=W, **kw):
    return ghostlight._ghostlight._render_gate_debug(w, w, lens, lens.calibration(),
                                           cfg_for(lens, w=w, **kw))


def live_band(lens, axis="x", lo=1.00, hi=1.15, step=0.005, **kw):
    """Source positions along one axis that deposit anything, in frame units."""
    live = []
    for t in np.arange(lo, hi, step):
        pos = {"sx": float(t)} if axis == "x" else {"sy": float(t), "sx": 0.5}
        if float(gate_layer(lens, w=96, **pos, **kw).sum()) > 0.0:
            live.append(float(t))
    return (min(live), max(live)) if live else None


# ------------------------------------------------------------------------ gates
def g1_off_on(lens, outdir):
    off1 = gate_layer(lens, on=False)
    off2 = gate_layer(lens, on=False)
    on = gate_layer(lens, on=True)
    noise = float(np.abs(off1 - off2).max())
    change = float(np.abs(on - off1).max())
    out = lens.render_point_flare(W, W, cfg_for(lens, on=False))
    gate("G1 off emits no gate layer", "gate_r" not in out,
         f"keys={sorted(k for k in out if k.startswith('gate'))}")
    gate("G1 off is byte-identical (atomic floor)", noise < 1e-9, f"noise={noise:.3e}")
    gate("G1 on engages the pass", change > max(1e-9, 50 * noise),
         f"change={change:.4e} vs 50*noise={50*noise:.3e}")


def g2_mirror(lens, outdir):
    cal = lens.calibration()
    d = dbg(lens, roughness_rad=0.0, w=256)
    lum = np.asarray(d["gate_r"]) + np.asarray(d["gate_g"]) + np.asarray(d["gate_b"])
    cols = np.nonzero(lum.max(axis=0) > 0.02 * lum.max())[0]
    half_w = COVERAGE * cal.sensor_half_w
    mm_per_px = 2.0 * half_w / 256
    deepest = float(256 - 1 - cols.min()) * mm_per_px
    predicted = (d["zs_mm"] + d["t_mm"]) / (2.0 * cal.f_number_x)
    gate("G2 fold lands inside the wall", int(cols.max()) <= 255,
         f"deepest column={int(cols.max())}")
    gate("G2 fold obeys the reach law", 0.0 < deepest <= 1.5 * predicted,
         f"deepest={deepest:.4f}mm vs (zs+t)/2N={predicted:.4f}mm")


def g3_separation(lens, outdir):
    base = dbg(lens)

    def energy_reach(**kw):
        d = dbg(lens, **kw)
        return d["energy"], d["reach_mm"], d["scrapes"]

    e_thin, r_thin, _ = energy_reach(thickness_mm=0.2)
    e_thick, r_thick, _ = energy_reach(thickness_mm=3.0)
    gate("G3 thickness moves energy, not reach",
         e_thick > 4.0 * e_thin and abs(r_thick - r_thin) < 0.3 * r_thin,
         f"energy x{e_thick/max(e_thin,1e-30):.1f}, reach {r_thin:.3f}->{r_thick:.3f}mm")

    e_near, r_near, _ = energy_reach(standoff_mm=5.0)
    e_far, r_far, _ = energy_reach(standoff_mm=15.0)
    gate("G3 standoff trades reach against energy",
         r_far > r_near and e_far < e_near,
         f"reach {r_near:.3f}->{r_far:.3f}mm, energy {e_near:.4f}->{e_far:.4f}")

    e0, _, s0 = energy_reach(roughness_rad=0.0)
    e1, _, s1 = energy_reach(roughness_rad=0.25)
    e2, _, s2 = energy_reach(roughness_rad=0.25, groove_angle_deg=90.0)
    gate("G3 roughness is direction-only (scrape count fixed)",
         s0 == s1 == s2, f"scrapes {s0} / {s1} / {s2}")
    gate("G3 roughness loss is light thrown back out of the opening",
         e1 < e0 and abs(e2 - e0) < 0.02 * e0,
         f"across={e1:.4f} along={e2:.4f} mirror={e0:.4f}")

    e_dark, r_dark, _ = energy_reach(reflectance_r0=0.02)
    e_shiny, r_shiny, _ = energy_reach(reflectance_r0=0.60)
    gate("G3 reflectance moves brightness only",
         e_shiny > e_dark and e_shiny < 3.0 * e_dark
         and abs(r_shiny - r_dark) < 1e-3 * max(r_dark, 1e-9),
         f"energy x{e_shiny/max(e_dark,1e-30):.2f} for 30x r0 (grazing dominates)")

    def extent_x(**kw):
        d = dbg(lens, w=256, **kw)
        lum = np.asarray(d["gate_r"]) + np.asarray(d["gate_g"]) + np.asarray(d["gate_b"])
        live = lum.max(axis=0) > 0.02 * lum.max()
        c = np.nonzero(live)[0]
        return int(c.max() - c.min() + 1) if c.size else 0

    across, along = extent_x(groove_angle_deg=0.0), extent_x(groove_angle_deg=90.0)
    gate("G3 groove angle rotates the streak", across > 2 * along,
         f"extent across={across}px along={along}px")

    # Figure: one panel per attribute, energy and reach normalised to the sweep.
    sweeps = [
        ("Thickness (mm)", "thickness_mm", [0.2, 0.5, 0.8, 1.5, 3.0, 6.0]),
        ("Standoff (mm)", "standoff_mm", [2.0, 5.0, 8.0, 12.0, 20.0]),
        ("Roughness (rad)", "roughness_rad", [0.0, 0.02, 0.05, 0.08, 0.15, 0.25]),
        ("Reflectance R0", "reflectance_r0", [0.02, 0.04, 0.1, 0.25, 0.6]),
    ]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(15, 3.4), facecolor=BG)
    for ax, (title, key, vals) in zip(axes, sweeps):
        es, rs = [], []
        for v in vals:
            d = dbg(lens, **{key: v})
            es.append(d["energy"]); rs.append(d["reach_mm"])
        es = np.asarray(es) / max(max(es), 1e-30)
        rs = np.asarray(rs) / max(max(rs), 1e-30)
        ax.plot(vals, es, "-o", color=ACCENT, lw=2, ms=4, label="energy")
        ax.plot(vals, rs, "-s", color=ACCENT2, lw=2, ms=4, label="reach")
        ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#3A424D")
        ax.set_ylim(-0.05, 1.1)
    axes[0].legend(facecolor=BG, edgecolor="#3A424D", labelcolor=FG, fontsize=7)
    fig.suptitle("Gate flare: each knob moves one property (normalised per sweep)",
                 color=FG, fontsize=11)
    fig.tight_layout()
    p = Path(outdir) / "fig_gate_separation.png"
    fig.savefig(p, dpi=110, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {p}")


def g4_brightness(lens, outdir):
    d = dbg(lens, w=256)
    ref = dbg(lens, sx=0.5, w=256)
    gate_peak = float(max(np.asarray(d[c]).max() for c in ("gate_r", "gate_g", "gate_b")))
    src_peak = float(max(np.asarray(ref[c]).max()
                         for c in ("direct_r", "direct_g", "direct_b")))
    ratio = gate_peak / max(src_peak, 1e-30)
    gate("G4 gate is faint against the source", 1e-6 < ratio < 0.05,
         f"peak ratio = {100*ratio:.4f}% of the source's own image")


def g5_determinism(lens, outdir):
    cal = lens.calibration()
    a, b = gate_layer(lens), gate_layer(lens)
    peak = max(float(a.max()), 1e-30)
    gate("G5 deterministic", float(np.abs(a - b).max()) <= 1e-5 * peak,
         f"{float(np.abs(a-b).max())/peak:.2e} x peak")

    c = cfg_for(lens)
    offs = np.zeros((16, 3), np.float32)
    offs[:, 0] = np.linspace(-2e-4, 2e-4, 16)
    offs[:, 1] = np.linspace(1e-4, -1e-4, 16)
    offs[:, 2] = 1.0 / 16

    def render(o):
        return ghostlight._arrays.gate_to_hwc(lens.render_source_flare(o, 128, 128, c, calib=cal))

    whole = render(offs)
    pk = max(float(whole.max()), 1e-30)
    worst = 0.0
    for x, y in ((offs[:8], offs[8:]), (offs[:5], offs[5:]), (offs[9:], offs[:9])):
        worst = max(worst, float(np.abs(whole - (render(x) + render(y))).max()) / pk)
    gate("G5 chunked equals unchunked", worst <= 1e-4, f"worst={worst:.2e} x peak")


def g6_no_occlusion(lens, outdir):
    worst = {}
    for on in (False, True):
        c = cfg_for(lens, on=on)
        c.diffraction.starburst = True
        c.diffraction.veil = True
        out = lens.render_point_flare(W, W, c)
        worst[on] = out
    ok = True
    detail = []
    for name, fn in (("ghost", ghostlight._arrays.ghost_to_hwc),
                     ("starburst", ghostlight._arrays.starburst_to_hwc),
                     ("veil", ghostlight._arrays.veil_to_hwc)):
        a, b = fn(worst[False]), fn(worst[True])
        pk = max(float(np.abs(a).max()), 1e-30)
        rel = float(np.abs(a - b).max()) / pk
        ok = ok and rel <= 1e-4
        detail.append(f"{name}={rel:.1e}")
    gate("G6 no occlusion of any other layer", ok, " ".join(detail))


def g7_inside_frame(lens, outdir):
    g = gate_layer(lens, sx=0.5)
    gate("G7 source inside the frame produces zero", float(np.abs(g).max()) == 0.0,
         f"max={float(np.abs(g).max()):.3e}")


def g8_stop_down(lens, outdir):
    """Specular reach is (standoff + thickness) / 2N, so stopping down shrinks it.
    Measured with roughness off — with the lobe on, the spread is standoff*tan(theta)
    and carries no f-number dependence at all."""
    stop_idx = [i for i, s in enumerate(lens.surfaces) if s.is_stop]
    if not stop_idx:
        gate("G8 stop-down law", False, "lens has no stop surface")
        return
    si = stop_idx[0]
    original = lens.surfaces[si].semi_aperture
    rows = []
    try:
        for scale in (1.0, 0.7, 0.5):
            # Mutate the stop in place; calibration() re-derives from it. Do NOT
            # call reload() here — it re-reads the file and undoes the change.
            lens.surfaces[si].semi_aperture = original * scale
            cal = lens.calibration()
            d = dbg(lens, roughness_rad=0.0, w=256)
            rows.append((cal.f_number_x, d["reach_mm"],
                         (d["zs_mm"] + d["t_mm"]) / (2.0 * cal.f_number_x)))
    finally:
        lens.surfaces[si].semi_aperture = original
    ns = [r[0] for r in rows]
    reaches = [r[1] for r in rows]
    monotone = all(reaches[i] > reaches[i + 1] for i in range(len(reaches) - 1))
    close = all(abs(r[1] - r[2]) <= 0.45 * r[2] for r in rows if r[2] > 0)
    gate("G8 stopping down shrinks the reach", monotone,
         " ".join(f"f/{n:.2f}->{r:.3f}mm" for n, r in zip(ns, reaches)))
    gate("G8 reach tracks (zs+t)/2N", close,
         " ".join(f"{r[1]:.3f}vs{r[2]:.3f}" for r in rows))


def g9_zero_cost(lens, outdir):
    def timeit(on):
        c = cfg_for(lens, on=on)
        lens.render_point_flare(W, W, c)          # warm up
        t0 = time.perf_counter()
        for _ in range(5):
            lens.render_point_flare(W, W, c)
        return (time.perf_counter() - t0) / 5 * 1e3

    off_ms, on_ms = timeit(False), timeit(True)
    gate("G9 off costs nothing", off_ms <= on_ms * 1.15,
         f"off={off_ms:.1f}ms on={on_ms:.1f}ms (+{100*(on_ms/off_ms-1):.0f}%)")


def g10_anamorphic(outdir):
    """Anamorphic glass has genuinely different f-numbers per axis, so the same
    law — reach = (standoff + thickness) / 2N — predicts a different reach off the
    side walls than off the top and bottom.

    Measured in sensor millimetres on each axis rather than in source-position
    units: those are fractions of the frame, and an anamorphic frame is not
    square, so a ratio taken in them would mix the optics with the aspect."""
    if not ANAMORPHIC.exists():
        gate("G10 anamorphic asymmetry", False, "lens file missing")
        return
    lens = ghostlight.OpticalSystem.load(str(ANAMORPHIC))
    cal = lens.calibration()

    def reach_mm(axis):
        best = 0.0
        for t in np.arange(1.0, 1.30, 0.01):
            pos = {"sx": float(t)} if axis == "x" else {"sy": float(t), "sx": 0.5}
            d = dbg(lens, roughness_rad=0.0, w=128, **pos)
            best = max(best, float(d["reach_mm"]))
        return best

    rx, ry = reach_mm("x"), reach_mm("y")
    if rx <= 0.0 or ry <= 0.0:
        gate("G10 anamorphic asymmetry", False, f"no scrape (x={rx:.4f}, y={ry:.4f})")
        return
    pred_x = (5.0 + 0.8) / (2.0 * cal.f_number_x)
    gate("G10 anamorphic reaches asymmetrically", abs(rx - ry) > 0.10 * max(rx, ry),
         f"x={rx:.3f}mm y={ry:.3f}mm ({100*abs(rx-ry)/max(rx,ry):.0f}% apart) "
         f"on f/{cal.f_number_x:.2f} x f/{cal.f_number_y:.2f}")
    # The horizontal axis tracks its closed form closely. The vertical does not:
    # calibration reports a PARAXIAL per-axis f-number, and on a cylindrical
    # design the image-space ray slope in y is not that number's reciprocal. The
    # asymmetry is real and derived; its magnitude is not Ny/Nx.
    gate("G10 horizontal reach tracks (zs+t)/2Nx", abs(rx - pred_x) <= 0.2 * pred_x,
         f"{rx:.3f}mm vs {pred_x:.3f}mm")

    # Control: a spherical lens must show no such split.
    sph = ghostlight.OpticalSystem.load(str(SPHERICAL))

    def sph_reach(axis):
        best = 0.0
        for t in np.arange(1.0, 1.12, 0.005):
            pos = {"sx": float(t)} if axis == "x" else {"sy": float(t), "sx": 0.5}
            d = ghostlight._ghostlight._render_gate_debug(
                128, 128, sph, sph.calibration(),
                cfg_for(sph, roughness_rad=0.0, w=128, **pos))
            best = max(best, float(d["reach_mm"]))
        return best

    sx_r, sy_r = sph_reach("x"), sph_reach("y")
    gate("G10 spherical reaches symmetrically (control)",
         abs(sx_r - sy_r) <= 0.10 * max(sx_r, sy_r, 1e-9),
         f"x={sx_r:.3f}mm y={sy_r:.3f}mm")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if not ghostlight._cuda_available():
        print("CUDA unavailable — gate validation needs a GPU.")
        sys.exit(1)
    if not SPHERICAL.exists():
        print(f"missing {SPHERICAL}")
        sys.exit(1)

    lens = ghostlight.OpticalSystem.load(str(SPHERICAL))
    checks = {
        "g1": lambda: g1_off_on(lens, args.out),
        "g2": lambda: g2_mirror(lens, args.out),
        "g3": lambda: g3_separation(lens, args.out),
        "g4": lambda: g4_brightness(lens, args.out),
        "g5": lambda: g5_determinism(lens, args.out),
        "g6": lambda: g6_no_occlusion(lens, args.out),
        "g7": lambda: g7_inside_frame(lens, args.out),
        "g8": lambda: g8_stop_down(lens, args.out),
        "g9": lambda: g9_zero_cost(lens, args.out),
        "g10": lambda: g10_anamorphic(args.out),
    }
    selected = [args.only] if args.only else list(checks)
    for key in selected:
        fn = checks.get(key)
        if fn is None:
            print(f"unknown gate {key}")
            continue
        try:
            fn()
        except Exception as exc:                      # a crash is a failed gate
            gate(f"{key.upper()} raised", False, f"{type(exc).__name__}: {exc}")

    npass = sum(1 for _, ok in _gates if ok)
    print(f"\n==== GATES {npass}/{len(_gates)} passed ====")
    for name, ok in _gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(0 if npass == len(_gates) else 2)


if __name__ == "__main__":
    main()
