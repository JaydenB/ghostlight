# ============================================================================
# validate_starburst_pupil.py — off-axis cat's-eye, matte box, and
# front-glass textures folded into the diffraction pupil.
#
# Produces validation figures against BOTH paths:
#   * the isolated debug path (_render_starburst_debug) — exposes the effective
#     pupil amplitude A(u,v), the mono PSF, and the RGB sprite + throughput T;
#   * the designer render path (render_point_flare) — the actual composited
#     ghost + starburst buffers the panels show.
#
# Figures (written next to this script's --out dir):
#   fig_catseye.png   pupil + sprite morph vs field angle (the off-axis fix)
#   fig_mattebox.png  barn-door clips -> crisp knife-edge diffraction fringes
#   fig_texture.png   front-glass dust -> pupil holes -> diffraction halos
#   fig_designer.png  render_point_flare on-axis vs off-axis vs matte (designer path)
#   fig_theory.png    throughput vs field (vignetting) + cat's-eye = circle ∩ circle
# ============================================================================
import os, sys, argparse
import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DBG = ghostlight._ghostlight._render_starburst_debug
MATTE_OPEN = float(ghostlight._ghostlight.MATTE_BOX_OPEN)   # a side left open never clips
LENS_PATH = str(lens_file("DoubleGauss.lens"))

# ---- shared display helpers -------------------------------------------------
INK, MUTED, ACC = "#e9e6df", "#8b8680", "#e0b048"
plt.rcParams.update({
    "figure.facecolor": "#14140f", "axes.facecolor": "#14140f",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#3a382f",
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
})

def crop(a, half):
    c = a.shape[0] // 2
    return a[c - half:c + half, c - half:c + half]

def show_pupil(ax, pupil, half=210, title=None):
    ax.imshow(crop(pupil, half), origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)

def tonemap_sprite(rgb, half=170, gamma=2.4):
    s = crop(rgb, half).copy()
    m = s.max()
    if m <= 0: return s
    return np.clip(s / m, 0, 1) ** (1.0 / gamma)

def show_sprite(ax, rgb, half=170, title=None):
    ax.imshow(tonemap_sprite(rgb, half), origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)

def load_lens():
    lens = ghostlight.OpticalSystem.load(LENS_PATH)
    return lens, lens.calibration()

def cfg_point(sx=0.5, sy=0.5, blades=8, rot=22.5, ns=20, surv=True, tex=True,
              matte=None):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 1.0
    c.spectral_samples = ns
    c.aperture_blades = blades; c.aperture_rotation = rot
    d = c.diffraction
    d.starburst = True; d.spectral_samples = ns
    d.use_survivor_mask = surv; d.use_surface_textures = tex
    if matte is not None:
        d.matte_box.enabled = True
        d.matte_box.top, d.matte_box.bottom, d.matte_box.left, d.matte_box.right = matte
    return c

def debug(lens, cal, grid=1024, **kw):
    c = cfg_point(**kw)
    return DBG(grid, grid, lens, cal, c)

def sprite_of(r):
    N = r["grid"]
    return np.array(r["sprite_rgb"]).reshape(N, N, 3)

def rms_axes(a):
    a = np.maximum(np.asarray(a, float), 0.0); s = a.sum()
    if s <= 0: return 0.0, 0.0
    N = a.shape[0]; y, x = np.mgrid[0:N, 0:N]
    cx = (a * x).sum() / s; cy = (a * y).sum() / s
    return (np.sqrt((a * (x - cx) ** 2).sum() / s),
            np.sqrt((a * (y - cy) ** 2).sum() / s))

# ---------------------------------------------------------------------------
def fig_catseye(lens, cal, out):
    xs = [0.5, 0.70, 0.85, 0.97]   # kept within the imaged field (chief survives)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for k, sx in enumerate(xs):
        r = debug(lens, cal, sx=sx, tex=False)
        P = np.array(r["pupil"]); S = sprite_of(r)
        ang = np.degrees(r["field_angle_x"])
        psx, psy = rms_axes(P); ssx, ssy = rms_axes(S.sum(-1))
        pr = (psx / psy) if psy > 0 else 0.0
        sr = (ssx / ssy) if ssy > 0 else 0.0
        show_pupil(ax[0, k], P,
                   title=f"{ang:.1f}°  pupil  x/y={pr:.2f}\nthroughput T={r['throughput']:.2f}")
        show_sprite(ax[1, k], S, title=f"starburst  x/y={sr:.2f}")
    ax[0, 0].set_ylabel("effective pupil A(u,v)", color=INK, fontsize=10)
    ax[1, 0].set_ylabel("diffraction starburst", color=INK, fontsize=10)
    fig.suptitle("Off-axis cat's-eye — the pupil vignettes to a lens shape and the "
                 "starburst morphs + dims (source swept left→right)",
                 color=INK, fontsize=12.5, y=0.99)
    fig.text(0.5, 0.005,
             "Pupil narrows in x (x/y ↓) ⇒ starburst widens in x (x/y ↑): correct Fourier duality. "
             "T is the physical pupil throughput — the burst dims with vignetting.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(out, "fig_catseye.png"), dpi=115)
    plt.close(fig); print("  -> fig_catseye.png")

def fig_mattebox(lens, cal, out):
    O = MATTE_OPEN
    cases = [
        ("no matte box", None),
        ("top / bottom barn doors\n(±6 mm)", (6, 6, O, O)),
        ("left / right barn doors\n(±6 mm)", (O, O, 6, 6)),
        ("letterbox (all four)\ntop/bot ±5, l/r ±9", (5, 5, 9, 9)),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for k, (name, mb) in enumerate(cases):
        r = debug(lens, cal, sx=0.5, blades=0, surv=False, tex=False, matte=mb)
        show_pupil(ax[0, k], np.array(r["pupil"]),
                   title=f"{name}\nT={r['throughput']:.2f}")
        show_sprite(ax[1, k], sprite_of(r), half=210)
    ax[0, 0].set_ylabel("effective pupil A(u,v)", color=INK)
    ax[1, 0].set_ylabel("diffraction starburst", color=INK)
    fig.suptitle("Matte box (barn doors) — a hard knife edge on the pupil, so it "
                 "produces its own edge-perpendicular diffraction fringes",
                 color=INK, fontsize=12.5, y=0.99)
    fig.text(0.5, 0.005,
             "The clip is applied crisply at full FFT resolution (not baked into the "
             "smooth vignette) precisely so the barn-door edge diffracts. "
             "The SAME matte box clips the geometric ghost trace.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(out, "fig_mattebox.png"), dpi=115)
    plt.close(fig); print("  -> fig_mattebox.png")

def make_dusty_lens():
    lens = ghostlight.OpticalSystem.load(LENS_PATH)
    front_R = lens.surfaces[0].semi_aperture
    # The pupil only samples the central ep/front_R (~0.53) of the front glass, so
    # to make the dust land in the pupil we keep it inside that footprint. (Dust
    # outside it is physically real but invisible to an on-axis beam.)
    ep_frac = 0.50
    Npx = 320
    yy, xx = np.mgrid[0:Npx, 0:Npx].astype(np.float32)
    u = (xx + 0.5) / Npx * 2 - 1; v = (yy + 0.5) / Npx * 2 - 1
    tex = np.ones((Npx, Npx), np.float32)
    rng = np.random.default_rng(3)
    placed = 0
    while placed < 16:
        cx, cy = rng.uniform(-ep_frac, ep_frac, 2); rr = rng.uniform(0.02, 0.055)
        if cx * cx + cy * cy < 0.16 * 0.16:   # keep centre clear for the axial cal ray
            continue
        tex[((u - cx) ** 2 + (v - cy) ** 2) < rr * rr] = 0.0
        placed += 1
    tex[(u * u + v * v) > 1.0] = 1.0
    img = ghostlight.ApertureImage()
    img.semi_diameter = float(front_R); img.pixels = tex
    lens.surfaces[0].aperture_shape = ghostlight.ApertureShape.IMAGE
    lens.surfaces[0].aperture_semi_diameter = float(front_R)
    lens.aperture_images[0] = img
    return lens, lens.calibration(), tex

def fig_texture(out):
    lens, cal, tex = make_dusty_lens()
    r_on = debug(lens, cal, sx=0.5, blades=0, surv=False, tex=True)
    r_off = debug(lens, cal, sx=0.5, blades=0, surv=False, tex=False)
    S_on = sprite_of(r_on).sum(-1); S_off = sprite_of(r_off).sum(-1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.0))
    ax[0].imshow(tex, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("front-glass texture\n(dust, transmission 0..1)", fontsize=9)
    show_pupil(ax[1], np.array(r_on["pupil"]),
               title="pupil — texture ON (dust holes)\nT=%.3f  (OFF: T=1.000)" % r_on["throughput"])
    # clean vs dusty starburst, log-scaled to reveal the scattered halo
    def logdisp(a, half=180):
        a = crop(a, half); a = np.maximum(a, 0); m = a.max()
        return np.log10(a / m + 1e-4) if m > 0 else a
    ax[2].imshow(logdisp(S_on), origin="lower", cmap="inferno", vmin=-4, vmax=0)
    ax[2].set_title("starburst with dust (log)", fontsize=9)
    # difference: where the dust throws light (scatter), log magnitude
    d = np.abs(crop(S_on, 180) - crop(S_off, 180)); m = d.max()
    ax[3].imshow(np.log10(d / m + 1e-4) if m > 0 else d, origin="lower", cmap="inferno", vmin=-4, vmax=0)
    ax[3].set_title("dust scatter |ON − OFF| (log)", fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Front-glass texture — dust on the front element modulates the pupil amplitude, "
                 "so it both dims the beam and diffracts a scatter halo",
                 color=INK, fontsize=12.5, y=1.02)
    fig.text(0.5, -0.03,
             "Same bitmap the ray tracer already vignettes with (matched UV convention); here it also enters "
             "the coherent pupil. The rightmost panel isolates the extra light the dust scatters into the field.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_texture.png"), dpi=115, bbox_inches="tight")
    plt.close(fig); print("  -> fig_texture.png")

def _meter(hwc):   # the designer panel's robust metering (sparse-frame aware)
    p99 = float(np.percentile(hwc, 99)); peak = float(hwc.max())
    if peak < 1e-8: return 0.0
    if p99 >= peak * 1e-3: return (0.9 / p99) if p99 >= 1e-8 else 0.0
    lum = hwc.max(-1) if hwc.ndim == 3 else hwc
    sig = lum[lum > peak * 1e-4]
    ref = float(np.percentile(sig, 90)) if sig.size else peak
    return (0.9 / ref) if ref >= 1e-8 else 0.0

def _composite(o):
    g = np.stack([o["ghost_r"], o["ghost_g"], o["ghost_b"]], -1)
    if "starburst_r" in o:
        g = g + np.stack([o["starburst_r"], o["starburst_g"], o["starburst_b"]], -1)
    return g

def _crop_to_burst(img, lum, z=150):
    # Centre the crop on the brightest pixel (the starburst core), so an off-axis
    # source that lands near the frame edge is still framed.
    H, W = lum.shape
    iy, ix = np.unravel_index(int(np.argmax(lum)), lum.shape)
    iy = min(max(iy, z), H - z); ix = min(max(ix, z), W - z)
    return img[iy - z:iy + z, ix - z:ix + z]

def fig_designer(out):
    lens = ghostlight.OpticalSystem.load(LENS_PATH)
    W = 512
    def render(sx, sy, matte=None):
        c = cfg_point(sx, sy, blades=8, rot=22.5, ns=16, surv=True, tex=False, matte=matte)
        c.source_r = c.source_g = c.source_b = 8.0
        c.flare_gain = 2000.0
        c.diffraction.scale_trim = 10.0
        return lens.render_point_flare(W, W, c)
    panels = [
        ("on-axis (source centred)", render(0.5, 0.5)),
        ("off-axis (source→edge)\ncat's-eye morph + dim", render(0.82, 0.5)),
        ("off-axis + matte box\ntop/bottom ±7 mm", render(0.82, 0.5, matte=(7, 7, MATTE_OPEN, MATTE_OPEN))),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.7))
    for a, (name, o) in zip(ax, panels):
        hwc = _composite(o); s = _meter(hwc)
        img = np.clip(hwc * s, 0, 1) ** (1 / 2.2)
        a.imshow(_crop_to_burst(img, hwc.max(-1), z=150), origin="upper")
        a.set_title(name, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Designer render path (render_point_flare) — ghost + starburst composited, "
                 "the actual buffers the panels show",
                 color=INK, fontsize=12.5, y=1.0)
    fig.text(0.5, -0.02, "Each panel framed on its starburst core (the off-axis source lands near the "
             "frame edge). Same robust metering the panel uses; note the off-axis burst morph + matte cut.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_designer.png"), dpi=115, bbox_inches="tight")
    plt.close(fig); print("  -> fig_designer.png")

def fig_cutoff(lens, cal, out):
    # Sweep a source out to (and past) the lens's imaged field on a large sensor.
    # The starburst renders — dimming and morphing — across the WHOLE field, then
    # switches off exactly when the chief ray is clipped (the source is no longer
    # imaged). The cut-off is gated on the real chief ray, not a throughput number.
    import math
    scale = 5.0
    mh = np.degrees(cal.max_half_angle_h)          # frame edge == chief-clip angle
    fracs = [0.0, 0.5, 0.78, 0.97, 1.18]           # ... last one is past the field
    def sx_for(deg):
        tw = scale * math.tan(math.radians(mh))
        ndc = (math.tan(math.radians(deg)) / tw) if tw > 0 else 0.0
        return 0.5 + 0.5 * ndc
    fig, ax = plt.subplots(2, len(fracs), figsize=(3.05 * len(fracs), 6.4))
    for k, f in enumerate(fracs):
        c = cfg_point(sx_for(f * mh), 0.5, blades=8, rot=22.5, ns=14, surv=True, tex=False)
        c.sensor_half_w = cal.sensor_half_w * scale
        c.sensor_half_h = cal.sensor_half_h * scale
        r = DBG(1024, 1024, lens, cal, c)
        P = np.array(r["pupil"]); S = sprite_of(r)
        ang = np.degrees(r["field_angle_x"]); T = r["throughput"]
        on = (np.array(r["starburst_r"]).sum() + np.array(r["starburst_g"]).sum()
              + np.array(r["starburst_b"]).sum()) > 1e-9
        show_pupil(ax[0, k], P, title=f"{ang:.1f}°   T={T:.3f}"
                   + ("" if on else "\nchief clipped"))
        if on:
            show_sprite(ax[1, k], S, half=200)
        else:
            ax[1, k].imshow(np.zeros((40, 40, 3))); ax[1, k].set_xticks([]); ax[1, k].set_yticks([])
            ax[1, k].text(0.5, 0.5, "OFF\n(chief ray\nclipped)", ha="center", va="center",
                          color=ACC, fontsize=12, family="monospace", transform=ax[1, k].transAxes)
    ax[0, 0].set_ylabel("effective pupil A(u,v)", color=INK)
    ax[1, 0].set_ylabel("diffraction starburst", color=INK)
    fig.suptitle("Cut-off tied to the chief ray — the starburst renders across the whole imaged "
                 "field, then switches off exactly when the source stops being imaged",
                 color=INK, fontsize=12.5, y=0.99)
    fig.text(0.5, 0.018,
             "Columns at 0 / 0.5 / 0.78 / 0.97 / 1.18 × the field-edge angle. Last panel is past the field: a "
             "crescent still survives, but the chief ray is clipped — source no longer imaged, so no starburst.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(out, "fig_cutoff.png"), dpi=115)
    plt.close(fig); print("  -> fig_cutoff.png")

def fig_theory(lens, cal, out):
    # (a) throughput vs field angle vs cos^4 natural falloff
    xs = np.linspace(0.5, 1.18, 16)
    ang, T = [], []
    for sx in xs:
        r = debug(lens, cal, grid=512, sx=float(sx), blades=0, tex=False)
        ang.append(np.degrees(r["field_angle_x"])); T.append(r["throughput"])
    ang = np.array(ang); T = np.array(T)
    cos4 = np.cos(np.radians(ang)) ** 4

    # (b) cat's-eye envelope boundary vs the two-circle vignetting model. The eye
    # is a vesica: two circular arcs meeting at the top/bottom corners. Fit each
    # arc exactly through 3 measured points (a side extreme + the two corners) —
    # a circle through 3 symmetric points has its centre on the u-axis.
    r = debug(lens, cal, grid=512, sx=1.05, blades=0, tex=False)
    P = np.array(r["pupil"]); N = P.shape[0]
    fill = 0.30
    edge = (P > 0.5 * P.max())
    ys, xs2 = np.where(edge)
    to_u = lambda i: ((i + 0.5) / N * 2 - 1) / fill
    U = to_u(xs2); V = to_u(ys)
    uL, uR = U.min(), U.max()
    vh = np.abs(V).max()
    uT = U[np.argmax(np.abs(V))]                    # u at the top/bottom corner
    def circ_uaxis(us, u0):                          # centre cx on u-axis through (us,0),(uT,±vh)
        cx = (us * us - uT * uT - vh * vh) / (2.0 * (us - uT))
        return cx, abs(us - cx)
    cxR, rR = circ_uaxis(uR, uT)                      # right arc (the stop)
    cxL, rL = circ_uaxis(uL, uT)                      # left arc (the shifted vignetting rim)
    th = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2))
    ax[0].plot(ang, T, "-o", color=ACC, lw=2, ms=5, label="traced pupil throughput T (mechanical vignetting)")
    ax[0].plot(ang, cos4, "--", color="#6fb2c9", lw=1.8, label="cos⁴θ natural illumination")
    ax[0].plot(ang, T * cos4, ":", color=MUTED, lw=1.6, label="product ≈ relative illumination")
    ax[0].set_xlabel("field angle (deg)"); ax[0].set_ylabel("relative")
    ax[0].set_ylim(0, 1.05); ax[0].grid(alpha=0.12)
    ax[0].legend(fontsize=8.2, facecolor="#14140f", edgecolor="#3a382f", labelcolor=INK)
    ax[0].set_title("Off-axis falloff: rim vignetting dominates cos⁴", fontsize=10.5)

    ax[1].scatter(U, V, s=3, color=ACC, alpha=0.30, label="traced survivor pupil")
    ax[1].plot(cxR + rR * np.cos(th), rR * np.sin(th), color="#6fb2c9", lw=1.4, label="aperture stop (circle)")
    ax[1].plot(cxL + rL * np.cos(th), rL * np.sin(th), color="#c98f6f", lw=1.4, label="shifted rim (circle)")
    ax[1].set_aspect("equal"); ax[1].set_xlim(-1.25, 1.25); ax[1].set_ylim(-1.25, 1.25)
    ax[1].set_xlabel("pupil u"); ax[1].set_ylabel("pupil v"); ax[1].grid(alpha=0.12)
    ax[1].legend(fontsize=8.2, facecolor="#14140f", edgecolor="#3a382f", labelcolor=INK, loc="upper right")
    ax[1].set_title("Cat's-eye = stop ∩ shifted rim (fitted to the trace)", fontsize=10.5)
    fig.suptitle("Comparison with vignetting theory", color=INK, fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_theory.png"), dpi=115, bbox_inches="tight")
    plt.close(fig); print("  -> fig_theory.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lens, cal = load_lens()
    print("Generating pupil validation figures ...")
    fig_catseye(lens, cal, args.out)
    fig_cutoff(lens, cal, args.out)
    fig_mattebox(lens, cal, args.out)
    fig_texture(args.out)
    fig_designer(args.out)
    fig_theory(lens, cal, args.out)
    print("done ->", args.out)

if __name__ == "__main__":
    main()
