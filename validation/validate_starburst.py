# ============================================================================
# starburst_validate.py — validate the C++/CUDA starburst against an
# independent numpy reference, and against closed-form optics.
#
# Gates:
#   1. Calibration first-order sanity (EFL / f-number / entrance pupil).
#   2. Mono PSF: C++ dbg.psf vs numpy reference (same pupil, same FFT).
#   3. Physical scale: dx_mm == lambda_ref * f_number * pupil_fill; Airy first
#      zero at 1.22 / pupil_fill texels for a circular pupil.
#   4. Spike geometry: an 8-blade pupil throws 8 spikes at the expected angles.
#   5. Dispersion: the red channel's radial extent exceeds blue (pattern ~ lambda).
#   6. Energy conservation: sprite total is independent of the FFT grid size.
# Saves figures next to this script.
# ============================================================================
import os, sys
import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
LENS = str(lens_file("DoubleGauss.lens"))

SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
S_BLUE, S_AQUA, S_YELLOW, S_RED = "#3987e5", "#199e70", "#c98500", "#e66767"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK2, "axes.edgecolor": "#383835", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "Segoe UI",
    "axes.titlecolor": INK, "font.size": 11,
})
GLOW = LinearSegmentedColormap.from_list("glow", [
    (0.0, "#0b0b0d"), (0.28, "#10233f"), (0.52, "#1c5cab"),
    (0.70, "#3987e5"), (0.87, "#9ec5f4"), (1.0, "#ffffff")])

results = []
def gate(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# ---------------------------------------------------------------------------
# numpy reference: mono PSF, mirroring starburst_render.cu exactly.
# ---------------------------------------------------------------------------
def numpy_mono_psf(N, fill, n_blades=0, rot_rad=0.0):
    ii = np.arange(N)
    cover = np.zeros((N, N))
    for sy in range(4):
        for sx in range(4):
            gu = ((ii[None, :] + (sx + 0.5) / 4) / N * 2 - 1) / fill
            gv = ((ii[:, None] + (sy + 0.5) / 4) / N * 2 - 1) / fill
            r2 = gu * gu + gv * gv
            inside = r2 <= 1.0
            if n_blades >= 3:
                ang = np.arctan2(gv, gu) - rot_rad
                sector_ang = 2 * np.pi / n_blades
                sector = np.mod(ang, sector_ang)
                apothem = np.cos(np.pi / n_blades)
                inside = inside & (np.sqrt(r2) * np.cos(sector - sector_ang * 0.5) <= apothem)
            cover = cover + inside
    cover /= 16.0
    F = np.fft.fft2(cover.astype(np.complex128))
    psf = np.fft.fftshift(np.abs(F) ** 2)
    s = psf.sum()
    return psf / s if s > 0 else psf

# ---------------------------------------------------------------------------
print("Loading lens + calibration ...")
lens = ghostlight.OpticalSystem.load(LENS)
cal = lens.calibration()
print(f"  sensor_half   = {cal.sensor_half_w:.3f} x {cal.sensor_half_h:.3f} mm")
print(f"  max_half_angle= {np.degrees(cal.max_half_angle_h):.2f} x {np.degrees(cal.max_half_angle_v):.2f} deg")
print(f"  focal_length  = {cal.focal_length_x:.3f} / {cal.focal_length_y:.3f} mm")
print(f"  entrance_pupil= {cal.entrance_pupil_semi_x:.3f} / {cal.entrance_pupil_semi_y:.3f} mm (semi)")
print(f"  f_number      = {cal.f_number_x:.3f} / {cal.f_number_y:.3f}")

# GATE 1: first-order sanity. Double Gauss ~50-100mm, f/2ish. Just require the
# solve produced positive, self-consistent numbers (f# = f / 2r).
f1 = cal.focal_length_x
r1 = cal.entrance_pupil_semi_x
fn1 = cal.f_number_x
consistent = abs(fn1 - f1 / (2 * r1)) < 1e-2 if r1 > 0 else False
gate("calibration first-order", f1 > 0 and r1 > 0 and 0.5 < fn1 < 32 and consistent,
     f"EFL={f1:.1f}mm f/{fn1:.2f} EP_semi={r1:.2f}mm")

# ---------------------------------------------------------------------------
def run_cpp(grid=1024, fill=0.30, blades=0, rot_deg=0.0, ns=24, source_x=0.5,
            scale_trim=1.0, w=256, h=256):
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = source_x; cfg.source_y = 0.5
    cfg.source_r = cfg.source_g = cfg.source_b = 1.0
    cfg.spectral_samples = ns
    cfg.aperture_blades = blades
    cfg.aperture_rotation = rot_deg
    cfg.diffraction.starburst = True
    cfg.diffraction.starburst_grid = grid
    cfg.diffraction.pupil_fill = fill
    cfg.diffraction.scale_trim = scale_trim
    cfg.diffraction.spectral_samples = ns
    return ghostlight._ghostlight._render_starburst_debug(w, h, lens, cal, cfg)

print("\nRendering circular pupil (C++) ...")
FILL = 0.30
d_circ = run_cpp(grid=1024, fill=FILL, blades=0)
psf_cpp = np.asarray(d_circ["psf"])
N = d_circ["grid"]
print(f"  grid={N}, dx={d_circ['dx_mm_x']*1e3:.4f} um/texel, "
      f"f#={d_circ['f_number_x']:.3f}, EFL={d_circ['focal_length_x']:.2f}mm")

# numpy reference
psf_np = numpy_mono_psf(N, FILL, n_blades=0)

# GATE 2: mono PSF match. Normalize both to unit sum; compare via correlation
# and peak-region relative error.
a = psf_cpp / psf_cpp.sum()
b = psf_np / psf_np.sum()
corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
# central 64x64 relative L1
c = N // 2
sl = slice(c - 32, c + 32)
num = np.abs(a[sl, sl] - b[sl, sl]).sum()
den = b[sl, sl].sum()
rel = float(num / den)
gate("mono PSF C++ vs numpy", corr > 0.999 and rel < 0.02,
     f"corr={corr:.6f}, central L1 rel={rel*100:.3f}%")

# GATE 3: physical scale. dx == lambda_ref * f# * fill.
lam_ref_mm = d_circ["lambda_ref_nm"] * 1e-6
dx_expected = lam_ref_mm * d_circ["f_number_x"] * FILL
dx_got = d_circ["dx_mm_x"]
scale_ok = abs(dx_got - dx_expected) / dx_expected < 1e-3
# Airy first zero: radial-average the circular PSF, find first minimum.
from scipy.ndimage import map_coordinates
th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
rr = np.arange(1, 60)
prof = np.zeros(len(rr))
for t in th:
    xs = c + rr * np.cos(t); ys = c + rr * np.sin(t)
    prof += map_coordinates(psf_cpp, np.stack([ys, xs]), order=1)
prof /= len(th)
# first local minimum after the peak
zero_idx = None
for i in range(2, len(prof) - 1):
    if prof[i] < prof[i - 1] and prof[i] <= prof[i + 1]:
        zero_idx = rr[i]; break
airy_expected = 1.22 / FILL
airy_ok = zero_idx is not None and abs(zero_idx - airy_expected) / airy_expected < 0.12
gate("physical scale (dx = lambda*f#*fill)", scale_ok,
     f"dx got={dx_got*1e3:.5f} exp={dx_expected*1e3:.5f} um")
gate("Airy first zero @ 1.22/fill texels", airy_ok,
     f"got={zero_idx} texels, expected~{airy_expected:.2f}")

# ---------------------------------------------------------------------------
print("\nRendering 8-blade pupil (C++) ...")
d_oct = run_cpp(grid=1024, fill=FILL, blades=8, rot_deg=22.5)
psf_oct = np.asarray(d_oct["psf"])

# GATE 4: spike geometry. Sum intensity in an annulus by azimuth, find peaks.
ann = np.zeros(720)
rlo, rhi = 40, 180
for k, t in enumerate(np.linspace(0, 2 * np.pi, 720, endpoint=False)):
    rr2 = np.arange(rlo, rhi)
    xs = c + rr2 * np.cos(t); ys = c + rr2 * np.sin(t)
    ann[k] = map_coordinates(np.log10(np.maximum(psf_oct, 1e-30)),
                             np.stack([ys, xs]), order=1).sum()
ann_s = ann - ann.min()
thr = ann_s.max() * 0.5
peaks = []
for k in range(720):
    if ann_s[k] > thr and ann_s[k] >= ann_s[(k - 1) % 720] and ann_s[k] > ann_s[(k + 1) % 720]:
        peaks.append(k)
# merge near-adjacent peaks
merged = []
for p in peaks:
    if not merged or min((p - merged[-1]) % 720, (merged[-1] - p) % 720) > 10:
        merged.append(p)
n_spikes = len(merged)
gate("8-blade -> 8 spikes", n_spikes == 8, f"found {n_spikes} spikes at "
     f"{sorted(int(round(m*0.5)) for m in merged)} deg")

# GATE 5: dispersion. Red channel radial extent > blue on the octagon sprite.
sprite = np.asarray(d_oct["sprite_rgb"])  # (N,N,3)
def channel_radial_extent(chan, frac=0.5):
    # azimuthally-averaged radial profile; radius where it drops below frac*peak-of-far
    prof = np.zeros(c - 2)
    rr3 = np.arange(1, c - 1)
    for t in np.linspace(0, 2 * np.pi, 180, endpoint=False):
        xs = c + rr3 * np.cos(t); ys = c + rr3 * np.sin(t)
        prof += map_coordinates(chan, np.stack([ys, xs]), order=1)
    prof /= 180
    # radius containing 90% of energy (energy-weighted extent)
    cum = np.cumsum(prof * rr3)
    cum /= cum[-1]
    return rr3[np.searchsorted(cum, 0.9)]
ext_r = channel_radial_extent(sprite[:, :, 0])
ext_b = channel_radial_extent(sprite[:, :, 2])
disp_ratio = ext_r / max(ext_b, 1)
gate("dispersion: red pattern larger than blue", disp_ratio > 1.05,
     f"R/B radial-extent ratio = {disp_ratio:.3f}")

# GATE 6: energy conservation across grid size.
d512 = run_cpp(grid=512, fill=FILL, blades=8, rot_deg=22.5)
s1024 = np.asarray(d_oct["sprite_rgb"]).sum()
s512 = np.asarray(d512["sprite_rgb"]).sum()
econs = abs(s1024 - s512) / max(s1024, s512)
gate("energy conservation across grid (512 vs 1024)", econs < 0.03,
     f"sum1024={s1024:.4g} sum512={s512:.4g} rel diff={econs*100:.2f}%")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def show_log(ax, img, lo=-7, hi=0, cmap=GLOW, ex=None):
    d = np.log10(np.maximum(img / img.max(), 10.0 ** lo))
    ax.imshow(d, cmap=cmap, vmin=lo, vmax=hi, origin="lower", interpolation="antialiased", extent=ex)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

# Figure V1: C++ vs numpy mono PSF + difference + radial profile
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
show_log(axes[0], psf_cpp); axes[0].set_title("C++ mono PSF (circular)", fontsize=11)
show_log(axes[1], psf_np);  axes[1].set_title("numpy reference", fontsize=11)
diff = np.abs(a - b)
axes[2].imshow(np.log10(np.maximum(diff/ a.max(), 1e-9)), cmap="magma", origin="lower")
axes[2].set_xticks([]); axes[2].set_yticks([])
for s in axes[2].spines.values(): s.set_visible(False)
axes[2].set_title(f"|C++ - numpy| (log)\ncorr={corr:.5f}", fontsize=11)
axes[3].plot(rr, prof/prof.max(), color=S_BLUE, lw=2)
axes[3].axvline(airy_expected, color=S_RED, ls="--", lw=1.5)
axes[3].text(airy_expected+1, 0.5, f"1.22/fill\n={airy_expected:.1f}px", color=S_RED, fontsize=9)
if zero_idx: axes[3].axvline(zero_idx, color=S_AQUA, ls=":", lw=1.5)
axes[3].set_yscale("log"); axes[3].set_ylim(1e-4, 1.4); axes[3].set_xlim(0, 40)
axes[3].grid(True, color=GRID, lw=0.7)
axes[3].set_xlabel("radius (texels)"); axes[3].set_title("radial profile + Airy zero", fontsize=11)
fig.suptitle(f"Starburst validation — circular pupil, {N}² FFT, fill={FILL} (Q={1/FILL:.1f})",
             color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "val_1_monopsf.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

# Figure V2: 8-blade PSF + azimuthal spike detection + sprite
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
show_log(axes[0], psf_oct); axes[0].set_title("C++ 8-blade mono PSF", fontsize=11)
axd = np.linspace(0, 360, 720, endpoint=False)
axes[1].plot(axd, ann_s, color=S_BLUE, lw=1.5)
for m in merged: axes[1].axvline(m*0.5, color=S_YELLOW, lw=1, alpha=0.8)
axes[1].set_xlim(0, 360); axes[1].grid(True, color=GRID, lw=0.7)
axes[1].set_xlabel("azimuth (deg)"); axes[1].set_title(f"annular intensity → {n_spikes} spikes", fontsize=11)
def disp_rgb(spr):
    m = spr.max()
    x = np.clip(spr / (m if m>0 else 1), 0, 1) ** 0.3
    return x
axes[2].imshow(disp_rgb(sprite), origin="lower", interpolation="antialiased")
axes[2].set_xticks([]); axes[2].set_yticks([])
for s in axes[2].spines.values(): s.set_visible(False)
axes[2].set_title(f"spectral sprite (24λ)\nR/B extent {disp_ratio:.2f}", fontsize=11)
fig.suptitle("Starburst validation — 8-blade dispersion + spike geometry", color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "val_2_blades.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\n{'='*60}\nGATES: {n_pass}/{len(results)} passed")
for name, ok, det in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print("figures -> val_1_monopsf.png, val_2_blades.png")
sys.exit(0 if n_pass == len(results) else 1)
