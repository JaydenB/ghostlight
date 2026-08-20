"""Point-flare sampling render baseline.

Renders deterministic Halton configurations and compares their ghost buffers
with saved arrays. Concentrated sampling is validated separately because it is
statistically equivalent rather than bit-stable.

    python point_flare_sampling_baseline.py
    python point_flare_sampling_baseline.py --save
"""
import pathlib, sys
import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight  # noqa

# Beside this script, not in Temp: these are the "before" of every rebaseline,
# so they have to outlive a temp-directory sweep.
GOLD = pathlib.Path(__file__).resolve().parent / "goldens" / "point_flare_sampling"
GOLD.mkdir(parents=True, exist_ok=True)
W = H = 320
SH = 14.0
CASES = [("double_gauss", "DoubleGauss.lens"),
         ("sirui", "Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens")]
# on-axis, off-axis, far-corner, + a polygon-override case (exercises the mask).
CONFIGS = [(0.5, 0.5, 0), (1.3, 0.5, 0), (1.6, 0.9, 0), (1.3, 0.5, 6)]


def cfg(sx, sy, cull, blades):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = 256
    c.spectral_samples = 16
    c.pupil_jitter = 2
    c.jitter_seed = 12345                 # Halton: deterministic
    c.sensor_half_w = SH
    c.sensor_half_h = SH
    c.min_ghost_intensity = 0.0
    if blades:
        c.aperture_blades = blades
    c.cull_dead_pairs = cull
    # Pin the classic path because concentrated sampling is not bit-stable.
    if hasattr(c, "concentrate_samples"):
        c.concentrate_samples = False
    return c


def render(lens, calib, sx, sy, cull, blades):
    o = lens.render_point_flare(W, H, cfg(sx, sy, cull, blades), calib=calib)
    return np.stack([np.asarray(o["ghost_r"]), np.asarray(o["ghost_g"]),
                     np.asarray(o["ghost_b"])], -1)


def main():
    save = "--save" in sys.argv
    if not ghostlight._cuda_available():
        print("No CUDA."); return
    n_ok = n_fail = 0
    for name, fn in CASES:
        lens = ghostlight.OpticalSystem.load(str(lens_file(fn))); calib = lens.calibration()
        render(lens, calib, 0.5, 0.5, True, 0)          # warmup
        for sx, sy, blades in CONFIGS:
            for cull in (False, True):
                img = render(lens, calib, sx, sy, cull, blades)
                key = f"{name}_{sx}_{sy}_b{blades}_c{int(cull)}"
                p = GOLD / f"{key}.npy"
                if p.exists() and not save:
                    ref = np.load(p)
                    d = float(np.abs(img.astype(np.float64) - ref.astype(np.float64)).max())
                    peak = float(ref.sum(-1).max())
                    ppm = d / (peak + 1e-30) * 1e6
                    # GPU atomicAdd order varies run-to-run, so byte-exact equality
                    # is unattainable; gate at the measured atomic-reorder floor
                    # (~0.1-0.3 ppm of peak) with margin to 1 ppm. 8-bit visibility
                    # sits 4 orders higher (3922 ppm).
                    ok = ppm < 1.0
                    n_ok += ok; n_fail += (not ok)
                    print(f"  {key:34} maxdiff={d:.3e} = {ppm:8.3f} ppm  {'OK' if ok else 'FAIL'}")
                else:
                    np.save(p, img)
                    print(f"  {key:34} saved  sum={img.sum(dtype=np.float64):.8e}")
    if not save:
        print(f"\n{n_ok} within the atomic floor (<1 ppm of peak), {n_fail} above")
        print("RESULT:", "UNCHANGED (atomic floor)" if n_fail == 0 else "CHANGED — investigate")


if __name__ == "__main__":
    main()
