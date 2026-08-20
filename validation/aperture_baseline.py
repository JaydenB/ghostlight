"""Render baseline for the bladed aperture profile.

Captures every renderer that reads the stop silhouette or the pupil calibration,
on five lenses: a spherical control, an anamorphic (exercises every per-axis
path), two plain bladed-stop fixtures, and one carrying every blade control at
once. Saved buffers are the "before"; re-running compares against them.

The plain hexagons pin the untouched-polygon path, which is what the landing
gates needed. blade_shape is the only case that runs the deformed branch, so it
is what a LATER regression in the profile maths would show up in — without it
this set would go on passing while the new geometry rotted.

    python aperture_baseline.py           # save if missing, else compare
    python aperture_baseline.py --save    # (re)save the whole set
    python aperture_baseline.py --only hex_stop

The splat kernels accumulate with
atomicAdd, whose summation order varies run to run, so byte-exact equality is
unattainable on GPU paths and the gate is the measured atomic-reorder floor
(1 ppm of peak). Layers that reduce deterministically do come back exactly
equal, and the report says which did, so a real change that happens to land
under 1 ppm on one layer still shows up as "was 0.0, now isn't".

The manifest carries the calibration scalars alongside the buffers, and the
report prints any that moved. A change that only shows up in the calibration —
a pupil measurement, an f-number — perturbs the renders too little to clear the
atomic floor, so without the manifest it would pass silently.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
GOLD = HERE / "goldens" / "aperture_baseline"
GOLD.mkdir(parents=True, exist_ok=True)

# The atomic-reorder floor is measured in ppm of the layer's own peak.
PPM_LIMIT = 1.0

W = H = 192
JITTER_SEED = 12345

# The gate's capture band is narrow — only landings that actually strike the
# wall scrape — so the source position that lights it up is per-lens. These were
# found by sweeping; a position outside the band records an all-zero layer,
# which gates nothing.
CASES = [
    ("double_gauss", lens_file("DoubleGauss.lens"), 1.03),
    ("sirui",        lens_file("Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens"), 1.05),
    ("hex_stop",     GOLD / "hex_stop.lens", 1.03),
    ("hex_rot30",    GOLD / "hex_stop_rot30.lens", 1.03),
    ("blade_shape",  GOLD / "blade_shape.lens", 1.03),
]


# --------------------------------------------------------------------------
# Configs. Every renderer gets the same deterministic sampling: Halton pupil,
# fixed seed, concentration off (statistically equivalent, not bit-stable).
# --------------------------------------------------------------------------
def _base_cfg(lens, calib, sx, sy, *, coverage=1.0):
    c = ghostlight.PointFlareConfig()
    c.source_x, c.source_y = sx, sy
    c.source_r = c.source_g = c.source_b = 10.0
    c.flare_gain = 5000.0
    c.ray_grid = 192
    c.spectral_samples = 8
    c.pupil_jitter = 2
    c.jitter_seed = JITTER_SEED
    c.sensor_half_w = coverage * calib.sensor_half_w
    c.sensor_half_h = coverage * calib.sensor_half_h
    c.min_ghost_intensity = 0.0
    c.concentrate_samples = False
    return c


def _hwc(out, prefix):
    keys = [f"{prefix}_r", f"{prefix}_g", f"{prefix}_b"]
    if not all(k in out for k in keys):
        return None
    return np.stack([np.asarray(out[k]) for k in keys], -1)


def _ghosts(lens, calib, sx, sy):
    cfg = _base_cfg(lens, calib, sx, sy)
    return _hwc(lens.render_point_flare(W, H, cfg, calib=calib), "ghost")


def _starburst(lens, calib, engine):
    cfg = _base_cfg(lens, calib, 0.5, 0.5)
    cfg.diffraction.starburst = True
    cfg.diffraction.starburst_engine = engine
    cfg.diffraction.starburst_grid = 512
    cfg.diffraction.pupil_fill = 0.30
    cfg.diffraction.scale_trim = 8.0
    cfg.diffraction.spectral_samples = 8
    return _hwc(lens.render_point_flare(W, H, cfg, calib=calib), "starburst")


def _gate(lens, calib, sx):
    # The gate only scrapes light that lands outside the opening, so the frame
    # has to sit inside the image circle and the source just outside the frame.
    cfg = _base_cfg(lens, calib, sx, 0.5, coverage=0.70)
    cfg.gate.enabled = True
    out = lens.render_point_flare(W, H, cfg, calib=calib)
    layer = _hwc(out, "gate")
    return np.zeros((H, W, 3), np.float32) if layer is None else layer


def _psf(lens, calib):
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = 3
    cfg.tile_w = cfg.tile_h = 48
    cfg.tile_extent_mm = 0.12
    cfg.spectral_samples = 4
    cfg.ray_grid = 48
    ax = calib.max_half_angle_h * 0.5
    ay = calib.max_half_angle_v * 0.5
    xs = np.linspace(-ax, ax, 3, dtype=np.float32)
    ys = np.linspace(-ay, ay, 3, dtype=np.float32)
    sources = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    out = lens.render_psf(sources.astype(np.float32), cfg)
    return np.stack([np.asarray(out["r"]), np.asarray(out["g"]),
                     np.asarray(out["b"])], -1)


# One fixed chunk of angular offsets (dx, dy, weight) — a 5-point cross around
# the source. Hand-written rather than drawn from source_sampling so the chunk
# can't drift if that module's sequence is ever retuned.
_SOURCE_CHUNK = np.array([
    [0.0000, 0.0000, 0.36],
    [0.0012, 0.0000, 0.16],
    [-0.0012, 0.0000, 0.16],
    [0.0000, 0.0012, 0.16],
    [0.0000, -0.0012, 0.16],
], dtype=np.float32)


def _source(lens, calib):
    cfg = _base_cfg(lens, calib, 0.7, 0.5)
    return _hwc(lens.render_source_flare(_SOURCE_CHUNK, W, H, cfg, calib=calib),
                "ghost")


def layers_for(lens, calib, gate_sx):
    """Every layer this set captures, as (name, zero-arg render callable)."""
    return [
        ("ghost_axial",      lambda: _ghosts(lens, calib, 0.5, 0.5)),
        ("ghost_offaxis",    lambda: _ghosts(lens, calib, 1.3, 0.5)),
        ("ghost_corner",     lambda: _ghosts(lens, calib, 1.6, 0.9)),
        ("starburst_sprite", lambda: _starburst(lens, calib, ghostlight.StarburstEngine.SPRITE)),
        ("starburst_mdft",   lambda: _starburst(lens, calib, ghostlight.StarburstEngine.MDFT)),
        ("gate",             lambda: _gate(lens, calib, gate_sx)),
        ("psf",              lambda: _psf(lens, calib)),
        ("source_chunk",     lambda: _source(lens, calib)),
    ]


CALIB_FIELDS = (
    "sensor_half_w", "sensor_half_h", "max_half_angle_h", "max_half_angle_v",
    "image_circle_semi_w", "image_circle_semi_h",
    "focal_length_x", "focal_length_y",
    "entrance_pupil_semi_x", "entrance_pupil_semi_y",
    "f_number_x", "f_number_y", "pupil_area_frac",
)


def _calib_dict(calib):
    return {f: float(getattr(calib, f)) for f in CALIB_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="(re)save the goldens")
    ap.add_argument("--only", default="", help="restrict to one case name")
    args = ap.parse_args()

    if not ghostlight._cuda_available():
        print("No CUDA — nothing to capture.")
        return 1

    manifest_path = GOLD / "manifest.json"
    old_manifest = {}
    if manifest_path.exists() and not args.save:
        old_manifest = json.loads(manifest_path.read_text())
    manifest = {}

    n_exact = n_floor = n_fail = 0
    for name, path, gate_sx in CASES:
        if args.only and args.only != name:
            continue
        lens = ghostlight.OpticalSystem.load(str(path))
        calib = lens.calibration()
        manifest[name] = _calib_dict(calib)
        print(f"\n{name}  ({path.name})")

        # Warm-up: the first render on a fresh context also allocates, and an
        # allocation failure mid-set would otherwise look like a diff.
        _ghosts(lens, calib, 0.5, 0.5)

        for layer, fn in layers_for(lens, calib, gate_sx):
            img = np.asarray(fn(), dtype=np.float32)
            p = GOLD / f"{name}__{layer}.npy"
            if p.exists() and not args.save:
                ref = np.load(p)
                if ref.shape != img.shape:
                    print(f"  {layer:18} SHAPE {ref.shape} -> {img.shape}  FAIL")
                    n_fail += 1
                    continue
                d = float(np.abs(img.astype(np.float64)
                                 - ref.astype(np.float64)).max())
                peak = float(np.abs(ref).max())
                ppm = d / (peak + 1e-30) * 1e6
                if d == 0.0:
                    n_exact += 1
                    verdict = "EXACT"
                elif ppm < PPM_LIMIT:
                    n_floor += 1
                    verdict = "floor"
                else:
                    n_fail += 1
                    verdict = "FAIL"
                print(f"  {layer:18} maxdiff={d:.3e}  {ppm:9.3f} ppm  {verdict}")
            else:
                np.save(p, img)
                print(f"  {layer:18} saved  sum={img.sum(dtype=np.float64):.8e}")

        if not args.save and name in old_manifest:
            for field, was in old_manifest[name].items():
                now = manifest[name].get(field)
                if now is None:
                    continue
                if was == 0.0:
                    moved = abs(now) > 0.0
                else:
                    moved = abs(now - was) / abs(was) > 1e-6
                if moved:
                    print(f"  calib {field:24} {was:.6g} -> {now:.6g}")

    if args.save or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"\nmanifest written: {manifest_path}")
    else:
        (GOLD / "manifest.current.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.save:
        print(f"\n{n_exact} exact, {n_floor} within the atomic floor "
              f"(<{PPM_LIMIT:g} ppm), {n_fail} above")
        print("RESULT:", "UNCHANGED" if n_fail == 0 else "CHANGED — investigate")
        return 0 if n_fail == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
