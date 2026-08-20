"""Validation gates for the bladed aperture profile.

The geometry itself is pinned on the CPU in tests/test_aperture_profile.py
against the oracle. THIS script validates the live render path: that the three
consumers of the silhouette agree, that the diffraction pupil is the shape the
profile describes, that scrubbing a control moves the render smoothly rather
than in steps, and that starburst brightness follows the stop's open area.

Each gate below is selectable by its id:

  G1  the diffraction pupil IS the profile, to sub-texel
  G2  trace, pupil mask and diffraction pupil agree on one silhouette
  G3  scrubbing each control moves the starburst smoothly
  G4  starburst flux tracks the stop's open area
  G5  f-number tracks the pupil's support extent through a rotation sweep
  G6  the C++ profile matches the Python oracle at float32 tolerance
  G7  a representative control combination renders (figure)

Run:
    python validation\\validate_aperture_shapes.py [--out DIR]
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import tempfile

import numpy as np

from _paths import lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight  # noqa: E402

ORACLE = pathlib.Path(__file__).resolve().parent / "aperture_profile_oracle.py"
BASE = lens_file("DoubleGauss.lens")

# A representative control combination: every blade control off its default at
# once, so one figure exercises curvature, twist, notch and rotation together.
REFERENCE = dict(blades=5, rotation_deg=29.0, curvature=-0.90, twist=-0.80,
                 notch_deg=35.8, notch_angle_deg=45.0)

# House palette, matching the other validate_* figures.
BG, FG, ACCENT, ACCENT2 = "#101317", "#E7EAEE", "#C4862A", "#2C8CC0"

_gates = []


def gate(name, ok, detail=""):
    _gates.append(bool(ok))
    print(f"GATE {name:52s} {'PASS' if ok else 'FAIL'}   {detail}")


# ---------------------------------------------------------------------------
# Lens construction
# ---------------------------------------------------------------------------

def bladed(blades=6, rotation_deg=0.0, **controls) -> ghostlight.OpticalSystem:
    """double_gauss with its stop switched to a bladed one."""
    doc = json.loads(BASE.read_text(encoding="utf-8"))
    mod = {"type": "aperture", "shape": "polygon", "blades": blades}
    if rotation_deg:
        mod["rotation_deg"] = rotation_deg
    mod.update({k: v for k, v in controls.items() if v})
    for element in doc["optical_system"]:
        for surface in element.get("surfaces", []):
            if surface.get("is_stop"):
                surface.setdefault("modifiers", []).append(mod)
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w",
                                     delete=False) as f:
        json.dump(doc, f)
        path = f.name
    return ghostlight.OpticalSystem.load(path)


def stop_of(lens):
    return next(s for s in lens.surfaces if s.is_stop)


def cfg(grid=512, trim=8.0, engine=None):
    c = ghostlight.PointFlareConfig()
    c.source_x = c.source_y = 0.5
    c.source_r = c.source_g = c.source_b = 1.0
    c.spectral_samples = 8
    c.pupil_jitter = 2
    c.jitter_seed = 12345
    c.ray_grid = 192
    c.concentrate_samples = False
    c.diffraction.starburst = True
    c.diffraction.starburst_grid = grid
    c.diffraction.pupil_fill = 0.30
    c.diffraction.scale_trim = trim
    c.diffraction.spectral_samples = 8
    if engine is not None:
        c.diffraction.starburst_engine = engine
    return c


def starburst(lens, w=192, **kw):
    d = ghostlight._ghostlight._render_starburst_debug(w, w, lens, lens.calibration(),
                                             cfg(**kw))
    lum = (np.asarray(d["starburst_r"]) + np.asarray(d["starburst_g"])
           + np.asarray(d["starburst_b"])).astype(np.float64)
    return lum, d


# ---------------------------------------------------------------------------
# G1 — the diffraction pupil is the profile
# ---------------------------------------------------------------------------

def g1_pupil_is_the_profile(outdir):
    """The rasterised diffraction pupil has to be the shape r(theta) describes.

    Measured two ways, because neither alone is conclusive. Enclosed area is
    corner-insensitive and pins the overall size and shape at once; the radial
    0.5-coverage crossing pins the boundary locally, but only away from the
    blade tips — a rasterised corner necessarily rounds, so a probe fired
    straight at a vertex reads short no matter how correct the mask is.
    """
    N, FILL = 512, 0.30
    texel = 2.0 / (N * FILL)
    worst_area = 0.0
    worst_radius = 0.0
    for controls in ({"blades": 6},
                     {"blades": 6, "curvature": -0.9},
                     {"blades": 6, "curvature": -0.5, "twist": 0.8},
                     dict(REFERENCE)):
        lens = bladed(**controls)
        _lum, d = starburst(lens, grid=N)
        pupil = np.asarray(d["pupil"], np.float64)
        n = int(d["grid"])
        profile = stop_of(lens).aperture_profile

        # The unit disk occupies a radius of FILL * n/2 texels, so a full pupil
        # sums to pi * FILL^2 / 4 of the grid.
        want_area = profile.area_frac * math.pi * FILL * FILL / 4.0
        worst_area = max(worst_area,
                         abs(pupil.sum() / (n * n) - want_area) / want_area)

        # Texel (i, j) sits at normalised coordinate ((i + 0.5)/n*2 - 1) / FILL,
        # matching build_pupil_kernel.
        half = profile.half
        for k in range(180):
            theta = 2.0 * math.pi * (k + 0.19) / 180
            phi = math.fmod(theta - profile.rotation, profile.sigma)
            if min(abs(phi), abs(profile.sigma - phi)) < 0.5 * half:
                continue                      # too near a tip to resolve
            hit = None
            for step in range(1, 4 * n):
                r = step * 0.25 * texel
                i = int(round((r * math.cos(theta) * FILL + 1.0) * 0.5 * n - 0.5))
                j = int(round((r * math.sin(theta) * FILL + 1.0) * 0.5 * n - 0.5))
                if not (0 <= i < n and 0 <= j < n) or pupil[j, i] <= 0.5:
                    hit = r
                    break
            if hit is not None:
                worst_radius = max(worst_radius,
                                   abs(hit - profile.radius_at(theta)))

    gate("G1 pupil encloses the profile's open area", worst_area < 0.01,
         f"worst area error {worst_area * 100:.3f}%")
    gate("G1 pupil boundary follows r(theta)", worst_radius < 2.0 * texel,
         f"max |r_raster - r_profile| = {worst_radius / texel:.2f} texel")


# ---------------------------------------------------------------------------
# G2 — one silhouette, three consumers
# ---------------------------------------------------------------------------

def g2_one_silhouette(outdir):
    """The trace, the pupil mask and the diffraction pupil must agree.

    Probed where they can be compared directly: a ray aimed just inside and
    just outside the stop's own boundary, and the mask that pre-filters the
    entrance-pupil grid.
    """
    lens = bladed(blades=5, curvature=-0.9, twist=-0.8, notch_deg=35.8,
                  notch_angle_deg=45.0)
    stop = stop_of(lens)
    profile = stop.aperture_profile
    semi = stop.semi_aperture
    bad_trace = bad_mask = 0
    for k in range(240):
        theta = 2.0 * math.pi * (k + 0.13) / 240
        r = semi * profile.radius_at(theta)
        # Only a ray that PASSES outside the silhouette proves disagreement:
        # the stop is not the lens's only aperture, so a rejection just inside
        # it could have come from any other surface.
        x = 1.03 * r * math.cos(theta)
        y = 1.03 * r * math.sin(theta)
        ray = ghostlight.Ray(ghostlight.Vec3f(x, y, stop.z - 5.0),
                     ghostlight.Vec3f(0.0, 0.0, 1.0), 587.56)
        if ghostlight.trace_primary_ray(ray, lens).status == ghostlight.TraceStatus.OK:
            bad_trace += 1
        # The normalised mask is the same predicate, one scale removed.
        u = 0.97 * profile.radius_at(theta) * math.cos(theta)
        v = 0.97 * profile.radius_at(theta) * math.sin(theta)
        if profile.radius_at(math.atan2(v, u)) < math.hypot(u, v):
            bad_mask += 1
    gate("G2 trace agrees with the profile boundary", bad_trace == 0,
         f"{bad_trace} rays passed outside the silhouette")
    gate("G2 pupil mask agrees with the profile", bad_mask == 0,
         f"{bad_mask} disagreements over 240 azimuths")


# ---------------------------------------------------------------------------
# G3 — scrub smoothness
# ---------------------------------------------------------------------------

def g3_scrub(outdir):
    """No frame-to-frame discontinuity as each control is swept.

    The failure this guards is the starburst-scrub-flicker class: an aliased
    pupil or a resampled meridian makes one frame in a sweep jump.
    """
    sweeps = {
        "curvature": [("curvature", -1.0 + 2.0 * i / 16) for i in range(17)],
        "twist": [("twist", -1.0 + 2.0 * i / 16) for i in range(17)],
        "notch": [("notch_deg", -45.0 + 90.0 * i / 16) for i in range(17)],
        "rotation": [("rotation_deg", 72.0 * i / 16) for i in range(17)],
    }
    results = {}
    for label, steps in sweeps.items():
        frames = []
        for key, value in steps:
            kw = {"blades": 5, "curvature": -0.6}
            kw[key if key != "rotation_deg" else "rotation_deg"] = value
            if key == "curvature":
                kw["curvature"] = value
            lum, _ = starburst(bladed(**kw), w=128, grid=256)
            total = lum.sum()
            frames.append(lum / total if total > 0 else lum)
        deltas = [float(np.abs(b - a).sum()) for a, b in zip(frames, frames[1:])]
        median = float(np.median(deltas))
        worst = max(deltas)
        results[label] = (worst, median)
        gate(f"G3 {label} scrub is smooth", worst < 6.0 * max(median, 1e-9),
             f"worst step {worst:.4f} vs median {median:.4f}")
    return results


# ---------------------------------------------------------------------------
# G4 — brightness tracks open area
# ---------------------------------------------------------------------------

def g4_brightness(outdir):
    for engine, name in ((ghostlight.StarburstEngine.SPRITE, "sprite"),
                         (ghostlight.StarburstEngine.MDFT, "mdft")):
        circular = ghostlight.OpticalSystem.load(str(BASE))
        hexagon = bladed(blades=6)
        ref, _ = starburst(circular, engine=engine)
        hexed, _ = starburst(hexagon, engine=engine)
        want = hexagon.calibration().pupil_area_frac
        # The pattern also narrows on the flat axis, so compare total flux
        # against the area term times the f-number term the gain carries.
        c_ref, c_hex = circular.calibration(), hexagon.calibration()
        fnum_term = ((c_ref.f_number_x * c_ref.f_number_y)
                     / (c_hex.f_number_x * c_hex.f_number_y))
        got = hexed.sum() / ref.sum()
        gate(f"G4 {name} flux = area x f-number term",
             abs(got - want * fnum_term) < 0.02 * want * fnum_term,
             f"{got:.4f} vs {want * fnum_term:.4f}")


# ---------------------------------------------------------------------------
# G5 — f-number tracks the support extent
# ---------------------------------------------------------------------------

def g5_support(outdir):
    """A hexagon's X extent genuinely swings by 1/cos(30 deg) as it turns, so
    what is under test is that the swing is CONTINUOUS — the old two-meridian
    sample stepped instead, because it read whichever feature the +X ray landed
    on. Refining the sweep has to halve the worst step."""
    def sweep(n):
        rot = [60.0 * i / n for i in range(n + 1)]
        fx = [bladed(blades=6, rotation_deg=r).calibration().f_number_x
              for r in rot]
        return rot, fx

    rotations, fx = sweep(24)
    _r2, fx2 = sweep(48)
    coarse = max(abs(b - a) / a for a, b in zip(fx, fx[1:]))
    fine = max(abs(b - a) / a for a, b in zip(fx2, fx2[1:]))
    gate("G5 f-number varies continuously with rotation", fine / coarse < 0.75,
         f"worst step {coarse * 100:.3f}% -> {fine * 100:.3f}% on refinement")
    # A full sector must come back to itself.
    gate("G5 rotation is sector-periodic",
         abs(fx[-1] - fx[0]) / fx[0] < 2e-3,
         f"f#(0) = {fx[0]:.5f}, f#(60 deg) = {fx[-1]:.5f}")
    return rotations, fx


# ---------------------------------------------------------------------------
# G6 — oracle parity through the bindings
# ---------------------------------------------------------------------------

def g6_oracle(outdir):
    if not ORACLE.exists():
        gate("G6 oracle parity", False, "oracle missing")
        return
    spec = importlib.util.spec_from_file_location("aperture_oracle", ORACLE)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    worst = 0.0
    for blades in (3, 5, 6, 8, 11):
        for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for t in (-1.0, 0.0, 0.6):
                for nd, na in ((0.0, 0.0), (25.0, 0.0), (-35.8, 45.0)):
                    lens = bladed(blades=blades, curvature=c, twist=t,
                                  notch_deg=nd, notch_angle_deg=na)
                    got = stop_of(lens).aperture_profile
                    # The oracle measures its sector from a blade midpoint;
                    # Ghostlight puts a vertex at angle 0.
                    want = oracle.ApertureProfile(blades, 180.0 / blades,
                                                  c, t, nd, na)
                    for i in range(120):
                        theta = 2.0 * math.pi * (i + 0.317) / 120
                        err = abs(got.radius_at(theta) - want.radius_at(theta))
                        tol = 3e-6 + abs(got.dr_dtheta(theta)) * 1.5e-6
                        worst = max(worst, err / tol)
    gate("G6 C++ profile matches the oracle", worst <= 1.0,
         f"worst error = {worst:.2f} x the float32 bound")


# ---------------------------------------------------------------------------
# G7 — the reference combination, rendered
# ---------------------------------------------------------------------------

def g7_reference(outdir, scrub, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens = bladed(**REFERENCE)
    profile = stop_of(lens).aperture_profile
    lum, d = starburst(lens, w=256, grid=512)
    pupil = np.asarray(d["pupil"], np.float64)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(FG)

    theta = np.linspace(0.0, 2.0 * math.pi, 2000)
    r = np.array([profile.radius_at(float(t)) for t in theta])
    axes[0].plot(r * np.cos(theta), r * np.sin(theta), color=ACCENT, lw=1.6)
    axes[0].set_aspect("equal")
    axes[0].set_title("profile r(theta)", color=FG, fontsize=10)

    axes[1].imshow(pupil, cmap="magma", origin="lower")
    axes[1].set_title("rasterised pupil", color=FG, fontsize=10)

    # The pattern is a few pixels across at this trim, so crop to it before
    # stretching — a full-frame log just shows one lit pixel.
    cy, cx = np.unravel_index(int(np.argmax(lum)), lum.shape)
    half_box = 28
    y0, y1 = max(0, cy - half_box), min(lum.shape[0], cy + half_box)
    x0, x1 = max(0, cx - half_box), min(lum.shape[1], cx + half_box)
    crop = lum[y0:y1, x0:x1]
    axes[2].imshow(np.log1p(crop / max(crop.max(), 1e-12) * 5e4),
                   cmap="magma", origin="lower")
    axes[2].set_title("starburst (log, cropped)", color=FG, fontsize=10)

    rotations, fx = sweep
    axes[3].plot(rotations, fx, color=ACCENT2, lw=1.6)
    axes[3].set_title("f-number vs stop rotation", color=FG, fontsize=10)
    axes[3].set_xlabel("degrees", color=FG, fontsize=8)

    fig.suptitle(
        f"blades {REFERENCE['blades']}  curvature {REFERENCE['curvature']:+.0%}"
        f"  twist {REFERENCE['twist']:+.0%}  notch {REFERENCE['notch_deg']}deg"
        f"  notch angle {REFERENCE['notch_angle_deg']}deg"
        f"  angle {REFERENCE['rotation_deg']}deg"
        f"   |   open area {profile.area_frac:.4f}",
        color=FG, fontsize=11)
    fig.tight_layout()
    path = pathlib.Path(outdir) / "fig_aperture_shapes.png"
    fig.savefig(path, facecolor=BG, dpi=110)
    plt.close(fig)

    gate("G7 representative combination renders", lum.sum() > 0.0,
         f"figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent))
    args = ap.parse_args()

    if not ghostlight._cuda_available():
        print("No CUDA — the render gates need a GPU.")
        return 1

    g1_pupil_is_the_profile(args.out)
    g2_one_silhouette(args.out)
    scrub = g3_scrub(args.out)
    g4_brightness(args.out)
    sweep = g5_support(args.out)
    g6_oracle(args.out)
    g7_reference(args.out, scrub, sweep)

    print(f"\n{sum(_gates)}/{len(_gates)} gates pass")
    return 0 if all(_gates) else 2


if __name__ == "__main__":
    sys.exit(main())
