"""Covered-field measurement harness.

The covered field is measured by probing the disc the RENDERERS spawn -- a disc
of surfaces[0].semi_aperture on the plane SPAWN_OFFSET ahead of the front
vertex, displaced off axis by spawn_shift() so it follows the beam --
and finding the field angle at which the surviving AREA of that disc falls to a
chosen fraction of its axial value.

Everything here mirrors the C++ probe it exists to validate:

  * the sample set is the canonical one -- default ApertureSamplerParams, so the
    mask is the lens's own stop (anamorphic aspect + polygon blades), with no
    blades_override and no baffles.  calibrate_lens() takes no FlareConfig, so
    that is the only config-free choice available to it.
  * survivors are AREA-weighted, not counted.  Cells on the survival boundary
    are subsampled for fractional coverage, because a count-quantised field
    strobes under parameter scrubs -- the same defect the starburst survivor
    envelope had before SURVIVOR_EDGE_SS.
  * sensor_half is the area-weighted mean |landing| of the survivors at the
    resolved angle, which is where the flare actually appears.

Modes (all write a markdown table to stdout):

    python probe_covered_field.py curve      # throughput vs field angle
    python probe_covered_field.py field      # resolved covered field per threshold
    python probe_covered_field.py threshold  # frame-edge error per threshold
    python probe_covered_field.py grid       # grid-convergence sweep
    python probe_covered_field.py sweep      # resolved field for every lens

Common options:  --lenses a,b,c   --n 41   --thresholds 0.95,0.9,0.8,0.7,0.5
                 --axis x|y|both  --frame-half 12.445
"""
import argparse
import math
import sys
import time

from _paths import LENSES, lens_file  # noqa: E402  (also puts the package on sys.path)
import ghostlight  # noqa: E402


LAMBDA_D = 587.56
SPAWN_OFFSET = 20.0          # keep in sync with src/spawn_plane.h

# Coarse ladder for the resolve search.  Wider and finer than the C++ probe's
# because it has to bracket both a helios44 (~4 deg at 90%) and an Atlas (~32
# deg) without assuming which kind of lens it is looking at.
COARSE_DEG = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0,
              20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
TOL_DEG = 0.01
EDGE_SS = 4                  # boundary-cell subsampling, EDGE_SS^2 per cell

# Representative lenses used by the covered-field validation.
DEFAULT_LENSES = [
    "helios44.lens",
    "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens",
    "00081_zeiss_100mm.lens",
    "DoubleGauss.lens",
    # Keep the focused prescription used by the point-flare render baseline.
    "Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens",
    "Leitz_Summicron_100mm.lens",
]

SHORT = {
    "helios44.lens": "helios44",
    "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens": "Atlas 1.5x ANA",
    "00081_zeiss_100mm.lens": "zeiss 100mm",
    "DoubleGauss.lens": "double_gauss",
    "Sirui_75mm_F1.8_1.33x_EP3936919NWA1.lens": "sirui75 ANA",
    "Leitz_Summicron_100mm.lens": "summicron 100",
    "00424_petzval_100mm.lens": "petzval 100",
}


# ---------------------------------------------------------------------------
# The canonical pupil mask: resolve_pupil_mask() + PupilMask::contains() from
# src/aperture_sampler.h, with default ApertureSamplerParams (no override).
# ---------------------------------------------------------------------------
class PupilMask:
    def __init__(self, lens):
        aspect, n_blades, rot_rad = 1.0, 0, 0.0
        for s in lens.surfaces:
            if s.is_stop:
                aspect = s.aperture_aspect
                # aperture_shape comes back from pybind11 as an int.
                if int(s.aperture_shape) == int(ghostlight.ApertureShape.POLYGON) \
                        and s.aperture_blades >= 3:
                    n_blades = s.aperture_blades
                    rot_rad = s.aperture_rotation_rad
                break
        self.aspect = aspect
        self.n_blades = n_blades
        self.rot_rad = rot_rad
        poly = n_blades >= 3
        self.apothem = math.cos(math.pi / n_blades) if poly else 1.0
        self.sector = 2.0 * math.pi / n_blades if poly else 1.0

    def contains(self, u, v):
        ux = u / self.aspect
        r2 = ux * ux + v * v
        if r2 > 1.0:
            return False
        if self.n_blades >= 3:
            angle = math.atan2(v, ux) - self.rot_rad
            sector = math.fmod(angle, self.sector)
            if sector < 0.0:
                sector += self.sector
            if math.sqrt(r2) * math.cos(sector - self.sector * 0.5) > self.apothem:
                return False
        return True


class Probe:
    """Area-weighted survivor probe over the renderers' shifted spawn disc."""

    def __init__(self, lens, n=41, spawn_offset=SPAWN_OFFSET, lam=LAMBDA_D,
                 edge_ss=EDGE_SS, ladder=None, tol_deg=TOL_DEG, max_refine=24):
        self.lens = lens
        self.n = n
        self.S = spawn_offset
        self.lam = lam
        self.edge_ss = edge_ss
        self.ladder = list(ladder) if ladder is not None else list(COARSE_DEG)
        self.tol_deg = tol_deg
        self.max_refine = max_refine
        self.mask = PupilMask(lens)
        self.front_R = lens.surfaces[0].semi_aperture
        self.z = lens.surfaces[0].z - spawn_offset
        self.traces = 0

    def _alive(self, u, v, direction, sdx, sdy):
        """Trace one pupil sample; return its sensor landing or None."""
        self.traces += 1
        origin = ghostlight.Vec3f(u * self.front_R + sdx, v * self.front_R + sdy, self.z)
        res = ghostlight.trace_primary_ray(ghostlight.Ray(origin, direction, self.lam), self.lens)
        if res.status != ghostlight.TraceStatus.OK:
            return None
        if not (math.isfinite(res.position.x) and math.isfinite(res.position.y)):
            return None
        return res.position

    def survivors(self, angle_h, angle_v):
        """(area, mean_x, mean_y) of the surviving disc at this field angle.

        `area` is in normalised pupil-cell units: a cell fully inside the mask
        whose ray survives contributes 1, a boundary cell contributes its
        subsampled coverage fraction.  Means are area-weighted.
        """
        n = self.n
        bx, by = math.tan(angle_h), math.tan(angle_v)
        # spawn_shift(bx, by): the front aperture's back-projection.
        sdx, sdy = -self.S * bx, -self.S * by
        direction = ghostlight.Vec3f(bx, by, 1.0).normalized()

        step = 2.0 / n
        # Pass 1: cell centres.
        live = {}
        for j in range(n):
            v = -1.0 + (j + 0.5) * step
            for i in range(n):
                u = -1.0 + (i + 0.5) * step
                if not self.mask.contains(u, v):
                    live[(i, j)] = None
                    continue
                live[(i, j)] = self._alive(u, v, direction, sdx, sdy)

        # Pass 2: any cell whose 4-neighbourhood disagrees about survival sits
        # on a boundary (mask edge or vignetting edge) and gets subsampled.
        area = 0.0
        sum_x = sum_y = 0.0
        for j in range(n):
            for i in range(n):
                here = live[(i, j)] is not None
                edge = False
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = live.get((i + di, j + dj))
                    if (i + di) < 0 or (i + di) >= n or (j + dj) < 0 or (j + dj) >= n:
                        # Off-grid neighbours count as dead: a live cell at the
                        # grid rim is a boundary cell (and a warning sign that
                        # the disc is clipping the grid).
                        if here:
                            edge = True
                        continue
                    if (nb is not None) != here:
                        edge = True
                        break
                if not edge:
                    if here:
                        p = live[(i, j)]
                        area += 1.0
                        sum_x += p.x
                        sum_y += p.y
                    continue

                # Subsample this cell.
                ss = self.edge_ss
                if ss <= 1:
                    if here:
                        p = live[(i, j)]
                        area += 1.0
                        sum_x += p.x
                        sum_y += p.y
                    continue
                u0 = -1.0 + i * step
                v0 = -1.0 + j * step
                sub = step / ss
                w = 1.0 / (ss * ss)
                for sj in range(ss):
                    v = v0 + (sj + 0.5) * sub
                    for si in range(ss):
                        u = u0 + (si + 0.5) * sub
                        if not self.mask.contains(u, v):
                            continue
                        p = self._alive(u, v, direction, sdx, sdy)
                        if p is None:
                            continue
                        area += w
                        sum_x += w * p.x
                        sum_y += w * p.y

        if area <= 0.0:
            return 0.0, float("nan"), float("nan")
        return area, sum_x / area, sum_y / area

    def at(self, angle_rad, axis):
        """survivors() along one axis, the other held at zero."""
        return self.survivors(angle_rad, 0.0) if axis == "x" else self.survivors(0.0, angle_rad)


def throughput_curve(probe, axis, angles_deg):
    """[(angle_deg, fraction_of_axial, mean_landing)] along one axis."""
    a0, _, _ = probe.at(0.0, axis)
    out = []
    for a in angles_deg:
        area, mx, my = probe.at(math.radians(a), axis)
        frac = (area / a0) if a0 > 0 else 0.0
        out.append((a, frac, mx if axis == "x" else my))
    return out


def resolve_field(probe, axis, threshold):
    """Resolve the covered field at `threshold` of axial surviving area.

    Returns (angle_rad, sensor_half_mm, status).  status is 'ok', 'degenerate'
    (nothing survives on axis) or 'no-crossing' (throughput never falls to the
    threshold inside the ladder -- the resolved angle is the ceiling and the
    number is a floor, not a measurement).
    """
    # ladder[0] is 0 deg, so the axial evaluation IS the first ladder step.
    ladder = probe.ladder
    a0, _, _ = probe.at(math.radians(ladder[0]), axis)
    if a0 <= 0.0:
        return 0.0, 0.0, "degenerate"

    # Walk out until the surviving fraction first falls below the threshold.
    # Area weighting makes the curve monotone in practice (it is the count-
    # quantised version that reads 104 % of axial and needs a last-crossing
    # rule), so the first crossing is the vignetting edge and the walk can stop
    # there instead of paying for the whole ladder.
    cross = None
    fracs_at_cross = None
    prev_frac = 1.0
    for i in range(1, len(ladder)):
        area, _, _ = probe.at(math.radians(ladder[i]), axis)
        frac = area / a0
        if frac < threshold:
            cross = i - 1
            fracs_at_cross = (prev_frac, frac)
            break
        prev_frac = frac
    if cross is None:
        ceiling = math.radians(ladder[-1])
        _, mx, my = probe.at(ceiling, axis)
        half = abs(mx if axis == "x" else my)
        return ceiling, half, "no-crossing"

    # Refine with the Illinois variant of regula falsi rather than bisection.
    # Each step costs a whole grid evaluation, and the area curve is smooth and
    # monotone once it is area-weighted, so interpolating the crossing converges
    # in a third of the evaluations bisection needs -- which is most of what
    # keeps this inside the calibration budget.
    lo = math.radians(ladder[cross])
    hi = math.radians(ladder[cross + 1])
    f = lambda a: (probe.at(a, axis)[0] / a0) - threshold
    f_lo, f_hi = fracs_at_cross[0] - threshold, fracs_at_cross[1] - threshold
    tol = math.radians(probe.tol_deg)

    best = lo
    for _ in range(probe.max_refine):
        if hi - lo <= tol:
            break
        denom = f_lo - f_hi
        if abs(denom) < 1e-12:
            mid = 0.5 * (lo + hi)
        else:
            mid = lo + (hi - lo) * f_lo / denom
            # Keep the step strictly inside the bracket.
            edge = 0.01 * (hi - lo)
            mid = min(max(mid, lo + edge), hi - edge)
        f_mid = f(mid)
        if f_mid >= 0.0:
            lo, f_lo = mid, f_mid
            f_hi *= 0.5              # Illinois: unstick the stagnant end
        else:
            hi, f_hi = mid, f_mid
            f_lo *= 0.5
        best = lo

    _, mx, my = probe.at(best, axis)
    half = abs(mx if axis == "x" else my)
    return best, half, "ok"


def frame_edge_error(probe, axis, threshold, frame_half):
    """Error of the source map at the frame edge, in % of the frame half-width.

    The renderer maps a source at ndc=1 to
        angle = atan(1 * scale * tan(covered)),  scale = frame_half / sensor_half
    i.e. atan(frame_half / f_eff) with f_eff = sensor_half / tan(covered).  The
    light should then land at frame_half.  This measures where it does land --
    the area-weighted survivor mean, which is where the flare appears.
    """
    ang, half, status = resolve_field(probe, axis, threshold)
    if status == "degenerate" or half <= 0.0 or ang <= 0.0:
        return float("nan"), status
    f_eff = half / math.tan(ang)
    probe_ang = math.atan(frame_half / f_eff)
    _, mx, my = probe.at(probe_ang, axis)
    landing = mx if axis == "x" else my
    if not math.isfinite(landing):
        return float("nan"), "no-landing"
    return (abs(landing) - frame_half) / frame_half * 100.0, status


def today_error(lens, calib, axis, frame_half, probe):
    """The same frame-edge error, against the map the renderers SHIP.

    This calls the shipped map (`_solve_source_map`) rather than reimplementing
    it. A harness that models the thing it is checking can only check the model.

    Note the asymmetry with frame_edge_error() above, which deliberately keeps
    its own arithmetic: that one is the INDEPENDENT reference the shipped map is
    measured against, so it must not call the binding.  This one is the
    measurement OF the shipped map, so it must.
    """
    ax = ay = 0.0
    if axis == "x":
        r = ghostlight._ghostlight._solve_source_map(lens, calib, 1.0, 0.5,
                                           frame_half, frame_half)
        ax, ay = r["angle_x"], r["angle_y"]
    else:
        r = ghostlight._ghostlight._solve_source_map(lens, calib, 0.5, 1.0,
                                           frame_half, frame_half)
        ax, ay = r["angle_x"], r["angle_y"]
    if not (math.isfinite(ax) and math.isfinite(ay)):
        return float("nan")
    _, mx, my = probe.at(ax if axis == "x" else ay, axis)
    landing = mx if axis == "x" else my
    if not math.isfinite(landing):
        return float("nan")
    return (abs(landing) - frame_half) / frame_half * 100.0


# ---------------------------------------------------------------------------
def load(name):
    p = lens_file(name)
    if not p.exists():
        raise SystemExit(f"lens not found: {p}")
    return ghostlight.OpticalSystem.load(str(p))


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode",
                    choices=["curve", "field", "threshold", "grid", "sweep", "budget"])
    ap.add_argument("--lenses", default=None,
                    help="comma-separated lens filenames (default: representative set)")
    ap.add_argument("--n", type=int, default=41, help="probe grid resolution")
    ap.add_argument("--thresholds", default="0.95,0.90,0.80,0.70,0.50")
    ap.add_argument("--axis", default="x", choices=["x", "y", "both"])
    ap.add_argument("--frame-half", type=float, default=12.445,
                    help="frame half-width in mm for the frame-edge error")
    ap.add_argument("--angles", default=None,
                    help="comma-separated field angles in deg for `curve`")
    ap.add_argument("--grids", default="21,41,81,121",
                    help="grid sizes for `grid`")
    ap.add_argument("--spawn-offset", type=float, default=SPAWN_OFFSET,
                    help="override SPAWN_OFFSET to test spawn-plane invariance")
    ap.add_argument("--edge-ss", type=int, default=EDGE_SS,
                    help="boundary subsampling (edge_ss^2 per boundary cell; 1 = off)")
    ap.add_argument("--ladder", default=None,
                    help="comma-separated coarse ladder in deg (for `budget`)")
    ap.add_argument("--configs", default=None,
                    help="budget mode: n:edge_ss pairs, e.g. 9:2,13:2,17:2,21:3")
    ap.add_argument("--tol", type=float, default=TOL_DEG,
                    help="refinement tolerance in deg")
    ap.add_argument("--max-refine", type=int, default=24,
                    help="cap on refinement steps (each step is one grid evaluation)")
    return ap.parse_args(argv)


def lens_list(args):
    if args.lenses:
        return [s.strip() for s in args.lenses.split(",") if s.strip()]
    return list(DEFAULT_LENSES)


def axes(args):
    return ["x", "y"] if args.axis == "both" else [args.axis]


def main(argv=None):
    args = parse_args(argv)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    names = lens_list(args)
    t_start = time.perf_counter()

    # Helios44 is framed at 10 mm; the rest use Super-35. --frame-half overrides this.
    frame_of = {"helios44.lens": 10.00}

    if args.mode == "curve":
        angles = [float(a) for a in (args.angles or "0,5,10,15,20,25,28,30,35,40,45").split(",")]
        print(f"\nSurviving fraction of the corrected disc (n={args.n}, "
              f"SPAWN_OFFSET={args.spawn_offset})\n")
        head = "| lens | axis | " + " | ".join(f"{a:g}°" for a in angles) + " |"
        print(head)
        print("|---" * (len(angles) + 2) + "|")
        for name in names:
            lens = load(name)
            for ax in axes(args):
                p = Probe(lens, n=args.n, spawn_offset=args.spawn_offset)
                row = throughput_curve(p, ax, angles)
                cells = " | ".join(f"{f * 100:.1f}%" for _, f, _ in row)
                print(f"| {SHORT.get(name, name)} | {ax} | {cells} |")

    elif args.mode == "field":
        print(f"\nResolved covered field and sensor_half (n={args.n}, "
              f"SPAWN_OFFSET={args.spawn_offset})\n")
        print("| lens | axis | today | " +
              " | ".join(f"{t * 100:g} %" for t in thresholds) + " |")
        print("|---" * (len(thresholds) + 3) + "|")
        for name in names:
            lens = load(name)
            calib = ghostlight.calibrate_lens(lens, LAMBDA_D)
            for ax in axes(args):
                p = Probe(lens, n=args.n, spawn_offset=args.spawn_offset)
                ang0 = calib.max_half_angle_h if ax == "x" else calib.max_half_angle_v
                half0 = calib.sensor_half_w if ax == "x" else calib.sensor_half_h
                cells = []
                for t in thresholds:
                    ang, half, status = resolve_field(p, ax, t)
                    flag = "" if status == "ok" else f" [{status}]"
                    cells.append(f"{half:.2f} mm / {math.degrees(ang):.2f}°{flag}")
                print(f"| {SHORT.get(name, name)} | {ax} | "
                      f"{half0:.2f} mm / {math.degrees(ang0):.2f}° | "
                      + " | ".join(cells) + " |")

    elif args.mode == "threshold":
        print(f"\nSource at the frame edge: error against the traced landing "
              f"(n={args.n})\n")
        print("| lens | axis | frame half | today | " +
              " | ".join(f"{t * 100:g} %" for t in thresholds) + " |")
        print("|---" * (len(thresholds) + 4) + "|")
        acc = {t: [] for t in thresholds}
        acc_today = []
        for name in names:
            lens = load(name)
            calib = ghostlight.calibrate_lens(lens, LAMBDA_D)
            fh = frame_of.get(name, args.frame_half)
            for ax in axes(args):
                p = Probe(lens, n=args.n, spawn_offset=args.spawn_offset)
                e0 = today_error(lens, calib, ax, fh, p)
                acc_today.append(abs(e0))
                cells = []
                for t in thresholds:
                    e, status = frame_edge_error(p, ax, t, fh)
                    acc[t].append(abs(e))
                    cells.append(f"{e:+.2f}" if math.isfinite(e) else "n/a")
                print(f"| {SHORT.get(name, name)} | {ax} | {fh:.3f} mm | "
                      f"{e0:+.2f} % | " + " | ".join(cells) + " |")

        def mean(xs):
            xs = [x for x in xs if math.isfinite(x)]
            return sum(xs) / len(xs) if xs else float("nan")
        print(f"| **mean \\|error\\|** | | | {mean(acc_today):.2f} % | "
              + " | ".join(f"{mean(acc[t]):.2f}" for t in thresholds) + " |")

    elif args.mode == "grid":
        grids = [int(g) for g in args.grids.split(",")]
        print(f"\nGrid convergence: resolved field / frame-edge error\n")
        print("| lens | axis | threshold | " + " | ".join(f"n={g}" for g in grids) + " |")
        print("|---" * (len(grids) + 3) + "|")
        for name in names:
            lens = load(name)
            fh = frame_of.get(name, args.frame_half)
            for ax in axes(args):
                for t in thresholds:
                    cells = []
                    for g in grids:
                        p = Probe(lens, n=g, spawn_offset=args.spawn_offset)
                        ang, half, status = resolve_field(p, ax, t)
                        e, _ = frame_edge_error(p, ax, t, fh)
                        cells.append(f"{math.degrees(ang):.2f}° / {e:+.2f}%")
                    print(f"| {SHORT.get(name, name)} | {ax} | {t * 100:g} % | "
                          + " | ".join(cells) + " |")

    elif args.mode == "budget":
        # Size the shipped probe: for each (grid, edge_ss, ladder) candidate,
        # how far does the resolved angle sit from a high-resolution reference,
        # and how many traces did it cost?  0.5 us/trace is the measured C++
        # cost, which is what turns a trace count into the wall-time budget.
        US_PER_TRACE = 0.5
        thr = thresholds[0]
        ladder = ([float(a) for a in args.ladder.split(",")]
                  if args.ladder else None)
        configs = []
        for spec in (args.configs or "9:2,13:2,17:2,21:2,21:3,29:2,41:4").split(","):
            n_s, _, ss_s = spec.partition(":")
            configs.append((int(n_s), int(ss_s or EDGE_SS)))

        print(f"\nProbe sizing at {thr * 100:g} % "
              f"(reference n=121 ss=4, ladder={'custom' if ladder else 'default'})\n")
        print("| lens | axis | reference | " +
              " | ".join(f"n={n} ss={ss}" for n, ss in configs) + " |")
        print("|---" * (len(configs) + 3) + "|")

        worst = {c: 0.0 for c in configs}
        cost = {c: 0 for c in configs}
        for name in names:
            lens = load(name)
            for ax in axes(args):
                ref = Probe(lens, n=121, spawn_offset=args.spawn_offset, edge_ss=4)
                ref_ang, ref_half, _ = resolve_field(ref, ax, thr)
                cells = []
                for c in configs:
                    n_c, ss_c = c
                    p = Probe(lens, n=n_c, spawn_offset=args.spawn_offset,
                              edge_ss=ss_c, ladder=ladder, tol_deg=args.tol,
                              max_refine=args.max_refine)
                    ang, half, status = resolve_field(p, ax, thr)
                    d_deg = abs(math.degrees(ang - ref_ang))
                    d_half = (abs(half - ref_half) / ref_half * 100.0
                              if ref_half > 0 else float("nan"))
                    worst[c] = max(worst[c], d_deg)
                    cost[c] = max(cost[c], p.traces)
                    cells.append(f"{d_deg:.2f}° / {d_half:+.1f}% / {p.traces}")
                print(f"| {SHORT.get(name, name)} | {ax} | "
                      f"{math.degrees(ref_ang):.2f}° / {ref_half:.2f} mm | "
                      + " | ".join(cells) + " |")

        print("\n| config | worst |Δangle| | worst traces/axis | est. both axes |")
        print("|---|---|---|---|")
        for c in configs:
            ms = 2 * cost[c] * US_PER_TRACE / 1000.0
            verdict = "OK" if (worst[c] <= 0.2 and ms <= 2.0) else \
                      ("slow" if worst[c] <= 0.2 else "inaccurate")
            print(f"| n={c[0]} ss={c[1]} | {worst[c]:.2f}° | {cost[c]} | "
                  f"{ms:.2f} ms — {verdict} |")

    elif args.mode == "sweep":
        thr = thresholds[0]
        print(f"\nEvery lens: resolved covered field at {thr * 100:g} % "
              f"(n={args.n})\n")
        print("| lens | today h | today v | new h | new v | status |")
        print("|---|---|---|---|---|---|")
        files = names if args.lenses else sorted(p.name for p in LENSES.rglob("*.lens"))
        for name in files:
            try:
                lens = load(name)
                calib = ghostlight.calibrate_lens(lens, LAMBDA_D)
                px = Probe(lens, n=args.n, spawn_offset=args.spawn_offset)
                py = Probe(lens, n=args.n, spawn_offset=args.spawn_offset)
                ax_, hx, sx = resolve_field(px, "x", thr)
                ay_, hy, sy = resolve_field(py, "y", thr)
                st = sx if sx == sy else f"{sx}/{sy}"
                print(f"| {name} | {calib.sensor_half_w:.2f} mm / "
                      f"{math.degrees(calib.max_half_angle_h):.2f}° | "
                      f"{calib.sensor_half_h:.2f} mm / "
                      f"{math.degrees(calib.max_half_angle_v):.2f}° | "
                      f"{hx:.2f} mm / {math.degrees(ax_):.2f}° | "
                      f"{hy:.2f} mm / {math.degrees(ay_):.2f}° | {st} |")
            except Exception as exc:  # a bad lens file must not kill the sweep
                print(f"| {name} | | | | | ERROR: {exc} |")

    print(f"\n({time.perf_counter() - t_start:.1f} s)")


if __name__ == "__main__":
    main()
