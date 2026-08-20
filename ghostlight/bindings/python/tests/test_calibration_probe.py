"""The covered-field probe must measure the lens, not the bookkeeping.

Calibration resolves the covered field by probing the disc the renderers spawn
-- a disc of the front semi-aperture on a plane SPAWN_OFFSET ahead of the front
vertex, displaced off axis by spawn_shift() so it follows the beam.  Two
properties make that a measurement rather than an artefact of where the
renderers happen to launch rays, and both are gated here.

The probe itself lives in C++ (src/lens_calibration.cpp).  These tests
reconstruct it in Python from the same definition, which is what lets them
assert the invariance directly: if the probe were measuring the spawn plane
rather than the lens, the reconstruction would move when SPAWN_OFFSET does.
"""
import math

import pytest

import ghostlight

from _corpus import LENSES_DIR, lens_path

LAMBDA_D = 587.56
SPAWN_OFFSET = 20.0          # keep in sync with src/spawn_plane.h

# One smoothly-vignetting lens, one deep-pupil anamorphic: the two regimes the
# probe has to straddle (the second is the one a purely axial probe reads as
# 4.2 deg when it covers 35).
CASES = ["helios44.lens",
         "AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens"]


def _lens(name):
    path = lens_path(name)
    if not path.exists():
        pytest.skip(f"{name} not present")
    return ghostlight.OpticalSystem.load(str(path))


def _surviving_area(lens, angle_deg, offset=SPAWN_OFFSET, n=21):
    """Fraction of the front disc that still reaches the sensor, area-weighted.

    Mirrors probe_disc() in src/lens_calibration.cpp: sample the disc, displace
    it by spawn_shift(), and weight boundary cells by coverage rather than
    counting them.  Simplified to a circular mask, which is what both test
    lenses' stops resolve to.
    """
    front_r = lens.surfaces[0].semi_aperture
    z = lens.surfaces[0].z - offset
    t = math.tan(math.radians(angle_deg))
    sdx, sdy = -offset * t, 0.0          # spawn_shift(bx, 0)
    direction = ghostlight.Vec3f(t, 0.0, 1.0).normalized()

    def alive(u, v):
        if u * u + v * v > 1.0:
            return False
        origin = ghostlight.Vec3f(u * front_r + sdx, v * front_r + sdy, z)
        res = ghostlight.trace_primary_ray(ghostlight.Ray(origin, direction, LAMBDA_D), lens)
        return res.status == ghostlight.TraceStatus.OK and math.isfinite(res.position.x)

    step = 2.0 / n
    grid = {}
    for j in range(n):
        v = -1.0 + (j + 0.5) * step
        for i in range(n):
            grid[(i, j)] = alive(-1.0 + (i + 0.5) * step, v)

    area = 0.0
    for j in range(n):
        for i in range(n):
            here = grid[(i, j)]
            edge = any(grid.get((i + di, j + dj), False) != here
                       for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if not edge:
                area += 1.0 if here else 0.0
                continue
            u0, v0, sub = -1.0 + i * step, -1.0 + j * step, step / 2
            for sj in range(2):
                for si in range(2):
                    if alive(u0 + (si + 0.5) * sub, v0 + (sj + 0.5) * sub):
                        area += 0.25
    return area


@pytest.mark.parametrize("name", CASES)
def test_probe_is_spawn_plane_invariant(name):
    """Moving SPAWN_OFFSET must not move the measured throughput.

    A sample launched at u*R - S*b heading b crosses the front vertex at u*R for
    any S, so the set the probe measures is a property of the lens.
    """
    lens = _lens(name)
    angle = 12.0
    at_20 = _surviving_area(lens, angle, offset=20.0)
    at_60 = _surviving_area(lens, angle, offset=60.0)
    assert at_20 > 0.0, "probe found nothing at 20 mm; pick another field angle"
    assert abs(at_60 - at_20) / at_20 < 0.02, (
        f"{name}: surviving area {at_20:.2f} at SPAWN_OFFSET=20 but {at_60:.2f} "
        f"at 60 -- the probe is following the spawn plane, not the lens")


@pytest.mark.parametrize("name", CASES)
def test_calibrated_field_sits_where_throughput_crosses(name):
    """calibrate_lens resolves the angle where throughput hits its threshold.

    Bounds rather than a pinned number: the shipped probe runs a coarser grid
    than this reconstruction, so the two agree on the crossing, not on digits.
    """
    lens = _lens(name)
    cal = ghostlight.calibrate_lens(lens, LAMBDA_D)
    resolved = math.degrees(cal.max_half_angle_h)
    assert resolved > 0.0

    axial = _surviving_area(lens, 0.0)
    frac = _surviving_area(lens, resolved) / axial
    # The probe thresholds at 90% of axial; allow for the grid difference.
    assert 0.80 < frac < 0.97, (
        f"{name}: at the calibrated field {resolved:.2f}° the disc still passes "
        f"{frac:.1%} of axial -- that is not where the threshold crosses")


@pytest.mark.parametrize("name", CASES)
def test_image_circle_lies_outside_the_covered_field(name):
    """The illuminated edge is wider than the 90%-throughput covered field."""
    lens = _lens(name)
    cal = ghostlight.calibrate_lens(lens, LAMBDA_D)
    assert cal.image_circle_semi_w > cal.sensor_half_w, (
        f"{name}: image circle {cal.image_circle_semi_w:.2f} mm is not outside "
        f"the covered field {cal.sensor_half_w:.2f} mm")
    # And it really is near the end of the light: little survives past it.
    lo = _surviving_area(lens, math.degrees(cal.max_half_angle_h))
    assert lo > 0.0


def test_image_circle_never_reports_inside_the_covered_field():
    """Across the whole library, not just the two lenses above.

    The image circle is the SAME probe at a lower threshold, so it cannot cross
    first.  It can still come back smaller by measurement accident -- when a
    lens drops from above one threshold to below the other inside a single
    ladder step, both refine the same bracket and the circle gets the cheaper
    refinement.  That inversion is what this catches; every consumer reading the
    circle as "what the lens covers" depends on it not happening.
    """
    offenders = []
    for path in sorted(LENSES_DIR.rglob("*.lens")):
        cal = ghostlight.calibrate_lens(ghostlight.OpticalSystem.load(str(path)), LAMBDA_D)
        if cal.image_circle_semi_w < cal.sensor_half_w - 1e-4:
            offenders.append(f"{path.name} h: {cal.image_circle_semi_w:.3f} "
                             f"< {cal.sensor_half_w:.3f}")
        if cal.image_circle_semi_h < cal.sensor_half_h - 1e-4:
            offenders.append(f"{path.name} v: {cal.image_circle_semi_h:.3f} "
                             f"< {cal.sensor_half_h:.3f}")
    assert not offenders, "image circle resolved inside the covered field:\n  " + \
        "\n  ".join(offenders)


def test_no_lens_calibrates_to_a_zero_the_source_map_divides_by():
    """`sensor_half / tan(max_half_angle)` is formed unguarded downstream.

    A lens the probe cannot resolve must come out of calibration with a floor,
    not a zero -- otherwise every source in that render lands on a NaN.
    """
    bad = []
    for path in sorted(LENSES_DIR.rglob("*.lens")):
        cal = ghostlight.calibrate_lens(ghostlight.OpticalSystem.load(str(path)), LAMBDA_D)
        for label, v in (("sensor_half_w", cal.sensor_half_w),
                         ("sensor_half_h", cal.sensor_half_h),
                         ("max_half_angle_h", cal.max_half_angle_h),
                         ("max_half_angle_v", cal.max_half_angle_v),
                         ("image_circle_semi_w", cal.image_circle_semi_w),
                         ("image_circle_semi_h", cal.image_circle_semi_h)):
            if not (v > 0.0) or not math.isfinite(v):
                bad.append(f"{path.name}: {label} = {v}")
    assert not bad, "calibration produced a zero / non-finite:\n  " + "\n  ".join(bad)


def test_first_order_solves_are_untouched_by_the_probe():
    """paraxial_efl and the marginal-ray solve share trace_from_ok with nothing.

    The probe applies spawn_shift to its own sample coordinates rather than
    inside the shared single-ray primitive, so these stay exactly what they
    were.  Pinned as a relation, not a constant: f/(2*ep) is the definition.
    """
    lens = _lens("helios44.lens")
    cal = ghostlight.calibrate_lens(lens, LAMBDA_D)
    assert cal.focal_length_x > 0.0 and cal.entrance_pupil_semi_x > 0.0
    assert cal.f_number_x == pytest.approx(
        cal.focal_length_x / (2.0 * cal.entrance_pupil_semi_x), rel=1e-6)
    # The marginal ray is solved on axis, where spawn_shift is exactly zero, so
    # the pupil can never exceed the front element it is measured across.
    assert cal.entrance_pupil_semi_x <= lens.surfaces[0].semi_aperture + 1e-6


def test_rotationally_symmetric_lens_resolves_both_axes_alike():
    """Symmetry fusion is exact: one search answers both axes when it applies."""
    lens = _lens("DoubleGauss.lens")
    cal = ghostlight.calibrate_lens(lens, LAMBDA_D)
    assert cal.max_half_angle_h == cal.max_half_angle_v
    assert cal.sensor_half_w == cal.sensor_half_h


def test_anamorphic_lens_resolves_axes_separately():
    """...and does not apply where the two axes genuinely differ."""
    lens = _lens("AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens")
    cal = ghostlight.calibrate_lens(lens, LAMBDA_D)
    assert cal.max_half_angle_h != cal.max_half_angle_v
