"""The spawn disc must follow the beam off axis.

Renderers launch pupil rays on a disc of surfaces[0].semi_aperture across the
plane SPAWN_OFFSET mm ahead of the front vertex.  Off axis that disc has to be
displaced by -SPAWN_OFFSET*tan(field) -- it is the front aperture's
back-projection onto the spawn plane, not an axis-centred copy of it.  Leave it
centred and the sampled set slides off the element: helios44 keeps a quarter of
its beam at 20 deg, tessar none at all, and the renders read as plausible-looking
vignetting rather than as a bug.

The property tested here is not an absolute capture percentage.  A planar disc
can never catch everything -- the front element is curved, so its aperture does
not project to a circle -- and the achievable ceiling varies per lens.  What must
hold is that the shifted disc reaches that ceiling, measured directly as the
capture of a disc launched right at the front vertex.
"""
import math

import pytest

import ghostlight

from _corpus import lens_path

SPAWN_OFFSET = 20.0          # keep in sync with src/spawn_plane.h
LAMBDA_D = 587.56


def _capture(lens, angle_deg, offset=SPAWN_OFFSET, n=61, axis="x"):
    """Fraction of the beam entering the lens that a disc on the spawn plane sees.

    Returns (centred, shifted) -- the capture of an axis-centred disc and of one
    displaced by -offset*tan(field).  The denominator is every launch on the
    plane that reaches the sensor, so both are measured against the whole beam.
    """
    front_r = lens.surfaces[0].semi_aperture
    z = lens.surfaces[0].z - offset
    t = math.tan(math.radians(angle_deg))
    bx, by = (t, 0.0) if axis == "x" else (0.0, t)
    shift_x, shift_y = -offset * bx, -offset * by
    span_x = 1.25 * front_r + abs(shift_x)
    span_y = 1.25 * front_r + abs(shift_y)
    direction = ghostlight.Vec3f(bx, by, 1.0).normalized()

    total = centred = shifted = 0
    for j in range(n):
        y = span_y * (2.0 * j / (n - 1) - 1.0)
        for i in range(n):
            x = span_x * (2.0 * i / (n - 1) - 1.0)
            res = ghostlight.trace_primary_ray(ghostlight.Ray(ghostlight.Vec3f(x, y, z), direction, LAMBDA_D), lens)
            if res.status != ghostlight.TraceStatus.OK:
                continue
            if not (math.isfinite(res.position.x) and math.isfinite(res.position.y)):
                continue
            total += 1
            if x * x + y * y <= front_r * front_r:
                centred += 1
            if (x - shift_x) ** 2 + (y - shift_y) ** 2 <= front_r * front_r:
                shifted += 1
    assert total > 0, "no ray reaches the sensor at this field; pick another angle"
    return centred / total, shifted / total


def _lens(name):
    path = lens_path(name)
    if not path.exists():
        pytest.skip(f"{name} not present")
    return ghostlight.OpticalSystem.load(str(path))


# (file, angle, axis) -- one tight-slack lens, one deep-pupil anamorphic.  The
# anamorphics clip vertically first, their horizontal pupil being squeezed.
CASES = [
    ("helios44.lens", 20.0, "x"),
    ("AtlasLensCo_40mm_F3.0_1.5x_US12379576B2_Table1.lens", 20.0, "y"),
]


@pytest.mark.parametrize("name,angle,axis", CASES)
def test_shifted_disc_reaches_the_planar_limit(name, angle, axis):
    """The shifted disc catches as much as a disc at the front vertex does."""
    lens = _lens(name)
    _, shifted = _capture(lens, angle, axis=axis)
    # offset -> 0 puts the disc on the vertex plane: the best a plane can do.
    limit, _ = _capture(lens, angle, offset=0.05, axis=axis)
    assert shifted >= limit - 0.03, (
        f"{name} at {angle} deg: shifted disc captures {shifted:.1%} against a "
        f"planar ceiling of {limit:.1%} -- the disc is not tracking the beam")


@pytest.mark.parametrize("name,angle,axis", CASES)
def test_centred_disc_would_lose_the_beam(name, angle, axis):
    """The lenses above really do exercise the defect, so the test above can fail."""
    centred, shifted = _capture(lens := _lens(name), angle, axis=axis)
    assert lens is not None
    assert shifted - centred > 0.25, (
        f"{name} at {angle} deg no longer discriminates: centred {centred:.1%} vs "
        f"shifted {shifted:.1%} -- pick a harsher field or another lens")


def test_on_axis_shift_is_exactly_zero():
    """tan(0) is exactly 0, so an axial launch is bit-for-bit what it always was."""
    lens = _lens("helios44.lens")
    centred, shifted = _capture(lens, 0.0, n=41)
    assert centred == shifted


@pytest.mark.gpu
def test_psf_pupil_fraction_follows_the_beam():
    """The shipped pre-pass reports the transmission of the SHIFTED disc.

    pupil_fraction counts hits over unit-disk samples, so it reads the disc the
    renderer actually launches.  Recomputing both candidate discs here means the
    bound moves with the lens rather than being pinned to a measured number.
    """
    import numpy as np

    lens = _lens("helios44.lens")
    cfg = ghostlight.PSFConfig()
    cfg.grid_nx = cfg.grid_ny = 1
    cfg.tile_w = cfg.tile_h = 64
    cfg.tile_extent_mm = 0.2
    cfg.center_mode = ghostlight.PSFCenterMode.CHIEF_CENTROID

    angle = math.radians(20.0)
    sources = np.array([[angle, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    reported = float(np.asarray(ghostlight.render_psf(sources, lens, cfg)["pupil_fraction"])[0])

    front_r = lens.surfaces[0].semi_aperture
    z = lens.surfaces[0].z - SPAWN_OFFSET
    t = math.tan(angle)
    direction = ghostlight.Vec3f(t, 0.0, 1.0).normalized()

    def disc_transmission(shift_x, n=17):
        hits = total = 0
        for iy in range(n):
            v = -1.0 + (iy + 0.5) * (2.0 / n)
            for ix in range(n):
                u = -1.0 + (ix + 0.5) * (2.0 / n)
                if u * u + v * v > 1.0:
                    continue
                total += 1
                origin = ghostlight.Vec3f(u * front_r + shift_x, v * front_r, z)
                res = ghostlight.trace_primary_ray(ghostlight.Ray(origin, direction, LAMBDA_D), lens)
                if res.status == ghostlight.TraceStatus.OK and math.isfinite(res.position.x):
                    hits += 1
        return hits / total

    centred = disc_transmission(0.0)
    shifted = disc_transmission(-SPAWN_OFFSET * t)
    assert shifted - centred > 0.1, "probe no longer discriminates the two discs"
    # Probe grid and pre-pass disagree by a few percent at the rim; the gap
    # between the two discs is far wider than that.
    assert abs(reported - shifted) < 0.5 * (shifted - centred), (
        f"PSF pre-pass reports {reported:.3f}, shifted disc gives {shifted:.3f}, "
        f"centred gives {centred:.3f} -- the pre-pass is sampling the wrong disc")
