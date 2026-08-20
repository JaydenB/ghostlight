"""Bladed aperture profile — geometry, freshness and the plain-polygon guard.

The geometry ORACLE is ``validation/aperture_profile_oracle.py``.
``PROFILE_VECTORS`` below is frozen from it (in float64), so these tests pin the
C++ port without depending on that file being importable; ``test_matches_oracle``
additionally re-derives the whole cross-check sweep from the oracle when it is.

Sector orientation: the oracle puts a blade MIDPOINT at angle 0, Ghostlight puts a
VERTEX there (the convention the file format documents and the plain-polygon
trace already implements). Oracle profiles are therefore built with
``ORACLE_ORIENTATION_DEG / blades`` added to the rotation.

Tolerance: the profile is evaluated in float32, and the steep ramps a fully
twisted blade produces reach |dr/dtheta| ~ 30 per radian, so a 1-ulp difference
in the angle alone moves the radius by ~1e-5. Comparisons are made against a
slope-aware bound rather than a flat epsilon — see ``_tol``.
"""
import importlib.util
import math
import pathlib
from dataclasses import dataclass

import pytest

import ghostlight

ORACLE = (pathlib.Path(__file__).resolve().parents[4]
          / "validation" / "aperture_profile_oracle.py")

# Half a sector, in the "degrees per blade" form the oracle's rotation takes.
ORACLE_ORIENTATION_DEG = 180.0


@dataclass(frozen=True)
class Case:
    blades: int
    rotation_deg: float
    curvature: float
    twist: float
    notch_deg: float
    notch_angle_deg: float
    area_frac: float
    radii: tuple


PROFILE_VECTORS = (
    Case(
        blades=6, rotation_deg=0.0, curvature=0.0,
        twist=0.0, notch_deg=0.0, notch_angle_deg=0.0,
        area_frac=0.826993343,
        radii=(
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
            1.000000000, 0.896575472, 0.866025404, 0.896575472,
        ),
    ),
    Case(
        blades=5, rotation_deg=29.0, curvature=-0.9,
        twist=-0.8, notch_deg=35.8, notch_angle_deg=45.0,
        area_frac=0.323039208,
        radii=(
            0.507270417, 0.657906050, 0.727888503, 0.413567990,
            0.447774808, 0.528722434, 0.706443512, 0.462663137,
            0.417644259, 0.459377468, 0.553715173, 0.765094741,
            0.408740362, 0.423008571, 0.472973178, 0.582933199,
            0.837009005, 0.409056589, 0.429742758, 0.488826842,
            0.617269098, 0.926821038, 0.410719248, 0.437953135,
        ),
    ),
    Case(
        blades=3, rotation_deg=0.0, curvature=-1.0,
        twist=-1.0, notch_deg=0.0, notch_angle_deg=0.0,
        area_frac=0.180985389,
        radii=(
            1.000000000, 0.227249623, 0.237976920, 0.259265408,
            0.295085291, 0.353432665, 0.451099646, 0.627812153,
            1.000000000, 0.227249623, 0.237976920, 0.259265408,
            0.295085291, 0.353432665, 0.451099646, 0.627812153,
            1.000000000, 0.227249623, 0.237976920, 0.259265408,
            0.295085291, 0.353432665, 0.451099646, 0.627812153,
        ),
    ),
    Case(
        blades=8, rotation_deg=12.5, curvature=0.5,
        twist=0.0, notch_deg=25.0, notch_angle_deg=0.0,
        area_frac=0.944258649,
        radii=(
            0.969302709, 0.979304028, 0.963773491, 0.969302709,
            0.979304028, 0.963773491, 0.969302709, 0.979304028,
            0.963773491, 0.969302709, 0.979304028, 0.963773491,
            0.969302709, 0.979304028, 0.963773491, 0.969302709,
            0.979304028, 0.963773491, 0.969302709, 0.979304028,
            0.963773491, 0.969302709, 0.979304028, 0.963773491,
        ),
    ),
    Case(
        blades=5, rotation_deg=0.0, curvature=-0.6,
        twist=0.7, notch_deg=0.0, notch_angle_deg=0.0,
        area_frac=0.456354318,
        radii=(
            1.000000000, 0.744607114, 0.621471847, 0.561751720,
            0.542146456, 0.931366377, 0.712745532, 0.605557428,
            0.554999245, 0.549045981, 0.873341101, 0.685023896,
            0.591813788, 0.549718682, 0.596092837, 0.823849152,
            0.660865185, 0.580037263, 0.545844833, 0.712473838,
            0.781334135, 0.639806107, 0.570061578, 0.543330773,
        ),
    ),
    Case(
        blades=6, rotation_deg=0.0, curvature=-0.4,
        twist=0.0, notch_deg=-30.0, notch_angle_deg=35.0,
        area_frac=0.582228954,
        radii=(
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
            0.908046163, 0.735415178, 0.675499815, 0.736264566,
        ),
    ),
    Case(
        blades=11, rotation_deg=0.0, curvature=1.0,
        twist=0.6, notch_deg=45.0, notch_angle_deg=20.0,
        area_frac=1.000000000,
        radii=(
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
            1.000000000, 1.000000000, 1.000000000, 1.000000000,
        ),
    ),
    Case(
        blades=4, rotation_deg=45.0, curvature=-0.25,
        twist=0.35, notch_deg=45.0, notch_angle_deg=0.0,
        area_frac=0.425999228,
        radii=(
            0.627862932, 0.609919462, 0.628525792, 0.643066041,
            0.707663118, 0.683151700, 0.627862932, 0.609919462,
            0.628525792, 0.643066041, 0.707663118, 0.683151700,
            0.627862932, 0.609919462, 0.628525792, 0.643066041,
            0.707663118, 0.683151700, 0.627862932, 0.609919462,
            0.628525792, 0.643066041, 0.707663118, 0.683151700,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_surface(blades, rotation_deg=0.0, curvature=0.0, twist=0.0,
                 notch_deg=0.0, notch_angle_deg=0.0, semi_aperture=10.0):
    s = ghostlight.Surface()
    s.semi_aperture = semi_aperture
    s.is_stop = True
    s.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
    s.aperture_blades = blades
    s.aperture_rotation_rad = math.radians(rotation_deg)
    s.aperture_curvature = curvature
    s.aperture_twist = twist
    s.aperture_notch_rad = math.radians(notch_deg)
    s.aperture_notch_angle_rad = math.radians(notch_angle_deg)
    return s


def profile_for(case_or_blades, **kw):
    if isinstance(case_or_blades, Case):
        c = case_or_blades
        return make_surface(c.blades, c.rotation_deg, c.curvature, c.twist,
                            c.notch_deg, c.notch_angle_deg).aperture_profile
    return make_surface(case_or_blades, **kw).aperture_profile


def _tol(profile, theta):
    """Slope-aware float32 bound at ``theta``.

    A float32 angle near 2*pi carries ~1e-6 rad of quantisation, which the
    profile's own slope turns into radius error; the constant term covers the
    evaluation itself.
    """
    return 3e-6 + abs(profile.dr_dtheta(theta)) * 1.5e-6


def sample(profile, n):
    return [profile.radius_at(2.0 * math.pi * i / n) for i in range(n)]


ALL_PARAMS = [
    (blades, c, t, nd, na)
    for blades in (3, 5, 6, 8)
    for c in (-1.0, -0.5, 0.0, 0.5, 1.0)
    for t in (-1.0, 0.0, 0.6)
    for nd, na in ((0.0, 0.0), (25.0, 0.0), (-35.8, 45.0), (45.0, 20.0))
]

def _ids(case):
    return (f"b{case.blades}_c{case.curvature}"
            f"_t{case.twist}_n{case.notch_deg}")


# ---------------------------------------------------------------------------
# Frozen vectors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", PROFILE_VECTORS, ids=_ids)
def test_frozen_radii(case):
    p = profile_for(case)
    for i, want in enumerate(case.radii):
        theta = 2.0 * math.pi * i / len(case.radii)
        assert p.radius_at(theta) == pytest.approx(want, abs=_tol(p, theta))


@pytest.mark.parametrize("case", PROFILE_VECTORS, ids=_ids)
def test_frozen_area_fraction(case):
    assert profile_for(case).area_frac == pytest.approx(case.area_frac, abs=1e-6)


def test_reference_combination():
    """The panel preset this feature was specified against."""
    p = profile_for(5, rotation_deg=29.0, curvature=-0.90, twist=-0.80,
                    notch_deg=35.8, notch_angle_deg=45.0)
    radii = sample(p, 4000)
    assert p.area_frac == pytest.approx(0.3230, abs=5e-5)
    assert p.r_w == pytest.approx(0.4086, abs=5e-5)
    assert p.p == pytest.approx(0.4040, abs=5e-5)
    assert min(radii) == pytest.approx(0.4086, abs=1e-4)
    assert max(radii) == pytest.approx(0.9992, abs=1e-4)


@pytest.mark.skipif(not ORACLE.exists(), reason="geometry oracle not present")
def test_matches_oracle():
    """Re-derive the whole cross-check sweep from the oracle itself."""
    spec = importlib.util.spec_from_file_location("aperture_oracle", ORACLE)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    for blades, c, t, nd, na in ALL_PARAMS:
        p = profile_for(blades, curvature=c, twist=t,
                        notch_deg=nd, notch_angle_deg=na)
        o = oracle.ApertureProfile(blades, ORACLE_ORIENTATION_DEG / blades,
                                   c, t, nd, na)
        assert p.area_frac == pytest.approx(o.area_fraction, abs=1e-6)
        for i in range(180):
            theta = 2.0 * math.pi * (i + 0.317) / 180
            assert p.radius_at(theta) == pytest.approx(
                o.radius_at(theta), abs=_tol(p, theta))


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blades", [3, 5, 6, 8, 11])
def test_defaults_are_the_regular_polygon(blades):
    """Same expression check_aperture()'s plain-polygon branch evaluates."""
    p = profile_for(blades)
    apothem = math.cos(math.pi / blades)
    sector = 2.0 * math.pi / blades
    assert p.plain == 1
    for i in range(500):
        theta = 2.0 * math.pi * i / 500
        offset = math.fmod(theta, sector)
        if offset < 0.0:
            offset += sector
        offset -= 0.5 * sector
        assert p.radius_at(theta) == pytest.approx(apothem / math.cos(offset),
                                                   abs=1e-6)


@pytest.mark.parametrize("blades", [3, 5, 6, 8])
def test_a_blade_tip_points_along_plus_x_at_rotation_zero(blades):
    """The polygon orientation documented by the file format and used by tracing."""
    p = profile_for(blades, curvature=-0.5)
    assert p.radius_at(0.0) == pytest.approx(1.0, abs=1e-6)
    assert p.radius_at(math.pi / blades) == pytest.approx(p.r_w, abs=1e-6)


def test_full_curvature_is_the_circle_under_any_twist_and_notch():
    p = profile_for(7, curvature=1.0, twist=-0.8, notch_deg=40.0,
                    notch_angle_deg=45.0)
    assert all(r == 1.0 for r in sample(p, 500))


@pytest.mark.parametrize("blades,c,t,nd,na", ALL_PARAMS)
def test_radius_never_exceeds_one(blades, c, t, nd, na):
    """Tips stay on the unit circle; a facet only ever removes material."""
    p = profile_for(blades, curvature=c, twist=t,
                    notch_deg=nd, notch_angle_deg=na)
    radii = sample(p, 2000)
    assert max(radii) <= 1.0 + 1e-6
    assert min(radii) > 0.0
    assert all(math.isfinite(r) for r in radii)
    if nd == 0.0:
        assert max(radii) == pytest.approx(1.0, abs=1e-5)


def test_notch_zero_is_a_bit_exact_no_op():
    """notch_angle alone must not move the silhouette by a single ulp."""
    a = profile_for(6, curvature=0.4, twist=0.3, notch_angle_deg=45.0)
    b = profile_for(6, curvature=0.4, twist=0.3)
    assert a.facets == 0
    assert sample(a, 500) == sample(b, 500)


def test_twist_zero_is_symmetric_and_nonzero_is_chiral():
    sym = profile_for(5, curvature=-0.6)
    chiral = profile_for(5, curvature=-0.6, twist=0.7)
    for i in range(500):
        theta = 2.0 * math.pi * i / 500
        assert sym.radius_at(theta) == pytest.approx(sym.radius_at(-theta),
                                                     abs=_tol(sym, theta))
    worst = max(abs(chiral.radius_at(2.0 * math.pi * i / 500)
                    - chiral.radius_at(-2.0 * math.pi * i / 500))
                for i in range(500))
    assert worst > 1e-2


@pytest.mark.parametrize("control", ["twist", "notch"])
def test_opposite_signs_are_mirror_images(control):
    if control == "twist":
        a = profile_for(6, curvature=-0.5, twist=0.7)
        b = profile_for(6, curvature=-0.5, twist=-0.7)
    else:
        a = profile_for(6, curvature=-0.5, notch_deg=30.0, notch_angle_deg=35.0)
        b = profile_for(6, curvature=-0.5, notch_deg=-30.0, notch_angle_deg=35.0)
    for i in range(500):
        theta = 2.0 * math.pi * (i + 0.37) / 500
        assert a.radius_at(theta) == pytest.approx(b.radius_at(-theta),
                                                   abs=_tol(a, theta))


@pytest.mark.parametrize("blades,c,t,nd,na", ALL_PARAMS)
def test_continuous_everywhere(blades, c, t, nd, na):
    """A real jump keeps its size as sampling refines; a steep ramp halves."""
    p = profile_for(blades, curvature=c, twist=t,
                    notch_deg=nd, notch_angle_deg=na)

    def max_jump(n):
        r = sample(p, n)
        return max(abs(b - a) for a, b in zip(r, r[1:] + r[:1]))

    coarse, fine = max_jump(4000), max_jump(8000)
    if coarse > 1e-5:
        assert fine / coarse < 0.75


@pytest.mark.parametrize("blades,c,t,nd,na", ALL_PARAMS)
def test_open_area_never_collapses(blades, c, t, nd, na):
    p = profile_for(blades, curvature=c, twist=t,
                    notch_deg=nd, notch_angle_deg=na)
    assert p.area_frac > 0.05


def test_area_is_monotonic_in_curvature():
    areas = [profile_for(5, curvature=c / 10.0).area_frac
             for c in range(-10, 11)]
    assert all(b >= a - 1e-6 for a, b in zip(areas, areas[1:]))
    assert areas[-1] == pytest.approx(1.0, abs=1e-4)


def test_hexagon_area_matches_the_analytic_value():
    want = (6.0 / (2.0 * math.pi)) * math.sin(math.pi / 3.0)
    assert profile_for(6).area_frac == pytest.approx(want, abs=1e-6)


def test_analytic_slope_matches_finite_differences():
    for blades, c, t, nd, na in ALL_PARAMS[:40]:
        p = profile_for(blades, curvature=c, twist=t,
                        notch_deg=nd, notch_angle_deg=na)
        for i in range(60):
            theta = 2.0 * math.pi * (i + 0.31) / 60
            h = 1e-4
            fd = (p.radius_at(theta + h) - p.radius_at(theta - h)) / (2.0 * h)
            assert p.dr_dtheta(theta) == pytest.approx(fd, abs=0.02, rel=0.05)


# ---------------------------------------------------------------------------
# Derived-block freshness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr,value", [
    ("aperture_blades", 7),
    ("aperture_rotation_rad", 0.3),
    ("aperture_curvature", -0.5),
    ("aperture_twist", 0.4),
    ("aperture_notch_rad", 0.2),
    ("aperture_notch_angle_rad", 0.5),
])
def test_setters_refresh_the_derived_block(attr, value):
    s = make_surface(6, curvature=-0.3, notch_deg=20.0)
    before = sample(s.aperture_profile, 64)
    setattr(s, attr, value)
    assert sample(s.aperture_profile, 64) != before


def test_shape_switch_clears_the_profile():
    s = make_surface(6, curvature=-0.5)
    assert s.aperture_profile.deformed()
    s.aperture_shape = int(ghostlight.ApertureShape.CIRCLE)
    assert s.aperture_profile.blades == 0
    assert not s.aperture_profile.deformed()


def test_plain_polygon_is_not_marked_deformed():
    assert profile_for(6).deformed() is False
    assert profile_for(6, notch_angle_deg=45.0).deformed() is False
    assert profile_for(6, curvature=1e-6).deformed() is True


def test_circular_stop_has_no_profile():
    s = ghostlight.Surface()
    assert s.aperture_profile.blades == 0
    assert s.aperture_profile.area_frac == 1.0
