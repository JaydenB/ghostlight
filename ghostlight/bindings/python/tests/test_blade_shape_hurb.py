"""HURB's edge distance on a bladed stop.

The kick itself is pinned in test_hurb.py. A deformed blade edge is a conic, so
an exact closest point would need a quartic solve; the profile supplies a
first-order gradient distance instead. These tests pin the two properties that
make it usable: it vanishes exactly where check_aperture() flips, and it stays
continuous.
"""
import math

import pytest

import ghostlight

_edge = ghostlight._ghostlight._aperture_edge_distance_debug
_D_LINE = 587.56
SEMI = 10.0


def _stop(**controls) -> ghostlight.Surface:
    s = ghostlight.Surface()
    s.radius = 0.0
    s.thickness = 0.0
    s.ior = 1.0
    s.semi_aperture = SEMI
    s.is_stop = True
    s.disp_model = ghostlight.DispersionModel.AIR
    s.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
    s.aperture_blades = 6
    for k, v in controls.items():
        setattr(s, k, v)
    return s


def _distance(surface, x, y):
    d, _n = _edge(ghostlight.Vec3f(x, y, 0.0), surface)
    return d


def _normal(surface, x, y):
    _d, n = _edge(ghostlight.Vec3f(x, y, 0.0), surface)
    return n


def _boundary(surface, theta):
    return SEMI * surface.aperture_profile.radius_at(theta)


DEFORMED = [
    {"aperture_curvature": -0.8},
    {"aperture_curvature": 0.6},
    {"aperture_curvature": -0.5, "aperture_twist": 0.9},
    {"aperture_curvature": -0.9, "aperture_twist": -0.8,
     "aperture_notch_rad": math.radians(35.8),
     "aperture_notch_angle_rad": math.radians(45.0)},
]
def _ids(controls):
    return "_".join(f"{k.split('_', 1)[1]}{v:.2g}"
                    for k, v in controls.items())


# ---------------------------------------------------------------------------
# The undeformed path is untouched
# ---------------------------------------------------------------------------

def test_plain_polygon_still_measures_to_the_side_line():
    s = _stop()
    apothem = SEMI * math.cos(math.pi / 6.0)
    assert _distance(s, 0.0, 0.0) == pytest.approx(apothem, abs=1e-5)
    # Straight to an edge midpoint: the remaining distance is what is left of
    # the apothem.
    assert _distance(s, 0.0, 4.0) == pytest.approx(apothem - 4.0, abs=1e-5)


def test_notch_angle_alone_does_not_change_the_distance():
    plain = _stop()
    raked = _stop(aperture_notch_angle_rad=math.radians(45.0))
    for i in range(64):
        theta = 2.0 * math.pi * i / 64
        x, y = 5.0 * math.cos(theta), 5.0 * math.sin(theta)
        assert _distance(raked, x, y) == _distance(plain, x, y)


# ---------------------------------------------------------------------------
# The deformed path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("controls", DEFORMED, ids=_ids)
def test_distance_vanishes_on_the_silhouette(controls):
    """Zero distance has to coincide with check_aperture()'s pass/fail edge."""
    s = _stop(**controls)
    for i in range(90):
        theta = 2.0 * math.pi * (i + 0.31) / 90
        r = _boundary(s, theta)
        assert _distance(s, r * math.cos(theta), r * math.sin(theta)) == \
            pytest.approx(0.0, abs=2e-4)


@pytest.mark.parametrize("controls", DEFORMED, ids=_ids)
def test_distance_is_positive_inside_and_clamped_outside(controls):
    """Callers only ever ask about surviving rays, so outside clamps to zero."""
    s = _stop(**controls)
    for i in range(90):
        theta = 2.0 * math.pi * (i + 0.31) / 90
        r = _boundary(s, theta)
        c, sn = math.cos(theta), math.sin(theta)
        assert _distance(s, 0.9 * r * c, 0.9 * r * sn) > 0.0
        assert _distance(s, 1.1 * r * c, 1.1 * r * sn) == 0.0


def _walk(surface, frac, n):
    """Edge distance at ``frac`` of the boundary radius, all the way round."""
    out = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        r = frac * _boundary(surface, theta)
        out.append(_distance(surface, r * math.cos(theta), r * math.sin(theta)))
    return out


def _max_jump(values):
    return max(abs(b - a) for a, b in zip(values, values[1:] + values[:1]))


SYMMETRIC = [
    {"aperture_curvature": -0.8},
    {"aperture_curvature": 0.6},
    {"aperture_curvature": -0.4, "aperture_notch_rad": math.radians(30.0)},
]


@pytest.mark.parametrize("controls", SYMMETRIC, ids=_ids)
def test_distance_is_continuous_on_a_symmetric_blade(controls):
    """Across waists, tips and facet joins alike: a jump halves as we refine."""
    s = _stop(**controls)
    coarse = _max_jump(_walk(s, 0.6, 2000))
    fine = _max_jump(_walk(s, 0.6, 4000))
    if coarse > 1e-4:
        assert fine / coarse < 0.75


@pytest.mark.parametrize("controls", DEFORMED, ids=_ids)
def test_distance_is_continuous_away_from_the_tips(controls):
    """Twist gives the two sides of a tip different steepness, so the linearised
    distance steps across the tip ray (see ApertureProfile::edge_gap). Everywhere
    else — including the waist and both facet joins — it is continuous."""
    s = _stop(**controls)
    blades = s.aperture_blades
    rot = s.aperture_rotation_rad

    def away_from_tips(n):
        values, keep = _walk(s, 0.6, n), []
        for i, d in enumerate(values):
            theta = 2.0 * math.pi * i / n
            phi = math.fmod(theta - rot, 2.0 * math.pi / blades)
            keep.append(None if min(phi, 2.0 * math.pi / blades - phi)
                        < 0.02 else d)
        return [abs(b - a) for a, b in zip(keep, keep[1:])
                if a is not None and b is not None]

    coarse, fine = max(away_from_tips(2000)), max(away_from_tips(4000))
    if coarse > 1e-4:
        assert fine / coarse < 0.75


@pytest.mark.parametrize("controls", DEFORMED, ids=_ids)
def test_the_step_at_a_tip_is_bounded_by_the_distance_itself(controls):
    """The linearisation is one-sided at a corner, but it can never turn a ray
    that is far from an edge into one that is near it."""
    s = _stop(**controls)
    for frac in (0.6, 0.9, 0.99):
        values = _walk(s, frac, 4000)
        assert _max_jump(values) <= max(values)


@pytest.mark.parametrize("controls", DEFORMED, ids=_ids)
def test_normal_is_a_unit_vector_perpendicular_to_the_boundary(controls):
    s = _stop(**controls)
    for i in range(90):
        theta = 2.0 * math.pi * (i + 0.31) / 90
        r = 0.8 * _boundary(s, theta)
        n = _normal(s, r * math.cos(theta), r * math.sin(theta))
        assert math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z) == \
            pytest.approx(1.0, abs=1e-5)
        assert n.z == 0.0


def test_full_curvature_measures_the_circle():
    """At curvature +1 the boundary is the circumscribed circle exactly."""
    s = _stop(aperture_curvature=1.0)
    for i in range(32):
        theta = 2.0 * math.pi * i / 32
        for r in (2.0, 6.0, 9.0):
            assert _distance(s, r * math.cos(theta), r * math.sin(theta)) == \
                pytest.approx(SEMI - r, abs=2e-4)


def test_dead_centre_reports_the_waist():
    s = _stop(aperture_curvature=-0.8)
    assert _distance(s, 0.0, 0.0) == pytest.approx(
        SEMI * s.aperture_profile.r_w, abs=1e-5)
