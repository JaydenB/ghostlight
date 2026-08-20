"""Tests for Vec3f and Ray foundational math."""

import math
import pytest
import ghostlight


# ---------------------------------------------------------------------------
# Vec3f construction
# ---------------------------------------------------------------------------

def test_vec3_default_zero():
    v = ghostlight.Vec3f()
    assert v.x == 0.0
    assert v.y == 0.0
    assert v.z == 0.0


def test_vec3_scalar_broadcast():
    v = ghostlight.Vec3f(3.0)
    assert v.x == 3.0
    assert v.y == 3.0
    assert v.z == 3.0


def test_vec3_components():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0


# ---------------------------------------------------------------------------
# Vec3f indexing and iteration
# ---------------------------------------------------------------------------

def test_vec3_index():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    assert v[0] == 1.0
    assert v[1] == 2.0
    assert v[2] == 3.0


def test_vec3_iteration():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    components = list(v)
    assert components == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Vec3f arithmetic
# ---------------------------------------------------------------------------

def test_vec3_add():
    a = ghostlight.Vec3f(1.0, 0.0, 0.0)
    b = ghostlight.Vec3f(0.0, 1.0, 0.0)
    c = a + b
    assert c.x == pytest.approx(1.0)
    assert c.y == pytest.approx(1.0)
    assert c.z == pytest.approx(0.0)


def test_vec3_sub():
    a = ghostlight.Vec3f(3.0, 2.0, 1.0)
    b = ghostlight.Vec3f(1.0, 1.0, 1.0)
    c = a - b
    assert c.x == pytest.approx(2.0)
    assert c.y == pytest.approx(1.0)
    assert c.z == pytest.approx(0.0)


def test_vec3_mul_scalar():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    w = v * 2.0
    assert w.x == pytest.approx(2.0)
    assert w.y == pytest.approx(4.0)
    assert w.z == pytest.approx(6.0)


def test_vec3_rmul_scalar():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    w = 2.0 * v
    assert w.x == pytest.approx(2.0)
    assert w.y == pytest.approx(4.0)
    assert w.z == pytest.approx(6.0)


def test_vec3_div_scalar():
    v = ghostlight.Vec3f(4.0, 8.0, 2.0)
    w = v / 2.0
    assert w.x == pytest.approx(2.0)
    assert w.y == pytest.approx(4.0)
    assert w.z == pytest.approx(1.0)


def test_vec3_negate():
    v = ghostlight.Vec3f(1.0, -2.0, 3.0)
    w = -v
    assert w.x == pytest.approx(-1.0)
    assert w.y == pytest.approx(2.0)
    assert w.z == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Vec3f magnitude
# ---------------------------------------------------------------------------

def test_vec3_length_sq():
    v = ghostlight.Vec3f(3.0, 4.0, 0.0)
    assert v.length_sq() == pytest.approx(25.0)


def test_vec3_length():
    v = ghostlight.Vec3f(3.0, 4.0, 0.0)
    assert v.length() == pytest.approx(5.0)


def test_vec3_length_unit():
    v = ghostlight.Vec3f(1.0, 0.0, 0.0)
    assert v.length() == pytest.approx(1.0)


def test_vec3_normalized_unit_length():
    v = ghostlight.Vec3f(3.0, 0.0, 0.0)
    n = v.normalized()
    assert n.length() == pytest.approx(1.0)


def test_vec3_normalized_direction():
    v = ghostlight.Vec3f(0.0, 4.0, 0.0)
    n = v.normalized()
    assert n.x == pytest.approx(0.0)
    assert n.y == pytest.approx(1.0)
    assert n.z == pytest.approx(0.0)


def test_vec3_normalized_diagonal():
    v = ghostlight.Vec3f(1.0, 1.0, 1.0)
    n = v.normalized()
    assert n.length() == pytest.approx(1.0, abs=1e-6)
    expected = 1.0 / math.sqrt(3.0)
    assert n.x == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Dot and cross products
# ---------------------------------------------------------------------------

def test_dot_orthogonal():
    a = ghostlight.Vec3f(1.0, 0.0, 0.0)
    b = ghostlight.Vec3f(0.0, 1.0, 0.0)
    assert ghostlight.dot(a, b) == pytest.approx(0.0)


def test_dot_parallel():
    a = ghostlight.Vec3f(2.0, 0.0, 0.0)
    b = ghostlight.Vec3f(3.0, 0.0, 0.0)
    assert ghostlight.dot(a, b) == pytest.approx(6.0)


def test_dot_self_equals_length_sq():
    v = ghostlight.Vec3f(1.0, 2.0, 3.0)
    assert ghostlight.dot(v, v) == pytest.approx(v.length_sq())


def test_cross_canonical_xy():
    x = ghostlight.Vec3f(1.0, 0.0, 0.0)
    y = ghostlight.Vec3f(0.0, 1.0, 0.0)
    z = ghostlight.cross(x, y)
    assert z.x == pytest.approx(0.0)
    assert z.y == pytest.approx(0.0)
    assert z.z == pytest.approx(1.0)


def test_cross_canonical_yz():
    y = ghostlight.Vec3f(0.0, 1.0, 0.0)
    z = ghostlight.Vec3f(0.0, 0.0, 1.0)
    x = ghostlight.cross(y, z)
    assert x.x == pytest.approx(1.0)
    assert x.y == pytest.approx(0.0)
    assert x.z == pytest.approx(0.0)


def test_cross_anticommutative():
    a = ghostlight.Vec3f(1.0, 2.0, 3.0)
    b = ghostlight.Vec3f(4.0, 5.0, 6.0)
    ab = ghostlight.cross(a, b)
    ba = ghostlight.cross(b, a)
    assert ab.x == pytest.approx(-ba.x)
    assert ab.y == pytest.approx(-ba.y)
    assert ab.z == pytest.approx(-ba.z)


def test_cross_parallel_is_zero():
    a = ghostlight.Vec3f(1.0, 2.0, 3.0)
    b = ghostlight.Vec3f(2.0, 4.0, 6.0)
    c = ghostlight.cross(a, b)
    assert c.length() == pytest.approx(0.0, abs=1e-6)


def test_cross_result_orthogonal_to_inputs():
    a = ghostlight.Vec3f(1.0, 2.0, 0.0)
    b = ghostlight.Vec3f(0.0, 1.0, 1.0)
    c = ghostlight.cross(a, b)
    assert ghostlight.dot(c, a) == pytest.approx(0.0, abs=1e-6)
    assert ghostlight.dot(c, b) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------

def test_ray_default_wavelength():
    o = ghostlight.Vec3f(0.0, 0.0, -200.0)
    d = ghostlight.Vec3f(0.0, 0.0, 1.0)
    r = ghostlight.Ray(o, d)
    assert r.wavelength == pytest.approx(587.56)


def test_ray_explicit_wavelength():
    o = ghostlight.Vec3f(0.0, 0.0, 0.0)
    d = ghostlight.Vec3f(0.0, 0.0, 1.0)
    r = ghostlight.Ray(o, d, 450.0)
    assert r.wavelength == pytest.approx(450.0)


def test_ray_origin():
    o = ghostlight.Vec3f(1.0, 2.0, 3.0)
    d = ghostlight.Vec3f(0.0, 0.0, 1.0)
    r = ghostlight.Ray(o, d)
    assert r.origin.x == pytest.approx(1.0)
    assert r.origin.y == pytest.approx(2.0)
    assert r.origin.z == pytest.approx(3.0)


def test_ray_direction():
    o = ghostlight.Vec3f(0.0, 0.0, 0.0)
    d = ghostlight.Vec3f(0.0, 0.0, 1.0)
    r = ghostlight.Ray(o, d)
    assert r.dir.x == pytest.approx(0.0)
    assert r.dir.y == pytest.approx(0.0)
    assert r.dir.z == pytest.approx(1.0)


def test_ray_wavelength_mutation():
    o = ghostlight.Vec3f(0.0, 0.0, 0.0)
    d = ghostlight.Vec3f(0.0, 0.0, 1.0)
    r = ghostlight.Ray(o, d, 550.0)
    r.wavelength = 650.0
    assert r.wavelength == pytest.approx(650.0)
