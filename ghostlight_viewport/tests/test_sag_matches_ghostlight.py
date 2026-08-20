"""geometry.sag matches ghostlight's asphere_sag 1:1.

Compares :func:`ghostlight_viewport.geometry.sag` against a reference Python port
of ghostlight's C++ ``asphere_sag`` over a randomized parameter grid.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from _helpers import reference_asphere_sag


@pytest.fixture(autouse=True)
def _seed():
    random.seed(20260613)


def test_sag_matches_reference_on_sphere():
    from ghostlight_viewport import geometry
    for R in (-30.0, -10.0, 10.0, 30.0, 1000.0):
        for K in (-1.0, 0.0, 0.5, 1.5):
            for r in np.linspace(0.0, abs(R) * 0.5, 25):
                want = reference_asphere_sag(float(r), R, K, [])
                if want > 1e29:
                    continue
                got = float(geometry.sag(np.array([r]), R, K, np.array([]))[0])
                assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), (
                    f"R={R} K={K} r={r}: want {want} got {got}"
                )


def test_sag_matches_reference_on_asphere():
    from ghostlight_viewport import geometry
    rng = random.Random(20260613)
    for _ in range(40):
        R = rng.uniform(-50.0, 50.0)
        if abs(R) < 1.0:
            R = math.copysign(1.0, R) * 1.0
        K = rng.uniform(-1.0, 1.0)
        terms = [rng.uniform(-1e-5, 1e-5) for _ in range(4)]  # A4..A10
        rs = np.linspace(0.0, abs(R) * 0.3, 30)
        for r in rs:
            want = reference_asphere_sag(float(r), R, K, terms)
            if want > 1e29:
                continue
            got = float(geometry.sag(np.array([r]), R, K, np.array(terms))[0])
            assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), (
                f"R={R} K={K} terms={terms} r={r}: want {want} got {got}"
            )


def test_flat_surface_emits_only_asphere_terms():
    from ghostlight_viewport import geometry
    # radius == 0 → conic term is dropped (matches reference's flat branch).
    terms = [1e-4, 0.0, 1e-6, 0.0]
    for r in np.linspace(0.0, 5.0, 10):
        want = reference_asphere_sag(float(r), 0.0, 0.0, terms)
        got = float(geometry.sag(np.array([r]), 0.0, 0.0, np.array(terms))[0])
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)


def test_cylinder_polygon_aperture_truncates_rim():
    """A polygonal-aperture cylinder must NOT render as a full rectangle.

    Builds a mock surface object and inspects the resulting silhouette in XY
    against an axis-aligned bounding box.
    """
    from ghostlight_viewport import geometry

    class _Form:
        name = "CYLINDRICAL"
    class _Axis:
        name = "AXIS_Y"
    class _Shape:
        name = "POLYGON"
    class _Surface:
        form = _Form()
        cyl_axis = _Axis()
        aperture_shape = _Shape()
        aperture_blades = 6
        aperture_rotation_rad = 0.0
        aperture_aspect = 1.0
        radius = 20.0
        conic_k = 0.0
        asphere_terms = []
        n_asphere_terms = 0
        semi_aperture = 5.0
        z = 0.0
        decenter_x = 0.0
        decenter_y = 0.0

    mesh = geometry.tessellate_cylinder(_Surface(), outward_sign=-1.0)
    assert mesh.vertex_count > 0
    xy = mesh.vertices[:, :2]
    # Half-diagonal of the rectangle = sqrt(2) * semi.  Polygon clipping must
    # push the corner-most vertices inside the apothem distance.
    rect_corner = float(np.linalg.norm([_Surface.semi_aperture] * 2))
    apothem = math.cos(math.pi / 6) * _Surface.semi_aperture
    rmax = float(np.linalg.norm(xy, axis=1).max())
    assert rmax < rect_corner - 1e-3, (
        f"polygon clipping inactive: rmax={rmax} >= rect_corner={rect_corner}"
    )
    # Should still extend most of the way to the polygon boundary.
    assert rmax > apothem * 0.9


def test_cylinder_dispatch_with_int_enum_fields():
    """Form/cyl_axis/aperture_shape may come back from pybind11 as plain ints.

    The C++ Surface struct stores these as ``int`` so the binding exposes
    integers, not enum instances.  A naive ``getattr(form, "name", "SPHERE")``
    silently falls through to the spherical path on every real loaded lens —
    which is exactly the bug that made anamorphic .lens files render as
    spheres.  This test pins the int-friendly dispatch.
    """
    from ghostlight_viewport import geometry

    class _IntSurface:
        form = 2            # FORM_CYLINDRICAL
        cyl_axis = 0        # CYL_AXIS_X — flat along X, curved along Y
        aperture_shape = 0  # APERTURE_CIRCLE
        aperture_blades = 0
        aperture_rotation_rad = 0.0
        aperture_aspect = 1.0
        radius = 100.0
        conic_k = 0.0
        asphere_terms = []
        n_asphere_terms = 0
        semi_aperture = 10.0
        z = 0.0
        decenter_x = 0.0
        decenter_y = 0.0

    # tessellate_surface must route to the cylindrical path.
    mesh = geometry.tessellate_surface(_IntSurface(), outward_sign=-1.0)
    v = mesh.vertices
    # Rectangular footprint reaches to ±semi in both axes (aspect == 1).
    assert math.isclose(float(v[:, 0].max()), 10.0, abs_tol=1e-3)
    assert math.isclose(float(v[:, 1].max()), 10.0, abs_tol=1e-3)
    # Sag must depend on the CURVED axis only (Y here).  At y=0 the sag is
    # zero; at |y| ~ semi the sag is ~semi**2/(2*R) = 100/200 = 0.5 mm.  If
    # the dispatch had fallen through to the spherical cap, sag would depend
    # on r = sqrt(x**2 + y**2) instead and the center column would also bulge.
    y_edge = np.abs(v[:, 1]) > 9.0
    y_center = np.abs(v[:, 1]) < 0.5
    assert y_edge.any() and y_center.any()
    assert float(np.abs(v[y_edge, 2]).mean()) > 0.4
    assert float(np.abs(v[y_center, 2]).mean()) < 0.05

    # Rim-vertex helper must also dispatch correctly so lofted side walls
    # follow the rectangle perimeter, not a circle inscribed in it.
    rim = geometry._surface_rim_vertices(_IntSurface(), n_theta=64)
    rim_corner = float(np.linalg.norm(rim[:, :2], axis=1).max())
    # The far rectangle corner is at distance sqrt(2)*semi ~ 14.14 — well
    # beyond the inscribed circle (radius=semi=10) that the old polar code
    # would have produced.
    assert rim_corner > 13.0, (
        f"side-wall rim still inscribed in circle: rmax={rim_corner}"
    )
