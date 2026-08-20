"""A tilted element's side wall must stay welded to its caps.

The cap tessellator applied the surface's ``rot`` matrix; the side-wall rim
builders applied only ``decenter_x/y`` and ``z``. On an untilted lens the two
agree, so nothing looked wrong for as long as nothing could be tilted. The
moment the editor could author a tilt, a thick element's wall stayed
axis-aligned and visibly tore away from the glass.

The invariant that catches it, and that these tests pin: **the side wall's rim
ring and the cap's outer boundary are the same points.** Every vertex on the
wall's leading edge must lie on the cap it joins.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from _helpers import example_doublet_path, require_ghostlight


_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LENS_PATH = example_doublet_path()


def _thick_glass_element(system):
    """The element with the largest axial thickness -- a thin one hides the bug."""
    import ghostlight
    best, best_thick = None, -1.0
    for el in system.elements:
        if el.kind != ghostlight.ElementKind.GLASS or len(el.surface_ids) < 2:
            continue
        idx = el.resolve_surfaces(system)
        thick = sum(float(system.surfaces[i].thickness) for i in idx[:-1])
        if thick > best_thick:
            best, best_thick = el, thick
    if best is None:
        pytest.skip("lens has no multi-surface glass element")
    return best


def _tilted_system(tilt=(-10.3, 0.0, 0.0), pivot=(0.0, 0.0, 0.0)):
    """Load the doublet and tilt its thickest element, exactly as the UI does."""
    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    el = _thick_glass_element(system)
    el.rotation_euler_deg = tuple(float(v) for v in tilt)
    el.pivot = tuple(float(v) for v in pivot)
    assert ghostlight.bake_system_poses(system) is True
    return system, el


def _min_distance_to(points: np.ndarray, cloud: np.ndarray) -> float:
    """Largest distance from any ``points`` row to its nearest ``cloud`` row."""
    d = np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=2)
    return float(d.min(axis=1).max())


# ---------------------------------------------------------------------------
# The reported bug
# ---------------------------------------------------------------------------

def test_wall_rim_lies_on_the_cap_when_tilted():
    """Every wall rim vertex must sit on the cap surface it joins."""
    from ghostlight_viewport import geometry

    system, el = _tilted_system()
    idx = el.resolve_surfaces(system)
    front, back = system.surfaces[idx[0]], system.surfaces[idx[-1]]

    rim_front = geometry._surface_rim_vertices(front)
    rim_back = geometry._surface_rim_vertices(back)
    cap_front = geometry.tessellate_surface(front, outward_sign=-1.0).vertices
    cap_back = geometry.tessellate_surface(back, outward_sign=+1.0).vertices

    semi = float(front.semi_aperture)
    tol = semi * 0.02  # tessellation spacing, not slop

    gap_front = _min_distance_to(rim_front.astype(np.float64),
                                 cap_front.astype(np.float64))
    gap_back = _min_distance_to(rim_back.astype(np.float64),
                                cap_back.astype(np.float64))
    assert gap_front < tol, (
        f"tilted front wall rim floats {gap_front:.4f}mm off its cap "
        f"(tolerance {tol:.4f}mm)"
    )
    assert gap_back < tol, (
        f"tilted back wall rim floats {gap_back:.4f}mm off its cap "
        f"(tolerance {tol:.4f}mm)"
    )


def test_wall_rim_is_actually_rotated():
    """Guard against 'passing' by making the cap wrong instead of the rim.

    A tilted rim must differ from the untilted one; if both builders simply
    stopped rotating, the test above would pass while the picture stayed broken.
    """
    from ghostlight_viewport import geometry

    ghostlight = require_ghostlight()
    flat = ghostlight.OpticalSystem.load(str(LENS_PATH))
    el_flat = _thick_glass_element(flat)
    rim_flat = geometry._surface_rim_vertices(
        flat.surfaces[el_flat.resolve_surfaces(flat)[-1]]
    )

    system, el = _tilted_system()
    rim_tilted = geometry._surface_rim_vertices(
        system.surfaces[el.resolve_surfaces(system)[-1]]
    )

    moved = float(np.abs(rim_tilted - rim_flat).max())
    assert moved > 0.5, f"rim barely moved under a 10.3 deg tilt ({moved:.4f}mm)"


def test_rim_normal_follows_the_tilted_axis():
    """The rim ring must lie in the surface's own plane, not the world's.

    Fit a plane to the rim and check its normal tracks the element's tilted
    optical axis. A rim that was translated but not rotated stays perpendicular
    to world +Z and fails here.
    """
    from ghostlight_viewport import geometry

    tilt_deg = -10.3
    system, el = _tilted_system(tilt=(tilt_deg, 0.0, 0.0))
    surface = system.surfaces[el.resolve_surfaces(system)[-1]]

    rim = geometry._surface_rim_vertices(surface).astype(np.float64)
    centred = rim - rim.mean(axis=0)
    # Smallest singular vector of the centred ring is the plane normal.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)

    axis = geometry.surface_axis(surface)
    axis /= np.linalg.norm(axis)
    # Sign is arbitrary out of the SVD.
    alignment = abs(float(np.dot(normal, axis)))
    assert alignment > 0.999, (
        f"rim plane is not perpendicular to the element axis (|cos| = {alignment:.5f})"
    )

    # And that axis really is tilted away from world +Z by the authored angle.
    world_z = np.array([0.0, 0.0, 1.0])
    got_deg = math.degrees(math.acos(abs(float(np.dot(axis, world_z)))))
    assert got_deg == pytest.approx(abs(tilt_deg), abs=0.05)


def test_cylinder_cap_and_rim_agree_when_tilted():
    """Same invariant for the anamorphic path, which had the same gap."""
    from ghostlight_viewport import geometry

    ghostlight = require_ghostlight()
    system, el = _tilted_system()
    idx = el.resolve_surfaces(system)
    surface = system.surfaces[idx[0]]
    # Force the cylindrical branch on a real, posed surface.
    surface.form = ghostlight.SurfaceForm.CYLINDRICAL

    rim = geometry._surface_rim_vertices(surface).astype(np.float64)
    cap = geometry.tessellate_surface(surface, outward_sign=-1.0).vertices.astype(
        np.float64
    )
    tol = float(surface.semi_aperture) * 0.05
    gap = _min_distance_to(rim, cap)
    assert gap < tol, f"tilted cylindrical rim floats {gap:.4f}mm off its cap"


def test_iris_follows_a_posed_stop():
    """The iris ring is flat in the stop's plane and must move with it."""
    from ghostlight_viewport import geometry

    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    stop = next(
        (el for el in system.elements if el.kind == ghostlight.ElementKind.STOP), None
    )
    if stop is None:
        pytest.skip("lens has no stop element")

    surface = system.surfaces[stop.resolve_surfaces(system)[0]]
    before = geometry.build_iris(surface).vertices.copy()

    stop.position = (stop.position[0] + 3.0, stop.position[1] + 2.0, stop.position[2])
    stop.rotation_euler_deg = (6.0, 0.0, 0.0)
    assert ghostlight.bake_system_poses(system) is True

    after = geometry.build_iris(surface).vertices
    assert float(np.abs(after - before).max()) > 1.0, (
        "iris ring ignored the stop's decenter / tilt"
    )
    # It should have moved by the decenter, in the right direction.
    assert float(after[:, 0].mean() - before[:, 0].mean()) == pytest.approx(3.0, abs=0.2)


# ---------------------------------------------------------------------------
# The untilted case must be untouched
# ---------------------------------------------------------------------------

def test_untilted_geometry_is_unchanged():
    """The identity fast path has to stay exactly that.

    ``place_surface_vertices`` skips the matmul when rot is identity, so an
    on-axis lens must produce bit-identical vertices to before the refactor.
    Checked against the analytic expectation rather than a stored baseline.
    """
    from ghostlight_viewport import geometry

    ghostlight = require_ghostlight()
    system = ghostlight.OpticalSystem.load(str(LENS_PATH))
    el = _thick_glass_element(system)
    surface = system.surfaces[el.resolve_surfaces(system)[0]]

    assert geometry.surface_rotation(surface) is None, "sample lens is not on axis"

    rim = geometry._surface_rim_vertices(surface)
    # An untilted rim sits at exactly the surface's z plus its sag, centred on
    # the decenter (which is zero here).
    theta = np.linspace(0.0, 2.0 * math.pi, rim.shape[0], endpoint=False)
    r = geometry._aperture_radial_test(theta, surface)
    expected_z = geometry.sag(
        r, float(surface.radius), float(surface.conic_k),
        geometry._asphere_array(surface),
    ) + float(surface.z)
    assert rim[:, 2] == pytest.approx(expected_z, abs=1e-5)
    assert rim[:, 0] == pytest.approx(r * np.cos(theta), abs=1e-5)


def test_roll_turns_everything_that_is_not_a_body_of_revolution():
    """Rot Z reaches the surface frame always; it only *looks* inert on a sphere.

    Rolled by an exact multiple of the azimuthal sample spacing, so a body of
    revolution maps its vertices onto themselves and the tessellation residual
    (up to half an arc step) can't be mistaken for a real shape change.
    """
    from ghostlight_viewport import geometry

    ghostlight = require_ghostlight()
    roll = 6 * (360.0 / geometry.N_AZIMUTH_DEFAULT)

    def cap_for(roll_deg, **surface_attrs):
        system = ghostlight.OpticalSystem.load(str(LENS_PATH))
        el = _thick_glass_element(system)
        el.rotation_euler_deg = (0.0, 0.0, roll_deg)
        assert ghostlight.bake_system_poses(system) is True
        s = system.surfaces[el.resolve_surfaces(system)[0]]
        for k, v in surface_attrs.items():
            setattr(s, k, v)
        return s, geometry.tessellate_surface(s, outward_sign=-1.0).vertices

    def shape_change(**surface_attrs):
        _sa, a = cap_for(0.0, **surface_attrs)
        _sb, b = cap_for(roll, **surface_attrs)
        return _min_distance_to(b.astype(np.float64), a.astype(np.float64))

    # A sphere with a circular aperture is symmetric about its own axis, so
    # rolling it is exactly a no-op — not "small", zero.
    assert shape_change() == pytest.approx(0.0, abs=1e-4)

    # Everything that breaks that symmetry genuinely turns.
    assert shape_change(aperture_aspect=2.0) > 1.0, "anamorphic stretch didn't roll"
    assert shape_change(form=ghostlight.SurfaceForm.CYLINDRICAL) > 1.0, "cylinder didn't roll"
    assert shape_change(
        aperture_shape=ghostlight.ApertureShape.POLYGON, aperture_blades=6
    ) > 1.0, "polygon aperture didn't roll"

    # And the roll is in the traced frame regardless of form.
    surface, _ = cap_for(roll)
    assert float(surface.rot[0]) == pytest.approx(math.cos(math.radians(roll)), abs=1e-5)


def test_surface_rotation_returns_none_for_identity():
    from ghostlight_viewport import geometry

    class _Stub:
        rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    assert geometry.surface_rotation(_Stub()) is None
    assert geometry.surface_axis(_Stub()) == pytest.approx([0.0, 0.0, 1.0])


def test_place_surface_vertices_rotates_about_the_vertex():
    """Rotation happens BEFORE the offset -- the vertex is the fixed point.

    Rotating after translating would swing the surface around the world origin,
    which on a lens sitting at z = -50 would fling it out of frame.
    """
    from ghostlight_viewport import geometry

    class _Stub:
        # 90 deg about X: local +Z -> world -Y.
        rot = [1.0, 0.0, 0.0,
               0.0, 0.0, -1.0,
               0.0, 1.0, 0.0]
        decenter_x = 2.0
        decenter_y = -1.0
        z = -50.0

    surface = _Stub()
    # The vertex itself (local origin) must land exactly on the pose.
    origin = geometry.place_surface_vertices(np.zeros((1, 3)), surface)
    assert origin[0] == pytest.approx([2.0, -1.0, -50.0])
    # A point 1mm along local +Z lands 1mm along world -Y from there.
    along = geometry.place_surface_vertices(np.array([[0.0, 0.0, 1.0]]), surface)
    assert along[0] == pytest.approx([2.0, -2.0, -50.0])
