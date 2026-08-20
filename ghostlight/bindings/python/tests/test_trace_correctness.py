"""Tests for ray tracing geometry and physics correctness."""

import math
import pytest
import ghostlight


_D_LINE = 587.56
_F_LINE = 486.13
_C_LINE = 656.27


def _primary_ray(x=0.0, y=0.0, z_start=-200.0, wavelength=_D_LINE):
    origin = ghostlight.Vec3f(x, y, z_start)
    direction = ghostlight.Vec3f(0.0, 0.0, 1.0)
    return ghostlight.Ray(origin, direction, wavelength)


def _make_bk7_singlet():
    """Two-surface BK7 singlet with a reasonable geometry."""
    sys = ghostlight.OpticalSystem()
    sys.name = "bk7_singlet"

    front = ghostlight.Surface()
    front.radius = 50.0
    front.thickness = 5.0
    front.ior = 1.5168
    front.abbe_v = 64.17
    front.semi_aperture = 20.0
    front.disp_model = ghostlight.DispersionModel.ABBE
    sys.surfaces.append(front)

    rear = ghostlight.Surface()
    rear.radius = -50.0
    rear.thickness = 45.0
    rear.ior = 1.0
    rear.abbe_v = 0.0
    rear.semi_aperture = 20.0
    rear.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(rear)

    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = 8.0
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(stop)

    sys.finalize()
    return sys


# ---------------------------------------------------------------------------
# Primary ray geometry
# ---------------------------------------------------------------------------

def test_on_axis_ray_lands_near_optical_axis(simple_system):
    """An on-axis paraxial ray must land very close to the optical axis."""
    ray = _primary_ray(0.0, 0.0)
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.OK
    assert abs(result.position.x) < 1e-4
    assert abs(result.position.y) < 1e-4


def test_off_axis_ray_displaced_from_axis(simple_system):
    """An off-axis ray must land displaced from the optical axis."""
    ray = _primary_ray(x=2.0)
    result = ghostlight.trace_primary_ray(ray, simple_system)
    # May be vignetted for large offsets; just check displacement when OK
    if result.status == ghostlight.TraceStatus.OK:
        assert abs(result.position.x) > 1e-3


def test_primary_position_lands_at_origin(simple_system):
    """Primary ray must land on the sensor plane at z=0."""
    ray = _primary_ray()
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.OK
    assert result.position.z == pytest.approx(0.0, abs=1e-3)


def test_status_ok_paraxial(simple_system):
    """Paraxial on-axis ray must have status OK."""
    ray = _primary_ray()
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.OK


def test_status_vignetted_outside_aperture(simple_system):
    """Ray starting far outside the semi-aperture must be vignetted."""
    # semi_aperture of simple_system is 15mm; start well outside it
    ray = _primary_ray(x=30.0)
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.VIGNETTED


# ---------------------------------------------------------------------------
# Diagnostic path geometry
# ---------------------------------------------------------------------------

def test_hit_point_z_ascending(simple_system):
    """Each surface hit must have strictly increasing z relative to the one before."""
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    zs = [ev.hit_point.z for ev in path.events]
    for i in range(1, len(zs)):
        assert zs[i] >= zs[i - 1], f"z not ascending at event {i}: {zs}"


def test_surface_normal_unit_length(simple_system):
    """All surface normals in the diagnostic path must be unit vectors."""
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    for ev in path.events:
        length = ev.surface_normal.length()
        assert length == pytest.approx(1.0, abs=1e-5), \
            f"Surface {ev.surface_index} normal length {length}"


def test_ior_before_matches_lens_system(simple_system):
    """Diagnostic event.ior_before must match OpticalSystem.ior_before()."""
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    for ev in path.events:
        expected = simple_system.ior_before(ev.surface_index)
        assert ev.ior_before == pytest.approx(expected, abs=1e-5)


def test_no_reflections_in_primary_path(simple_system):
    """Primary tracing must not produce any reflected events."""
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    for ev in path.events:
        assert not ev.reflected


def test_surface_indices_sequential(simple_system):
    """Diagnostic events must be in sequential surface order."""
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    indices = [ev.surface_index for ev in path.events]
    assert indices == list(range(simple_system.num_surfaces()))


# ---------------------------------------------------------------------------
# Ghost ray structure
# ---------------------------------------------------------------------------

def test_ghost_reflected_events_at_correct_surfaces(simple_system):
    """Reflected events must occur exactly at bounce_a and bounce_b."""
    ray = _primary_ray()
    path = ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, 0, 1)
    reflected_indices = [ev.surface_index for ev in path.events if ev.reflected]
    assert 0 in reflected_indices
    assert 1 in reflected_indices


def test_ghost_exactly_two_reflections(simple_system):
    """Ghost path must have exactly 2 reflected events."""
    ray = _primary_ray()
    path = ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, 0, 1)
    reflected_count = sum(1 for ev in path.events if ev.reflected)
    assert reflected_count == 2


def test_ghost_weight_positive_for_valid_pair(simple_system):
    """A valid ghost pair on-axis must produce positive weight."""
    ray = _primary_ray()
    result = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    if result.status == ghostlight.TraceStatus.OK:
        assert result.weight > 0.0


def test_ghost_position_z_on_axis(simple_system):
    """Ghost result position z must land on the sensor at z=0 (or miss sensor)."""
    ray = _primary_ray()
    result = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    if result.status == ghostlight.TraceStatus.OK:
        assert result.position.z == pytest.approx(0.0, abs=1e-3)


def test_ghost_z_changes_direction(simple_system):
    """A ghost path travels forward, backward, then forward through the lens."""
    ray = _primary_ray()
    path = ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, 0, 1)
    zs = [ev.hit_point.z for ev in path.events]
    # Just verify there's a direction change: not all ascending or all descending
    ascending = all(zs[i + 1] >= zs[i] for i in range(len(zs) - 1))
    descending = all(zs[i + 1] <= zs[i] for i in range(len(zs) - 1))
    # A ghost path cannot be purely monotone unless it's a degenerate case
    # (e.g., 2-surface lens). For simple_system (3 surfaces) it must reverse.
    if len(zs) >= 3:
        assert not (ascending or descending), \
            "Ghost z path should reverse direction but didn't"


# ---------------------------------------------------------------------------
# Chromatic dispersion shifts landing position
# ---------------------------------------------------------------------------

def test_chromatic_dispersion_shifts_landing_position():
    """Different wavelengths through a dispersive singlet must land at different positions."""
    sys = _make_bk7_singlet()
    ray_f = _primary_ray(x=3.0, wavelength=_F_LINE)
    ray_c = _primary_ray(x=3.0, wavelength=_C_LINE)
    result_f = ghostlight.trace_primary_ray(ray_f, sys)
    result_c = ghostlight.trace_primary_ray(ray_c, sys)
    if result_f.status == ghostlight.TraceStatus.OK and result_c.status == ghostlight.TraceStatus.OK:
        # Positions should differ due to chromatic aberration
        dx = abs(result_f.position.x - result_c.position.x)
        dy = abs(result_f.position.y - result_c.position.y)
        assert dx + dy > 1e-4, "Chromatic shift should be detectable"


def test_shorter_wavelength_bends_more():
    """For a converging singlet, shorter wavelength (F) must converge sooner (lower IOR at focus)."""
    sys = _make_bk7_singlet()
    # On-axis: F-line IOR is higher → more bending
    # This is just an IOR check through ior_before_at
    ior_f = sys.ior_before_at(1, _F_LINE)
    ior_c = sys.ior_before_at(1, _C_LINE)
    assert ior_f > ior_c


# ---------------------------------------------------------------------------
# Surface geometry variants
# ---------------------------------------------------------------------------

def test_decentered_surface_shifts_hit_point():
    """Decentering a surface must shift the ray hit point."""
    sys_a = _make_bk7_singlet()
    sys_b = _make_bk7_singlet()
    sys_b.surfaces[0].decenter_x = 2.0
    sys_b.finalize()

    ray = _primary_ray()
    path_a = ghostlight.trace_primary_ray_diagnostic(ray, sys_a)
    path_b = ghostlight.trace_primary_ray_diagnostic(ray, sys_b)

    # An on-axis ray travels along z, so hit_point.x is 0 in world space regardless
    # of decenter.  The observable effect of decenter is a shifted surface normal
    # (which changes refraction) and therefore a shifted sensor landing position.
    sensor_x_a = path_a.result.position.x
    sensor_x_b = path_b.result.position.x
    assert abs(sensor_x_a - sensor_x_b) > 0.1, "Decenter should shift the sensor hit position"


def test_asphere_differs_from_sphere():
    """An aspheric surface with non-zero conic must land at different position."""
    def make_sys(form, conic_k=0.0):
        sys = ghostlight.OpticalSystem()
        front = ghostlight.Surface()
        front.radius = 50.0
        front.thickness = 5.0
        front.ior = 1.5168
        front.abbe_v = 64.17
        front.semi_aperture = 20.0
        front.disp_model = ghostlight.DispersionModel.ABBE
        front.form = form
        front.conic_k = conic_k
        sys.surfaces.append(front)

        rear = ghostlight.Surface()
        rear.radius = 0.0
        rear.thickness = 0.0
        rear.ior = 1.0
        rear.semi_aperture = 20.0
        rear.disp_model = ghostlight.DispersionModel.AIR
        sys.surfaces.append(rear)
        sys.finalize()
        return sys

    sys_sphere = make_sys(ghostlight.SurfaceForm.SPHERE, conic_k=0.0)
    sys_asphere = make_sys(ghostlight.SurfaceForm.ASPHERE, conic_k=-1.0)

    ray = _primary_ray(x=5.0)
    r_sph = ghostlight.trace_primary_ray(ray, sys_sphere)
    r_asp = ghostlight.trace_primary_ray(ray, sys_asphere)

    if r_sph.status == ghostlight.TraceStatus.OK and r_asp.status == ghostlight.TraceStatus.OK:
        dx = abs(r_sph.position.x - r_asp.position.x)
        assert dx > 1e-4, "Asphere with conic_k=-1 should differ from sphere"
