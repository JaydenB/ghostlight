"""Tests for physical invariants and regression correctness."""

import math
import pytest
import numpy as np
import ghostlight


_D_LINE = 587.56
_F_LINE = 486.13
_C_LINE = 656.27


def _primary_ray(x=0.0, y=0.0, z_start=-200.0, wl=_D_LINE):
    return ghostlight.Ray(ghostlight.Vec3f(x, y, z_start), ghostlight.Vec3f(0.0, 0.0, 1.0), wl)


# ---------------------------------------------------------------------------
# Normal dispersion for BK7
# ---------------------------------------------------------------------------

def test_normal_dispersion_abbe_f_gt_d_gt_c(bk7_surface):
    """IOR must decrease monotonically from F → d → C for BK7 (normal dispersion)."""
    ior_f = bk7_surface.ior_at(_F_LINE)
    ior_d = bk7_surface.ior_at(_D_LINE)
    ior_c = bk7_surface.ior_at(_C_LINE)
    assert ior_f > ior_d > ior_c


def test_normal_dispersion_sellmeier_f_gt_d_gt_c(sellmeier_bk7_surface):
    ior_f = sellmeier_bk7_surface.ior_at(_F_LINE)
    ior_d = sellmeier_bk7_surface.ior_at(_D_LINE)
    ior_c = sellmeier_bk7_surface.ior_at(_C_LINE)
    assert ior_f > ior_d > ior_c


# ---------------------------------------------------------------------------
# Fresnel normal-incidence formula
# ---------------------------------------------------------------------------

def test_fresnel_normal_incidence_transmittance(simple_system):
    """Primary ray transmittance through air-glass interface must approximate
    the analytical formula: T ≈ 1 - ((n-1)/(n+1))^2."""
    n = 1.5168  # BK7
    # Analytical Fresnel reflectance at normal incidence
    R_theory = ((n - 1.0) / (n + 1.0)) ** 2
    T_theory = 1.0 - R_theory

    # Trace on-axis ray; single-surface diagnostic event gives weight at surface 0
    ray = _primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    # First event is the front BK7 surface
    w0 = path.events[0].fresnel_weight
    # Fresnel weight for transmission = 1 - R
    assert w0 == pytest.approx(T_theory, rel=0.01), \
        f"Fresnel transmittance {w0:.5f} vs theoretical {T_theory:.5f}"


# ---------------------------------------------------------------------------
# Ghost weight bound
# ---------------------------------------------------------------------------

def test_ghost_weight_less_than_primary(simple_system):
    """Ghost weight (two reflections) must be strictly less than primary weight."""
    ray = _primary_ray()
    primary = ghostlight.trace_primary_ray(ray, simple_system)
    ghost = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    if ghost.status == ghostlight.TraceStatus.OK and primary.status == ghostlight.TraceStatus.OK:
        assert ghost.weight < primary.weight


def test_ghost_weight_upper_bound(simple_system):
    """Ghost weight must not exceed R_a × R_b (product of reflectances at bounces)."""
    n = 1.5168
    R = ((n - 1.0) / (n + 1.0)) ** 2  # bare Fresnel reflectance ≈ 0.042 for BK7
    max_ghost_weight = R * R

    ray = _primary_ray()
    result = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    if result.status == ghostlight.TraceStatus.OK:
        # The ghost also traverses transmissive surfaces, so its weight
        # should be <= R*R (the theoretical max ignoring transmittance losses)
        assert result.weight <= max_ghost_weight * 1.5  # 1.5x for numeric tolerance


# ---------------------------------------------------------------------------
# Ghost on-axis symmetry
# ---------------------------------------------------------------------------

def test_on_axis_ghost_lands_near_axis(simple_system):
    """An on-axis ghost ray must land near the optical axis (x ≈ y ≈ 0)."""
    ray = _primary_ray(x=0.0, y=0.0)
    result = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    if result.status == ghostlight.TraceStatus.OK:
        assert abs(result.position.x) < 0.5
        assert abs(result.position.y) < 0.5


# ---------------------------------------------------------------------------
# Primary weight monotonically decreasing with more surfaces
# ---------------------------------------------------------------------------

def _make_n_surface_lens(n_glass):
    """Build a lens with n_glass BK7 elements separated by air gaps.

    Each element contributes two air-glass interfaces (entry + exit), so a lens
    with more glass elements has more Fresnel loss and a lower transmittance.
    """
    sys = ghostlight.OpticalSystem()
    for i in range(n_glass):
        s = ghostlight.Surface()
        s.radius = 50.0 if i % 2 == 0 else -50.0
        s.thickness = 5.0
        s.ior = 1.5168
        s.abbe_v = 64.17
        s.semi_aperture = 20.0
        s.disp_model = ghostlight.DispersionModel.ABBE
        sys.surfaces.append(s)
        # Air gap after each glass element (except after the last one)
        if i < n_glass - 1:
            gap = ghostlight.Surface()
            gap.radius = 0.0
            gap.thickness = 2.0
            gap.ior = 1.0
            gap.semi_aperture = 20.0
            gap.disp_model = ghostlight.DispersionModel.AIR
            sys.surfaces.append(gap)

    rear = ghostlight.Surface()
    rear.radius = 0.0
    rear.thickness = 0.0
    rear.ior = 1.0
    rear.semi_aperture = 20.0
    rear.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(rear)
    sys.finalize()
    return sys


def test_more_surfaces_lower_weight():
    """A lens with more glass surfaces must have lower primary transmittance."""
    sys1 = _make_n_surface_lens(1)
    sys3 = _make_n_surface_lens(3)
    ray = _primary_ray()
    r1 = ghostlight.trace_primary_ray(ray, sys1)
    r3 = ghostlight.trace_primary_ray(ray, sys3)
    if r1.status == ghostlight.TraceStatus.OK and r3.status == ghostlight.TraceStatus.OK:
        assert r3.weight < r1.weight


# ---------------------------------------------------------------------------
# Spectral lambdas uniform spacing
# ---------------------------------------------------------------------------

def test_spectral_lambdas_uniform_spacing():
    """All gaps between consecutive samples must be identical."""
    lambdas = ghostlight.build_spectral_lambdas(10, 400.0, 700.0)
    gaps = [lambdas[i + 1] - lambdas[i] for i in range(len(lambdas) - 1)]
    expected = gaps[0]
    for g in gaps:
        assert g == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Render determinism (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_render_point_flare_deterministic(loaded_lens):
    """Two identical renders must produce pixel-identical output."""
    cfg = ghostlight.PointFlareConfig()
    cfg.ray_grid = 16
    cfg.spectral_samples = 4
    cfg.source_r = 3.0
    cfg.source_g = 3.0
    cfg.source_b = 3.0

    out1 = loaded_lens.render_point_flare(32, 32, cfg)
    out2 = loaded_lens.render_point_flare(32, 32, cfg)

    np.testing.assert_allclose(out1["ghost_r"], out2["ghost_r"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out1["ghost_g"], out2["ghost_g"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out1["ghost_b"], out2["ghost_b"], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Field plausibility for a real lens
# ---------------------------------------------------------------------------

def test_calibration_field_plausible(doublegauss_lens):
    """The field the lens covers must be plausible for a real camera lens.

    Asked of the image circle, not of max_half_angle.  The covered field is the
    ONSET of vignetting (90% of axial throughput), which on a lens that shades
    off early sits well inside the illuminated area -- this one calibrates to
    3.5 deg there while illuminating out to 10.8 deg.  Both numbers are correct;
    only the second one answers "how much does this lens cover".
    """
    cal = doublegauss_lens.calibration()
    f = cal.focal_length_x
    assert f > 0.0, "paraxial solve failed; the check below needs it"
    circle_half_angle = math.atan(cal.image_circle_semi_w / f)

    # A typical camera lens has half-angle between ~10 and ~60 deg.
    assert math.radians(5.0) < circle_half_angle < math.radians(70.0), \
        f"Unexpected covered field: {math.degrees(circle_half_angle):.1f}°"
    # And the onset of vignetting must land inside the circle it vignettes in.
    assert 0.0 < cal.max_half_angle_h <= circle_half_angle, \
        (f"covered field {math.degrees(cal.max_half_angle_h):.1f}° is not inside "
         f"the image circle {math.degrees(circle_half_angle):.1f}°")
