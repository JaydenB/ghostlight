"""Tests for IOR/dispersion models and Fresnel weight invariants."""

import math
import pytest
import ghostlight


# Spectral reference wavelengths (nm)
_F_LINE = 486.13   # hydrogen F
_D_LINE = 587.56   # sodium d
_C_LINE = 656.27   # hydrogen C


def _make_singlet(surface):
    """Wrap a single optical surface into a finalized OpticalSystem (singlet in air)."""
    sys = ghostlight.OpticalSystem()
    sys.name = "test_singlet"
    sys.surfaces.append(surface)

    rear = ghostlight.Surface()
    rear.radius = 0.0
    rear.thickness = 0.0
    rear.ior = 1.0
    rear.abbe_v = 0.0
    rear.semi_aperture = surface.semi_aperture
    rear.disp_model = ghostlight.DispersionModel.AIR
    sys.surfaces.append(rear)

    sys.finalize()
    return sys


# ---------------------------------------------------------------------------
# Abbe/Cauchy dispersion model
# ---------------------------------------------------------------------------

def test_ior_at_d_line(bk7_surface):
    """IOR at d-line must match the nominal n_d value."""
    assert bk7_surface.ior_at(_D_LINE) == pytest.approx(1.5168, abs=1e-4)


def test_ior_at_f_line_greater_than_d_line(bk7_surface):
    """Normal dispersion: shorter wavelength → higher IOR."""
    assert bk7_surface.ior_at(_F_LINE) > bk7_surface.ior_at(_D_LINE)


def test_ior_at_c_line_less_than_d_line(bk7_surface):
    """Normal dispersion: longer wavelength → lower IOR."""
    assert bk7_surface.ior_at(_C_LINE) < bk7_surface.ior_at(_D_LINE)


def test_ior_monotone_decreasing(bk7_surface):
    """IOR must strictly decrease from F to C for normal-dispersion glass."""
    ior_f = bk7_surface.ior_at(_F_LINE)
    ior_d = bk7_surface.ior_at(_D_LINE)
    ior_c = bk7_surface.ior_at(_C_LINE)
    assert ior_f > ior_d > ior_c


# ---------------------------------------------------------------------------
# Air (no dispersion)
# ---------------------------------------------------------------------------

def test_ior_air_constant_across_wavelengths():
    """Air surfaces always return IOR = 1.0, regardless of wavelength."""
    s = ghostlight.Surface()
    s.ior = 1.0
    s.abbe_v = 0.0
    s.disp_model = ghostlight.DispersionModel.AIR
    assert s.ior_at(400.0) == pytest.approx(1.0)
    assert s.ior_at(587.56) == pytest.approx(1.0)
    assert s.ior_at(700.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sellmeier dispersion model
# ---------------------------------------------------------------------------

def test_sellmeier_ior_at_d_line(sellmeier_bk7_surface):
    """Sellmeier model for BK7 must reproduce the catalogue n_d within 0.001."""
    ior = sellmeier_bk7_surface.ior_at(_D_LINE)
    assert ior == pytest.approx(1.5168, abs=1e-3)


def test_sellmeier_normal_dispersion(sellmeier_bk7_surface):
    """Sellmeier BK7 must show normal dispersion (F > d > C)."""
    ior_f = sellmeier_bk7_surface.ior_at(_F_LINE)
    ior_d = sellmeier_bk7_surface.ior_at(_D_LINE)
    ior_c = sellmeier_bk7_surface.ior_at(_C_LINE)
    assert ior_f > ior_d > ior_c


def test_abbe_vs_sellmeier_agreement(bk7_surface, sellmeier_bk7_surface):
    """Both dispersion models for BK7 must agree at d-line within 1e-4."""
    ior_abbe = bk7_surface.ior_at(_D_LINE)
    ior_sell = sellmeier_bk7_surface.ior_at(_D_LINE)
    assert ior_abbe == pytest.approx(ior_sell, abs=1e-4)


# ---------------------------------------------------------------------------
# OpticalSystem.ior_before / ior_before_at
# ---------------------------------------------------------------------------

def test_ior_before_first_surface_is_air(simple_system):
    """The medium before the first surface must be air (IOR = 1.0)."""
    assert simple_system.ior_before(0) == pytest.approx(1.0)


def test_ior_before_glass_surface_greater_than_one(simple_system):
    """Inside a glass element, ior_before returns > 1.0."""
    # simple_system surface 1 is glass (ior=1.5168)
    ior = simple_system.ior_before(1)
    assert ior > 1.0


def test_ior_before_at_wavelength_dispersion(simple_system):
    """ior_before_at must return wavelength-dependent IOR for glass surfaces."""
    ior_f = simple_system.ior_before_at(1, _F_LINE)
    ior_c = simple_system.ior_before_at(1, _C_LINE)
    # For normal dispersion glass the F-line IOR must be higher
    assert ior_f > ior_c


# ---------------------------------------------------------------------------
# Fresnel weight invariants via trace
# ---------------------------------------------------------------------------

def _on_axis_primary(z_start=-200.0, wavelength=_D_LINE):
    return ghostlight.Ray(ghostlight.Vec3f(0.0, 0.0, z_start), ghostlight.Vec3f(0.0, 0.0, 1.0), wavelength)


def test_primary_trace_weight_in_unit_interval(simple_system):
    """Primary ray weight (cumulative transmittance) must be in (0, 1]."""
    ray = _on_axis_primary()
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.OK
    assert 0.0 < result.weight <= 1.0


def test_primary_weight_less_than_one_due_to_reflection(simple_system):
    """Even anti-reflection-coated surfaces lose some energy to reflection."""
    ray = _on_axis_primary()
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.weight < 1.0


def test_ar_coating_increases_primary_weight(simple_system):
    """Adding AR coating layers must increase primary transmittance."""
    # Uncoated
    simple_system.surfaces[0].coating.ar_layers = 0
    ray = _on_axis_primary()
    weight_bare = ghostlight.trace_primary_ray(ray, simple_system).weight

    # Single-layer AR
    simple_system.surfaces[0].coating.ar_layers = 1
    weight_ar = ghostlight.trace_primary_ray(ray, simple_system).weight

    assert weight_ar > weight_bare


def test_ghost_weight_less_than_primary_weight(simple_system):
    """Ghost (double-reflection) must carry far less energy than primary."""
    ray = _on_axis_primary()
    primary = ghostlight.trace_primary_ray(ray, simple_system)
    ghost = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)

    if ghost.status == ghostlight.TraceStatus.OK:
        assert ghost.weight < primary.weight


def test_diagnostic_fresnel_weights_in_01(simple_system):
    """Every per-surface Fresnel weight must be in (0, 1]."""
    ray = _on_axis_primary()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    for ev in path.events:
        assert 0.0 < ev.fresnel_weight <= 1.0
