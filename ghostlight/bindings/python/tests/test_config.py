"""Tests for config class hierarchy, defaults, and enum accessibility."""

import pytest
import numpy as np
import ghostlight


# ---------------------------------------------------------------------------
# RenderConfig defaults
# ---------------------------------------------------------------------------

def test_render_config_ray_grid_default():
    cfg = ghostlight.RenderConfig()
    assert cfg.ray_grid == 64


def test_render_config_spectral_samples_default():
    cfg = ghostlight.RenderConfig()
    assert cfg.spectral_samples == 16


def test_render_config_verbose_default():
    cfg = ghostlight.RenderConfig()
    assert cfg.verbose is False


def test_render_config_mutation():
    cfg = ghostlight.RenderConfig()
    cfg.ray_grid = 32
    assert cfg.ray_grid == 32


def test_render_config_spectral_samples_mutation():
    cfg = ghostlight.RenderConfig()
    cfg.spectral_samples = 8
    assert cfg.spectral_samples == 8


# ---------------------------------------------------------------------------
# FlareConfig inherits RenderConfig + has extra fields
# ---------------------------------------------------------------------------

def test_flare_config_has_ray_grid():
    cfg = ghostlight.FlareConfig()
    assert hasattr(cfg, "ray_grid")
    assert cfg.ray_grid == 64


def test_flare_config_flare_gain_default():
    cfg = ghostlight.FlareConfig()
    assert cfg.flare_gain == pytest.approx(1000.0)


def test_flare_config_min_ghost_intensity_default():
    cfg = ghostlight.FlareConfig()
    assert cfg.min_ghost_intensity == pytest.approx(1e-7, rel=0.01)


def test_flare_config_ghost_normalize_default():
    cfg = ghostlight.FlareConfig()
    assert cfg.ghost_normalize is True


def test_flare_config_max_area_boost_default():
    cfg = ghostlight.FlareConfig()
    assert cfg.max_area_boost == pytest.approx(100.0)


def test_flare_config_mutation():
    cfg = ghostlight.FlareConfig()
    cfg.flare_gain = 500.0
    assert cfg.flare_gain == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# PointFlareConfig inherits FlareConfig + has source fields
# ---------------------------------------------------------------------------

def test_point_flare_config_has_ray_grid():
    cfg = ghostlight.PointFlareConfig()
    assert hasattr(cfg, "ray_grid")


def test_point_flare_config_has_flare_gain():
    cfg = ghostlight.PointFlareConfig()
    assert hasattr(cfg, "flare_gain")


def test_point_flare_config_source_x_default():
    cfg = ghostlight.PointFlareConfig()
    assert cfg.source_x == pytest.approx(0.5)


def test_point_flare_config_source_y_default():
    cfg = ghostlight.PointFlareConfig()
    assert cfg.source_y == pytest.approx(0.5)


def test_point_flare_config_source_rgb_mutation():
    cfg = ghostlight.PointFlareConfig()
    cfg.source_r = 2.0
    cfg.source_g = 1.5
    cfg.source_b = 0.5
    assert cfg.source_r == pytest.approx(2.0)
    assert cfg.source_g == pytest.approx(1.5)
    assert cfg.source_b == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Enum accessibility
# ---------------------------------------------------------------------------

def test_output_cs_acescg():
    _ = ghostlight.OutputColorSpace.ACESCG


def test_output_cs_srgb_linear():
    _ = ghostlight.OutputColorSpace.SRGB_LINEAR


def test_output_cs_p3_d65():
    _ = ghostlight.OutputColorSpace.P3_D65


def test_output_cs_p3_d60():
    _ = ghostlight.OutputColorSpace.P3_D60


def test_output_cs_xyz():
    _ = ghostlight.OutputColorSpace.XYZ


def test_sensor_model_cie1931():
    _ = ghostlight.SensorModel.CIE_1931


def test_input_cs_acescg():
    _ = ghostlight.InputColorSpace.ACESCG


def test_input_cs_srgb_linear():
    _ = ghostlight.InputColorSpace.SRGB_LINEAR


def test_surface_form_enum():
    _ = ghostlight.SurfaceForm.SPHERE
    _ = ghostlight.SurfaceForm.ASPHERE
    _ = ghostlight.SurfaceForm.CYLINDRICAL


def test_dispersion_model_enum():
    _ = ghostlight.DispersionModel.AIR
    _ = ghostlight.DispersionModel.ABBE
    _ = ghostlight.DispersionModel.SELLMEIER


def test_coating_model_enum():
    _ = ghostlight.CoatingModel.SIMPLE
    _ = ghostlight.CoatingModel.ATTENUATOR_GAUSS


def test_trace_status_enum():
    _ = ghostlight.TraceStatus.OK
    _ = ghostlight.TraceStatus.VIGNETTED
    _ = ghostlight.TraceStatus.TIR
    _ = ghostlight.TraceStatus.MISSED_SURFACE


# ---------------------------------------------------------------------------
# Custom matrix settable
# ---------------------------------------------------------------------------

def test_custom_input_to_xyz_settable():
    cfg = ghostlight.RenderConfig()
    identity = np.eye(3, dtype=np.float32)
    cfg.custom_input_to_xyz = identity


def test_custom_xyz_to_output_settable():
    cfg = ghostlight.RenderConfig()
    identity = np.eye(3, dtype=np.float32)
    cfg.custom_xyz_to_output = identity


# ---------------------------------------------------------------------------
# GhostAovMode enum + FlareConfig AOV fields
# ---------------------------------------------------------------------------

def test_ghost_aov_mode_enum():
    _ = ghostlight.GhostAovMode.NONE
    _ = ghostlight.GhostAovMode.PER_PAIR


def test_flare_config_aov_mode_default():
    assert ghostlight.FlareConfig().aov_mode == ghostlight.GhostAovMode.NONE


def test_flare_config_aov_max_pairs_default():
    assert ghostlight.FlareConfig().aov_max_pairs == -1


def test_flare_config_aov_mode_mutation():
    cfg = ghostlight.FlareConfig()
    cfg.aov_mode = ghostlight.GhostAovMode.PER_PAIR
    assert cfg.aov_mode == ghostlight.GhostAovMode.PER_PAIR


def test_flare_config_aov_max_pairs_mutation():
    cfg = ghostlight.FlareConfig()
    cfg.aov_max_pairs = 5
    assert cfg.aov_max_pairs == 5
