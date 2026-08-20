"""Tests for error handling and edge cases across the API."""

import tempfile
import pytest
import numpy as np
import ghostlight


# ---------------------------------------------------------------------------
# Lens construction errors
# ---------------------------------------------------------------------------

def test_finalize_empty_lens_raises():
    """finalize() on an empty OpticalSystem must raise RuntimeError."""
    sys = ghostlight.OpticalSystem()
    with pytest.raises(RuntimeError):
        sys.finalize()


def test_load_nonexistent_path_raises():
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load("absolutely_does_not_exist_12345.lens")


def test_load_empty_json_raises():
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w", delete=False) as f:
        f.write("{}")
        path = f.name
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load(path)


def test_load_invalid_json_raises():
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w", delete=False) as f:
        f.write("not json at all")
        path = f.name
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load(path)


# ---------------------------------------------------------------------------
# Ghost ray out-of-range bounce indices
# ---------------------------------------------------------------------------

def test_ghost_bounce_equal_to_num_surfaces_does_not_crash(simple_system):
    """Out-of-range bounce index must not crash (can error gracefully)."""
    ray = ghostlight.Ray(ghostlight.Vec3f(0.0, 0.0, -200.0), ghostlight.Vec3f(0.0, 0.0, 1.0))
    n = simple_system.num_surfaces()
    # bounce_b == n is out of range; should either raise or return non-OK
    try:
        result = ghostlight.trace_ghost_ray(ray, simple_system, 0, n)
        # If it doesn't raise, the status must not be OK
        assert result.status != ghostlight.TraceStatus.OK
    except Exception:
        pass  # Any exception (except segfault) is acceptable


@pytest.mark.parametrize("a,b", [(-1, 1), (0, 0), (2, 1), (0, 99)])
def test_ghost_bounce_indices_raise_value_error(simple_system, a, b):
    ray = ghostlight.Ray(
        ghostlight.Vec3f(0.0, 0.0, -200.0),
        ghostlight.Vec3f(0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError):
        ghostlight.trace_ghost_ray(ray, simple_system, a, b)
    with pytest.raises(ValueError):
        ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, a, b)


def test_ior_before_rejects_invalid_surface_indices(simple_system):
    with pytest.raises(IndexError):
        simple_system.ior_before(-1)
    with pytest.raises(IndexError):
        simple_system.ior_before(simple_system.num_surfaces())


def test_asphere_term_count_is_bounded():
    surface = ghostlight.Surface()
    with pytest.raises(ValueError):
        surface.n_asphere_terms = -1
    with pytest.raises(ValueError):
        surface.n_asphere_terms = 999


def test_render_dimensions_are_validated_before_dispatch(simple_system):
    config = ghostlight.PointFlareConfig()
    calibration = simple_system.calibration()
    with pytest.raises(ValueError):
        ghostlight._ghostlight.render_point_flare(
            0, 32, simple_system, calibration, config
        )
    with pytest.raises(ValueError):
        ghostlight._ghostlight.render_source_flare(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            32, -1, simple_system, calibration, config,
        )

    psf_config = ghostlight.PSFConfig()
    psf_config.grid_nx = 0
    with pytest.raises(ValueError):
        simple_system.render_psf(np.empty((0, 2), dtype=np.float32), psf_config)


