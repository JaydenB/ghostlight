"""Tests for CPU ray tracing."""

import math

import pytest
import ghostlight


def _on_axis_primary_ray(z_start=-200.0, wavelength=587.56):
    origin = ghostlight.Vec3f(0.0, 0.0, z_start)
    direction = ghostlight.Vec3f(0.0, 0.0, 1.0)
    return ghostlight.Ray(origin, direction, wavelength)


def test_primary_ray_fast_path(simple_system):
    ray = _on_axis_primary_ray()
    result = ghostlight.trace_primary_ray(ray, simple_system)
    assert result.status == ghostlight.TraceStatus.OK
    assert result.weight > 0.0


def test_primary_ray_diagnostic_events(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    # On-axis paraxial ray should hit every surface
    assert path.result.status == ghostlight.TraceStatus.OK
    assert len(path.events) == simple_system.num_surfaces()


def test_primary_ray_diagnostic_no_reflected(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    # Primary path: no reflections
    assert all(not ev.reflected for ev in path.events)


def test_primary_ray_diagnostic_surface_indices(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    indices = [ev.surface_index for ev in path.events]
    assert indices == list(range(simple_system.num_surfaces()))


def test_ghost_ray_fast_path(simple_system):
    # Ghost between surface 0 and surface 1
    ray = _on_axis_primary_ray()
    result = ghostlight.trace_ghost_ray(ray, simple_system, 0, 1)
    # May be OK, vignetted, or TIR depending on geometry — just check it runs
    assert isinstance(result.status, ghostlight.TraceStatus)


def test_ghost_ray_diagnostic_two_reflections(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, 0, 1)
    reflected_count = sum(1 for ev in path.events if ev.reflected)
    # Ghost (a, b) must have exactly 2 reflecting events (at bounce_a and bounce_b)
    assert reflected_count == 2


def test_ghost_ray_diagnostic_vignette_result_defined(simple_system):
    """A vignetted ghost diagnostic trace returns a fully-defined result.

    Regression guard for A1: the diagnostic trace_ghost_ray overload's six
    early-exit paths must each assign `path_out.result` before returning, or
    Python read an indeterminate TraceResult.
    """
    # x = 20 mm is outside surface 0's 15 mm semi-aperture, so the ray vignettes
    # on the outbound leg at the first surface.
    ray = ghostlight.Ray(ghostlight.Vec3f(20.0, 0.0, -200.0), ghostlight.Vec3f(0.0, 0.0, 1.0), 587.56)
    path = ghostlight.trace_ghost_ray_diagnostic(ray, simple_system, 0, 1)

    # The result is defined, reports the vignette (not garbage), and mirrors
    # the last recorded event.
    assert len(path.events) > 0
    assert path.events[-1].status == ghostlight.TraceStatus.VIGNETTED
    assert path.result.status == ghostlight.TraceStatus.VIGNETTED
    assert 0.0 <= path.result.weight <= 1.0
    for c in (path.result.position.x, path.result.position.y, path.result.position.z):
        assert math.isfinite(c)


def test_primary_diagnostic_raypath_repr(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    assert "RayPath(" in repr(path)


def test_trace_event_repr(simple_system):
    ray = _on_axis_primary_ray()
    path = ghostlight.trace_primary_ray_diagnostic(ray, simple_system)
    assert len(path.events) > 0
    r = repr(path.events[0])
    assert "TraceEvent(" in r
