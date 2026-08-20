"""Tests for circle / polygon aperture shapes and the aspect-ratio knob.

Covers the Surface POD fields, the parser
(`"shape": "circular"` / `"polygon"` modifiers), and the geometry tables
documented in the spec.
"""

import json
import math
import pathlib
import tempfile
import pytest
import ghostlight


_D_LINE = 587.56


# ---------------------------------------------------------------------------
# Surface defaults
# ---------------------------------------------------------------------------

def test_surface_default_aperture_shape_is_circle():
    s = ghostlight.Surface()
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE


def test_surface_default_aperture_blades_is_zero():
    s = ghostlight.Surface()
    assert s.aperture_blades == 0


def test_surface_default_aperture_rotation_is_zero():
    s = ghostlight.Surface()
    assert s.aperture_rotation_rad == 0.0


def test_surface_default_aperture_aspect_is_one():
    s = ghostlight.Surface()
    assert s.aperture_aspect == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Helpers for trace-based geometry probing
# ---------------------------------------------------------------------------

def _stop_only_system(*,
                      shape: int = ghostlight.ApertureShape.CIRCLE,
                      blades: int = 0,
                      rotation_rad: float = 0.0,
                      aspect: float = 1.0,
                      semi_aperture: float = 10.0) -> ghostlight.OpticalSystem:
    """Single-surface system: flat aperture stop at z = 0.

    A ray fired parallel to +Z from z = -10 with (x, y) origin will hit the
    plane at the same (x, y); the only test that determines OK vs VIGNETTED
    is check_aperture().
    """
    sys = ghostlight.OpticalSystem()
    sys.name = "stop_only"

    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.abbe_v = 0.0
    stop.semi_aperture = semi_aperture
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(shape)
    stop.aperture_blades = blades
    stop.aperture_rotation_rad = rotation_rad
    stop.aperture_aspect = aspect
    sys.surfaces.append(stop)

    sys.finalize()
    return sys


def _hit_status(system: ghostlight.OpticalSystem, x: float, y: float) -> ghostlight.TraceStatus:
    """Trace a parallel +Z ray through `system` from (x, y, -10) and return the
    final status — OK if the ray made it past the aperture, VIGNETTED otherwise."""
    ray = ghostlight.Ray(ghostlight.Vec3f(x, y, -10.0), ghostlight.Vec3f(0.0, 0.0, 1.0), _D_LINE)
    return ghostlight.trace_primary_ray(ray, system).status


# ---------------------------------------------------------------------------
# Circle aperture, aspect = 1.0
# ---------------------------------------------------------------------------

def test_circle_inside_passes():
    sys = _stop_only_system(semi_aperture=10.0)
    assert _hit_status(sys, 0.0, 0.0) == ghostlight.TraceStatus.OK


def test_circle_outside_vignetted():
    sys = _stop_only_system(semi_aperture=10.0)
    assert _hit_status(sys, 11.0, 0.0) == ghostlight.TraceStatus.VIGNETTED


# ---------------------------------------------------------------------------
# Hexagon aperture, aspect = 1.0
# ---------------------------------------------------------------------------

def test_hex_vertex_at_radius_9_passes():
    """Vertex direction (angle 0°): boundary is at semi_aperture = 10."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, semi_aperture=10.0)
    assert _hit_status(sys, 9.0, 0.0) == ghostlight.TraceStatus.OK


def test_hex_edge_midpoint_inside_passes():
    """Edge midpoint direction (angle 90°): apothem = 10 * cos(30°) ≈ 8.66."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, semi_aperture=10.0)
    assert _hit_status(sys, 0.0, 8.0) == ghostlight.TraceStatus.OK


def test_hex_edge_midpoint_outside_vignetted():
    """At y = 9.0 we're past the apothem (8.66) but inside the bounding circle."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, semi_aperture=10.0)
    assert _hit_status(sys, 0.0, 9.0) == ghostlight.TraceStatus.VIGNETTED


def test_hex_beyond_bounding_circle_vignetted():
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, semi_aperture=10.0)
    assert _hit_status(sys, 0.0, 11.0) == ghostlight.TraceStatus.VIGNETTED


def test_hex_rotation_swaps_vertex_and_edge_axis():
    """With rotation = π/6 (= 30°), vertex direction is +Y and edge midpoint
    direction is +X — so the same y = 9.0 point that was VIGNETTED becomes OK
    (and x = 9.0 stays OK), while the y = 8.0 / x = 8.0 picture inverts."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, rotation_rad=math.pi / 6.0,
                            semi_aperture=10.0)
    # Vertex direction is now +Y: y = 9.0 should be inside.
    assert _hit_status(sys, 0.0, 9.0) == ghostlight.TraceStatus.OK
    # Edge midpoint direction is now +X: x = 9.0 should be outside the apothem.
    assert _hit_status(sys, 9.0, 0.0) == ghostlight.TraceStatus.VIGNETTED


# ---------------------------------------------------------------------------
# Hexagon aperture, aspect = 2.0
# ---------------------------------------------------------------------------

def test_hex_aspect2_vertex_at_x_18_passes():
    """X is stretched 2x: a vertex 9 units in canonical space lands at x = 18."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, aspect=2.0, semi_aperture=10.0)
    assert _hit_status(sys, 18.0, 0.0) == ghostlight.TraceStatus.OK


def test_hex_aspect2_beyond_x_bound_vignetted():
    """X half-axis = semi_aperture * aspect = 20; x = 21 is outside."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, aspect=2.0, semi_aperture=10.0)
    assert _hit_status(sys, 21.0, 0.0) == ghostlight.TraceStatus.VIGNETTED


def test_hex_aspect2_y_unchanged():
    """Y is not stretched: y = 8 still passes, y = 9 is still past the apothem."""
    sys = _stop_only_system(shape=ghostlight.ApertureShape.POLYGON,
                            blades=6, aspect=2.0, semi_aperture=10.0)
    assert _hit_status(sys, 0.0, 8.0) == ghostlight.TraceStatus.OK
    assert _hit_status(sys, 0.0, 9.0) == ghostlight.TraceStatus.VIGNETTED


# ---------------------------------------------------------------------------
# Parser round-trip
# ---------------------------------------------------------------------------

# Minimal V1 doublet template — same skeleton as example_doublet.lens but
# stripped down so we can mutate one surface's modifiers in-place per test.
def _doublet_doc_with_modifier(stop_modifier: dict) -> dict:
    return {
        "format": "ghostlight-optical",
        "version": {"major": 1, "minor": 0},
        "metadata": {"name": "Aperture test", "focal_length_mm": 50.0},
        "glass_catalogue": {
            "N-BK7": {
                "name": "N-BK7",
                "dispersion": {
                    "model": "sellmeier",
                    "B": [1.03961212, 0.23179234, 1.01046945],
                    "C": [0.00600069867, 0.02001791440, 103.560653],
                },
            },
        },
        "optical_system": [
            {
                "type": "element",
                "name": "Front",
                "transform": {"position": {"x": 0, "y": 0, "z": 0}},
                "materials": [{"glass": "N-BK7"}],
                "surfaces": [
                    {"semi_aperture": 25.0, "thickness": 4.0,
                     "form": {"type": "sphere", "radius": 50.0}},
                    {"semi_aperture": 25.0,
                     "form": {"type": "sphere", "radius": -50.0}},
                ],
            },
            {
                "type": "element",
                "name": "Stop",
                "transform": {"position": {"x": 0, "y": 0, "z": 10.0}},
                "materials": [],
                "surfaces": [
                    {
                        "semi_aperture": 10.0,
                        "is_stop": True,
                        "form": {"type": "sphere", "radius": 0.0},
                        "modifiers": [stop_modifier],
                    },
                ],
            },
        ],
    }


def _write_and_load(doc: dict) -> ghostlight.OpticalSystem:
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w",
                                     delete=False) as f:
        json.dump(doc, f)
        path = f.name
    return ghostlight.OpticalSystem.load(path)


def _find_stop_surface(lens: ghostlight.OpticalSystem) -> ghostlight.Surface:
    for s in lens.surfaces:
        if s.is_stop:
            return s
    raise AssertionError("no stop surface")


def test_parse_polygon_modifier_round_trip():
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "polygon",
        "blades": 6,
        "rotation_deg": 45.0,
        "aperture_aspect": 1.5,
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_shape == ghostlight.ApertureShape.POLYGON
    assert s.aperture_blades == 6
    assert s.aperture_rotation_rad == pytest.approx(math.pi / 4.0, abs=1e-5)
    assert s.aperture_aspect == pytest.approx(1.5)


def test_parse_circular_modifier_with_aspect():
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "circular",
        "aperture_aspect": 2.0,
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE
    assert s.aperture_blades == 0
    assert s.aperture_aspect == pytest.approx(2.0)


def test_parse_polygon_bad_blade_count_falls_back_to_circle():
    """`blades: 2` is invalid — parser must warn and fall back to CIRCLE."""
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "polygon",
        "blades": 2,
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE


def test_parse_aspect_zero_clamps_to_one():
    """aperture_aspect ≤ 0 must clamp to 1.0 with a warning."""
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "circular",
        "aperture_aspect": 0.0,
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_aspect == pytest.approx(1.0)


def test_parse_unknown_shape_falls_back_to_circle():
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "octahedron",
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE


def test_parse_image_shape_falls_back_to_circle():
    doc = _doublet_doc_with_modifier({
        "type": "aperture",
        "shape": "image",
        "image_path": "nonexistent.png",
    })
    lens = _write_and_load(doc)
    s = _find_stop_surface(lens)
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE


# ---------------------------------------------------------------------------
# Cache-key invalidation
# ---------------------------------------------------------------------------

def test_lens_cache_invalidates_on_aperture_shape_change(simple_lens):
    """Mutating an aperture field on a surface must invalidate the cached
    calibration, so the next calibration() call rebuilds it."""
    cal1 = simple_lens.calibration()
    # Mutate the stop surface's aperture shape.
    for s in simple_lens.surfaces:
        if s.is_stop:
            s.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
            s.aperture_blades = 6
            break
    cal2 = simple_lens.calibration()
    assert cal1 is not cal2
