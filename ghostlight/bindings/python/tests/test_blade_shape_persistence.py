"""Blade-shape controls through the loader, the writer and the trace.

Geometry itself is pinned in test_aperture_profile.py; this file covers the
round trip (file -> Surface -> file), the clamps and their warnings, and that
the deformed silhouette is what the ray trace actually applies.
"""
import json
import math
import pathlib
import tempfile

import pytest

import ghostlight

_D_LINE = 587.56


def _doc(stop_modifier: dict | None) -> dict:
    """Minimal two-element system whose stop carries ``stop_modifier``."""
    surface = {
        "semi_aperture": 10.0,
        "is_stop": True,
        "form": {"type": "sphere", "radius": 0.0},
    }
    if stop_modifier is not None:
        surface["modifiers"] = [stop_modifier]
    return {
        "format": "ghostlight-optical",
        "version": {"major": 1, "minor": 0},
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
                "name": "Singlet",
                "transform": {"position": {"x": 0, "y": 0, "z": -20.0}},
                "materials": [{"glass": "N-BK7", "nd": 1.5168, "vd": 64.17}],
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
                "surfaces": [surface],
            },
        ],
    }


def _load(doc: dict) -> ghostlight.OpticalSystem:
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w",
                                     delete=False) as f:
        json.dump(doc, f)
        path = f.name
    return ghostlight.OpticalSystem.load(path)


def _stop(lens: ghostlight.OpticalSystem) -> ghostlight.Surface:
    for s in lens.surfaces:
        if s.is_stop:
            return s
    raise AssertionError("no stop surface")


def _polygon(**extra) -> dict:
    return {"type": "aperture", "shape": "polygon", "blades": 6, **extra}


def _saved_stop_modifier(lens: ghostlight.OpticalSystem) -> dict | None:
    """Round-trip ``lens`` through the writer and return the stop's aperture."""
    out = pathlib.Path(tempfile.mkdtemp()) / "saved.lens"
    lens.save(str(out))
    doc = json.loads(out.read_text(encoding="utf-8"))
    for element in doc["optical_system"]:
        for surface in element.get("surfaces", []):
            if not surface.get("is_stop"):
                continue
            for mod in surface.get("modifiers", []):
                if mod.get("type") == "aperture":
                    return mod
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_absent_controls_load_as_a_plain_polygon():
    s = _stop(_load(_doc(_polygon())))
    assert s.aperture_curvature == 0.0
    assert s.aperture_twist == 0.0
    assert s.aperture_notch_rad == 0.0
    assert s.aperture_notch_angle_rad == 0.0
    assert s.aperture_profile.plain == 1


def test_controls_load_with_degrees_converted():
    s = _stop(_load(_doc(_polygon(curvature=-0.9, twist=-0.8,
                                  notch_deg=35.8, notch_angle_deg=45.0))))
    assert s.aperture_curvature == pytest.approx(-0.9)
    assert s.aperture_twist == pytest.approx(-0.8)
    assert s.aperture_notch_rad == pytest.approx(math.radians(35.8), abs=1e-6)
    assert s.aperture_notch_angle_rad == pytest.approx(math.radians(45.0),
                                                       abs=1e-6)
    assert s.aperture_profile.deformed()


def test_load_builds_the_derived_block():
    """sync_aperture_profiles() runs on load — no consumer sees a stale block."""
    s = _stop(_load(_doc(_polygon(blades=5, curvature=-0.9, twist=-0.8,
                                  notch_deg=35.8, notch_angle_deg=45.0))))
    assert s.aperture_profile.blades == 5
    assert s.aperture_profile.r_w == pytest.approx(0.4086, abs=5e-5)
    assert s.aperture_profile.area_frac == pytest.approx(0.3230, abs=5e-5)


@pytest.mark.parametrize("key,given,want", [
    ("curvature", 2.5, 1.0),
    ("curvature", -4.0, -1.0),
    ("twist", 1.5, 1.0),
    ("notch_deg", 90.0, 45.0),
    ("notch_deg", -90.0, -45.0),
    ("notch_angle_deg", 70.0, 45.0),
    ("notch_angle_deg", -10.0, 0.0),
])
def test_out_of_range_controls_clamp_and_warn(key, given, want, capfd):
    s = _stop(_load(_doc(_polygon(**{key: given}))))
    got = {
        "curvature": lambda: s.aperture_curvature,
        "twist": lambda: s.aperture_twist,
        "notch_deg": lambda: math.degrees(s.aperture_notch_rad),
        "notch_angle_deg": lambda: math.degrees(s.aperture_notch_angle_rad),
    }[key]()
    assert got == pytest.approx(want, abs=1e-5)
    assert key in capfd.readouterr().err


def test_a_later_circular_modifier_clears_blade_shape():
    """Last modifier wins, and it must not inherit the polygon's blade shape."""
    doc = _doc(_polygon(curvature=-0.9))
    stop = doc["optical_system"][1]["surfaces"][0]
    stop["modifiers"].append({"type": "aperture", "shape": "circular"})
    s = _stop(_load(doc))
    assert s.aperture_shape == ghostlight.ApertureShape.CIRCLE
    assert s.aperture_curvature == 0.0
    assert s.aperture_profile.blades == 0


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_writer_omits_controls_at_default():
    """A plain polygon has to save as the file it was loaded from."""
    mod = _saved_stop_modifier(_load(_doc(_polygon(rotation_deg=30.0))))
    assert mod == {"type": "aperture", "shape": "polygon", "blades": 6,
                   "rotation_deg": pytest.approx(30.0, abs=1e-4)}


def test_writer_emits_authored_controls_in_degrees():
    mod = _saved_stop_modifier(_load(_doc(
        _polygon(curvature=-0.9, twist=-0.8,
                 notch_deg=35.8, notch_angle_deg=45.0))))
    assert mod["curvature"] == pytest.approx(-0.9, abs=1e-6)
    assert mod["twist"] == pytest.approx(-0.8, abs=1e-6)
    assert mod["notch_deg"] == pytest.approx(35.8, abs=1e-4)
    assert mod["notch_angle_deg"] == pytest.approx(45.0, abs=1e-4)


def test_save_load_save_is_stable():
    first = _load(_doc(_polygon(blades=5, curvature=-0.9, twist=-0.8,
                                notch_deg=35.8, notch_angle_deg=45.0)))
    mod_a = _saved_stop_modifier(first)
    out = pathlib.Path(tempfile.mkdtemp()) / "once.lens"
    first.save(str(out))
    mod_b = _saved_stop_modifier(ghostlight.OpticalSystem.load(str(out)))
    assert mod_a == mod_b


# ---------------------------------------------------------------------------
# The trace applies it
# ---------------------------------------------------------------------------

def _status(lens, x, y):
    ray = ghostlight.Ray(ghostlight.Vec3f(x, y, -400.0), ghostlight.Vec3f(0.0, 0.0, 1.0), _D_LINE)
    return ghostlight.trace_primary_ray(ray, lens).status


def _stop_only(**controls) -> ghostlight.OpticalSystem:
    system = ghostlight.OpticalSystem()
    system.name = "stop_only"
    stop = ghostlight.Surface()
    stop.radius = 0.0
    stop.thickness = 0.0
    stop.ior = 1.0
    stop.semi_aperture = 10.0
    stop.is_stop = True
    stop.disp_model = ghostlight.DispersionModel.AIR
    stop.aperture_shape = int(ghostlight.ApertureShape.POLYGON)
    stop.aperture_blades = 6
    for k, v in controls.items():
        setattr(stop, k, v)
    system.surfaces.append(stop)
    system.finalize()
    return system


def test_full_curvature_opens_the_polygon_to_a_circle():
    """A hexagon rejects (0, 9.0); at curvature +1 the edge reaches the tips."""
    assert _status(_stop_only(), 0.0, 9.0) == ghostlight.TraceStatus.VIGNETTED
    assert _status(_stop_only(aperture_curvature=1.0),
                   0.0, 9.0) == ghostlight.TraceStatus.OK


def test_negative_curvature_scoops_the_edge_inward():
    """Inside a plain hexagon (r < apothem 8.66) but outside a scooped one."""
    assert _status(_stop_only(), 0.0, 8.0) == ghostlight.TraceStatus.OK
    assert _status(_stop_only(aperture_curvature=-1.0),
                   0.0, 8.0) == ghostlight.TraceStatus.VIGNETTED


def test_twist_makes_the_blade_handed():
    """Mirror-image points agree on a plain hexagon and disagree once twisted."""
    plain = _stop_only()
    twisted = _stop_only(aperture_curvature=-0.6, aperture_twist=0.8)
    profile = twisted.surfaces[0].aperture_profile
    # Probe where the two mirror images disagree most, at a radius between the
    # two boundaries: a handed blade has to accept one side and reject the other.
    theta = max((2.0 * math.pi * i / 720 for i in range(720)),
                key=lambda t: abs(profile.radius_at(t) - profile.radius_at(-t)))
    r = 10.0 * 0.5 * (profile.radius_at(theta) + profile.radius_at(-theta))
    x, y = r * math.cos(theta), r * math.sin(theta)
    assert _status(plain, x, y) == _status(plain, x, -y)
    assert _status(twisted, x, y) != _status(twisted, x, -y)


def test_finalize_refreshes_programmatic_edits():
    """A Surface built in Python and appended still traces its own shape."""
    system = _stop_only()
    system.surfaces[0].aperture_curvature = 1.0
    system.finalize()
    assert _status(system, 0.0, 9.0) == ghostlight.TraceStatus.OK
