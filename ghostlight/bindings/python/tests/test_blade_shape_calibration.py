"""Pupil calibration on a bladed stop.

Two claims are under test. First, that a lens without a bladed stop calibrates
without running the bladed-stop azimuth sweep. Second,
that a bladed stop's entrance pupil is measured as a per-axis SUPPORT extent
rather than sampled on two arbitrary meridians, which is what stops the pattern
size and brightness from jumping while a blade control is scrubbed.
"""
import json
import math
import pathlib
import tempfile

import pytest

import ghostlight

LENSES = pathlib.Path(__file__).resolve().parents[4] / "lenses"
DOUBLE_GAUSS = LENSES / "DoubleGauss.lens"


def _bladed(blades=6, rotation_deg=0.0, **controls) -> ghostlight.OpticalSystem:
    """double_gauss with its stop switched to a bladed one."""
    doc = json.loads(DOUBLE_GAUSS.read_text(encoding="utf-8"))
    mod = {"type": "aperture", "shape": "polygon", "blades": blades}
    if rotation_deg:
        mod["rotation_deg"] = rotation_deg
    mod.update(controls)
    for element in doc["optical_system"]:
        for surface in element.get("surfaces", []):
            if surface.get("is_stop"):
                surface.setdefault("modifiers", []).append(mod)
    with tempfile.NamedTemporaryFile(suffix=".lens", mode="w",
                                     delete=False) as f:
        json.dump(doc, f)
        path = f.name
    return ghostlight.OpticalSystem.load(path)


def _stop(lens):
    return next(s for s in lens.surfaces if s.is_stop)


# ---------------------------------------------------------------------------
# Round stops are untouched
# ---------------------------------------------------------------------------

def test_circular_stop_keeps_its_calibrated_pupil():
    """Pinned from the pre-change build: the gated path must not perturb it."""
    calib = ghostlight.OpticalSystem.load(str(DOUBLE_GAUSS)).calibration()
    assert calib.entrance_pupil_semi_x == pytest.approx(17.61361, abs=1e-5)
    assert calib.entrance_pupil_semi_y == pytest.approx(17.61361, abs=1e-5)
    assert calib.f_number_x == pytest.approx(3.14414, abs=1e-5)
    assert calib.pupil_area_frac == 1.0


def test_circular_stop_reports_a_full_pupil_area():
    assert ghostlight.OpticalSystem.load(str(DOUBLE_GAUSS)).calibration() \
        .pupil_area_frac == 1.0


# ---------------------------------------------------------------------------
# Open area
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blades,want", [
    (3, 3.0 / (2.0 * math.pi) * math.sin(2.0 * math.pi / 3.0)),
    (6, 6.0 / (2.0 * math.pi) * math.sin(math.pi / 3.0)),
    (8, 8.0 / (2.0 * math.pi) * math.sin(math.pi / 4.0)),
])
def test_polygon_area_fraction_is_the_analytic_value(blades, want):
    lens = _bladed(blades=blades)
    assert lens.calibration().pupil_area_frac == pytest.approx(want, abs=1e-6)


def test_hexagon_dims_by_the_documented_amount():
    """The documented hexagon worked example, and what the starburst gain is
    multiplied by."""
    assert _bladed(blades=6).calibration().pupil_area_frac == pytest.approx(
        0.826993, abs=1e-6)


def test_area_fraction_tracks_the_stop_profile():
    lens = _bladed(blades=5, curvature=-0.9, twist=-0.8,
                   notch_deg=35.8, notch_angle_deg=45.0)
    assert lens.calibration().pupil_area_frac == pytest.approx(
        _stop(lens).aperture_profile.area_frac, abs=1e-7)


def test_full_curvature_restores_a_full_pupil():
    """At curvature +1 the stop is a circle again, so nothing should dim."""
    assert _bladed(blades=6, curvature=1.0).calibration() \
        .pupil_area_frac == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Support extent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("controls", [
    {},
    {"rotation_deg": 30.0},
    {"curvature": -0.9},
    {"curvature": -0.5, "twist": 0.9},
    {"curvature": -0.9, "twist": -0.8, "notch_deg": 35.8,
     "notch_angle_deg": 45.0},
])
def test_support_is_never_smaller_than_the_meridian_sample(controls):
    """The sweep includes the two original meridians, so it can only widen."""
    lens = _bladed(**controls)
    calib = lens.calibration()
    profile = _stop(lens).aperture_profile
    circular = ghostlight.OpticalSystem.load(str(DOUBLE_GAUSS)).calibration()
    for axis, theta in (("x", 0.0), ("y", math.pi / 2.0)):
        # What the old two-probe path would have measured on this meridian:
        # the stop's own radius there, capped by what the rest of the lens
        # admits.
        sampled = min(circular.entrance_pupil_semi_x
                      * profile.radius_at(theta) / 1.0,
                      circular.entrance_pupil_semi_x)
        assert getattr(calib, f"entrance_pupil_semi_{axis}") >= sampled - 1e-4


def test_rotating_a_hexagon_moves_f_number_by_the_geometric_factor():
    """A hexagon's X extent really does differ between vertex- and flat-up; the
    fix is that the number now reports the extent instead of one meridian."""
    upright = _bladed(blades=6).calibration()
    turned = _bladed(blades=6, rotation_deg=30.0).calibration()
    assert turned.f_number_x / upright.f_number_x == pytest.approx(
        1.0 / math.cos(math.pi / 6.0), rel=2e-3)
    assert turned.f_number_y / upright.f_number_y == pytest.approx(
        math.cos(math.pi / 6.0), rel=2e-3)


@pytest.mark.parametrize("control", ["curvature", "twist"])
def test_f_number_is_continuous_while_a_control_is_scrubbed(control):
    """The wobble this replaces: a sampled meridian sliding off a steep ramp
    stepped f-number between adjacent scrub frames."""
    values = []
    for i in range(21):
        v = -1.0 + 2.0 * i / 20.0
        kw = {control: v}
        if control == "twist":
            kw["curvature"] = -0.6
        values.append(_bladed(blades=5, **kw).calibration().f_number_x)
    steps = [abs(b - a) / a for a, b in zip(values, values[1:])]
    assert max(steps) < 0.05


def test_curvature_plus_one_matches_the_circular_pupil():
    """The stop is a circle there, so the sweep has to land on the round answer."""
    circular = ghostlight.OpticalSystem.load(str(DOUBLE_GAUSS)).calibration()
    curved = _bladed(blades=6, curvature=1.0).calibration()
    assert curved.entrance_pupil_semi_x == pytest.approx(
        circular.entrance_pupil_semi_x, rel=1e-3)
    assert curved.entrance_pupil_semi_y == pytest.approx(
        circular.entrance_pupil_semi_y, rel=1e-3)


def test_the_sweep_keeps_the_axes_independent():
    """Collapsing to one scalar would break anamorphic lenses; the split stays."""
    lens = _bladed(blades=6, rotation_deg=30.0)
    calib = lens.calibration()
    assert calib.entrance_pupil_semi_x != calib.entrance_pupil_semi_y
