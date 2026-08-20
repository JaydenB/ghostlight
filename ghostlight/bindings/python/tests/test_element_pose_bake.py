"""Per-element pivot (centre of rotation) + the Python pose-bake mirror.

Two things are under test here:

1. ``transform.pivot`` — the element's own centre of rotation. Absent or
   all-zero must reproduce the historical "rotate about the front vertex"
   behaviour exactly; non-zero must move the rotation centre and nothing else.
2. ``ghostlight.pose.bake_system_poses`` — the Python re-implementation of the
   loader's pose bake that editors use for live feedback. It must agree with a
   real C++ ``reload()``, or the viewport silently drifts from what a save
   would produce.

Test (2) is the important one: two implementations of the same maths in two
languages only stay in step if something checks.
"""

import copy
import json
import math
from pathlib import Path

import pytest

import ghostlight
from ghostlight.pose import bake_system_poses, element_world_pivot, make_rotation


FIXTURE_DIR = Path(__file__).parent / "fixtures"
DOUBLET_WITH_PIVOT = FIXTURE_DIR / "doublet_with_pivot.lens"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_doc(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write(tmp_path: Path, doc: dict, name: str = "case.lens") -> str:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return str(p)


def _poses(system) -> list[tuple]:
    """(decenter_x, decenter_y, z, rot[9]) for every surface, in trace order."""
    out = []
    for s in system.surfaces:
        out.append((
            float(s.decenter_x),
            float(s.decenter_y),
            float(s.z),
            tuple(float(v) for v in s.rot),
        ))
    return out


def _assert_poses_close(a, b, tol=2e-4):
    assert len(a) == len(b), "surface count differs"
    for i, (pa, pb) in enumerate(zip(a, b)):
        assert pa[0] == pytest.approx(pb[0], abs=tol), f"surface {i} decenter_x"
        assert pa[1] == pytest.approx(pb[1], abs=tol), f"surface {i} decenter_y"
        assert pa[2] == pytest.approx(pb[2], abs=tol), f"surface {i} z"
        for k in range(9):
            assert pa[3][k] == pytest.approx(pb[3][k], abs=tol), f"surface {i} rot[{k}]"


def _set_transform(doc: dict, element_name: str, **blocks) -> dict:
    """Merge transform sub-blocks into the named element. Returns the doc."""
    for entry in doc["optical_system"]:
        if entry.get("name") == element_name:
            xform = entry.setdefault("transform", {})
            for key, value in blocks.items():
                xform[key] = value
            return doc
    raise AssertionError(f"no element named {element_name!r}")


# ---------------------------------------------------------------------------
# 1. transform.pivot semantics
# ---------------------------------------------------------------------------

def test_zero_pivot_is_identical_to_no_pivot_block(tmp_path):
    """An all-zero pivot must be a no-op, byte-for-byte in the baked poses.

    This is the backward-compatibility guarantee: every .lens file written
    before the field existed behaves as if it carries a zero pivot.
    """
    base = _load_doc(DOUBLET_WITH_PIVOT)
    with_zero = _set_transform(
        copy.deepcopy(base), "Front Doublet",
        rotation={"tilt_x": 2.0, "tilt_y": -1.5, "roll": 10.0},
        pivot={"x": 0.0, "y": 0.0, "z": 0.0},
    )
    without = _set_transform(
        copy.deepcopy(base), "Front Doublet",
        rotation={"tilt_x": 2.0, "tilt_y": -1.5, "roll": 10.0},
    )

    a = ghostlight.OpticalSystem.load(_write(tmp_path, with_zero, "a.lens"))
    b = ghostlight.OpticalSystem.load(_write(tmp_path, without, "b.lens"))
    # Exact, not approximate — the loader's `piv_corr` stays at literal zero.
    assert _poses(a) == _poses(b)


def test_pivot_without_rotation_does_not_move_anything(tmp_path):
    """A centre of rotation with no rotation to apply must not translate."""
    base = _load_doc(DOUBLET_WITH_PIVOT)
    moved = _set_transform(
        copy.deepcopy(base), "Front Doublet",
        pivot={"x": 3.0, "y": -2.0, "z": 4.0},
    )
    a = ghostlight.OpticalSystem.load(_write(tmp_path, base, "base.lens"))
    b = ghostlight.OpticalSystem.load(_write(tmp_path, moved, "moved.lens"))
    assert _poses(a) == _poses(b)


def test_pivot_at_front_vertex_matches_default(tmp_path):
    """Pivot (0,0,0) *is* the front vertex — tilting about it is the default."""
    base = _load_doc(DOUBLET_WITH_PIVOT)
    tilted = _set_transform(
        copy.deepcopy(base), "Rear Singlet",
        rotation={"tilt_y": 5.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, tilted, "t.lens"))
    rear = next(el for el in system.elements if el.name == "Rear Singlet")
    idx = rear.resolve_surfaces(system)

    # Untilted reference for the front vertex position.
    ref = ghostlight.OpticalSystem.load(_write(tmp_path, base, "r.lens"))
    ref_idx = next(
        el for el in ref.elements if el.name == "Rear Singlet"
    ).resolve_surfaces(ref)

    s0, r0 = system.surfaces[idx[0]], ref.surfaces[ref_idx[0]]
    assert float(s0.decenter_x) == pytest.approx(float(r0.decenter_x), abs=1e-5)
    assert float(s0.z) == pytest.approx(float(r0.z), abs=1e-5)
    # ...while the rear surface has swung out laterally.
    s1 = system.surfaces[idx[1]]
    assert abs(float(s1.decenter_x)) > 0.1


def test_pivot_at_rear_vertex_pins_the_rear_surface(tmp_path):
    """Put the pivot on the back vertex and the *back* vertex is what holds still.

    The clearest possible statement of what the field does: the same tilt,
    with the rotation centre moved from front to back, swaps which end of the
    element stays put.
    """
    base = _load_doc(DOUBLET_WITH_PIVOT)
    # Rear Singlet is 4.0 thick between its two surfaces.
    tilted = _set_transform(
        copy.deepcopy(base), "Rear Singlet",
        rotation={"tilt_y": 5.0},
        pivot={"x": 0.0, "y": 0.0, "z": 4.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, tilted, "t.lens"))
    ref = ghostlight.OpticalSystem.load(_write(tmp_path, base, "r.lens"))

    idx = next(
        el for el in system.elements if el.name == "Rear Singlet"
    ).resolve_surfaces(system)
    ref_idx = next(
        el for el in ref.elements if el.name == "Rear Singlet"
    ).resolve_surfaces(ref)

    # Back vertex pinned...
    s1, r1 = system.surfaces[idx[1]], ref.surfaces[ref_idx[1]]
    assert float(s1.decenter_x) == pytest.approx(float(r1.decenter_x), abs=1e-4)
    assert float(s1.decenter_y) == pytest.approx(float(r1.decenter_y), abs=1e-4)
    assert float(s1.z) == pytest.approx(float(r1.z), abs=1e-4)
    # ...front vertex swung out instead.
    s0 = system.surfaces[idx[0]]
    assert abs(float(s0.decenter_x)) > 0.1


def test_pivot_does_not_disturb_the_thickness_chain(tmp_path):
    """Nominal axial bookkeeping must ignore tilt and pivot entirely.

    Air gaps and the sensor rebase are derived from the untilted layout on
    purpose. If `piv_corr` ever leaks into `Transform.pos`, this is what
    catches it.
    """
    base = _load_doc(DOUBLET_WITH_PIVOT)
    ref = ghostlight.OpticalSystem.load(_write(tmp_path, base, "r.lens"))
    ref_thick = [float(s.thickness) for s in ref.surfaces]

    tilted = _set_transform(
        copy.deepcopy(base), "Front Doublet",
        rotation={"tilt_x": 4.0, "tilt_y": 3.0},
        pivot={"x": 1.0, "y": 2.0, "z": 6.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, tilted, "t.lens"))
    got_thick = [float(s.thickness) for s in system.surfaces]

    assert got_thick == pytest.approx(ref_thick, abs=1e-5)
    # And the chain still ends at the sensor.
    assert float(system.surfaces[-1].z) + float(system.surfaces[-1].thickness) \
        == pytest.approx(0.0, abs=1e-4)


def test_pivot_round_trips_through_the_writer(tmp_path):
    """Element.pivot must survive save → load, or undo destroys it."""
    base = _load_doc(DOUBLET_WITH_PIVOT)
    doc = _set_transform(
        copy.deepcopy(base), "Rear Singlet",
        rotation={"tilt_y": 3.0},
        pivot={"x": 0.5, "y": -1.25, "z": 2.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "in.lens"))
    rear = next(el for el in system.elements if el.name == "Rear Singlet")
    assert rear.pivot == (0.5, -1.25, 2.0)

    out = ghostlight.build_optical_system_doc(
        system=system,
        metadata=system._raw_metadata,
        glass_catalogue=system._raw_glass_catalogue,
    )
    reloaded = ghostlight.OpticalSystem.load(_write(tmp_path, out, "out.lens"))
    rear2 = next(el for el in reloaded.elements if el.name == "Rear Singlet")
    assert rear2.pivot == pytest.approx((0.5, -1.25, 2.0), abs=1e-5)
    _assert_poses_close(_poses(system), _poses(reloaded))


def test_writer_omits_the_pivot_block_when_zero(tmp_path):
    """Files with no pivot must not grow one on save."""
    system = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    doc = ghostlight.build_optical_system_doc(
        system=system,
        metadata=system._raw_metadata,
        glass_catalogue=system._raw_glass_catalogue,
    )
    for entry in doc["optical_system"]:
        assert "pivot" not in entry.get("transform", {})
    assert doc["version"] == {"major": ghostlight.LENS_FORMAT_MAJOR,
                              "minor": ghostlight.LENS_FORMAT_MINOR}


def test_a_pivot_round_trips_without_changing_the_version(tmp_path):
    """A non-zero element pivot must not change the emitted format version.

    That content-derived version is gone: the pivot simply round-trips, and
    the version is the same one every other file carries.
    """
    doc = _set_transform(
        copy.deepcopy(_load_doc(DOUBLET_WITH_PIVOT)), "Rear Singlet",
        pivot={"x": 0.0, "y": 0.0, "z": 2.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "p.lens"))
    out = ghostlight.build_optical_system_doc(
        system=system,
        metadata=system._raw_metadata,
        glass_catalogue=system._raw_glass_catalogue,
    )
    assert out["version"] == {"major": ghostlight.LENS_FORMAT_MAJOR,
                              "minor": ghostlight.LENS_FORMAT_MINOR}
    entry = next(e for e in out["optical_system"] if e["name"] == "Rear Singlet")
    assert entry["transform"]["pivot"]["z"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 2. The Python bake must agree with the C++ loader
# ---------------------------------------------------------------------------

_BAKE_CASES = [
    pytest.param({}, id="identity"),
    pytest.param({"position": {"mode": "absolute", "x": 2.0, "y": -1.0, "z": -34.0}},
                 id="decenter"),
    pytest.param({"rotation": {"tilt_y": 3.0}}, id="tilt_y"),
    pytest.param({"rotation": {"tilt_x": -2.5, "tilt_y": 4.0, "roll": 30.0}},
                 id="tilt_xyz"),
    pytest.param({"rotation": {"tilt_y": 5.0}, "pivot": {"x": 0, "y": 0, "z": 4.0}},
                 id="pivot_rear"),
    pytest.param({"position": {"mode": "absolute", "x": 1.5, "y": 0.75, "z": -34.0},
                  "rotation": {"tilt_x": 2.0, "tilt_y": -3.0, "roll": 12.0},
                  "pivot": {"x": -0.5, "y": 1.0, "z": 2.0}},
                 id="everything"),
]


@pytest.mark.parametrize("blocks", _BAKE_CASES)
def test_python_bake_matches_cpp_loader(tmp_path, blocks):
    """The whole point of the mirror: same input, same surface poses.

    Loads the case through C++, then re-bakes the *same* system in Python and
    checks nothing moved. If the two ever disagree — Euler order, composition
    order, the sensor rebase — this fails.
    """
    doc = copy.deepcopy(_load_doc(DOUBLET_WITH_PIVOT))
    if blocks:
        _set_transform(doc, "Rear Singlet", **blocks)
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "case.lens"))

    expected = _poses(system)
    assert bake_system_poses(system) is True
    _assert_poses_close(_poses(system), expected)


def test_python_bake_matches_cpp_after_an_in_memory_edit(tmp_path):
    """The live-editing path: mutate Element, bake, and compare to a reload.

    This is what the designer actually does — change the dataclass, bake for
    instant feedback, and rely on the save/undo round-trip agreeing later.
    """
    system = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    rear = next(el for el in system.elements if el.name == "Rear Singlet")

    x, y, z = rear.position
    rear.position = (x + 1.75, y - 0.5, z)
    rear.rotation_euler_deg = (1.0, -4.0, 15.0)
    rear.pivot = (0.25, -0.75, 2.0)

    assert bake_system_poses(system) is True
    live = _poses(system)

    # Now take the authoritative route: write it out and let C++ do the bake.
    doc = ghostlight.build_optical_system_doc(
        system=system,
        metadata=system._raw_metadata,
        glass_catalogue=system._raw_glass_catalogue,
    )
    reloaded = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "rt.lens"))

    _assert_poses_close(live, _poses(reloaded))


def test_python_bake_reproduces_the_rig(tmp_path):
    """The mirror must also compose the group-level pivots[] rig."""
    doc = copy.deepcopy(_load_doc(DOUBLET_WITH_PIVOT))
    doc["pivots"][0]["offset"] = {
        "position": {"x": 0.5, "y": -0.25, "z": 1.5},
        "rotation": {"tilt_x": 1.0, "tilt_y": 2.0, "roll": 0.0},
    }
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "rig.lens"))
    expected = _poses(system)
    assert bake_system_poses(system) is True
    _assert_poses_close(_poses(system), expected)


def test_bake_refuses_a_structurally_broken_system():
    """A stale UUID must abort the whole bake, not repose half the system."""
    system = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    before = _poses(system)
    rear = next(el for el in system.elements if el.name == "Rear Singlet")
    rear.surface_ids = ["00000000-0000-0000-0000-000000000000"]

    assert bake_system_poses(system) is False
    assert _poses(system) == before


# ---------------------------------------------------------------------------
# 3. element_world_pivot — what the viewport marker draws
# ---------------------------------------------------------------------------

def test_world_pivot_is_none_when_zero():
    system = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    for el in system.elements:
        assert element_world_pivot(system, el) is None


def test_world_pivot_sits_on_the_axis_for_an_untilted_element(tmp_path):
    """With no rotation, a pivot of (0,0,d) is d past the front vertex."""
    doc = _set_transform(
        copy.deepcopy(_load_doc(DOUBLET_WITH_PIVOT)), "Rear Singlet",
        pivot={"x": 0.0, "y": 0.0, "z": 4.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "p.lens"))
    rear = next(el for el in system.elements if el.name == "Rear Singlet")
    idx = rear.resolve_surfaces(system)
    s0 = system.surfaces[idx[0]]

    got = element_world_pivot(system, rear)
    assert got is not None
    assert got[0] == pytest.approx(float(s0.decenter_x), abs=1e-5)
    assert got[1] == pytest.approx(float(s0.decenter_y), abs=1e-5)
    assert got[2] == pytest.approx(float(s0.z) + 4.0, abs=1e-5)
    # It lands exactly on the rear vertex, which is 4.0 along.
    assert got[2] == pytest.approx(float(system.surfaces[idx[1]].z), abs=1e-4)


def test_world_pivot_is_the_fixed_point_of_the_rotation(tmp_path):
    """Defining property: tilting about the pivot must leave the pivot alone.

    Same element, same pivot, two different tilts — the marker must land in
    the same world position both times.
    """
    base = _load_doc(DOUBLET_WITH_PIVOT)
    pivot = {"x": 0.5, "y": -1.0, "z": 2.0}

    def world_pivot_for(rotation, name):
        doc = _set_transform(
            copy.deepcopy(base), "Rear Singlet", rotation=rotation, pivot=pivot,
        )
        system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, name))
        rear = next(el for el in system.elements if el.name == "Rear Singlet")
        return element_world_pivot(system, rear)

    a = world_pivot_for({"tilt_x": 0.0, "tilt_y": 0.0, "roll": 0.0}, "a.lens")
    b = world_pivot_for({"tilt_x": 3.0, "tilt_y": -6.0, "roll": 20.0}, "b.lens")
    assert a is not None and b is not None
    for axis in range(3):
        assert a[axis] == pytest.approx(b[axis], abs=1e-4), f"pivot moved on axis {axis}"


def test_make_rotation_matches_the_loader_convention():
    """R = Ry(tilt_y)·Rx(tilt_x)·Rz(roll), right-handed, row-major."""
    assert make_rotation(0, 0, 0) == pytest.approx(
        (1, 0, 0, 0, 1, 0, 0, 0, 1), abs=1e-12
    )
    # A pure tilt_x of 90° sends +Z to −Y (row-major: out[5] = −sin(tx)).
    r = make_rotation(90.0, 0.0, 0.0)
    assert r[5] == pytest.approx(-1.0, abs=1e-6)
    assert r[8] == pytest.approx(0.0, abs=1e-6)
    # A pure tilt_y of 90° sends +Z to +X (out[2] = sin(ty)).
    r = make_rotation(0.0, 90.0, 0.0)
    assert r[2] == pytest.approx(1.0, abs=1e-6)


def test_make_rotation_agrees_with_the_baked_surface_rot(tmp_path):
    """Cross-check the Python Euler build against what C++ actually baked."""
    doc = _set_transform(
        copy.deepcopy(_load_doc(DOUBLET_WITH_PIVOT)), "Rear Singlet",
        rotation={"tilt_x": 7.0, "tilt_y": -11.0, "roll": 23.0},
    )
    system = ghostlight.OpticalSystem.load(_write(tmp_path, doc, "e.lens"))
    rear = next(el for el in system.elements if el.name == "Rear Singlet")
    s0 = system.surfaces[rear.resolve_surfaces(system)[0]]
    expected = make_rotation(7.0, -11.0, 23.0)
    for k in range(9):
        assert float(s0.rot[k]) == pytest.approx(expected[k], abs=1e-5)
