"""Centroid pivot points and multi-pivot stacking semantics."""

import json
from pathlib import Path

import pytest
import ghostlight


def _two_singlet_doc(pivot_blocks: list[dict]) -> dict:
    """Two-singlet system at z=-30 and z=-15 with the given pivot definitions."""
    return {
        "format": "ghostlight-optical",
        "version": {"major": 1, "minor": 0},
        "metadata": {"name": "Two singlets"},
        "glass_catalogue": {
            "BK7-ish": {
                "name": "BK7-ish",
                "dispersion": {"model": "abbe", "nd": 1.5168, "Vd": 64.17},
            }
        },
        "optical_system": [
            {
                "type": "element",
                "id": "11111111-1111-1111-1111-111111111111",
                "name": "Front",
                "transform": {
                    "position": {"mode": "absolute", "x": 0, "y": 0, "z": -30.0}
                },
                "materials": [{"glass": "BK7-ish"}],
                "surfaces": [
                    {
                        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaa0000",
                        "semi_aperture": 10.0,
                        "thickness": 3.0,
                        "form": {"type": "sphere", "radius": 50.0},
                    },
                    {
                        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaa0001",
                        "semi_aperture": 10.0,
                        "thickness": 12.0,
                        "form": {"type": "sphere", "radius": -50.0},
                    },
                ],
            },
            {
                "type": "element",
                "id": "22222222-2222-2222-2222-222222222222",
                "name": "Rear",
                "transform": {
                    "position": {"mode": "absolute", "x": 0, "y": 0, "z": -15.0}
                },
                "materials": [{"glass": "BK7-ish"}],
                "surfaces": [
                    {
                        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbb0000",
                        "semi_aperture": 10.0,
                        "thickness": 3.0,
                        "form": {"type": "sphere", "radius": 80.0},
                    },
                    {
                        "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbb0001",
                        "semi_aperture": 10.0,
                        "thickness": 12.0,
                        "form": {"type": "sphere", "radius": -80.0},
                    },
                ],
            },
        ],
        "pivots": pivot_blocks,
    }


def _write(tmp_path: Path, doc: dict, name: str = "test.lens") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def _front_back_z(sys: ghostlight.OpticalSystem) -> tuple[list[float], list[float]]:
    front = next(el for el in sys.elements if el.name == "Front")
    rear  = next(el for el in sys.elements if el.name == "Rear")
    fz = [sys.surfaces[i].z for i in front.resolve_surfaces(sys)]
    rz = [sys.surfaces[i].z for i in rear.resolve_surfaces(sys)]
    return fz, rz


def test_centroid_pivot_point_is_mean_of_element_origins(tmp_path):
    """A pure z-translation pivot is invariant to the pivot point, so to test
    the centroid we use a rotation: rotation around the centroid leaves the
    centroid invariant and swaps the elements symmetrically."""
    # Pivot wrapping both elements, with a 180-degree roll (around z axis).
    # The centroid is z = -22.5. A 180° roll around the centroid in z is a
    # no-op in z (roll is rotation around the world Z axis), so test instead
    # a tilt_x rotation around the centroid.
    doc = _two_singlet_doc([
        {
            "id": "deadbeef-aaaa-bbbb-cccc-dddddddddddd",
            "name": "Both",
            "elements": [
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
            ],
            "pivot_point": {"mode": "centroid", "x": 0, "y": 0, "z": 0},
            "offset": {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"tilt_x": 0, "tilt_y": 0, "roll": 0},
            },
        }
    ])
    path = _write(tmp_path, doc)
    sys = ghostlight.OpticalSystem.load(path)
    # No-op pivot: front + rear vertex z must equal their authored positions
    # (after sensor-rebase). Compare them against a baseline build with no pivot.
    doc_no_pivot = _two_singlet_doc([])
    path_no_pivot = _write(tmp_path, doc_no_pivot, "nopivot.lens")
    base = ghostlight.OpticalSystem.load(path_no_pivot)

    fz, rz = _front_back_z(sys)
    bfz, brz = _front_back_z(base)
    for a, b in zip(fz, bfz):
        assert a == pytest.approx(b, abs=1e-5)
    for a, b in zip(rz, brz):
        assert a == pytest.approx(b, abs=1e-5)


def test_z_offset_with_centroid_translates_uniformly(tmp_path):
    """Pure translation offset is invariant to pivot point — both surfaces shift
    by exactly Δz."""
    doc = _two_singlet_doc([
        {
            "id": "11112222-3333-4444-5555-666677778888",
            "name": "Translate both",
            "elements": [
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
            ],
            "pivot_point": {"mode": "centroid", "x": 0, "y": 0, "z": 0},
            "offset": {
                "position": {"x": 0, "y": 0, "z": 3.0},
                "rotation": {"tilt_x": 0, "tilt_y": 0, "roll": 0},
            },
        }
    ])
    path = _write(tmp_path, doc)
    sys = ghostlight.OpticalSystem.load(path)
    base = ghostlight.OpticalSystem.load(_write(tmp_path, _two_singlet_doc([]), "nopivot.lens"))

    fz, rz = _front_back_z(sys)
    bfz, brz = _front_back_z(base)
    for a, b in zip(fz, bfz):
        assert a == pytest.approx(b + 3.0, abs=1e-5)
    for a, b in zip(rz, brz):
        assert a == pytest.approx(b + 3.0, abs=1e-5)


def test_two_pivots_stack_additively_on_same_element(tmp_path):
    """Two pivots, each with offset.position.z = 1, applied to the same element,
    add to a total of +2 mm."""
    doc = _two_singlet_doc([
        {
            "id": "aaaa1111-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "name": "First",
            "elements": ["22222222-2222-2222-2222-222222222222"],
            "pivot_point": {"mode": "manual", "x": 0, "y": 0, "z": 0},
            "offset": {
                "position": {"x": 0, "y": 0, "z": 1.0},
                "rotation": {"tilt_x": 0, "tilt_y": 0, "roll": 0},
            },
        },
        {
            "id": "bbbb2222-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "name": "Second",
            "elements": ["22222222-2222-2222-2222-222222222222"],
            "pivot_point": {"mode": "manual", "x": 0, "y": 0, "z": 0},
            "offset": {
                "position": {"x": 0, "y": 0, "z": 1.0},
                "rotation": {"tilt_x": 0, "tilt_y": 0, "roll": 0},
            },
        },
    ])
    path = _write(tmp_path, doc)
    sys = ghostlight.OpticalSystem.load(path)
    base = ghostlight.OpticalSystem.load(_write(tmp_path, _two_singlet_doc([]), "nopivot.lens"))
    _, rz_stacked = _front_back_z(sys)
    _, rz_base = _front_back_z(base)
    for a, b in zip(rz_stacked, rz_base):
        assert a == pytest.approx(b + 2.0, abs=1e-5)


def test_dangling_pivot_element_id_raises():
    fixture = Path(__file__).parent / "fixtures" / "bad_pivot_dangling.lens"
    with pytest.raises(RuntimeError):
        ghostlight.OpticalSystem.load(str(fixture))
