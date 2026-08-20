"""Tests for the Element grouping recovered from .lens files."""

import pytest
import ghostlight


def test_from_lens_file_returns_three_elements(example_lens_path):
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    assert len(elements) == 3
    names = [e.name for e in elements]
    assert names == ["Front Doublet", "Aperture Stop", "Rear Singlet"]


def test_element_kinds(example_lens_path):
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    assert elements[0].kind == ghostlight.ElementKind.GLASS
    assert elements[1].kind == ghostlight.ElementKind.STOP
    assert elements[2].kind == ghostlight.ElementKind.GLASS


def test_front_doublet_surface_ids(example_lens_path):
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    front = elements[0]
    assert front.surface_ids == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "cccccccc-cccc-cccc-cccc-cccccccccccc",
    ]
    assert front.material_glasses == ["N-BK7", "SF5"]
    assert front.element_id == "11111111-1111-1111-1111-111111111111"


def test_stop_has_no_materials(example_lens_path):
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    stop = elements[1]
    assert stop.material_glasses == []
    assert len(stop.surface_ids) == 1
    assert stop.surface_ids[0] == "dddddddd-dddd-dddd-dddd-dddddddddddd"


def test_rear_singlet(example_lens_path):
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    rear = elements[2]
    assert rear.material_glasses == ["N-BK7"]
    assert rear.surface_ids == [
        "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
    ]
    assert rear.position == (0.0, 0.0, -34.0)


def test_resolve_surfaces_against_loaded_system(example_lens_path):
    lens = ghostlight.OpticalSystem.load(example_lens_path)
    elements = ghostlight.Element.from_lens_file(example_lens_path)
    # The doublet has 3 surfaces in the file -> indices [0, 1, 2]
    assert elements[0].resolve_surfaces(lens) == [0, 1, 2]
    # The stop is one surface -> index [3]
    assert elements[1].resolve_surfaces(lens) == [3]
    # Rear singlet is two surfaces -> indices [4, 5]
    assert elements[2].resolve_surfaces(lens) == [4, 5]


def test_resolve_surfaces_raises_for_unknown_uuid(simple_system):
    el = ghostlight.Element(
        name="bogus",
        surface_ids=["00000000-0000-0000-0000-000000000000"],
    )
    with pytest.raises(KeyError):
        el.resolve_surfaces(simple_system)


def test_stop_classmethod():
    el = ghostlight.Element.stop(surface_id="abc", position=(0.0, 0.0, 5.0))
    assert el.kind == ghostlight.ElementKind.STOP
    assert el.surface_ids == ["abc"]
    assert el.material_glasses == []
    assert el.name == "Aperture Stop"
    assert el.position == (0.0, 0.0, 5.0)


def test_elements_from_uuids_live_construction(simple_system):
    # simple_system from conftest has 3 surfaces: glass-air-stop
    uuids = list(simple_system.surface_ids)
    assert len(uuids) == 3
    # Group the glass singlet together (2 surfaces) and the stop separately
    elements = ghostlight.Element.elements_from_uuids(
        simple_system,
        [[uuids[0], uuids[1]], [uuids[2]]],
        names=["Singlet", "Stop"],
    )
    assert len(elements) == 2
    assert elements[0].name == "Singlet"
    assert elements[0].kind == ghostlight.ElementKind.GLASS
    assert elements[0].surface_ids == [uuids[0], uuids[1]]
    assert elements[1].name == "Stop"
    assert elements[1].kind == ghostlight.ElementKind.STOP
    assert elements[1].surface_ids == [uuids[2]]
