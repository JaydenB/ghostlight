"""Tests for `transform.position.mode` resolution.

`relative_to_preceding` rewrites the element's z to `prev_z + z` before
the loader sees it. The first element with relative mode is treated as
absolute with a warning.
"""

from pathlib import Path

import pytest
import ghostlight

FIXTURE = Path(__file__).parent / "fixtures" / "relative_chain.lens"


def test_relative_chain_loads():
    sys = ghostlight.OpticalSystem.load(str(FIXTURE))
    assert sys.num_surfaces() == 6
    assert len(sys.elements) == 3


def test_resolved_element_positions_are_absolute():
    """Element.position is always resolved-absolute, even when authored relative."""
    elements = ghostlight.Element.from_lens_file(str(FIXTURE))
    # Authored z values: -50 (absolute), +10 (relative), +5 (relative).
    # Expected resolved: -50, -40, -35.
    assert elements[0].position[2] == pytest.approx(-50.0)
    assert elements[1].position[2] == pytest.approx(-40.0)
    assert elements[2].position[2] == pytest.approx(-35.0)


def test_position_mode_preserved_for_round_trip():
    elements = ghostlight.Element.from_lens_file(str(FIXTURE))
    assert elements[0].position_mode == "absolute"
    assert elements[1].position_mode == "relative_to_preceding"
    assert elements[2].position_mode == "relative_to_preceding"


def test_first_element_relative_emits_warning_and_treats_as_absolute(tmp_path):
    """A first element with relative mode is treated as absolute with a warning."""
    bad = {
        "format": "ghostlight-optical",
        "version": {"major": 1, "minor": 0},
        "metadata": {"name": "first-relative"},
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
                "name": "First-relative",
                "transform": {
                    "position": {
                        "mode": "relative_to_preceding",
                        "x": 0,
                        "y": 0,
                        "z": -25.0,
                    }
                },
                "materials": [{"glass": "BK7-ish"}],
                "surfaces": [
                    {
                        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaa0000",
                        "semi_aperture": 10.0,
                        "thickness": 3.0,
                        "form": {"type": "sphere", "radius": 30.0},
                    },
                    {
                        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaa0001",
                        "semi_aperture": 10.0,
                        "thickness": 22.0,
                        "form": {"type": "sphere", "radius": -30.0},
                    },
                ],
            }
        ],
        "pivots": [],
    }
    import json
    path = tmp_path / "first_relative.lens"
    path.write_text(json.dumps(bad))

    with pytest.warns(UserWarning, match="relative_to_preceding"):
        elements = ghostlight.Element.from_lens_file(str(path))
    # Treated as absolute → resolved z stays at the authored value.
    assert elements[0].position[2] == pytest.approx(-25.0)
