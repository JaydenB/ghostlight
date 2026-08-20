"""Tests for pivot loading + baking into surface poses.

The C++ loader composes pivot transforms onto element transforms before
flattening surfaces. With zero offset, the baked surfaces should match
the no-pivot baseline; non-zero offsets translate/rotate the surfaces
exactly.
"""

import math
from pathlib import Path

import pytest
import ghostlight

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DOUBLET_WITH_PIVOT = FIXTURE_DIR / "doublet_with_pivot.lens"


def test_loads_pivots():
    sys = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    assert len(sys.pivots) == 1
    p = sys.pivots[0]
    assert p.name == "Focus Group"
    assert p.element_ids == ["33333333-3333-3333-3333-333333333333"]
    assert p.pivot_point_mode == "centroid"
    assert p.offset_position == (0.0, 0.0, 0.0)
    assert p.offset_rotation == (0.0, 0.0, 0.0)
    assert len(p.exposed) == 1
    e = p.exposed[0]
    assert e.name == "focus"
    assert e.attr == "offset.position.z"
    assert e.min == -5.0 and e.max == 5.0


def _rear_singlet_indices(sys: ghostlight.OpticalSystem) -> list[int]:
    """Indices in `sys.surfaces` of the rear singlet's surfaces."""
    rear = next(el for el in sys.elements if el.name == "Rear Singlet")
    return rear.resolve_surfaces(sys)


def test_zero_offset_pivot_is_identity():
    """A pivot with zero offset must not change any surface pose."""
    sys = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    indices = _rear_singlet_indices(sys)
    surfs = [sys.surfaces[i] for i in indices]
    # With zero offset, baked decenter / rotation match the on-axis identity:
    # decenter_x == 0, decenter_y == 0, rot is identity.
    for s in surfs:
        assert abs(s.decenter_x) < 1e-5
        assert abs(s.decenter_y) < 1e-5
        # Identity rotation: rot[0]=rot[4]=rot[8]=1, off-diagonals 0.
        assert abs(s.rot[0] - 1.0) < 1e-5
        assert abs(s.rot[4] - 1.0) < 1e-5
        assert abs(s.rot[8] - 1.0) < 1e-5
        for i in (1, 2, 3, 5, 6, 7):
            assert abs(s.rot[i]) < 1e-5


def test_nonzero_z_offset_shifts_surface_z(tmp_path):
    """offset.position.z == +2 shifts the targeted element's surfaces by +2 in z."""
    # Load baseline (zero offset), record z's.
    base = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    indices = _rear_singlet_indices(base)
    base_z = [base.surfaces[i].z for i in indices]

    # Mutate pivot offset, save, reload.
    base.pivots[0].set_attr("offset.position.z", 2.0)
    new_path = tmp_path / "shifted.lens"
    base.save(str(new_path))

    shifted = ghostlight.OpticalSystem.load(str(new_path))
    shifted_indices = _rear_singlet_indices(shifted)
    shifted_z = [shifted.surfaces[i].z for i in shifted_indices]

    for bz, sz in zip(base_z, shifted_z):
        assert sz == pytest.approx(bz + 2.0, abs=1e-4)


def test_rotation_offset_changes_rot_matrix(tmp_path):
    """offset.rotation.tilt_x rotates the targeted element's surfaces."""
    base = ghostlight.OpticalSystem.load(str(DOUBLET_WITH_PIVOT))
    base.pivots[0].set_attr("offset.rotation.tilt_x", 5.0)  # degrees
    new_path = tmp_path / "tilted.lens"
    base.save(str(new_path))

    tilted = ghostlight.OpticalSystem.load(str(new_path))
    indices = _rear_singlet_indices(tilted)
    # rot is no longer identity for the tilted element's surfaces.
    s = tilted.surfaces[indices[0]]
    # tilt_x = 5° => rot[4] = cos(5°), rot[5] = -sin(5°), rot[7] = sin(5°), rot[8] = cos(5°)
    cx = math.cos(math.radians(5.0))
    sx = math.sin(math.radians(5.0))
    assert s.rot[4] == pytest.approx(cx, abs=1e-4)
    assert s.rot[5] == pytest.approx(-sx, abs=1e-4)
    assert s.rot[7] == pytest.approx(sx, abs=1e-4)
    assert s.rot[8] == pytest.approx(cx, abs=1e-4)
