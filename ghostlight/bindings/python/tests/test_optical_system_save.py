"""Round-trip writer: mutate pivot, save, reload, observe the new pose."""

from pathlib import Path

import pytest
import ghostlight

FIXTURE = Path(__file__).parent / "fixtures" / "doublet_with_pivot.lens"


def test_save_reload_roundtrip_preserves_state(tmp_path):
    sys = ghostlight.OpticalSystem.load(str(FIXTURE))
    out = tmp_path / "roundtrip.lens"
    sys.save(str(out))

    reloaded = ghostlight.OpticalSystem.load(str(out))
    assert reloaded.num_surfaces() == sys.num_surfaces()
    assert len(reloaded.elements) == len(sys.elements)
    assert len(reloaded.pivots) == len(sys.pivots)

    # Per-element identity preserved.
    for a, b in zip(sys.elements, reloaded.elements):
        assert a.element_id == b.element_id
        assert a.name == b.name
        assert a.position == pytest.approx(b.position)
        assert a.position_mode == b.position_mode

    # Pivot identity + state preserved.
    pa, pb = sys.pivots[0], reloaded.pivots[0]
    assert pa.pivot_id == pb.pivot_id
    assert pa.name == pb.name
    assert pa.element_ids == pb.element_ids
    assert pa.offset_position == pytest.approx(pb.offset_position)
    assert pa.offset_rotation == pytest.approx(pb.offset_rotation)


def test_save_after_pivot_mutation_changes_reloaded_pose(tmp_path):
    sys = ghostlight.OpticalSystem.load(str(FIXTURE))
    # Find the rear singlet's first surface z baseline.
    rear = next(el for el in sys.elements if el.name == "Rear Singlet")
    baseline_z = sys.surfaces[rear.resolve_surfaces(sys)[0]].z

    sys.pivots[0].set_attr("offset.position.z", -2.5)
    out = tmp_path / "edited.lens"
    sys.save(str(out))

    edited = ghostlight.OpticalSystem.load(str(out))
    rear2 = next(el for el in edited.elements if el.name == "Rear Singlet")
    edited_z = edited.surfaces[rear2.resolve_surfaces(edited)[0]].z

    assert edited_z == pytest.approx(baseline_z - 2.5, abs=1e-4)


def test_save_preserves_pivot_exposed_controls(tmp_path):
    sys = ghostlight.OpticalSystem.load(str(FIXTURE))
    out = tmp_path / "exposed.lens"
    sys.save(str(out))
    reloaded = ghostlight.OpticalSystem.load(str(out))
    a = sys.pivots[0].exposed
    b = reloaded.pivots[0].exposed
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert ea.name == eb.name
        assert ea.attr == eb.attr
        assert ea.min == pytest.approx(eb.min)
        assert ea.max == pytest.approx(eb.max)
