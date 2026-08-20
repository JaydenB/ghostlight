"""Tests for the reusable element-level toolbar actions."""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.project import Project


# ---------------------------------------------------------------------------
# Add into an empty system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder,expected_surfaces,expected_materials,expected_kind", [
    (element_actions.add_singlet,        2, 1, ghostlight.ElementKind.GLASS),
    (element_actions.add_doublet,        3, 2, ghostlight.ElementKind.GLASS),
    (element_actions.add_triplet,        4, 3, ghostlight.ElementKind.GLASS),
    (element_actions.add_aperture_stop,  1, 0, ghostlight.ElementKind.STOP),
])
def test_add_into_empty_system(
    qapp, builder, expected_surfaces, expected_materials, expected_kind,
):
    project = Project()
    assert project.system.num_surfaces() == 0
    assert project.system.elements == []

    new_el = builder(project)

    assert new_el.kind == expected_kind
    assert len(new_el.surface_ids) == expected_surfaces
    assert len(new_el.material_glasses) == expected_materials

    assert project.system.num_surfaces() == expected_surfaces
    assert len(project.system.surface_ids) == expected_surfaces
    assert len(project.system.aperture_images) == expected_surfaces
    # Parallel-vector alignment: every element UUID must be in surface_ids
    for uuid_str in new_el.surface_ids:
        assert uuid_str in list(project.system.surface_ids)

    # Finalize ran — chain ends at z=0.
    last = project.system.surfaces[-1]
    assert last.z + last.thickness == pytest.approx(0.0)


def test_add_aperture_stop_marks_dirty_and_pushes_undo(qapp):
    project = Project()
    assert project.is_dirty is False
    assert project.can_undo is False

    element_actions.add_aperture_stop(project)

    assert project.is_dirty is True
    assert project.can_undo is True
    assert project.undo_label == "Add Aperture Stop"


def test_add_then_undo_restores_prior_state(qapp, sample_lens_path):
    """Undo of an Add reverts surfaces + element count to pre-add state.

    Loaded from a sample so the BEFORE snapshot is non-empty — the C++
    loader rejects empty optical_system blocks, so undoing back to a totally
    empty system isn't supported (a documented pre-existing limitation of
    the snapshot-via-tempfile undo path)."""
    project = Project()
    project.load(str(sample_lens_path))
    n_surfaces_before = project.system.num_surfaces()
    n_elements_before = len(project.system.elements)

    element_actions.add_singlet(project)
    assert project.system.num_surfaces() == n_surfaces_before + 2
    assert len(project.system.elements) == n_elements_before + 1

    project.undo()
    assert project.system.num_surfaces() == n_surfaces_before
    assert len(project.system.elements) == n_elements_before


# ---------------------------------------------------------------------------
# Add into a loaded system, after an existing element
# ---------------------------------------------------------------------------


def test_add_after_inserts_in_trace_order(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))

    first_el = project.system.elements[0]
    last_surface_of_first = max(first_el.resolve_surfaces(project.system))
    surface_count_before = project.system.num_surfaces()

    new_el = element_actions.add_singlet(project, after=first_el)

    # New element appears at index 1 (right after first_el at 0).
    assert project.system.elements[1] is new_el
    # Its surfaces sit immediately after first_el's last surface.
    new_indices = new_el.resolve_surfaces(project.system)
    assert min(new_indices) == last_surface_of_first + 1
    assert max(new_indices) - min(new_indices) == len(new_indices) - 1
    # Total surface count grew by exactly the new element's surface count.
    assert project.system.num_surfaces() == surface_count_before + len(new_indices)


def test_add_at_front_does_not_shift_existing_surfaces(qapp, sample_lens_path):
    """Regression-style guard for the contract: adding a new element at
    the default front position must leave every prior surface's absolute
    z untouched. End-insertion would shift them all by the new element's
    total thickness, which is exactly the 'accidentally move other
    lenses' problem the user wanted gone."""
    project = Project()
    project.load(str(sample_lens_path))
    z_by_uuid = {
        uuid: project.system.surfaces[i].z
        for i, uuid in enumerate(project.system.surface_ids)
    }

    element_actions.add_singlet(project)

    for i, uuid in enumerate(project.system.surface_ids):
        if uuid in z_by_uuid:
            assert project.system.surfaces[i].z == pytest.approx(
                z_by_uuid[uuid]
            ), f"existing surface {uuid[:6]} shifted in absolute z"


def test_add_with_no_after_inserts_at_front(qapp, sample_lens_path):
    """Default-position adds go to the FRONT of the chain (furthest from
    sensor) so existing surfaces don't shift in absolute z. End-insertion
    would re-anchor the chain backward from sensor=0 and visibly move
    every existing element — the user's contract is explicitly to avoid
    that. They can drag the new row to reorder afterwards."""
    project = Project()
    project.load(str(sample_lens_path))
    total_elements_before = len(project.system.elements)
    z_before = {
        uuid: project.system.surfaces[i].z
        for i, uuid in enumerate(project.system.surface_ids)
    }

    new_el = element_actions.add_doublet(project, after=None)

    assert project.system.elements[0] is new_el
    assert len(project.system.elements) == total_elements_before + 1

    # Every existing surface's absolute z is preserved.
    for i, uuid in enumerate(project.system.surface_ids):
        if uuid in z_before:
            assert project.system.surfaces[i].z == pytest.approx(z_before[uuid])


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def test_remove_element_drops_surfaces_and_aperture_images(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))

    victim = project.system.elements[1]
    victim_surfaces = victim.resolve_surfaces(project.system)
    n_before = project.system.num_surfaces()
    surface_count_to_remove = len(victim_surfaces)

    assert element_actions.remove_element(project, victim) is True

    assert project.system.num_surfaces() == n_before - surface_count_to_remove
    assert len(project.system.surface_ids) == project.system.num_surfaces()
    assert len(project.system.aperture_images) == project.system.num_surfaces()
    # Element is gone from the system list.
    assert victim not in project.system.elements


def test_remove_unknown_element_is_noop(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    n_before = project.system.num_surfaces()

    stranger = ghostlight.Element(name="stranger", surface_ids=[], material_glasses=[])
    assert element_actions.remove_element(project, stranger) is False
    assert project.system.num_surfaces() == n_before


def test_add_singlet_registers_glass_in_catalogue(qapp):
    """Adding a glass element must drop an entry into ``glass_catalogue``
    so the undo-snapshot round-trip through the C++ loader passes.

    With the material-catalogue rollout the singlet now uses
    ``Schott_N-BK7`` (catalogue-resolved Sellmeier), not the legacy
    ``N-BK7`` Abbe stub.
    """
    project = Project()
    element_actions.add_singlet(project)
    assert "Schott_N-BK7" in project.system._raw_glass_catalogue
    entry = project.system._raw_glass_catalogue["Schott_N-BK7"]
    assert entry["catalogue_ref"] == "Schott:N-BK7"
    assert entry["dispersion"]["model"] == "sellmeier"
    # Sanity-check the first Sellmeier B coefficient matches the source.
    assert entry["dispersion"]["B"][0] == pytest.approx(1.03961212)


def test_undo_after_remove_with_referenced_glass_does_not_fail(qapp):
    """Regression: add stop, add singlet, remove stop, undo → must not raise.

    Without this, the undo's snapshot reload throws because the in-memory
    glass_catalogue was empty even though the singlet's material referenced
    the default glass key."""
    project = Project()
    element_actions.add_aperture_stop(project)
    element_actions.add_singlet(project)
    stop = next(
        el for el in project.system.elements if el.kind == ghostlight.ElementKind.STOP
    )
    element_actions.remove_element(project, stop)

    # The bug fired on this line.
    project.undo()

    # And the restored state contains both elements again.
    assert any(el.kind == ghostlight.ElementKind.STOP for el in project.system.elements)
    assert any(el.kind == ghostlight.ElementKind.GLASS for el in project.system.elements)


def test_remove_then_undo_restores(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    n_before = project.system.num_surfaces()
    el_count_before = len(project.system.elements)
    victim_name = project.system.elements[1].name

    element_actions.remove_element(project, project.system.elements[1])
    project.undo()

    assert project.system.num_surfaces() == n_before
    assert len(project.system.elements) == el_count_before
    # Restored element is a *new* instance after reload, but same name.
    assert any(el.name == victim_name for el in project.system.elements)
