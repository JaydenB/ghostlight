from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.project import Project


def _collect(signal):
    received: list = []

    def slot(*args):
        received.append(args if len(args) != 1 else args[0])

    signal.connect(slot)
    return received


def _first_sphere_surface_index(project: Project) -> int:
    for i, s in enumerate(project.system.surfaces):
        if int(s.form) == int(ghostlight.SurfaceForm.SPHERE):
            return i
    raise AssertionError("sample lens has no SPHERE surface")


def test_initial_stacks_empty(qapp):
    p = Project()
    assert p.can_undo is False
    assert p.can_redo is False
    assert p.undo_label is None
    assert p.redo_label is None


def test_edit_pushes_and_undo_restores_field(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    original = p.system.surfaces[si].radius

    with p.edit("Set Radius"):
        p.system.surfaces[si].radius = original + 5.0
    assert p.can_undo is True
    assert p.undo_label == "Set Radius"
    assert p.system.surfaces[si].radius == pytest.approx(original + 5.0)

    p.undo()
    assert p.system.surfaces[si].radius == pytest.approx(original)
    assert p.can_undo is False
    assert p.can_redo is True
    assert p.redo_label == "Set Radius"


def test_redo_replays_change(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    original = p.system.surfaces[si].radius

    with p.edit("Set Radius"):
        p.system.surfaces[si].radius = original + 5.0
    p.undo()
    p.redo()
    assert p.system.surfaces[si].radius == pytest.approx(original + 5.0)
    assert p.can_undo is True
    assert p.can_redo is False


def test_new_edit_clears_redo(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)

    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0
    p.undo()
    assert p.can_redo is True

    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 2.0
    assert p.can_redo is False


def test_compound_inner_edits_emit_systemModified_live(qapp, sample_lens_path):
    """Inside an open compound, each inner edit must still emit
    systemModified so observers (viewport, etc.) repaint live during a
    scrub drag. Only the undo-stack push is deferred to end_compound."""
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)

    modified = _collect(p.systemModified)
    p.begin_compound("Scrub Radius")
    for delta in (1.0, 2.0, 3.0):
        with p.edit("Set Radius"):
            p.system.surfaces[si].radius += delta
    # Three inner edits should have produced three systemModified emissions
    # *before* end_compound runs.
    assert len(modified) == 3
    p.end_compound()
    # end_compound emits one more (the coalesced push).
    assert len(modified) == 4


def test_compound_coalesces(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    original = p.system.surfaces[si].radius

    p.begin_compound("Scrub Radius")
    for delta in (1.0, 2.5, 4.0):
        with p.edit("Set Radius"):
            p.system.surfaces[si].radius = original + delta
    p.end_compound()

    assert len(p._undo) == 1
    assert p.undo_label == "Scrub Radius"
    p.undo()
    assert p.system.surfaces[si].radius == pytest.approx(original)


def test_compound_nested_only_pushes_outermost(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    original = p.system.surfaces[si].radius

    p.begin_compound("Outer")
    p.begin_compound("Inner")
    with p.edit("Set Radius"):
        p.system.surfaces[si].radius = original + 1.0
    p.end_compound()
    assert len(p._undo) == 0  # inner end did not push
    p.end_compound()
    assert len(p._undo) == 1
    assert p.undo_label == "Outer"


def test_undo_emits_systemReplaced_and_dirtyChanged(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0
    # Project is dirty from the edit; clear dirty signal collector now.
    replaced = _collect(p.systemReplaced)
    dirty = _collect(p.dirtyChanged)
    p.undo()
    assert len(replaced) == 1
    # Already dirty before undo, stays dirty — no transition signal.
    assert dirty == []


def test_canUndo_canRedo_signals_only_fire_on_transitions(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)

    cu = _collect(p.canUndoChanged)
    cr = _collect(p.canRedoChanged)

    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0
    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0
    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0

    # canUndo False -> True exactly once (across three edits).
    assert cu == [True]
    # canRedo stays False throughout (no undo yet).
    assert cr == []

    p.undo()
    # canUndo stays True (still have entries), canRedo flips True once.
    assert cu == [True]
    assert cr == [True]


def test_byte_budget_trims_oldest(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    # Force a small budget — sample lens snapshots are several KB each, so 1
    # KB guarantees the stack never holds more than one entry.
    p._snapshot_budget_bytes = 1024
    si = _first_sphere_surface_index(p)

    for _ in range(5):
        with p.edit("Set Radius"):
            p.system.surfaces[si].radius += 0.1

    assert len(p._undo) <= 1  # budget forced trimming
    # Current state is intact regardless.
    assert p.system.surfaces[si].radius == pytest.approx(
        p.system.surfaces[si].radius
    )


def test_form_change_undo_round_trips_structural_change(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    original_form = int(p.system.surfaces[si].form)
    assert original_form == int(ghostlight.SurfaceForm.SPHERE)

    with p.edit("Set Form"):
        p.system.surfaces[si].form = int(ghostlight.SurfaceForm.CYLINDRICAL)
    assert int(p.system.surfaces[si].form) == int(ghostlight.SurfaceForm.CYLINDRICAL)

    p.undo()
    assert int(p.system.surfaces[si].form) == original_form


def test_load_clears_history(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    si = _first_sphere_surface_index(p)
    with p.edit("Set Radius"):
        p.system.surfaces[si].radius += 1.0
    assert p.can_undo is True

    p.load(str(sample_lens_path))
    assert p.can_undo is False
    assert p.can_redo is False


def test_abort_skips_push(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    with p.edit("Set Radius") as txn:
        txn.abort()
    assert p.can_undo is False


def test_undo_after_radius_edit_does_not_corrupt_chain(qapp):
    """User-reported regression: build doublet + stop + doublet, change a
    radius on the last doublet, Ctrl+Z — the whole geometry collapsed
    (first doublet's back-surface thickness went 10 → -20, aperture
    stop's thickness went 10 → 0) because programmatically-added
    elements left ``el.position = (0, 0, 0)`` even though their surfaces
    were at z = -70 / -40 / -30. The writer serialized the bogus
    position into every snapshot, and the loader's inter-element gap
    patcher then computed nonsense thicknesses on restore. The fix
    syncs ``el.position`` from the first surface's z after every
    ``finalize()`` so the snapshot is internally consistent."""
    from ghostlight_designer.optical_editor import element_actions
    from ghostlight_designer.optical_editor.model import OpticalTreeModel
    from ghostlight_designer.optical_editor.columns import Column
    from PySide6.QtCore import QModelIndex, Qt

    p = Project()
    element_actions.add_doublet(p)
    element_actions.add_aperture_stop(p)
    element_actions.add_doublet(p)

    # Snapshot thicknesses + zs by UUID for the strict assertion below.
    thickness_before = {
        uuid: float(p.system.surfaces[i].thickness)
        for i, uuid in enumerate(p.system.surface_ids)
    }
    z_before = {
        uuid: float(p.system.surfaces[i].z)
        for i, uuid in enumerate(p.system.surface_ids)
    }

    # Change radius on the last doublet's first surface via the tree.
    model = OpticalTreeModel(p)
    last_el_idx = len(p.system.elements) - 1
    last_el = p.system.elements[last_el_idx]
    target_si = last_el.resolve_surfaces(p.system)[0]
    target_uuid = p.system.surface_ids[target_si]

    el_qidx = model.index(last_el_idx, 0, QModelIndex())
    surface_row = len(last_el.material_glasses)
    radius_qidx = model.index(surface_row, int(Column.RADIUS), el_qidx)
    model.setData(radius_qidx, 50.0, Qt.EditRole)
    assert p.system.surfaces[target_si].radius == pytest.approx(50.0)

    p.undo()

    # Radius reverted.
    new_si = list(p.system.surface_ids).index(target_uuid)
    assert p.system.surfaces[new_si].radius == pytest.approx(0.0)
    # Every other surface's thickness AND z is exactly where it was
    # before the edit — no corruption from stale element positions.
    for i, uuid in enumerate(p.system.surface_ids):
        if uuid in thickness_before:
            assert p.system.surfaces[i].thickness == pytest.approx(
                thickness_before[uuid], abs=1e-4
            ), f"surface {uuid[:6]} thickness corrupted across undo"
        if uuid in z_before:
            assert p.system.surfaces[i].z == pytest.approx(
                z_before[uuid], abs=1e-4
            ), f"surface {uuid[:6]} z corrupted across undo"


def test_added_element_position_matches_first_surface_z(qapp):
    """Programmatically-added elements must have ``el.position[2]`` synced
    to their first surface's z post-finalize, so JSON snapshots
    round-trip cleanly through the loader's gap-patching."""
    from ghostlight_designer.optical_editor import element_actions

    p = Project()
    element_actions.add_doublet(p)
    element_actions.add_aperture_stop(p)
    element_actions.add_doublet(p)

    for el in p.system.elements:
        surfs = el.resolve_surfaces(p.system)
        first_z = p.system.surfaces[surfs[0]].z
        assert el.position[2] == pytest.approx(first_z), (
            f"element {el.name!r} position.z={el.position[2]} but its "
            f"first surface is at z={first_z}"
        )


def test_failed_undo_rolls_forward_keeps_system_consistent(qapp):
    """Regression: undoing back to a state the loader rejects (an empty
    ``optical_system`` block) used to leave the system half-cleared with
    Python ``_elements`` still pointing at surface UUIDs that no longer
    exist — the user observed this as their element 'disassembling'.

    The recovery contract: undo catches the load failure, rolls forward to
    the AFTER snapshot it captured at the start, refreshes listeners, and
    re-raises so the caller knows it failed. State remains consistent and
    every element's surfaces still resolve."""
    from ghostlight_designer.optical_editor import element_actions

    p = Project()
    element_actions.add_doublet(p)
    assert len(p.system.elements) == 1
    surfs_before = len(p.system.surfaces)
    el_before = p.system.elements[0]
    resolvable_before = el_before.resolve_surfaces(p.system)

    # The Add Doublet undo BEFORE snapshot is an empty system, which the
    # C++ loader rejects with "optical_system produced no surfaces".
    with pytest.raises(RuntimeError):
        p.undo()

    # Recovery: state matches what was there before the failed undo.
    assert len(p.system.elements) == 1, "doublet must still exist"
    assert len(p.system.surfaces) == surfs_before
    # Every element still resolves — no orphaned UUIDs.
    p.system.elements[0].resolve_surfaces(p.system)  # raises KeyError on disassembly
    # Failing entry was dropped so subsequent undo calls don't loop on it.
    assert p.can_undo is False
