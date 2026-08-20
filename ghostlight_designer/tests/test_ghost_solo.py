"""Tests for the per-surface ghost-solo designer integration.

Covers:
- Project state + signal
- surface_actions.set_surface_ghost_solo and compute_ghost_filter helpers
- Stale-UUID prune on system mutation
- Tree icon switches to solo glyph
- Reset on systemReplaced (load / new)
"""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import element_actions, surface_actions
from ghostlight_designer.optical_editor.model import OpticalTreeModel
from ghostlight_designer.optical_editor.nodes import SurfaceNode, build_tree
from ghostlight_designer.project import Project


def _load(qapp, sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    return project


def _stop_element(project: Project) -> ghostlight.Element:
    return next(el for el in project.system.elements if el.kind == ghostlight.ElementKind.STOP)


def _first_surface_uuid(project: Project) -> str:
    return str(project.system.surface_ids[0])


# ---------------------------------------------------------------------------
# Project state + signal
# ---------------------------------------------------------------------------

def test_project_starts_with_no_solo(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert project.ghost_solo_surface_uuids == frozenset()


def test_set_surface_ghost_solo_emits_signal(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    uuid = _first_surface_uuid(project)

    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))

    changed = project.set_surface_ghost_solo(uuid, True)
    assert changed is True
    assert project.is_surface_ghost_solo(uuid)
    assert received == [True]


def test_set_surface_ghost_solo_idempotent(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    uuid = _first_surface_uuid(project)
    project.set_surface_ghost_solo(uuid, True)
    # Re-applying the same state returns False and emits no signal.
    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))
    changed = project.set_surface_ghost_solo(uuid, True)
    assert changed is False
    assert received == []


def test_clear_ghost_solo(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    uuids = [str(u) for u in list(project.system.surface_ids)[:3]]
    for u in uuids:
        project.set_surface_ghost_solo(u, True)
    assert len(project.ghost_solo_surface_uuids) == 3

    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))
    cleared = project.clear_ghost_solo()
    assert cleared is True
    assert project.ghost_solo_surface_uuids == frozenset()
    assert received == [True]


def test_load_clears_ghost_solo(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    project.set_surface_ghost_solo(_first_surface_uuid(project), True)
    assert project.ghost_solo_surface_uuids
    project.load(str(sample_lens_path))
    assert project.ghost_solo_surface_uuids == frozenset()


# ---------------------------------------------------------------------------
# Stale-UUID prune
# ---------------------------------------------------------------------------

def test_prune_dead_ghost_solo_on_remove(qapp, sample_lens_path):
    """Removing an element with a solo'd surface drops the dead UUID."""
    project = _load(qapp, sample_lens_path)
    # Solo a surface that belongs to a removable element.
    glass_el = next(
        el for el in project.system.elements if el.kind == ghostlight.ElementKind.GLASS
    )
    target_uuid = glass_el.surface_ids[0]
    project.set_surface_ghost_solo(target_uuid, True)
    assert target_uuid in project.ghost_solo_surface_uuids

    element_actions.remove_element(project, glass_el)
    assert target_uuid not in project.ghost_solo_surface_uuids


# ---------------------------------------------------------------------------
# surface_actions helpers
# ---------------------------------------------------------------------------

def test_set_surface_ghost_solo_via_index(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ok = surface_actions.set_surface_ghost_solo(project, 0, True)
    assert ok is True
    assert project.is_surface_ghost_solo(_first_surface_uuid(project))


def test_set_surface_ghost_solo_oob_returns_false(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    n = len(project.system.surfaces)
    ok = surface_actions.set_surface_ghost_solo(project, n + 5, True)
    assert ok is False


def test_compute_ghost_filter_unsolo_is_all_mode(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    gf = surface_actions.compute_ghost_filter(project)
    assert gf.mode == ghostlight.GhostFilter.Mode.ALL


def test_compute_ghost_filter_solo_one_surface(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    surface_actions.set_surface_ghost_solo(project, 1, True)
    gf = surface_actions.compute_ghost_filter(project)
    assert gf.mode == ghostlight.GhostFilter.Mode.INCLUDE
    # Every pair in the filter must involve surface 1.
    pairs = list(gf.pairs)
    assert pairs, "expected at least one ghost pair involving surface 1"
    for a, b in pairs:
        assert 1 in (a, b)


def test_compute_ghost_filter_multi_solo_union(qapp, sample_lens_path):
    """Soloing two surfaces gives pairs where EITHER surface participates."""
    project = _load(qapp, sample_lens_path)
    surface_actions.set_surface_ghost_solo(project, 0, True)
    surface_actions.set_surface_ghost_solo(project, 3, True)
    gf = surface_actions.compute_ghost_filter(project)
    for a, b in gf.pairs:
        assert (a == 0 or b == 0 or a == 3 or b == 3)


def test_compute_ghost_filter_all_dead_uuids_falls_back_to_all(
    qapp, sample_lens_path,
):
    """If every solo'd UUID has gone stale (the window between undo/redo
    and the next mutation), compute_ghost_filter returns mode ALL rather
    than an empty INCLUDE — a render dispatched against an empty INCLUDE
    would silently produce a black frame."""
    project = _load(qapp, sample_lens_path)
    # Poke a fabricated UUID directly into the solo set, bypassing the
    # normal API so we can simulate the "dead UUID lingering" window.
    project._ghost_solo_surface_uuids.add("not-a-real-uuid-abcd-1234")
    gf = surface_actions.compute_ghost_filter(project)
    assert gf.mode == ghostlight.GhostFilter.Mode.ALL
    assert list(gf.pairs) == []


# ---------------------------------------------------------------------------
# Tree icon switches
# ---------------------------------------------------------------------------

def test_build_tree_marks_solo_icon(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    uuid = _first_surface_uuid(project)
    project.set_surface_ghost_solo(uuid, True)
    root = build_tree(
        project.system, solo_uuids=project.ghost_solo_surface_uuids
    )
    # Find the SurfaceNode for our uuid.
    found = []
    def walk(node):
        if isinstance(node, SurfaceNode) and node.surface_uuid == uuid:
            found.append(node)
        for c in node.children:
            walk(c)
    walk(root)
    assert found, "SurfaceNode not found in tree"
    assert found[0].ghost_solo is True
    assert found[0].icon_name == "node-surface-solo"


def test_build_tree_default_surface_icon_when_not_solo(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    root = build_tree(project.system, solo_uuids=frozenset())
    def walk(node):
        if isinstance(node, SurfaceNode):
            assert node.ghost_solo is False
            assert node.icon_name == "node-surface"
        for c in node.children:
            walk(c)
    walk(root)


# ---------------------------------------------------------------------------
# Element-level solo
# ---------------------------------------------------------------------------

def test_set_element_ghost_solo_solos_every_surface(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    glass = next(el for el in project.system.elements if el.kind == ghostlight.ElementKind.GLASS)

    ok = element_actions.set_element_ghost_solo(project, glass, True)
    assert ok is True
    for uuid in glass.surface_ids:
        assert project.is_surface_ghost_solo(uuid)


def test_set_element_ghost_solo_unsolo_clears_every_surface(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    glass = next(el for el in project.system.elements if el.kind == ghostlight.ElementKind.GLASS)
    element_actions.set_element_ghost_solo(project, glass, True)

    ok = element_actions.set_element_ghost_solo(project, glass, False)
    assert ok is True
    for uuid in glass.surface_ids:
        assert not project.is_surface_ghost_solo(uuid)


def test_set_element_ghost_solo_refuses_stop(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    stop = _stop_element(project)
    ok = element_actions.set_element_ghost_solo(project, stop, True)
    assert ok is False
    # Nothing should have been added.
    for uuid in stop.surface_ids:
        assert not project.is_surface_ghost_solo(uuid)


def test_set_element_ghost_solo_emits_one_signal_per_call(qapp, sample_lens_path):
    """Element-level solo must batch — a doublet (3 surfaces) firing one
    signal per surface would thrash listeners (each does a full model
    rebuild)."""
    project = _load(qapp, sample_lens_path)
    glass = next(
        el for el in project.system.elements
        if el.kind == ghostlight.ElementKind.GLASS and len(el.surface_ids) >= 2
    )
    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))

    element_actions.set_element_ghost_solo(project, glass, True)
    assert len(received) == 1


def test_is_element_ghost_solo_reflects_state(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    glass = next(el for el in project.system.elements if el.kind == ghostlight.ElementKind.GLASS)

    assert element_actions.is_element_ghost_solo(project, glass) is False
    element_actions.set_element_ghost_solo(project, glass, True)
    assert element_actions.is_element_ghost_solo(project, glass) is True

    # Partial state — flip one surface off — is NOT considered solo'd.
    project.set_surface_ghost_solo(glass.surface_ids[0], False)
    assert element_actions.is_element_ghost_solo(project, glass) is False


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def test_set_surfaces_ghost_solo_batches_signal(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    uuids = list(project.system.surface_ids)[:3]

    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))
    changed = project.set_surfaces_ghost_solo(uuids, True)

    assert changed is True
    assert len(received) == 1
    for u in uuids:
        assert project.is_surface_ghost_solo(u)


def test_set_surfaces_ghost_solo_no_op_emits_nothing(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    received = []
    project.ghostSoloChanged.connect(lambda: received.append(True))

    changed = project.set_surfaces_ghost_solo([], True)
    assert changed is False
    assert received == []


def test_element_solo_from_mixed_state_fills_missing(qapp, sample_lens_path):
    """Mixed state → is_element_ghost_solo False → the menu shows Solo,
    and clicking it fills in the surfaces that weren't yet solo'd instead
    of clearing the ones that were."""
    project = _load(qapp, sample_lens_path)
    glass = next(
        el for el in project.system.elements
        if el.kind == ghostlight.ElementKind.GLASS and len(el.surface_ids) >= 2
    )
    # Solo just the first surface — element is in mixed state.
    project.set_surface_ghost_solo(glass.surface_ids[0], True)
    assert element_actions.is_element_ghost_solo(project, glass) is False

    element_actions.set_element_ghost_solo(project, glass, True)
    for uuid in glass.surface_ids:
        assert project.is_surface_ghost_solo(uuid)


def test_model_rebuilds_on_ghost_solo_changed(qapp, sample_lens_path):
    """OpticalTreeModel listens to ghostSoloChanged and rebuilds the tree,
    so the solo icon flips without anyone calling mark_modified."""
    project = _load(qapp, sample_lens_path)
    model = OpticalTreeModel(project)
    uuid = _first_surface_uuid(project)

    def find_surface_node():
        # Find the index for the surface UUID in the tree.
        for el_row in range(model.rowCount()):
            el_idx = model.index(el_row, 0)
            for child_row in range(model.rowCount(el_idx)):
                child_idx = model.index(child_row, 0, el_idx)
                node = child_idx.internalPointer()
                if isinstance(node, SurfaceNode) and node.surface_uuid == uuid:
                    return node
        return None

    before = find_surface_node()
    assert before is not None and before.ghost_solo is False
    surface_actions.set_surface_ghost_solo(project, 0, True)
    after = find_surface_node()
    assert after is not None and after.ghost_solo is True
    assert after.icon_name == "node-surface-solo"
