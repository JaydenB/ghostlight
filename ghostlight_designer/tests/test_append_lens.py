"""Tests for element_actions.append_lens_from_file — splice a whole lens
file's element chain onto the front (object side) or back (sensor side) of
the current system, stored as a relative-to-preceding chain."""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import body as body_mod
from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.optical_editor.body import OpticalEditorBody
from ghostlight_designer.optical_editor.toolbar import (
    ADD_IMPORT_LENS_BACK,
    ADD_IMPORT_LENS_FRONT,
)
from ghostlight_designer.project import Project


def _load(path) -> Project:
    project = Project()
    project.load(str(path))
    return project


def _geometry(system) -> list[float]:
    """Flat [z0, t0, z1, t1, ...] of the resolved axial surface layout."""
    out: list[float] = []
    for s in system.surfaces:
        out.append(float(s.z))
        out.append(float(s.thickness))
    return out


def _sensor_gap(system) -> float:
    """Distance from the last vertex + its BFD to the sensor (should be 0)."""
    last = system.surfaces[-1]
    return float(last.z) + float(last.thickness)


# ---------------------------------------------------------------------------
# Ordering & counts
# ---------------------------------------------------------------------------


def test_append_front_prepends_incoming(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    base_n = len(project.system.elements)
    base_surf_n = len(project.system.surfaces)

    new = element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=True,
    )
    els = project.system.elements
    assert len(els) == 2 * base_n
    assert len(project.system.surfaces) == 2 * base_surf_n
    # Incoming elements occupy the object-side (front) block.
    new_ids = [e.element_id for e in new]
    assert [e.element_id for e in els[:base_n]] == new_ids
    # The virtual sensor plane is still at z = 0.
    assert _sensor_gap(project.system) == pytest.approx(0.0, abs=1e-4)


def test_append_back_appends_incoming(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    base_n = len(project.system.elements)

    new = element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=False,
    )
    els = project.system.elements
    assert len(els) == 2 * base_n
    new_ids = [e.element_id for e in new]
    # Incoming elements occupy the sensor-side (back) block.
    assert [e.element_id for e in els[-base_n:]] == new_ids
    assert _sensor_gap(project.system) == pytest.approx(0.0, abs=1e-4)


def test_front_append_preserves_existing_absolute_z(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    before = _geometry(project.system)
    base_surf_n = len(project.system.surfaces)

    element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=True,
    )
    after = _geometry(project.system)
    # Front-insertion keeps the original system's surfaces at the same
    # absolute z — they occupy the sensor-side tail of the merged chain.
    assert after[-2 * base_surf_n:] == pytest.approx(before, abs=1e-4)


# ---------------------------------------------------------------------------
# Relative storage + round-trip (validates the writer's delta conversion)
# ---------------------------------------------------------------------------


def test_appended_chain_is_relative(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=True,
    )
    els = project.system.elements
    # First element is the absolute anchor; every following element is
    # stored relative-to-preceding so the stack reads as placed relatively.
    assert els[0].position_mode == "absolute"
    assert all(el.position_mode == "relative_to_preceding" for el in els[1:])


@pytest.mark.parametrize("to_front", [True, False])
def test_append_round_trips_through_save(qapp, sample_lens_path, tmp_path, to_front):
    project = _load(sample_lens_path)
    element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=to_front,
    )
    geom = _geometry(project.system)

    out = tmp_path / "combined.lens"
    project.system.save(str(out))
    reloaded = ghostlight.OpticalSystem.load(str(out))
    # The relative-mode chain must reload to identical geometry — this is
    # exactly what the writer's absolute→delta conversion guarantees; without
    # it the loader would re-add each predecessor's z and the chain would blow
    # apart on the first save.
    assert _geometry(reloaded) == pytest.approx(geom, abs=1e-4)


# ---------------------------------------------------------------------------
# Undo / redo (snapshots also go through the writer, so exercises the fix)
# ---------------------------------------------------------------------------


def test_append_undo_redo(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    base_geom = _geometry(project.system)
    base_n = len(project.system.elements)

    element_actions.append_lens_from_file(
        project, str(sample_lens_path), to_front=False,
    )
    assert len(project.system.elements) == 2 * base_n

    project.undo()
    assert len(project.system.elements) == base_n
    assert _geometry(project.system) == pytest.approx(base_geom, abs=1e-4)

    project.redo()
    assert len(project.system.elements) == 2 * base_n
    assert _sensor_gap(project.system) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Robustness — UUID regen + glass-catalogue merge
# ---------------------------------------------------------------------------


def test_append_regenerates_uuids(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    # Append the SAME file twice — surface/element ids would collide (and
    # corrupt resolve_surfaces' uuid→index map) without regeneration.
    element_actions.append_lens_from_file(project, str(sample_lens_path), to_front=True)
    element_actions.append_lens_from_file(project, str(sample_lens_path), to_front=False)

    surf_ids = list(project.system.surface_ids)
    assert len(surf_ids) == len(set(surf_ids)), "surface UUIDs must stay unique"
    el_ids = [e.element_id for e in project.system.elements]
    assert len(el_ids) == len(set(el_ids)), "element UUIDs must stay unique"


def test_append_merges_glass_catalogue(qapp, sample_lens_path):
    project = _load(sample_lens_path)
    incoming = ghostlight.OpticalSystem.load(str(sample_lens_path))
    inc_glasses = set(incoming._raw_glass_catalogue)

    element_actions.append_lens_from_file(project, str(sample_lens_path), to_front=True)
    assert inc_glasses <= set(project.system._raw_glass_catalogue)


def test_append_empty_incoming_raises(qapp, sample_lens_path, monkeypatch):
    project = _load(sample_lens_path)

    class _Empty:
        elements = []

    monkeypatch.setattr(ghostlight.OpticalSystem, "load", staticmethod(lambda _p: _Empty()))
    with pytest.raises(ValueError):
        element_actions.append_lens_from_file(
            project, "whatever.lens", to_front=True,
        )


# ---------------------------------------------------------------------------
# Body wiring — the Add-menu handler drives the file picker + selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, front", [(ADD_IMPORT_LENS_FRONT, True), (ADD_IMPORT_LENS_BACK, False)],
)
def test_body_import_handler_appends_and_selects(
    qapp, sample_lens_path, monkeypatch, kind, front,
):
    project = Project()
    project.load(str(sample_lens_path))
    base_n = len(project.system.elements)
    body = OpticalEditorBody(project)
    try:
        monkeypatch.setattr(
            body_mod.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(sample_lens_path), "")),
        )
        body._on_add_element_requested(kind)

        els = project.system.elements
        assert len(els) == 2 * base_n
        # The front-most of the appended block is selected.
        expected = els[0] if front else els[base_n]
        assert project.selected_element is expected
    finally:
        body.deleteLater()


def test_body_import_handler_cancel_is_noop(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    base_n = len(project.system.elements)
    body = OpticalEditorBody(project)
    try:
        # Empty path == user cancelled the dialog.
        monkeypatch.setattr(
            body_mod.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
        body._on_add_element_requested(ADD_IMPORT_LENS_FRONT)
        assert len(project.system.elements) == base_n
        assert not project.can_undo
    finally:
        body.deleteLater()
