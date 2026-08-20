"""Tests for the optical-editor coating row: conditional tree attachment
(coated + non-stop only), per-model cell gating, the Model discriminator
seeding, the surface right-click apply/remove path, and undo round-trip
(which exercises the writer→parser completeness that designer undo depends on).

In the sample doublet, surfaces 0/2/4/5 are coated (ar_layers=1), surface 1 is
an uncoated cemented interface, and surface 3 is the aperture stop.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import ghostlight

from ghostlight_designer.project import Project
from ghostlight_designer.optical_editor import coating_actions
from ghostlight_designer.optical_editor.nodes import (
    CoatingFormNode,
    NodeKind,
    SurfaceNode,
    build_tree,
    surface_uuid_for,
)
from ghostlight_designer.optical_editor import row_schemas as rs
from ghostlight_designer.optical_editor.row_schemas import CoatingModelUI

_COATED = 0    # a coated glass surface (ar_layers=1)
_UNCOATED = 1  # the cemented interface — no coating
_STOP = 3      # the aperture stop


def _load(qapp, sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    return project


def _coating_node(project, surface_index: int) -> CoatingFormNode:
    root = build_tree(project.system)
    for el in root.children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == surface_index:
                for child in surf.children:
                    if isinstance(child, CoatingFormNode):
                        return child
    raise AssertionError(f"no coating node for surface {surface_index}")


def _has_coating_node(project, surface_index: int) -> bool:
    root = build_tree(project.system)
    for el in root.children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == surface_index:
                return any(isinstance(c, CoatingFormNode) for c in surf.children)
    return False


def _ctx(project, node):
    return rs.SlotContext(node=node, system=project.system, project=project)


# ---------------------------------------------------------------------------
# Conditional tree attachment
# ---------------------------------------------------------------------------

def test_coating_node_only_under_coated_non_stop_surfaces(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    root = build_tree(project.system)

    coated_indices = []
    for el in root.children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode):
                if any(isinstance(c, CoatingFormNode) for c in surf.children):
                    coated_indices.append(surf.surface_index)
    # Exactly the coated glass surfaces; not the cemented interface, not the stop.
    assert sorted(coated_indices) == [0, 2, 4, 5]


def test_uncoated_surface_has_no_coating_node(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert not _has_coating_node(project, _UNCOATED)


def test_stop_never_has_coating_node_even_if_coated(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    # Force a coating onto the stop surface directly; the tree must still not
    # show a coating row for it (stops aren't coatable in the UI).
    project.system.surfaces[_STOP].coating.ar_layers = 1
    assert not _has_coating_node(project, _STOP)


def test_surface_uuid_resolves_through_coating_node(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    node = _coating_node(project, _COATED)
    assert node.kind == NodeKind.COATING_FORM
    uuid = surface_uuid_for(node)
    assert uuid
    assert uuid == list(project.system.surface_ids)[_COATED]


# ---------------------------------------------------------------------------
# Surface right-click apply / remove (the add path for bare surfaces)
# ---------------------------------------------------------------------------

def test_apply_preset_to_uncoated_surface_adds_row(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert not _has_coating_node(project, _UNCOATED)

    payload = {"type": "coating", "model": "artist",
               "tint": [1.0, 0.5, 0.2], "strength": 0.05}
    assert coating_actions.apply_coating_preset(project, _UNCOATED, payload)

    assert _has_coating_node(project, _UNCOATED)
    c = project.system.surfaces[_UNCOATED].coating
    assert int(c.model) == int(ghostlight.CoatingModel.ARTIST)
    assert project.can_undo


def test_remove_coating_drops_row(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert _has_coating_node(project, _COATED)
    assert coating_actions.remove_coating(project, _COATED)
    assert not _has_coating_node(project, _COATED)


def test_remove_coating_on_uncoated_is_noop(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert coating_actions.remove_coating(project, _UNCOATED) is False


# ---------------------------------------------------------------------------
# Per-model cell gating + Model discriminator seeding
# ---------------------------------------------------------------------------

def _model_slot():
    return next(s for s in rs.COATING_SCHEMA.slots if s.key == "coating_model")


def test_coated_surface_reads_simple(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    assert rs._coating_ui_model_of(ctx) == int(CoatingModelUI.SIMPLE)


def test_switch_to_artist_seeds_defaults(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    node = _coating_node(project, _COATED)
    ctx = _ctx(project, node)
    with project.edit("Set Coating Model"):
        res = _model_slot().write(ctx, int(CoatingModelUI.ARTIST))
    assert res.changed and res.requires_reset

    surf = project.system.surfaces[_COATED]
    assert int(surf.coating.model) == int(ghostlight.CoatingModel.ARTIST)
    assert surf.coating.tint_strength == pytest.approx(0.04)
    strength_slot = next(
        s for s in rs.COATING_SCHEMA.slots if s.key == "coating_strength")
    layers_slot = next(
        s for s in rs.COATING_SCHEMA.slots if s.key == "coating_ar_layers")
    assert strength_slot.editable(ctx) is True
    assert layers_slot.get(ctx) is None


def test_switch_to_spectral_seeds_table(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(CoatingModelUI.SPECTRAL))
    assert int(project.system.surfaces[_COATED].coating.model) == \
        int(ghostlight.CoatingModel.SPECTRAL)
    data_slot = next(s for s in rs.COATING_SCHEMA.slots if s.key == "coating_data")
    assert "λ table" in (data_slot.get(ctx) or "")


def test_switch_to_layer_stack_bakes(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(CoatingModelUI.LAYER_STACK))
    assert int(project.system.surfaces[_COATED].coating.model) == \
        int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)
    assert rs._coating_ui_model_of(ctx) == int(CoatingModelUI.LAYER_STACK)
    assert len(project.system.get_coating_layers(_COATED)) == 1


# ---------------------------------------------------------------------------
# Packed column layout — live slots sit in consecutive columns, no blank runs
# ---------------------------------------------------------------------------

def _packed_layout(project, surface_index: int) -> list[tuple[int, str]]:
    """``[(column, slot.label), ...]`` for the coating row, in column order."""
    node = _coating_node(project, surface_index)
    return [
        (col, rs.slot_at(node, col).label)
        for col in range(rs.column_count(project.system))
        if rs.slot_at(node, col) is not None
    ]


def test_simple_coating_row_has_no_column_gaps(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    assert _packed_layout(project, _COATED) == [
        (1, "Model"), (2, "AR Layers"), (3, "Preset"),
    ]


def test_artist_coating_row_packs_strength_and_tint(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(CoatingModelUI.ARTIST))
    assert _packed_layout(project, _COATED) == [
        (1, "Model"), (2, "Strength"), (3, "Tint"), (4, "Preset"),
    ]


@pytest.mark.parametrize("ui_model", [
    CoatingModelUI.SPECTRAL,
    CoatingModelUI.ANGULAR,
    CoatingModelUI.SPECTRAL_ANGULAR,
    CoatingModelUI.LAYER_STACK,
    CoatingModelUI.ATTENUATOR,
])
def test_table_backed_coating_rows_pack_data(qapp, sample_lens_path, ui_model):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(ui_model))
    assert _packed_layout(project, _COATED) == [
        (1, "Model"), (2, "Data"), (3, "Preset"),
    ]


def test_coating_row_reserves_no_trailing_columns(qapp, sample_lens_path):
    """A coated lens with no aspheres is exactly the canonical strip wide —
    the coating row must not push it past that."""
    project = _load(qapp, sample_lens_path)
    assert any(
        coating_actions.surface_has_coating(s) and not bool(s.is_stop)
        for s in project.system.surfaces
    )
    assert rs.max_asphere_terms_in_system(project.system) == 0
    assert rs.column_count(project.system) == len(rs.CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Tint hex write
# ---------------------------------------------------------------------------

def test_tint_hex_write(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(CoatingModelUI.ARTIST))
    tint_slot = next(s for s in rs.COATING_SCHEMA.slots if s.key == "coating_tint")
    with project.edit("Set Coating Tint"):
        res = tint_slot.write(ctx, "#FF8800")
    assert res.changed
    c = project.system.surfaces[_COATED].coating
    assert c.tint_r == pytest.approx(1.0, abs=2e-3)
    assert c.tint_g == pytest.approx(0x88 / 255, abs=2e-3)
    assert c.tint_b == pytest.approx(0.0, abs=2e-3)
    assert tint_slot.get(ctx).upper() == "#FF8800"


# ---------------------------------------------------------------------------
# Data payload write
# ---------------------------------------------------------------------------

def test_data_payload_write_spectral(qapp, sample_lens_path):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))
    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(CoatingModelUI.SPECTRAL))
    data_slot = next(s for s in rs.COATING_SCHEMA.slots if s.key == "coating_data")
    payload = json.dumps({"kind": "spectral",
                          "data": [[400, 0.05], [550, 0.01], [700, 0.05]],
                          "out_of_range_discard": False})
    with project.edit("Edit Coating Data"):
        res = data_slot.write(ctx, payload)
    assert res.changed
    table = project.system.get_coating_table(_COATED)
    assert table.shape == (3, 2)
    np.testing.assert_allclose(table[1], [550, 0.01], atol=1e-4)


# ---------------------------------------------------------------------------
# Undo round-trip (writer completeness enforcement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ui_model", [
    CoatingModelUI.ARTIST,
    CoatingModelUI.SPECTRAL,
    CoatingModelUI.ANGULAR,
    CoatingModelUI.SPECTRAL_ANGULAR,
    CoatingModelUI.LAYER_STACK,
    CoatingModelUI.ATTENUATOR,
])
def test_undo_restores_prior_coating(qapp, sample_lens_path, ui_model):
    project = _load(qapp, sample_lens_path)
    ctx = _ctx(project, _coating_node(project, _COATED))

    model_before = int(project.system.surfaces[_COATED].coating.model)

    with project.edit("Set Coating Model"):
        _model_slot().write(ctx, int(ui_model))
    assert int(project.system.surfaces[_COATED].coating.model) != model_before

    project.undo()
    # After undo the surface must be back to its original coating model —
    # proving the writer emitted the new coating and the parser read the prior
    # state back on restore.
    assert int(project.system.surfaces[_COATED].coating.model) == model_before
