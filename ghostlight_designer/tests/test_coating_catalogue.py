"""Tests for the coating catalogue: bundled parse, search, user overlay, and
applying a preset payload through the coating-row preset writer."""
from __future__ import annotations

import json

import numpy as np
import pytest

import ghostlight

from ghostlight_designer.coating_catalogue import (
    CatalogueCoating,
    CoatingCatalogue,
    get_coating_catalogue,
    reset_singleton,
)
from ghostlight_designer.project import Project
from ghostlight_designer.optical_editor import row_schemas as rs
from ghostlight_designer.optical_editor.nodes import (
    CoatingFormNode,
    SurfaceNode,
    build_tree,
)


# ---------------------------------------------------------------------------
# Bundled catalogue
# ---------------------------------------------------------------------------

def test_bundled_catalogue_loads():
    cat = CoatingCatalogue.load_bundled()
    assert len(cat) >= 6
    # A few keys we ship.
    assert cat.by_key("uncoated") is not None
    assert cat.by_key("artist_vintage_amber") is not None
    assert cat.by_key("layers_mgf2_quarterwave") is not None


def test_search_by_tag_and_query():
    cat = CoatingCatalogue.load_bundled()
    artist = cat.search(tag="artist")
    assert artist and all("artist" in c.tags for c in artist)
    amber = cat.search(query="amber")
    assert any(c.key == "artist_vintage_amber" for c in amber)


def test_payload_shapes_are_valid():
    cat = CoatingCatalogue.load_bundled()
    for c in cat.all():
        assert c.payload.get("type") == "coating"
        # Either a model discriminator or a layer stack.
        assert "model" in c.payload or "layers" in c.payload


def test_singleton_resets():
    reset_singleton()
    a = get_coating_catalogue()
    b = get_coating_catalogue()
    assert a is b
    reset_singleton()
    c = get_coating_catalogue()
    assert c is not a


# ---------------------------------------------------------------------------
# User overlay wins
# ---------------------------------------------------------------------------

def test_user_overlay_wins(tmp_path):
    user = {
        "format": "ghostlight-coating-catalogue",
        "version": {"major": 1, "minor": 0},
        "source": "User",
        "coatings": [
            {
                "key": "uncoated",  # collides with a bundled key
                "display_name": "My Override",
                "tags": ["user"],
                "description": "overridden",
                "payload": {"type": "coating", "model": "simple", "ar_layers": 0},
            }
        ],
    }
    p = tmp_path / "user_coatings.json"
    p.write_text(json.dumps(user), encoding="utf-8")
    cat = CoatingCatalogue.load_with_user(p)
    entry = cat.by_key("uncoated")
    assert entry is not None
    assert entry.display_name == "My Override"


# ---------------------------------------------------------------------------
# Applying a preset through the coating-row preset writer
# ---------------------------------------------------------------------------

def _coating_ctx(project, surface_index):
    root = build_tree(project.system)
    for el in root.children:
        for surf in el.children:
            if isinstance(surf, SurfaceNode) and surf.surface_index == surface_index:
                for child in surf.children:
                    if isinstance(child, CoatingFormNode):
                        return rs.SlotContext(node=child, system=project.system,
                                              project=project)
    raise AssertionError("no coating node")


def _preset_slot():
    return next(s for s in rs.COATING_SCHEMA.slots if s.key == "coating_preset")


def test_apply_artist_preset(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ctx = _coating_ctx(project, 0)
    preset = get_coating_catalogue().by_key("artist_vintage_amber")
    with project.edit("Apply Coating Preset"):
        res = _preset_slot().write(ctx, preset.payload_json())
    assert res.changed
    c = project.system.surfaces[0].coating
    assert int(c.model) == int(ghostlight.CoatingModel.ARTIST)
    assert c.tint_r == pytest.approx(1.0, abs=2e-3)
    assert c.tint_g == pytest.approx(0.6, abs=2e-3)


def test_apply_layer_stack_preset(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ctx = _coating_ctx(project, 0)
    preset = get_coating_catalogue().by_key("layers_v_coat_broadband")
    with project.edit("Apply Coating Preset"):
        res = _preset_slot().write(ctx, preset.payload_json())
    assert res.changed
    assert int(project.system.surfaces[0].coating.model) == \
        int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)
    assert len(project.system.get_coating_layers(0)) == 3


def test_apply_spectral_preset(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    ctx = _coating_ctx(project, 0)
    preset = get_coating_catalogue().by_key("broadband_ar_spectral")
    with project.edit("Apply Coating Preset"):
        _preset_slot().write(ctx, preset.payload_json())
    c = project.system.surfaces[0].coating
    assert int(c.model) == int(ghostlight.CoatingModel.SPECTRAL)
    assert c.table_count == 7
