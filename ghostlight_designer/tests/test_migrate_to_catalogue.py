"""Tests for ``migrate_to_catalogue`` — passive identification + active re-link.

Builds a synthetic catalogue (no .mtrl roundtrip) plus a small Project with
hand-authored ``_raw_glass_catalogue`` entries, then exercises every match
rule (catalogue_ref, exact dispersion, loose nd/Vd, name token) and the
active relink path including undo.
"""
from __future__ import annotations

import ghostlight
import pytest

from ghostlight_designer.material_catalogue import (
    CatalogueMaterial,
    MaterialCatalogue,
)
from ghostlight_designer.migrate_to_catalogue import (
    analyze_system,
    apply_relink,
    apply_relink_bulk,
    match_entry,
)
from ghostlight_designer.optical_editor import element_actions
from ghostlight_designer.project import Project


# ---------------------------------------------------------------------------
# Catalogue / project fixtures
# ---------------------------------------------------------------------------

def _make_catalogue() -> MaterialCatalogue:
    """A tiny hand-rolled catalogue covering a Sellmeier glass + Abbe glass."""
    bk7 = CatalogueMaterial(
        key="Schott_N-BK7",
        display_name="N-BK7",
        catalogue_ref="Schott:N-BK7",
        source_vendor="Schott",
        tags=("schott", "crown"),
        dispersion={
            "model": "sellmeier",
            "B": [1.03961212, 0.231792344, 1.01046945],
            "C": [0.00600069867, 0.0200179144, 103.560653],
        },
        nd=1.5168, vd=64.17,
        wavelength_range_nm=(300.0, 2500.0),
        glass_code="517642.251", glass_status="standard",
        density_g_cm3=2.51, description="BK7 crown glass.",
        comments="", references="", source_file="",
        raw_source=None,
    )
    sf5 = CatalogueMaterial(
        key="Schott_SF5",
        display_name="SF5",
        catalogue_ref="Schott:SF5",
        source_vendor="Schott",
        tags=("schott", "flint"),
        dispersion={"model": "abbe", "nd": 1.6727, "Vd": 32.21},
        nd=1.6727, vd=32.21,
        wavelength_range_nm=None,
        glass_code="", glass_status="standard",
        density_g_cm3=None, description="SF5 flint glass.",
        comments="", references="", source_file="",
        raw_source=None,
    )
    return MaterialCatalogue([bk7, sf5])


# ---------------------------------------------------------------------------
# match_entry per-rule coverage
# ---------------------------------------------------------------------------

def test_match_by_catalogue_ref_wins_over_weaker_rules():
    cat = _make_catalogue()
    entry = {
        "name":          "Some Name That Won't Match",
        "catalogue_ref": "Schott:N-BK7",
        "dispersion":    {"model": "abbe", "nd": 1.0, "Vd": 1.0},  # wrong on purpose
    }
    report = match_entry("Local_BK7", entry, cat)
    assert report.is_linked
    assert report.best.rule == "catalogue_ref"
    assert report.best.candidate.key == "Schott_N-BK7"


def test_match_by_exact_sellmeier_dispersion():
    cat = _make_catalogue()
    entry = {
        "name": "Different Name",
        "dispersion": {
            "model": "sellmeier",
            "B": [1.03961212, 0.231792344, 1.01046945],
            "C": [0.00600069867, 0.0200179144, 103.560653],
        },
    }
    report = match_entry("Local_X", entry, cat)
    assert report.best is not None
    assert report.best.rule == "dispersion"
    assert report.best.candidate.key == "Schott_N-BK7"


def test_match_by_loose_nd_vd_resolves_auto_generated_key():
    """Auto-generated keys like ``G_ND1.6727_VD32.21`` from Zemax conversion
    encode the Abbe pair the catalogue indexes on at glass_code precision."""
    cat = _make_catalogue()
    # Drift the pair past the tight dispersion-equality tolerance (1e-4) but
    # within the loose nd/Vd-only tolerance — that's the band rule 3 owns.
    entry = {
        "dispersion": {"model": "abbe", "nd": 1.6724, "Vd": 32.23},
    }
    report = match_entry("G_ND1.6727_VD32.21", entry, cat)
    assert report.best is not None
    assert report.best.rule == "nd_vd"
    assert report.best.candidate.key == "Schott_SF5"


def test_match_by_name_token_strips_vendor_prefix():
    cat = _make_catalogue()
    entry = {"name": "N-BK7", "dispersion": {"model": "abbe", "nd": 99.0, "Vd": 99.0}}
    report = match_entry("Hand-Typed-BK7", entry, cat)
    # nd/Vd are nonsense; the only thing that hits is the name token.
    name_match = next((m for m in report.matches if m.rule == "name"), None)
    assert name_match is not None
    assert name_match.candidate.key == "Schott_N-BK7"


def test_ambiguous_nd_vd_match_is_flagged():
    """Two catalogue entries within tolerance ⇒ report flags ambiguity so
    the picker can't silently auto-pick the wrong one."""
    # Build a catalogue where two materials share nd/Vd within 1e-3.
    a = CatalogueMaterial(
        key="A", display_name="A", catalogue_ref="X:A", source_vendor="X",
        tags=(), dispersion={"model": "abbe", "nd": 1.5, "Vd": 60.0},
        nd=1.5, vd=60.0, wavelength_range_nm=None,
        glass_code="", glass_status="", density_g_cm3=None,
        description="", comments="", references="", source_file="",
        raw_source=None,
    )
    b = CatalogueMaterial(
        key="B", display_name="B", catalogue_ref="X:B", source_vendor="X",
        tags=(), dispersion={"model": "abbe", "nd": 1.5004, "Vd": 60.0003},
        nd=1.5004, vd=60.0003, wavelength_range_nm=None,
        glass_code="", glass_status="", density_g_cm3=None,
        description="", comments="", references="", source_file="",
        raw_source=None,
    )
    cat = MaterialCatalogue([a, b])
    # Drift the entry so neither candidate hits dispersion-equality (1e-4)
    # but both still fall inside the loose nd/Vd tolerance — that forces
    # both matches into the same rule and exercises the ambiguity flag.
    report = match_entry("Local", {
        "dispersion": {"model": "abbe", "nd": 1.5002, "Vd": 60.0002},
    }, cat)
    assert len(report.matches) == 2
    assert all(m.rule == "nd_vd" for m in report.matches)
    assert report.is_ambiguous


# ---------------------------------------------------------------------------
# analyze_system / apply_relink
# ---------------------------------------------------------------------------

def test_analyze_system_walks_full_catalogue(qapp):
    project = Project()
    # Insert a singlet so the system has at least one material + catalogue entry.
    element_actions.add_singlet(project)
    cat = _make_catalogue()
    reports = analyze_system(project.system, cat)
    assert len(reports) == 1
    report = reports[0]
    assert report.project_key == "Schott_N-BK7"
    # The singlet was registered with the catalogue_ref already set, so this
    # report is the "linked" case.
    assert report.is_linked


def test_apply_relink_rewrites_catalogue_and_material_references(qapp):
    project = Project()
    element_actions.add_doublet(project)  # uses Schott_N-BK7 + Schott_SF5

    cat = _make_catalogue()
    sf5_target = cat.by_key("Schott_SF5")

    # Pretend the project ships a legacy auto-generated key for the flint.
    system = project.system
    system._raw_glass_catalogue["G_ND1.6727_VD32.21"] = {
        "name": "G_ND1.6727_VD32.21",
        "dispersion": {"model": "abbe", "nd": 1.6727, "Vd": 32.21},
    }
    el = system._elements[0]
    flint_index = el.material_glasses.index("Schott_SF5")
    el.material_glasses[flint_index] = "G_ND1.6727_VD32.21"
    # Mirror the on-disk shape: the new key now refers to the auto-generated
    # entry but the old one is still in catalogue.
    project._set_dirty(False)
    project._clear_history()

    assert apply_relink(project, "G_ND1.6727_VD32.21", sf5_target) is True
    # The element now references the catalogue key.
    assert "Schott_SF5" in el.material_glasses
    assert "G_ND1.6727_VD32.21" not in el.material_glasses
    # The legacy entry is gone, replaced by the catalogue's canonical entry.
    assert "G_ND1.6727_VD32.21" not in system._raw_glass_catalogue
    relinked = system._raw_glass_catalogue["Schott_SF5"]
    assert relinked["catalogue_ref"] == "Schott:SF5"
    assert relinked["dispersion"] == sf5_target.dispersion


def test_apply_relink_is_undoable(qapp):
    project = Project()
    element_actions.add_singlet(project)
    cat = _make_catalogue()
    bk7 = cat.by_key("Schott_N-BK7")

    # Force a re-link from the project key to itself with a new catalogue
    # entry so undo has something visible to restore.
    project.system._raw_glass_catalogue["Schott_N-BK7"]["dispersion"] = {
        "model": "abbe", "nd": 1.5, "Vd": 60.0,
    }
    project._set_dirty(False)
    project._clear_history()

    assert apply_relink(project, "Schott_N-BK7", bk7) is True
    # Now the catalogue entry holds the bundled Sellmeier dispersion.
    assert project.system._raw_glass_catalogue["Schott_N-BK7"]["dispersion"]["model"] == "sellmeier"

    project.undo()
    restored = project.system._raw_glass_catalogue["Schott_N-BK7"]
    assert restored["dispersion"]["model"] == "abbe"


def test_apply_relink_bulk_collapses_into_one_undo_entry(qapp):
    project = Project()
    element_actions.add_doublet(project)
    cat = _make_catalogue()

    # Drop two legacy keys into the catalogue + re-point element references
    # so bulk relink has something to do.
    sys = project.system
    sys._raw_glass_catalogue["Legacy_BK7"] = {
        "name": "Legacy_BK7",
        "dispersion": {"model": "abbe", "nd": 1.5168, "Vd": 64.17},
    }
    sys._raw_glass_catalogue["Legacy_SF5"] = {
        "name": "Legacy_SF5",
        "dispersion": {"model": "abbe", "nd": 1.6727, "Vd": 32.21},
    }
    el = sys._elements[0]
    el.material_glasses[el.material_glasses.index("Schott_N-BK7")] = "Legacy_BK7"
    el.material_glasses[el.material_glasses.index("Schott_SF5")] = "Legacy_SF5"
    project._set_dirty(False)
    project._clear_history()

    decisions = [
        ("Legacy_BK7", cat.by_key("Schott_N-BK7")),
        ("Legacy_SF5", cat.by_key("Schott_SF5")),
    ]
    n = apply_relink_bulk(project, decisions)
    assert n == 2
    assert el.material_glasses == ["Schott_N-BK7", "Schott_SF5"]
    # One bulk undo should restore both legacy keys at once. After the
    # snapshot-reload, ``el`` is orphaned — re-fetch the live element from
    # the system, which now holds the restored material list.
    assert project.can_undo
    project.undo()
    restored_el = project.system._elements[0]
    assert restored_el.material_glasses == ["Legacy_BK7", "Legacy_SF5"]


def test_apply_relink_returns_false_when_key_not_present(qapp):
    project = Project()
    cat = _make_catalogue()
    # System has no glasses at all yet.
    assert apply_relink(project, "Nonexistent", cat.by_key("Schott_N-BK7")) is False
