"""Tests for ``material_catalogue.MaterialCatalogue``.

Covers the loader, the dataclass surface, and the Sellmeier-vs-source-SPECS
backfill-agreement sweep. The backfill sweep
re-derives (nd, Vd) from each Sellmeier entry's coefficients via the same
formula ghostlight's C++ ``sellmeier_to_abbe`` uses, and asserts the result
matches the source SPECS values.
"""
from __future__ import annotations

import json
import math

import pytest

from ghostlight_designer.material_catalogue import (
    CATALOGUE_FORMAT,
    CATALOGUE_MAJOR_VERSION,
    CatalogueMaterial,
    MaterialCatalogue,
    get_catalogue,
    reset_singleton,
)


# ---------------------------------------------------------------------------
# Reference: Python copy of fresnel.h:53 sellmeier_n / optical_system.cpp:62
# sellmeier_to_abbe — used for the backfill agreement sweep.
# ---------------------------------------------------------------------------

_LAMBDA_D_NM = 587.56
_LAMBDA_F_NM = 486.13
_LAMBDA_C_NM = 656.27


def _sellmeier_n(B, C, lambda_nm: float) -> float:
    lam_um = lambda_nm * 1e-3
    lam2 = lam_um * lam_um
    n2 = 1.0
    for b, c in zip(B, C):
        n2 += float(b) * lam2 / (lam2 - float(c))
    return math.sqrt(n2) if n2 > 0.0 else 1.0


def _sellmeier_to_abbe(B, C) -> tuple[float, float]:
    nd = _sellmeier_n(B, C, _LAMBDA_D_NM)
    nF = _sellmeier_n(B, C, _LAMBDA_F_NM)
    nC = _sellmeier_n(B, C, _LAMBDA_C_NM)
    return nd, (nd - 1.0) / (nF - nC)


# ---------------------------------------------------------------------------
# Bundled-catalogue tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton_each_test():
    reset_singleton()
    yield
    reset_singleton()


def test_bundled_catalogue_loads_and_indexes_correctly():
    cat = MaterialCatalogue.load_bundled()
    # 162 Schott + 410 Ohara + 190 Hoya + 237 CDGM + 363 Hikari + 51 LZOS
    # ≈ 1413 at time of writing.
    assert len(cat) > 1400
    # Schott_N-BK7 is the canonical sanity-check entry.
    bk7 = cat.by_key("Schott_N-BK7")
    assert bk7 is not None
    assert bk7.display_name == "N-BK7"
    assert bk7.catalogue_ref == "Schott:N-BK7"
    assert bk7.source_vendor == "Schott"
    assert bk7.dispersion["model"] == "sellmeier"
    assert bk7.nd == pytest.approx(1.5168)
    assert bk7.vd == pytest.approx(64.17)
    # by_ref index hit too.
    assert cat.by_ref("Schott:N-BK7") is bk7


def test_every_bundled_vendor_is_present_and_keys_do_not_collide():
    """All six shipped vendor files load, and no two claim the same key.

    Keys are vendor-namespaced precisely so that same-named glasses across
    vendors (F1/F4 exist in both CDGM and LZOS; BAF4 in both CDGM and
    Hikari) stay distinct. A collision would silently drop one entry,
    since the catalogue is last-write-wins.
    """
    cat = MaterialCatalogue.load_bundled()
    counts: dict[str, int] = {}
    for m in cat:
        counts[m.source_vendor] = counts.get(m.source_vendor, 0) + 1
    assert set(counts) == {"Schott", "Ohara", "Hoya", "CDGM", "Hikari", "LZOS"}

    # A representative from each of the three newer families.
    for key, ref, vendor in (
        ("CDGM_H-K9L",   "CDGM:H-K9L",   "CDGM"),
        ("Hikari_BAF10", "Hikari:BAF10", "Hikari"),
        ("LZOS_K8",      "LZOS:K8",      "LZOS"),
    ):
        m = cat.by_key(key)
        assert m is not None, key
        assert m.catalogue_ref == ref
        assert m.source_vendor == vendor
        assert m.dispersion["model"] in ("abbe", "sellmeier")
        assert m.nd is not None and m.vd is not None

    # Same display name, different vendor, distinct keys — the namespacing
    # rule doing its job.
    assert cat.by_key("CDGM_F1").nd != cat.by_key("LZOS_F1").nd

    # by_key is a dict so collisions can't be counted after the fact; walk
    # the raw files instead.
    from importlib import resources
    import json as _json
    seen: dict[str, str] = {}
    for entry in resources.files("ghostlight_designer.resources.materials").iterdir():
        if not entry.name.endswith(".json"):
            continue
        with entry.open("r", encoding="utf-8") as fh:
            envelope = _json.load(fh)
        for key in envelope["materials"]:
            assert key not in seen, f"{key} in both {seen.get(key)} and {entry.name}"
            seen[key] = entry.name
    assert len(seen) == len(cat)


def test_bundled_abbe_entries_are_physically_sane():
    """Guard against a conversion bug producing nonsense nd/Vd.

    Vd derived off a tabulated n(λ) curve (LZOS) is the path most likely to
    go wrong silently — a sign flip or a near-zero nF-nC would land here.
    """
    cat = MaterialCatalogue.load_bundled()
    for m in cat:
        if m.dispersion["model"] != "abbe":
            continue
        nd, vd = m.dispersion["nd"], m.dispersion["Vd"]
        assert 1.3 < nd < 2.3, f"{m.key}: implausible nd {nd}"
        assert 15.0 < vd < 100.0, f"{m.key}: implausible Vd {vd}"
        # ui.nd / ui.vd mirror the active model for every abbe entry.
        assert m.nd == pytest.approx(nd)
        assert m.vd == pytest.approx(vd)


def test_get_catalogue_singleton_is_cached():
    a = get_catalogue()
    b = get_catalogue()
    assert a is b
    reset_singleton()
    c = get_catalogue()
    assert c is not a


def test_search_filters_compose():
    cat = MaterialCatalogue.load_bundled()
    schott = cat.search(vendor="schott")
    assert all(m.source_vendor == "Schott" for m in schott)
    # ANDed with tag.
    flints_at_schott = cat.search(vendor="schott", tag="flint")
    assert flints_at_schott
    assert all("flint" in m.tags for m in flints_at_schott)
    # Free-text query hits display_name + description haystack.
    bk_hits = cat.search(query="N-BK7")
    assert any(m.key == "Schott_N-BK7" for m in bk_hits)


def test_lens_catalogue_entry_omits_raw_source():
    """The writer must only see ``name`` / ``catalogue_ref`` / ``dispersion``
    — raw_source MUST NOT leak into .lens files."""
    cat = MaterialCatalogue.load_bundled()
    bk7 = cat.by_key("Schott_N-BK7")
    entry = bk7.lens_catalogue_entry()
    assert set(entry.keys()) == {"name", "catalogue_ref", "dispersion"}
    assert entry["dispersion"] == bk7.dispersion
    # Mutating the returned dict must not affect the catalogue's copy.
    entry["dispersion"]["B"][0] = 999.0
    assert cat.by_key("Schott_N-BK7").dispersion["B"][0] != 999.0


def test_user_catalogue_overrides_bundled_by_key(tmp_path):
    user_path = tmp_path / "user.json"
    user_path.write_text(json.dumps({
        "format":  CATALOGUE_FORMAT,
        "version": {"major": CATALOGUE_MAJOR_VERSION, "minor": 0},
        "source":  {"vendor": "User"},
        "materials": {
            "Schott_N-BK7": {
                "display_name":  "Customised BK7",
                "catalogue_ref": "User:BK7",
                "tags":          ["user"],
                "dispersion":    {"model": "abbe", "nd": 1.6, "Vd": 50.0},
                "ui":            {"nd": 1.6, "vd": 50.0},
                "raw_source":    None,
            },
        },
    }), encoding="utf-8")

    cat = MaterialCatalogue.load_with_user(user_path)
    bk7 = cat.by_key("Schott_N-BK7")
    assert bk7.display_name == "Customised BK7"
    assert bk7.dispersion == {"model": "abbe", "nd": 1.6, "Vd": 50.0}
    assert bk7.source_vendor == "User"


def test_invalid_envelope_version_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "format":  CATALOGUE_FORMAT,
        "version": {"major": 999},
        "materials": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="major version"):
        MaterialCatalogue.load_from_files([bad])


# ---------------------------------------------------------------------------
# Backfill agreement: derived (nd, Vd) ≈ SPECS (nd, Vd) for every Sellmeier
# entry in the bundled catalogue. This is the §5e sweep — it catches both
# upstream transcription errors and Sellmeier-vs-source-formula confusion
# from the converter.
# ---------------------------------------------------------------------------

def test_sellmeier_entries_agree_with_source_specs_nd_vd():
    cat = MaterialCatalogue.load_bundled()
    drifted: list[tuple[str, float, float]] = []
    n_checked = 0
    for m in cat:
        if m.dispersion.get("model") != "sellmeier":
            continue
        if m.nd is None or m.vd is None:
            continue
        derived_nd, derived_vd = _sellmeier_to_abbe(
            m.dispersion["B"], m.dispersion["C"],
        )
        n_checked += 1
        if abs(derived_nd - m.nd) > 1e-3 or abs(derived_vd - m.vd) > 5e-2:
            drifted.append((m.key, abs(derived_nd - m.nd), abs(derived_vd - m.vd)))

    # Sanity: we ran across at least a few hundred entries.
    assert n_checked > 300, f"only {n_checked} sellmeier entries checked"
    # Tolerance for nd: 1e-3 (per design doc §3b). Vd is more sensitive
    # because (nd-1) / (nF-nC) amplifies small absolute errors in nF/nC, so
    # we allow 5e-2; both upstream rounding (SPECS values are typically
    # quoted to 2 decimal places) and float32 vs. float64 arithmetic
    # contribute to the looser bound.
    assert not drifted, (
        f"{len(drifted)}/{n_checked} Sellmeier entries drifted; first 5: "
        + ", ".join(f"{k}(Δnd={a:.2e}, ΔVd={b:.2e})" for k, a, b in drifted[:5])
    )
