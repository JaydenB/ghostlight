"""Tests for tools/build_material_catalogue.py.

Builds a tiny synthetic ``.mtrl`` fixture tree under ``tmp_path``, runs the
converter, and asserts the shape / counts / per-rule decisions in the
output JSON. Covers formula 2 (Sellmeier), formula 3 (Abbe-fallback),
formula 1 (skip), and a malformed file (skip).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_TOOLS_DIR = _HERE.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

import build_material_catalogue as bmc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FORMULA2_NBK7 = """\
# this file is part of refractiveindex.info database
# refractiveindex.info database is in the public domain
DESCRIPTION: BK7 is a very popular crown glass.
REFERENCES: "<a href=\\"http://example.com/bk7\\">BK7 catalog</a>"
COMMENTS: "step 0.5 available"
DATA:
  - type: formula 2
    wavelength_range: 0.3 2.5
    coefficients: 0 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
  - type: tabulated k
    data: |
        0.300 2.8607E-06
        0.700 8.9305E-09
SPECS:
    n_absolute: false
    wavelength_vacuum: false
    temperature: 20.0 °C
    nd: 1.5168
    Vd: 64.17
    glass_code: 517642.251
    glass_status: standard
    density: 2.51 g/cm<sup>3</sup>
"""

_FORMULA3_BACD11 = """\
# this file is part of refractiveindex.info database
REFERENCES: "<a href=\\"http://example.com/bacd11\\">BACD11 catalog</a>"
DATA:
  - type: formula 3
    wavelength_range: 0.36501 1.01398
    coefficients: 2.4095163 -0.0091904415 2 0.012939968 -2 0.0002346143 -4 -1.1130589e-05 -6 1.0131863e-06 -8
SPECS:
    nd: 1.56384
    Vd: 60.83
    glass_code: 564608
    glass_status: special
    thermal_expansion:
      - temperature_range: -30 70 °C
        coefficient: 6.6e-06 K<sup>-1</sup>
"""

_FORMULA1_IRG23 = """\
# this file is part of refractiveindex.info database
REFERENCES: "IRG23 IR chalcogenide"
DATA:
  - type: formula 1
    wavelength_range: 1.0 12
    coefficients: 3.74971 3.07065 0.498392 0.953206 40.7557
"""

_NO_DISPERSION_B270 = """\
REFERENCES: "B270 tabulated only"
DATA:
  - type: tabulated n
    data: |
        0.5 1.5230
        0.6 1.5207
"""

# Real LZOS K8 rows. The LZOS catalogue publishes no dispersion formula and
# no SPECS block at all — just n(λ) — and every table lists the d/F/C lines
# as exact rows, so the Abbe pair comes straight off the table.
_TABULATED_N_K8 = """\
# this file is part of refractiveindex.info database
REFERENCES: "<a href=\\"http://lzos.ru/\\">LZOS Clear optical glass catalogue</a>"
DATA:
  - type: tabulated n
    data: |
        0.3650 1.53582
        0.48613 1.521955
        0.58756 1.516373
        0.65627 1.513895
        2.3254 1.48878
"""


def _build_fixture(tmp_path: Path) -> Path:
    """Lay out a vendor-dump-shaped scratch tree with 4 files.

    Returns the path to the synthetic root.
    """
    root = tmp_path / "vendor_dump"
    (root / "Schott" / "Infrared").mkdir(parents=True)
    (root / "Hoya").mkdir(parents=True)

    (root / "Schott" / "N-BK7.mtrl").write_text(_FORMULA2_NBK7, encoding="utf-8")
    (root / "Schott" / "B270.mtrl").write_text(_NO_DISPERSION_B270, encoding="utf-8")
    (root / "Schott" / "Infrared" / "IRG23.mtrl").write_text(_FORMULA1_IRG23, encoding="utf-8")
    (root / "Hoya" / "BACD11.mtrl").write_text(_FORMULA3_BACD11, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_converter_end_to_end(tmp_path):
    root = _build_fixture(tmp_path)
    out = tmp_path / "out"

    rc = bmc.main([
        "--in", str(root),
        "--out", str(out),
        "--retrieved", "2026-06-25",
    ])
    assert rc == 0

    schott = json.loads((out / "schott.json").read_text(encoding="utf-8"))
    hoya = json.loads((out / "hoya.json").read_text(encoding="utf-8"))

    # Envelope
    assert schott["format"] == "ghostlight-material-catalogue"
    assert schott["version"] == {"major": 1, "minor": 0}
    assert schott["source"]["vendor"] == "Schott"
    assert schott["source"]["retrieved"] == "2026-06-25"

    # Schott: N-BK7 succeeds; B270 (tabulated-n only) and IRG23 (formula 1) skip.
    assert set(schott["materials"].keys()) == {"Schott_N-BK7"}

    bk7 = schott["materials"]["Schott_N-BK7"]
    assert bk7["display_name"] == "N-BK7"
    assert bk7["catalogue_ref"] == "Schott:N-BK7"

    # Formula 2 with c0 = 0 → emit Sellmeier verbatim.
    disp = bk7["dispersion"]
    assert disp["model"] == "sellmeier"
    assert disp["B"] == pytest.approx([1.03961212, 0.231792344, 1.01046945])
    assert disp["C"] == pytest.approx([0.00600069867, 0.0200179144, 103.560653])

    # ui block keeps SPECS-derived metadata.
    assert bk7["ui"]["nd"] == pytest.approx(1.5168)
    assert bk7["ui"]["vd"] == pytest.approx(64.17)
    assert bk7["ui"]["wavelength_range_nm"] == [300.0, 2500.0]
    assert bk7["ui"]["glass_status"] == "standard"
    assert bk7["ui"]["density_g_cm3"] == pytest.approx(2.51)

    # raw_source keeps the original formula coefficients + tabulated_k.
    assert bk7["raw_source"]["formula"]["type"] == "formula_2"
    assert bk7["raw_source"]["formula"]["wavelength_range_um"] == [0.3, 2.5]
    assert "tabulated_k" in bk7["raw_source"]
    assert bk7["raw_source"]["tabulated_k"]["wavelength_um"][0] == 0.3

    # Tags: vendor + crown (nd < 1.6) + glass_status + wavelength bins + 'crown'
    # (description match).
    assert "schott" in bk7["tags"]
    assert "crown" in bk7["tags"]
    assert "standard" in bk7["tags"]

    # Hoya: BACD11 falls back to Abbe (formula 3 isn't representable).
    assert set(hoya["materials"].keys()) == {"Hoya_BACD11"}
    bacd = hoya["materials"]["Hoya_BACD11"]
    assert bacd["dispersion"] == {"model": "abbe", "nd": 1.56384, "Vd": 60.83}
    # Polynomial coefficients preserved under raw_source.formula for plotting.
    assert bacd["raw_source"]["formula"]["type"] == "formula_3"
    assert len(bacd["raw_source"]["formula"]["coefficients"]) == 11


def test_tabulated_n_only_derives_abbe_from_the_table(tmp_path):
    """LZOS-shaped input: no formula, no SPECS — nd/Vd come off the n(λ) table."""
    root = tmp_path / "src"
    (root / "Lzos").mkdir(parents=True)
    (root / "Lzos" / "K8.mtrl").write_text(_TABULATED_N_K8, encoding="utf-8")

    assert bmc.main(["--in", str(root), "--out", str(tmp_path / "out"),
                     "--retrieved", "2026-08-03"]) == 0
    lzos = json.loads((tmp_path / "out" / "lzos.json").read_text(encoding="utf-8"))

    # Directory "Lzos" is presented as the acronym the vendor actually uses.
    assert lzos["source"]["vendor"] == "LZOS"
    k8 = lzos["materials"]["LZOS_K8"]
    assert k8["catalogue_ref"] == "LZOS:K8"

    # nd is the d-line row verbatim; Vd = (nd - 1) / (nF - nC).
    assert k8["dispersion"]["model"] == "abbe"
    assert k8["dispersion"]["nd"] == pytest.approx(1.516373)
    assert k8["dispersion"]["Vd"] == pytest.approx(
        (1.516373 - 1.0) / (1.521955 - 1.513895), rel=1e-9
    )
    assert k8["ui"]["nd"] == pytest.approx(1.516373)

    # Wavelength range comes from the table's own extent, not a formula range.
    assert k8["ui"]["wavelength_range_nm"] == pytest.approx([365.0, 2325.4])

    # No formula existed, so raw_source carries the table and says where
    # nd/Vd came from rather than implying they were vendor-published.
    assert "formula" not in k8["raw_source"]
    assert k8["raw_source"]["nd_vd_derived_from"] == "tabulated_n"
    assert k8["raw_source"]["tabulated_n"]["n"][0] == pytest.approx(1.53582)


def test_tabulated_n_not_spanning_dfc_lines_is_skipped(tmp_path):
    """Never extrapolate: a short table that misses F/C yields no entry.

    B270 in the main fixture is exactly this case — its table stops at
    0.6 µm, short of the C line at 0.65627 µm.
    """
    root = tmp_path / "src"
    (root / "Vendor").mkdir(parents=True)
    (root / "Vendor" / "B270.mtrl").write_text(_NO_DISPERSION_B270, encoding="utf-8")

    warnings: list[str] = []
    envelope, counts = bmc.build_vendor_catalogue(
        vendor="Vendor", vendor_dir=root / "Vendor", repo_root=root.parent,
        retrieved="2026-08-03", warnings=warnings,
    )
    assert envelope["materials"] == {}
    assert sum(counts.values()) == 0
    assert any("no tabulated n spanning the d/F/C lines" in w for w in warnings)


def test_interp_n_refuses_to_extrapolate():
    xs, ns = [0.5, 0.6, 0.7], [1.5230, 1.5207, 1.5190]
    assert bmc._interp_n(xs, ns, 0.6) == pytest.approx(1.5207)      # exact node
    assert bmc._interp_n(xs, ns, 0.55) == pytest.approx(1.52185)    # midpoint
    assert bmc._interp_n(xs, ns, 0.49) is None                      # below
    assert bmc._interp_n(xs, ns, 0.71) is None                      # above


def test_vendor_dirs_are_discovered_not_hardcoded(tmp_path):
    """A dump holding only some vendors converts those without warning about
    the absent ones, and an unknown vendor directory still converts."""
    root = tmp_path / "src"
    (root / "Cdgm").mkdir(parents=True)
    (root / "Newvendor").mkdir(parents=True)
    (root / "NotAVendor").mkdir(parents=True)  # no .mtrl files — ignored
    (root / "Cdgm" / "H-K9L.mtrl").write_text(_FORMULA3_BACD11, encoding="utf-8")
    (root / "Newvendor" / "X.mtrl").write_text(_FORMULA3_BACD11, encoding="utf-8")

    out = tmp_path / "out"
    assert bmc.main(["--in", str(root), "--out", str(out),
                     "--retrieved", "2026-08-03"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["cdgm.json", "newvendor.json"]
    cdgm = json.loads((out / "cdgm.json").read_text(encoding="utf-8"))
    assert "CDGM_H-K9L" in cdgm["materials"]


def test_converter_missing_input_errors_with_clear_message(tmp_path, capsys):
    rc = bmc.main([
        "--in", str(tmp_path / "does-not-exist"),
        "--out", str(tmp_path / "out"),
    ])
    err = capsys.readouterr().err
    assert rc == 2
    assert "does not exist" in err
    assert "one" in err and "per vendor" in err


def test_converter_overwrites_existing_output(tmp_path):
    """A second run must produce identical output (deterministic) and
    overwrite — no stale entries left behind from a prior run."""
    root = _build_fixture(tmp_path)
    out = tmp_path / "out"

    assert bmc.main(["--in", str(root), "--out", str(out), "--retrieved", "2026-06-25"]) == 0
    first = (out / "schott.json").read_text(encoding="utf-8")
    # Pollute the output with a stale extra material that should be wiped.
    polluted = json.loads(first)
    polluted["materials"]["Schott_GHOST"] = {"display_name": "Ghost"}
    (out / "schott.json").write_text(json.dumps(polluted), encoding="utf-8")

    assert bmc.main(["--in", str(root), "--out", str(out), "--retrieved", "2026-06-25"]) == 0
    after = (out / "schott.json").read_text(encoding="utf-8")
    assert after == first  # deterministic
    parsed = json.loads(after)
    assert "Schott_GHOST" not in parsed["materials"]


def test_formula2_with_nonzero_c0_falls_back_to_abbe(tmp_path):
    """A 5-coefficient formula-2 entry can't fit our 3-term Sellmeier slot,
    but if SPECS has nd/Vd we still emit Abbe instead of dropping it."""
    root = tmp_path / "vendor_dump"
    (root / "Vendor").mkdir(parents=True)
    (root / "Vendor" / "Weird.mtrl").write_text(
        "DATA:\n"
        "  - type: formula 2\n"
        "    wavelength_range: 0.4 0.7\n"
        "    coefficients: 0.5 2.0 0.01 1.0 0.05\n"  # c0 != 0, 5 coeffs
        "SPECS:\n"
        "    nd: 1.55\n"
        "    Vd: 50.0\n",
        encoding="utf-8",
    )
    # Vendor isn't in VENDORS, so bypass the CLI and call build_vendor_catalogue
    # directly so we exercise the formula-2 fallback path.
    warnings: list[str] = []
    envelope, counts = bmc.build_vendor_catalogue(
        vendor="Vendor",
        vendor_dir=root / "Vendor",
        repo_root=root.parent,
        retrieved="2026-06-25",
        warnings=warnings,
    )
    assert counts["abbe"] == 1
    assert envelope["materials"]["Vendor_Weird"]["dispersion"] == {
        "model": "abbe", "nd": 1.55, "Vd": 50.0,
    }
    # The fallback should be noted in the warnings stream for visibility.
    assert any("formula 2 not representable" in w for w in warnings)
