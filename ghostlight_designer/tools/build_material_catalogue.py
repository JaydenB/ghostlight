#!/usr/bin/env python3
"""Convert refractiveindex.info ``.mtrl`` files to ghostlight material-catalogue JSON.

Dev tool only; ``tools/`` is a sibling of the ``ghostlight_designer/`` package
and is **not** packaged into the runtime wheel. Run once during the initial
material-catalogue rollout, then re-run later if/when we re-vendor an
updated source dump. Each run overwrites ``<out>/<vendor>.json`` in place.

Usage:
    python ghostlight_designer/tools/build_material_catalogue.py \
        --in  <dump-directory> \
        --out ghostlight_designer/ghostlight_designer/resources/materials

``--in`` may be any dump directory: vendor subdirectories are **discovered**
(any immediate subdir holding ``.mtrl`` files), not looked up from a fixed
list, so a partial dump converts what it holds and leaves the JSON of
vendors it doesn't contain alone, so each vendor family can be converted
from its own dump without disturbing the others.

The contract for the output envelope and the per-material shape is in
``ghostlight_designer/ghostlight_designer/resources/materials/README.md``.

Requires PyYAML (dev-only dependency). Every ``.mtrl`` file in both source
dumps parses as valid YAML, so we use ``yaml.safe_load`` rather than
rolling our own line-by-line parser.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "build_material_catalogue requires PyYAML. Install with: pip install pyyaml"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VENDORS = ("Schott", "Ohara", "Hoya", "Cdgm", "Hikari", "Lzos")

# Source directories are named in Title case; a few vendors are acronyms and
# should read that way in the picker's vendor column (and in the catalogue
# keys, which embed the vendor name). Unlisted vendors use the directory
# name verbatim. The output filename is always ``<display>.lower().json``.
_VENDOR_DISPLAY_NAMES = {
    "Cdgm": "CDGM",
    "Lzos": "LZOS",
}

# Spectral lines used to derive an Abbe pair from a tabulated n(λ) curve
# when a source file carries no dispersion formula at all (LZOS). Values in
# µm, matching the `wavelength_um` axis of the parsed tables.
_LINE_D_UM = 0.58756
_LINE_F_UM = 0.48613
_LINE_C_UM = 0.65627

# Treat below this cutoff as "crown", at/above as "flint". A coarse default;
# vendor-specific naming conventions are richer but this is a useful filter.
_CROWN_FLINT_CUTOFF_ND = 1.6

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Substring-keyed tags applied against DESCRIPTION + COMMENTS (lower-cased).
_DESCRIPTION_TAGS: tuple[tuple[str, str], ...] = (
    ("chalcogenide", "chalcogenide"),
    ("fluoride",     "fluoride"),
    ("fused silica", "silica"),
    ("borosilicate", "borosilicate"),
    ("crown",        "crown"),
    ("flint",        "flint"),
    ("lead-free",    "eco"),
    ("eco",          "eco"),
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _strip_html(text: Any) -> str:
    """Drop HTML tags and collapse whitespace. Returns empty for ``None``.

    Replaces tags with a space (not the empty string) so consecutive
    ``</a><br>`` doesn't fuse adjacent link texts together.
    """
    if not text:
        return ""
    out = _HTML_TAG_RE.sub(" ", str(text))
    return re.sub(r"\s+", " ", out).strip()


def _parse_num(tok: str):
    """Parse ``tok`` as int if it has no decimal/exponent, else float."""
    if "." in tok or "e" in tok.lower():
        return float(tok)
    return int(tok)


def _parse_num_list(value: Any) -> List:
    """Whitespace-separated numbers ``"0 1.039 0.006"`` → ``[0, 1.039, 0.006]``.

    Stops at the first non-numeric token, so ``"-30 70 °C"`` parses cleanly
    as ``[-30, 70]``. Used both for dispersion coefficients (always pure
    numbers) and SPECS fields that trail a unit (``temperature_range``).
    """
    if value is None:
        return []
    out: list = []
    for tok in str(value).split():
        try:
            out.append(_parse_num(tok))
        except ValueError:
            break
    return out


def _parse_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value))
    return float(m.group(0)) if m else None


def _parse_wavelength_range(value: Any) -> Optional[List[float]]:
    """Parse ``"0.3 2.5"`` → ``[0.3, 2.5]`` (µm)."""
    nums = _parse_num_list(value)
    if len(nums) < 2:
        return None
    return [float(nums[0]), float(nums[1])]


def _normalise_formula_type(raw: Any) -> Optional[str]:
    """``"formula 2"`` → ``"formula_2"``; unknown → ``None``."""
    if not raw:
        return None
    s = str(raw).strip().lower().replace(" ", "_")
    if s in ("formula_1", "formula_2", "formula_3"):
        return s
    return None


# ---------------------------------------------------------------------------
# Per-block parsers
# ---------------------------------------------------------------------------

def _parse_tabulated_block(block: Any) -> Optional[dict]:
    """Convert the ``data: |`` text of a tabulated_k / tabulated_n entry to
    parallel ``wavelength_um`` / ``k`` (or ``n``) arrays."""
    if not block:
        return None
    xs, ys = [], []
    for line in str(block).splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            xs.append(float(toks[0]))
            ys.append(float(toks[1]))
        except ValueError:
            continue
    if not xs:
        return None
    return {"wavelength_um": xs, "k": ys}


def _find_tabulated_n(data_list: Any) -> Optional[Tuple[List[float], List[float]]]:
    """First ``tabulated n`` entry as sorted ``(wavelength_um, n)`` arrays."""
    if not isinstance(data_list, list):
        return None
    for entry in data_list:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type", "")).strip().lower() != "tabulated n":
            continue
        block = _parse_tabulated_block(entry.get("data"))
        if not block or len(block["wavelength_um"]) < 2:
            continue
        pairs = sorted(zip(block["wavelength_um"], block["k"]))
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return None


def _interp_n(xs: List[float], ns: List[float], lam: float) -> Optional[float]:
    """Linear n(λ) interpolation. ``None`` if ``lam`` is outside the table.

    Refuses to extrapolate on purpose: a d/F/C line just past the end of a
    short table would otherwise yield a plausible-looking but invented Abbe
    number.
    """
    if lam < xs[0] or lam > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if lam <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            if x1 == x0:
                return ns[i]
            t = (lam - x0) / (x1 - x0)
            return ns[i - 1] + t * (ns[i] - ns[i - 1])
    return ns[-1]


def _abbe_from_tabulated_n(
    table: Tuple[List[float], List[float]],
) -> Optional[Tuple[float, float]]:
    """Derive ``(nd, Vd)`` from a tabulated n(λ) curve.

    Used only for sources that publish no dispersion formula and no
    ``SPECS.nd`` / ``SPECS.Vd`` (LZOS). Every LZOS table lists the d, F and
    C lines as exact rows, so in practice this is a lookup rather than an
    interpolation. Returns ``None`` when any line falls outside the table
    or the F/C pair is degenerate.
    """
    xs, ns = table
    n_d = _interp_n(xs, ns, _LINE_D_UM)
    n_F = _interp_n(xs, ns, _LINE_F_UM)
    n_C = _interp_n(xs, ns, _LINE_C_UM)
    if n_d is None or n_F is None or n_C is None:
        return None
    denom = n_F - n_C
    if abs(denom) < 1e-9:
        return None
    return float(n_d), float((n_d - 1.0) / denom)


def _parse_thermal_dispersion(entries: Any) -> Optional[dict]:
    """First entry of a ``thermal_dispersion`` list, normalised."""
    if not entries:
        return None
    first = entries[0] if isinstance(entries, list) and entries else None
    if not isinstance(first, dict):
        return None
    return {
        "type": str(first.get("type", "")).strip().strip('"').lower().replace(" ", "_"),
        "coefficients": _parse_num_list(first.get("coefficients")),
    }


def _parse_thermal_expansion(entries: Any) -> List[dict]:
    if not isinstance(entries, list):
        return []
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rng = _parse_num_list(entry.get("temperature_range"))
        coef = _parse_first_float(entry.get("coefficient"))
        if len(rng) >= 2 and coef is not None:
            out.append({
                "temperature_range_c": [float(rng[0]), float(rng[1])],
                "coefficient_per_k": coef,
            })
    return out


# ---------------------------------------------------------------------------
# Tag derivation
# ---------------------------------------------------------------------------

def _wavelength_tags(rng_um: Optional[List[float]]) -> List[str]:
    if not rng_um or len(rng_um) != 2:
        return []
    lo, hi = float(rng_um[0]), float(rng_um[1])
    tags = []
    if lo < 0.4:
        tags.append("uv")
    if lo < 0.7 and hi > 0.4:
        tags.append("visible")
    if hi > 0.7 and lo < 2.0:
        tags.append("nir")
    if hi > 2.0:
        tags.append("ir")
    return tags


def _derive_tags(
    *,
    vendor: str,
    subdir_tag: Optional[str],
    nd: Optional[float],
    glass_status: Optional[str],
    wavelength_range_um: Optional[List[float]],
    description: str,
    comments: str,
) -> List[str]:
    tags: list[str] = [vendor.lower()]
    if subdir_tag:
        tags.append(subdir_tag.lower())
    if glass_status:
        tags.append(str(glass_status).strip().lower())

    if nd is not None:
        tags.append("crown" if nd < _CROWN_FLINT_CUTOFF_ND else "flint")

    tags.extend(_wavelength_tags(wavelength_range_um))

    haystack = f"{description} {comments}".lower()
    if "lead" in haystack and "lead-free" not in haystack:
        tags.append("lead-containing")
    for needle, tag in _DESCRIPTION_TAGS:
        if needle in haystack and tag not in tags:
            tags.append(tag)

    # De-dup while preserving order.
    seen, out = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------

class SkipFile(Exception):
    """Raised when a .mtrl file has no ghostlight-compatible dispersion entry."""


def _pick_dispersion_entry(data_list: Any) -> Optional[dict]:
    """First ``DATA`` entry whose type is a known dispersion formula."""
    if not isinstance(data_list, list):
        return None
    for entry in data_list:
        if not isinstance(entry, dict):
            continue
        if _normalise_formula_type(entry.get("type")):
            return entry
    return None


def _collect_tabulated(data_list: Any) -> dict:
    """Pull tabulated_k / tabulated_n into a small dict (omit if empty)."""
    out: dict = {}
    if not isinstance(data_list, list):
        return out
    for entry in data_list:
        if not isinstance(entry, dict):
            continue
        t = str(entry.get("type", "")).strip().lower()
        block = _parse_tabulated_block(entry.get("data"))
        if not block:
            continue
        if t == "tabulated k":
            out["tabulated_k"] = block
        elif t == "tabulated n":
            out["tabulated_n"] = {
                "wavelength_um": block["wavelength_um"],
                "n":             block["k"],
            }
        elif t == "tabulated nk" and len(block["wavelength_um"]) == len(block["k"]):
            # Format here is `λ n k` per line; the simple two-column parser
            # only grabbed the first two columns, so n is what's in `k`.
            out["tabulated_nk_partial"] = block
    return out


def _build_dispersion(
    *,
    formula_type: str,
    coefficients: List,
    specs: dict,
    source_file: str,
    warnings: List[str],
) -> Optional[dict]:
    """Return the active ``dispersion`` block, or ``None`` to skip."""
    nd = specs.get("nd")
    vd = specs.get("Vd")

    def _abbe_from_specs(*, silent: bool = False) -> Optional[dict]:
        if nd is None or vd is None:
            if not silent:
                warnings.append(f"{source_file}: missing SPECS.nd/Vd -- skipping")
            return None
        return {"model": "abbe", "nd": float(nd), "Vd": float(vd)}

    if formula_type == "formula_2":
        # Layout: c0 B1 C1 B2 C2 B3 C3 -> 7 numbers for 3-term sellmeier.
        if len(coefficients) == 7 and abs(float(coefficients[0])) < 1e-12:
            return {
                "model": "sellmeier",
                "B": [float(coefficients[1]), float(coefficients[3]), float(coefficients[5])],
                "C": [float(coefficients[2]), float(coefficients[4]), float(coefficients[6])],
            }
        abbe = _abbe_from_specs(silent=True)
        c0_str = (
            f"c0={coefficients[0]}" if coefficients else "no coeffs"
        )
        if abbe is None:
            warnings.append(
                f"{source_file}: formula 2 not representable as 3-term sellmeier "
                f"({len(coefficients)} coeffs, {c0_str}) and no SPECS nd/Vd -- skipping"
            )
        else:
            warnings.append(
                f"{source_file}: formula 2 not representable as 3-term sellmeier "
                f"({len(coefficients)} coeffs, {c0_str}) -- fell back to Abbe from SPECS"
            )
        return abbe

    if formula_type == "formula_3":
        return _abbe_from_specs()

    if formula_type == "formula_1":
        warnings.append(f"{source_file}: formula 1 (IR Sellmeier) -- skipping")
        return None

    warnings.append(f"{source_file}: unknown formula '{formula_type}' -- skipping")
    return None


def convert_mtrl(
    *,
    path: Path,
    vendor: str,
    subdir_tag: Optional[str],
    repo_root: Path,
    warnings: List[str],
) -> Optional[Tuple[str, dict]]:
    """Convert one ``.mtrl`` file to ``(key, material_dict)``.

    Returns ``None`` when the file has no ghostlight-compatible dispersion
    entry (formula 1, missing SPECS, etc.) — the caller skips silently
    aside from the warning already appended.
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        warnings.append(f"{path}: not a YAML mapping, skipping")
        return None

    source_rel = path.name
    data_list = doc.get("DATA")
    specs = doc.get("SPECS") if isinstance(doc.get("SPECS"), dict) else {}

    nd = specs.get("nd")
    vd = specs.get("Vd")
    formula_type: Optional[str] = None
    coefficients: List = []
    wavelength_range_um: Optional[List[float]] = None
    nd_vd_derived = False

    dispersion_entry = _pick_dispersion_entry(data_list)
    if dispersion_entry is not None:
        formula_type = _normalise_formula_type(dispersion_entry.get("type"))
        coefficients = _parse_num_list(dispersion_entry.get("coefficients"))
        wavelength_range_um = _parse_wavelength_range(
            dispersion_entry.get("wavelength_range")
        )
        dispersion = _build_dispersion(
            formula_type=formula_type,
            coefficients=coefficients,
            specs=specs,
            source_file=source_rel,
            warnings=warnings,
        )
    else:
        # No dispersion formula at all. The LZOS catalogue publishes a bare
        # n(λ) table with no SPECS block, so nd/Vd have to come off the table
        # itself — see _abbe_from_tabulated_n. Files whose table doesn't span
        # the d/F/C lines still skip rather than get an extrapolated Abbe pair.
        table = _find_tabulated_n(data_list)
        derived = _abbe_from_tabulated_n(table) if table else None
        if derived is None:
            warnings.append(
                f"{source_rel}: no recognised DATA formula entry and no tabulated n "
                f"spanning the d/F/C lines -- skipping"
            )
            return None
        nd, vd = derived[0], round(derived[1], 6)
        wavelength_range_um = [table[0][0], table[0][-1]]
        dispersion = {"model": "abbe", "nd": float(nd), "Vd": float(vd)}
        nd_vd_derived = True

    if dispersion is None:
        return None

    description = _strip_html(doc.get("DESCRIPTION"))
    comments    = _strip_html(doc.get("COMMENTS"))
    references  = _strip_html(doc.get("REFERENCES"))
    glass_code  = specs.get("glass_code")
    glass_status = specs.get("glass_status")
    density_g_cm3 = _parse_first_float(specs.get("density"))
    temperature_c = _parse_first_float(specs.get("temperature"))

    stem = path.stem
    key = f"{vendor}_{stem}"
    catalogue_ref = f"{vendor}:{stem}"

    ui = {
        "nd": float(nd) if nd is not None else None,
        "vd": float(vd) if vd is not None else None,
        "wavelength_range_nm": (
            [wavelength_range_um[0] * 1000.0, wavelength_range_um[1] * 1000.0]
            if wavelength_range_um else None
        ),
        "glass_code":   str(glass_code) if glass_code is not None else "",
        "glass_status": str(glass_status) if glass_status is not None else "",
        "density_g_cm3": density_g_cm3,
        "description":   description,
        "comments":      comments,
        "references":    references,
        "source_file":   source_rel,
    }
    # Drop None / empty optional fields to keep the JSON tight.
    ui = {k: v for k, v in ui.items() if v not in (None, "")}

    raw_source: dict = {}
    if formula_type is not None:
        raw_source["formula"] = {
            "type": formula_type,
            "wavelength_range_um": wavelength_range_um,
            "coefficients": coefficients,
        }
    raw_source.update(_collect_tabulated(data_list))
    if nd_vd_derived:
        # Provenance: ui.nd / ui.vd are normally the vendor-published SPECS
        # values. Here they were computed off the n(λ) table, so say so
        # rather than let a consumer assume they're vendor-quoted.
        raw_source["nd_vd_derived_from"] = "tabulated_n"

    thermal_dispersion = _parse_thermal_dispersion(specs.get("thermal_dispersion"))
    if thermal_dispersion:
        raw_source["thermal_dispersion"] = thermal_dispersion
    thermal_expansion = _parse_thermal_expansion(specs.get("thermal_expansion"))
    if thermal_expansion:
        raw_source["thermal_expansion"] = thermal_expansion

    for key_in, key_out in (
        ("dPgF",                 "dPgF"),
        ("climatic_resistance",  "climatic_resistance"),
        ("stain_resistance",     "stain_resistance"),
        ("acid_resistance",      "acid_resistance"),
        ("alkali_resistance",    "alkali_resistance"),
        ("phosphate_resistance", "phosphate_resistance"),
    ):
        v = specs.get(key_in)
        if v is not None:
            raw_source[key_out] = float(v) if isinstance(v, (int, float)) else v

    if "n_absolute" in specs:
        raw_source["n_absolute"] = bool(specs["n_absolute"])
    if "wavelength_vacuum" in specs:
        raw_source["wavelength_vacuum"] = bool(specs["wavelength_vacuum"])
    if temperature_c is not None:
        raw_source["temperature_c"] = temperature_c

    tags = _derive_tags(
        vendor=vendor,
        subdir_tag=subdir_tag,
        nd=float(nd) if nd is not None else None,
        glass_status=glass_status,
        wavelength_range_um=wavelength_range_um,
        description=description,
        comments=comments,
    )

    material = {
        "display_name":  stem,
        "catalogue_ref": catalogue_ref,
        "tags":          tags,
        "dispersion":    dispersion,
        "ui":            ui,
        "raw_source":    raw_source,
    }
    return key, material


# ---------------------------------------------------------------------------
# Per-vendor build
# ---------------------------------------------------------------------------

def _iter_vendor_files(vendor_dir: Path) -> Iterable[Tuple[Path, Optional[str]]]:
    """Yield ``(file, subdir_tag)`` for every ``.mtrl`` under ``vendor_dir``.

    ``subdir_tag`` is the lowercased one-level subdirectory name (e.g.
    ``infrared``, ``misc``) for files under a vendor's sub-buckets, or
    ``None`` for top-level files. The picker uses these as filter tags so
    Schott IR / Misc glasses stay discoverable in a flat list.
    """
    if not vendor_dir.is_dir():
        return
    # Top-level
    for entry in sorted(vendor_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".mtrl":
            yield entry, None
    # One level of subdirs (Schott has Infrared/ and Misc/)
    for entry in sorted(vendor_dir.iterdir()):
        if entry.is_dir():
            tag = entry.name.lower()
            for f in sorted(entry.iterdir()):
                if f.is_file() and f.suffix == ".mtrl":
                    yield f, tag


def build_vendor_catalogue(
    *,
    vendor: str,
    vendor_dir: Path,
    repo_root: Path,
    retrieved: str,
    warnings: List[str],
) -> Tuple[dict, Counter]:
    """Build the per-vendor envelope. Returns ``(envelope, model_counts)``."""
    materials: dict[str, dict] = {}
    model_counts: Counter = Counter()
    for path, subdir_tag in _iter_vendor_files(vendor_dir):
        try:
            result = convert_mtrl(
                path=path,
                vendor=vendor,
                subdir_tag=subdir_tag,
                repo_root=repo_root,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover — diagnostic, not fatal
            warnings.append(f"{path}: parse error — {exc!r}")
            continue
        if result is None:
            continue
        key, material = result
        materials[key] = material
        model_counts[material["dispersion"]["model"]] += 1

    envelope = {
        "format":  "ghostlight-material-catalogue",
        "version": {"major": 1, "minor": 0},
        "source": {
            "vendor":    vendor,
            "origin":    "refractiveindex.info",
            "license":   "CC0-1.0",
            "retrieved": retrieved,
        },
        "materials": dict(sorted(materials.items())),
    }
    return envelope, model_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_vendor_dirs(in_dir: Path) -> List[str]:
    """Immediate subdirectories of ``in_dir`` that hold ``.mtrl`` files.

    Known vendors (``VENDORS``) come first in declared order so re-running
    against a familiar dump produces a stable summary; anything else follows
    alphabetically. Discovery rather than a fixed list means a source dump
    that carries only some vendors doesn't warn about the absent ones, and a
    newly vendored family needs no code edit here.
    """
    if not in_dir.is_dir():
        return []
    present = {p.name for p in in_dir.iterdir() if p.is_dir()}
    with_mtrl = {name for name in present if any((in_dir / name).rglob("*.mtrl"))}
    known = [v for v in VENDORS if v in with_mtrl]
    extra = sorted(with_mtrl - set(known))
    return known + extra


def _write_json(envelope: dict, out_path: Path) -> None:
    """Deterministic JSON dump (sorted keys handled at the materials level;
    here we just keep the envelope readable). Trailing newline + 2-space
    indent matches the rest of the repo."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(envelope, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert refractiveindex.info .mtrl glasses to ghostlight "
                    "material-catalogue JSON.",
    )
    parser.add_argument(
        "--in", dest="in_dir", required=True,
        help="Directory of refractiveindex.info .mtrl files, one subdirectory per vendor",
    )
    parser.add_argument(
        "--out", dest="out_dir", required=True,
        help="Destination directory for <vendor>.json files (created if absent)",
    )
    parser.add_argument(
        "--retrieved", default=_dt.date.today().isoformat(),
        help="Retrieval date stamped into the envelope (YYYY-MM-DD). "
             "Defaults to today.",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not in_dir.is_dir():
        print(
            f"error: --in directory does not exist: {in_dir}\n"
            f"       Point --in at a refractiveindex.info source dump: one\n"
            f"       subdirectory per vendor, each holding .mtrl files.",
            file=sys.stderr,
        )
        return 2

    repo_root = in_dir.parent

    total_warnings: List[str] = []
    grand_counts: Counter = Counter()
    per_vendor_summary: list[str] = []

    vendor_dirs = _discover_vendor_dirs(in_dir)
    if not vendor_dirs:
        print(
            f"error: no vendor subdirectories containing .mtrl files under {in_dir}",
            file=sys.stderr,
        )
        return 2

    for dir_name in vendor_dirs:
        vendor = _VENDOR_DISPLAY_NAMES.get(dir_name, dir_name)
        vendor_dir = in_dir / dir_name
        envelope, model_counts = build_vendor_catalogue(
            vendor=vendor,
            vendor_dir=vendor_dir,
            repo_root=repo_root,
            retrieved=args.retrieved,
            warnings=total_warnings,
        )
        out_path = out_dir / f"{vendor.lower()}.json"
        _write_json(envelope, out_path)
        n_total = len(envelope["materials"])
        n_sell = model_counts["sellmeier"]
        n_abbe = model_counts["abbe"]
        grand_counts.update(model_counts)
        per_vendor_summary.append(
            f"  {vendor:<7s} -> {out_path.name:<14s} "
            f"{n_total:>4d} materials  (sellmeier={n_sell}, abbe={n_abbe})"
        )

    print("Material catalogue build:")
    for line in per_vendor_summary:
        print(line)
    print(
        f"  total: {sum(grand_counts.values())} emitted  "
        f"(sellmeier={grand_counts['sellmeier']}, abbe={grand_counts['abbe']}), "
        f"{len(total_warnings)} warnings/skips"
    )
    if total_warnings:
        print("\nWarnings / skipped files:")
        for w in total_warnings:
            print(f"  - {w}")

    # Non-zero skips are expected (formula 1 IR glass) — exit 0 unless we
    # produced zero materials, which would indicate a structural problem.
    return 0 if sum(grand_counts.values()) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
