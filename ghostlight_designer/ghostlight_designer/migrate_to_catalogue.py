"""Match a project's in-file glass catalogue against the bundled material catalogue.

Two flows live here:

* **Passive identification** — walk the
  project's ``_raw_glass_catalogue`` and *report* which entries already
  align with bundled vendor materials. No mutation; the picker uses this
  to badge entries as "From: Schott" / "drifted" / "project-local".

* **Active re-link** (§5c) — given a per-entry user decision, rewrite both
  ``system._raw_glass_catalogue`` and every ``el.material_glasses[i]``
  reference so the project uses the catalogue key (and shares its
  dispersion data) instead of the project-local key.

Both pure-functional over ``(system, catalogue)``; the only mutation
helper is :func:`apply_relink`, which wraps the rewrite in
``project.edit(...)`` so the existing undo path catches it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .material_catalogue import CatalogueMaterial, MaterialCatalogue

logger = logging.getLogger(__name__)


# Tolerances. The Sellmeier bound matches what ghostlight's writer round-trips
# to (full float64 precision); the Abbe equality bound is tight so it
# discriminates between true catalogue twins and merely-close auto-generated
# keys.
_SELLMEIER_ATOL = 1e-9
_DISPERSION_ABBE_ATOL = 1e-4

# Loose (nd, Vd) match used as the fallback rule when neither catalogue_ref
# nor exact dispersion equality fires. Tolerances reflect the precision of
# Zemax-style six-digit ``glass_code`` (e.g. ``517642`` → nd 1.517 ± 5e-4,
# Vd 64.2 ± 5e-2): tight on nd, loose on Vd. Auto-generated keys like
# ``G_ND1.6727_VD32.21`` quantise both at this precision.
_LOOSE_ND_ATOL = 1e-3
_LOOSE_VD_ATOL = 1e-1


# Numeric rank for sort/threshold use. Higher = more confident.
RULE_RANK = {
    "catalogue_ref": 4,
    "dispersion":    3,
    "nd_vd":         2,
    "name":          1,
}


@dataclass(frozen=True)
class Match:
    """One catalogue candidate for a project-local glass entry."""

    rule: str                        # one of RULE_RANK's keys
    confidence: int                  # numeric rank (4 highest, 1 lowest)
    candidate: CatalogueMaterial
    note: str = ""                   # human-readable detail for the UI


@dataclass(frozen=True)
class GlassMatchReport:
    """Match results for a single project-local glass key."""

    project_key: str
    project_entry: dict              # verbatim ``_raw_glass_catalogue[key]``
    matches: List[Match]             # sorted by descending confidence

    @property
    def best(self) -> Optional[Match]:
        return self.matches[0] if self.matches else None

    @property
    def is_linked(self) -> bool:
        """True iff there's a top-confidence catalogue_ref match (rule 1)."""
        return bool(self.matches) and self.matches[0].rule == "catalogue_ref"

    @property
    def is_ambiguous(self) -> bool:
        """True when ≥ 2 matches share the top confidence and rule.

        A single best hit (even if there are weaker also-rans) is NOT
        ambiguous; this guards the bulk auto-link path from silently
        picking among genuine ties.
        """
        if len(self.matches) < 2:
            return False
        top = self.matches[0]
        return (
            self.matches[1].confidence == top.confidence
            and self.matches[1].rule == top.rule
        )


# ---------------------------------------------------------------------------
# Per-rule matchers
# ---------------------------------------------------------------------------

def _match_by_catalogue_ref(
    entry: dict, catalogue: MaterialCatalogue,
) -> Optional[Match]:
    ref = str(entry.get("catalogue_ref", "") or "").strip()
    if not ref:
        return None
    candidate = catalogue.by_ref(ref)
    if candidate is None:
        return None
    return Match(
        rule="catalogue_ref",
        confidence=RULE_RANK["catalogue_ref"],
        candidate=candidate,
        note=f"linked via catalogue_ref={ref!r}",
    )


def _close(a: float, b: float, atol: float) -> bool:
    return abs(float(a) - float(b)) <= atol


def _dispersion_equal(a: dict, b: dict) -> bool:
    """Bit-near equality between two dispersion blocks (same model required)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    model = a.get("model")
    if model != b.get("model"):
        return False
    if model == "sellmeier":
        try:
            for k in ("B", "C"):
                av, bv = a.get(k) or [], b.get(k) or []
                if len(av) != 3 or len(bv) != 3:
                    return False
                for x, y in zip(av, bv):
                    if not _close(x, y, _SELLMEIER_ATOL):
                        return False
            return True
        except (TypeError, ValueError):
            return False
    if model == "abbe":
        try:
            return (
                _close(a.get("nd"), b.get("nd"), _DISPERSION_ABBE_ATOL)
                and _close(a.get("Vd"), b.get("Vd"), _DISPERSION_ABBE_ATOL)
            )
        except (TypeError, ValueError):
            return False
    return False


def _match_by_dispersion_equality(
    entry: dict, catalogue: MaterialCatalogue,
) -> List[Match]:
    dispersion = entry.get("dispersion")
    if not isinstance(dispersion, dict):
        return []
    out: list[Match] = []
    for cand in catalogue:
        if _dispersion_equal(dispersion, cand.dispersion):
            out.append(Match(
                rule="dispersion",
                confidence=RULE_RANK["dispersion"],
                candidate=cand,
                note=f"exact dispersion match ({dispersion.get('model')})",
            ))
    return out


def _entry_nd_vd(entry: dict) -> Optional[tuple[float, float]]:
    """Extract (nd, Vd) from a lens-side glass entry, regardless of model.

    Only the Abbe branch is currently mined — Sellmeier-only entries would
    require us to re-derive nd/Vd which is the C++ side's job; for
    matching purposes such entries are already covered by rule 2
    (exact dispersion equality).
    """
    dispersion = entry.get("dispersion")
    if isinstance(dispersion, dict) and dispersion.get("model") == "abbe":
        try:
            return float(dispersion["nd"]), float(dispersion["Vd"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _match_by_nd_vd(
    entry: dict, catalogue: MaterialCatalogue,
) -> List[Match]:
    nd_vd = _entry_nd_vd(entry)
    if nd_vd is None:
        return []
    nd, vd = nd_vd
    out: list[Match] = []
    for cand in catalogue:
        if cand.nd is None or cand.vd is None:
            continue
        if _close(nd, cand.nd, _LOOSE_ND_ATOL) and _close(vd, cand.vd, _LOOSE_VD_ATOL):
            out.append(Match(
                rule="nd_vd",
                confidence=RULE_RANK["nd_vd"],
                candidate=cand,
                note=f"nd within {_LOOSE_ND_ATOL}, Vd within {_LOOSE_VD_ATOL}",
            ))
    return out


_VENDOR_PREFIX_RE = re.compile(r"^(schott|ohara|hoya|user)[-_:]", re.IGNORECASE)


def _normalise_name(name: str) -> str:
    """Strip vendor prefix and case-normalise for token comparison."""
    s = _VENDOR_PREFIX_RE.sub("", name or "")
    return s.strip().lower()


def _match_by_name(
    project_key: str, entry: dict, catalogue: MaterialCatalogue,
) -> List[Match]:
    """Lowest-confidence fallback: case-insensitive name token match.

    Compares the project entry's ``name`` (and its catalogue key as a
    fallback) against each candidate's display name.
    """
    candidates_names: list[str] = []
    name = entry.get("name")
    if name:
        candidates_names.append(_normalise_name(str(name)))
    candidates_names.append(_normalise_name(project_key))

    out: list[Match] = []
    seen_keys: set[str] = set()
    for cand_name in candidates_names:
        if not cand_name:
            continue
        for cand in catalogue:
            if cand.key in seen_keys:
                continue
            if _normalise_name(cand.display_name) == cand_name:
                seen_keys.add(cand.key)
                out.append(Match(
                    rule="name",
                    confidence=RULE_RANK["name"],
                    candidate=cand,
                    note=f"display-name match ({cand.display_name!r})",
                ))
    return out


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def match_entry(
    project_key: str,
    entry: dict,
    catalogue: MaterialCatalogue,
) -> GlassMatchReport:
    """Match a single ``_raw_glass_catalogue`` entry against the catalogue.

    Returns a report whose ``matches`` is sorted by descending confidence.
    Rules are short-circuited only for the catalogue_ref hit (rule 1): if
    a ref-match exists we don't bother running the cheaper rules. The
    weaker rules still run when there's no ref, so the picker can suggest
    a plausible link even for unprovenanced entries.
    """
    if not isinstance(entry, dict):
        return GlassMatchReport(project_key=project_key, project_entry={}, matches=[])

    matches: list[Match] = []

    ref_match = _match_by_catalogue_ref(entry, catalogue)
    if ref_match is not None:
        matches.append(ref_match)
    else:
        # Run the weaker rules. De-dup by candidate key — a single bundled
        # entry can match by dispersion AND by nd/Vd, but the report should
        # surface it once at its highest-confidence rule.
        seen: dict[str, Match] = {}
        for m in (
            _match_by_dispersion_equality(entry, catalogue)
            + _match_by_nd_vd(entry, catalogue)
            + _match_by_name(project_key, entry, catalogue)
        ):
            existing = seen.get(m.candidate.key)
            if existing is None or m.confidence > existing.confidence:
                seen[m.candidate.key] = m
        matches.extend(seen.values())

    matches.sort(key=lambda m: (-m.confidence, m.candidate.key))
    return GlassMatchReport(
        project_key=project_key, project_entry=entry, matches=matches,
    )


def analyze_system(
    system, catalogue: MaterialCatalogue,
) -> List[GlassMatchReport]:
    """Build a match report for every entry in ``system._raw_glass_catalogue``."""
    out: list[GlassMatchReport] = []
    for key, entry in (system._raw_glass_catalogue or {}).items():
        out.append(match_entry(str(key), entry, catalogue))
    return out


# ---------------------------------------------------------------------------
# Active re-link
# ---------------------------------------------------------------------------

def apply_relink(
    project,
    project_key: str,
    target: CatalogueMaterial,
    *,
    label: Optional[str] = None,
) -> bool:
    """Rewrite ``project`` to use catalogue ``target`` instead of ``project_key``.

    Walks both ``system._raw_glass_catalogue`` and every element's
    ``material_glasses[i]`` to replace the key. If the target key already
    exists in the project catalogue with a different dispersion, this
    function overwrites it — callers should preview that before invoking.

    Wrapped in ``project.edit(...)`` so undo / redo / dirty tracking work.
    Returns ``True`` when at least one reference was rewritten, ``False``
    when ``project_key`` wasn't actually used by the system (stale UI).
    """
    system = project.system
    catalogue = system._raw_glass_catalogue or {}
    if project_key not in catalogue and not any(
        project_key in (el.material_glasses or [])
        for el in (system._elements or [])
    ):
        return False

    label = label or f"Re-link to {target.display_name}"

    with project.edit(label):
        # 1. Catalogue: drop the old entry (if any), install the target.
        if project_key in catalogue and project_key != target.key:
            del catalogue[project_key]
        catalogue[target.key] = target.lens_catalogue_entry()

        # 2. Element references: every material_glasses slot referencing the
        # old key now points at the catalogue key.
        for el in (system._elements or []):
            mats = el.material_glasses or []
            for i, glass in enumerate(mats):
                if glass == project_key:
                    mats[i] = target.key
    return True


def apply_relink_bulk(
    project,
    decisions: List[tuple[str, CatalogueMaterial]],
    *,
    label: str = "Re-link glasses to catalogue",
) -> int:
    """Apply many re-links in one compound undo entry.

    ``decisions`` is a list of ``(project_key, target)`` tuples — the caller
    is expected to have filtered out ambiguous reports already. Returns
    the number of decisions that actually mutated the project (mirrors
    :func:`apply_relink`'s contract).
    """
    if not decisions:
        return 0
    n_applied = 0
    project.begin_compound(label)
    try:
        for project_key, target in decisions:
            if apply_relink(project, project_key, target, label=label):
                n_applied += 1
    finally:
        project.end_compound()
    return n_applied
