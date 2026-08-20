"""Catalogue-hammer helpers: enumerate candidates, find nearest glass, hull.

Used by the optimization panel's hammer path
(:mod:`ghostlight_designer.optimization_panel.optimizer`) and by the ODE
Material-row right-click submenu when it needs the vendor list.

Deliberately Qt-free: everything here takes a
:class:`~ghostlight_designer.material_catalogue.MaterialCatalogue` and
returns plain data so unit tests can run headless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .material_catalogue import CatalogueMaterial, MaterialCatalogue


# ---------------------------------------------------------------------------
# Vendor enumeration
# ---------------------------------------------------------------------------


def vendors_in_catalogue(catalogue: MaterialCatalogue) -> List[str]:
    """Distinct ``source_vendor``s present, sorted alphabetically.

    Matches the ordering used by the ODE Material row's Designer combo so
    the two lists never drift.
    """
    seen = {m.source_vendor for m in catalogue.all() if m.source_vendor}
    return sorted(seen)


def candidates_for_vendor(
    catalogue: MaterialCatalogue, vendor: str,
) -> List[CatalogueMaterial]:
    """Every catalogue material with the given ``source_vendor``.

    Filters to entries that have both ``nd`` and ``vd`` populated —
    substitution ranks by (nd, Vd) distance, and a glass without those
    fields can't be scored.
    """
    if not vendor:
        return []
    return [
        m for m in catalogue.all()
        if m.source_vendor == vendor and m.nd is not None and m.vd is not None
    ]


# ---------------------------------------------------------------------------
# Vendor bounding hulls (used as the default search region)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VendorHull:
    """Axis-aligned bounding box of one vendor's catalogue in (nd, Vd) space.

    Both bounds are inclusive; when a vendor has zero valid glasses (all
    ``nd`` or ``vd`` missing) every field is ``None``. The hammer treats
    a fully-``None`` hull as "vendor unusable" and skips it.
    """
    vendor: str
    count: int
    nd_min: Optional[float]
    nd_max: Optional[float]
    vd_min: Optional[float]
    vd_max: Optional[float]

    def is_empty(self) -> bool:
        return self.count == 0


def vendor_hull(catalogue: MaterialCatalogue, vendor: str) -> VendorHull:
    """(nd, Vd) axis-aligned bounding box for ``vendor``.

    Used as the default search region when the user hasn't tightened the
    SubstitutionSpec bounds. Missing values inside candidate entries are
    silently skipped — they can't participate in either min or max.
    """
    matches = candidates_for_vendor(catalogue, vendor)
    if not matches:
        return VendorHull(vendor=vendor, count=0, nd_min=None, nd_max=None,
                          vd_min=None, vd_max=None)
    nds = [float(m.nd) for m in matches if m.nd is not None]
    vds = [float(m.vd) for m in matches if m.vd is not None]
    return VendorHull(
        vendor=vendor,
        count=len(matches),
        nd_min=min(nds) if nds else None,
        nd_max=max(nds) if nds else None,
        vd_min=min(vds) if vds else None,
        vd_max=max(vds) if vds else None,
    )


# ---------------------------------------------------------------------------
# Nearest-glass lookup — used to seed the hammer with a good starting glass
# ---------------------------------------------------------------------------


def _score_distance(
    a_nd: float, a_vd: float, b_nd: float, b_vd: float,
) -> float:
    """Normalised (nd, Vd) distance.

    ``Vd`` values in the catalogue span ~20…80 while ``nd`` values span
    ~1.4…2.0. Scaling ``Vd`` by 0.01 gives the two axes comparable weight
    in Euclidean distance — matches what
    :func:`ghostlight_designer.optical_editor.row_schemas.convert_material_to_vendor`
    uses so the two "nearest glass" paths agree.
    """
    dnd = a_nd - b_nd
    dvd = 0.01 * (a_vd - b_vd)
    return math.sqrt(dnd * dnd + dvd * dvd)


def nearest_glass(
    catalogue: MaterialCatalogue,
    nd: float,
    vd: float,
    *,
    vendor: Optional[str] = None,
) -> Optional[CatalogueMaterial]:
    """Closest catalogue entry to ``(nd, Vd)``, optionally scoped to ``vendor``.

    Returns ``None`` when no candidate has usable ``nd``/``vd`` values.
    Ties are broken by insertion order (Python ``min`` semantics).
    """
    if vendor is not None:
        candidates = candidates_for_vendor(catalogue, vendor)
    else:
        candidates = [
            m for m in catalogue.all()
            if m.nd is not None and m.vd is not None
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda m: _score_distance(float(m.nd), float(m.vd), nd, vd),
    )


# ---------------------------------------------------------------------------
# Candidate filtering — spec bounds → concrete glass list
# ---------------------------------------------------------------------------


def candidates_within_spec(
    catalogue: MaterialCatalogue,
    spec,
) -> List[CatalogueMaterial]:
    """Vendor candidates whose (nd, Vd) fall inside ``spec``'s bounds.

    ``spec`` is a :class:`~ghostlight_designer.project.SubstitutionSpec`;
    ``None`` bounds default to the vendor's convex hull (i.e. every
    vendor candidate passes on that axis). Returned in catalogue order.
    """
    vendor = getattr(spec, "vendor", "") or ""
    matches = candidates_for_vendor(catalogue, vendor)
    if not matches:
        return []
    hull = vendor_hull(catalogue, vendor)

    def bound(user_lo, user_hi, hull_lo, hull_hi) -> Tuple[float, float]:
        lo = user_lo if user_lo is not None else hull_lo
        hi = user_hi if user_hi is not None else hull_hi
        lo = -math.inf if lo is None else float(lo)
        hi = math.inf if hi is None else float(hi)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    nd_lo, nd_hi = bound(spec.nd_lo, spec.nd_hi, hull.nd_min, hull.nd_max)
    vd_lo, vd_hi = bound(spec.vd_lo, spec.vd_hi, hull.vd_min, hull.vd_max)

    out: List[CatalogueMaterial] = []
    for m in matches:
        try:
            mnd = float(m.nd)
            mvd = float(m.vd)
        except (TypeError, ValueError):
            continue
        if not (nd_lo <= mnd <= nd_hi):
            continue
        if not (vd_lo <= mvd <= vd_hi):
            continue
        out.append(m)
    return out
