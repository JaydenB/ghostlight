"""Ghost enumeration, brightness metering and labelling for the ghost explorer.

Qt-free on purpose — everything here is a computation over a
:class:`ghostlight.OpticalSystem`, so the panel's ghost bookkeeping is unit
testable without a ``QApplication``.

Three jobs:

* **Which ghosts exist** — :func:`enumerate_entries` starts from
  ``ghostlight.filter_ghost_pairs``, the renderer's *own* pre-filter, rather
  than the raw ``enumerate_ghost_pairs`` list, so the scrubber only offers
  pairs the renderer can actually draw. Pairs that fail the IOR-contrast or
  ``min_ghost_intensity`` test never reach the GPU, and offering them would
  mean scrubbing onto guaranteed-black frames. Ghost *numbers* are 1-based
  indices into that filtered list — the same order the renderer assigns
  per-pair AOV layers.

* **What to call the surfaces** — :func:`build_surface_labels` names each
  surface index from the lens file's element groupings, so the readout can say
  "Front Doublet rear" instead of just "surface 3".

* **What the whole flare looks like** — :func:`render_rough_survey`. One
  low-resolution ``render_point_flare`` of the lens with *every* ghost active.
  The panel uses it for two things: metering the viewer exposure (so the
  scrubber holds one exposure across every ghost, and a dim ghost genuinely
  reads as dim against a bright one), and — when ``want_peaks`` is set —
  scoring each ghost for the cull.

  Brightness scoring rides on ``aov_mode = PER_PAIR``, which makes the same
  render also emit every pair as its own layer; the peak of a layer is that
  ghost's brightness. Measuring the renderer is the only approach that can't
  disagree with what the panel paints. An earlier version of this module
  estimated brightness on the CPU by tracing a pupil grid and summing the
  reflected weight that landed on the sensor. It was cheap but simply wrong —
  measured against real renders its ranking bore almost no relation to the
  rendered result, because the renderer's area normalisation, sample
  concentration and spectral integration dominate raw Fresnel throughput.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ghostlight

_log = logging.getLogger("ghostlight_designer.ghost_explorer_panel.ghost_survey")

# Resolution and sampling for the rough whole-flare pass. Deliberately far
# below display quality: it feeds an exposure meter and a relative brightness
# ranking, neither of which needs detail, and with ``want_peaks`` it is the one
# place in this panel whose cost scales with the ghost count.
ROUGH_WIDTH_PX = 96
ROUGH_RAY_GRID = 192
ROUGH_SPECTRAL = 4

# Default cull threshold: a ghost is hidden when its peak falls below this
# fraction of the brightest ghost's peak. 1% is roughly seven stops down — in
# a full render you would not notice it — while still erring towards showing
# too much rather than hiding something the user wanted.
DEFAULT_CULL_REL = 0.01
CULL_REL_MIN = 1.0e-6
CULL_REL_MAX = 0.5


@dataclass(frozen=True)
class GhostEntry:
    """One renderable ghost: the two reflecting surfaces plus its brightness.

    ``number`` is 1-based and stable for a given lens + sensor, so it survives
    the cull toggle — turning culling on hides entries but never renumbers the
    survivors.
    """

    number: int
    surf_a: int
    surf_b: int
    label_a: str
    label_b: str
    # Peak pixel value of this ghost's own render layer, in the panel's
    # scene-linear output space. 0.0 until a metering pass has run.
    peak: float = 0.0
    # ``peak`` relative to the brightest ghost of the survey, in [0, 1].
    rel: float = 0.0
    # False until metered. An unmetered entry is never culled, so a missing or
    # failed metering pass can only ever show too much.
    metered: bool = False

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.surf_a, self.surf_b)

    def surfaces_text(self) -> str:
        """``"surface 3 (Front Doublet rear) -> surface 8 (Rear Group front)"``."""
        return (
            f"surface {self.surf_a} ({self.label_a})"
            f"  →  surface {self.surf_b} ({self.label_b})"
        )

    def brightness_text(self) -> str:
        if not self.metered:
            return "brightness not measured"
        if self.peak <= 0.0:
            return "nothing on sensor"
        return f"{self.rel * 100.0:.2f}% of brightest"


# ---------------------------------------------------------------------------
# Surface naming
# ---------------------------------------------------------------------------


def build_surface_labels(system: "ghostlight.OpticalSystem") -> List[str]:
    """Readable name per surface index, resolved from the element groupings.

    A two-surface element names its surfaces ``front`` / ``rear``; a longer one
    (a cemented doublet, say) numbers them ``s1..sN``; a single-surface element
    such as a stop just uses the element name. Surfaces no element claims — and
    every surface of a system built programmatically, which has no element
    layer at all — keep the generic ``"surface N"`` fallback.
    """
    try:
        n = int(system.num_surfaces())
    except Exception:
        return []
    labels = [f"surface {i}" for i in range(n)]
    try:
        elements = system.elements
    except Exception:
        _log.exception("build_surface_labels: element list unavailable")
        return labels
    for el in elements:
        try:
            indices = el.resolve_surfaces(system)
        except Exception:
            # Stale surface UUID — leave those surfaces on the fallback name.
            continue
        name = (getattr(el, "name", "") or "").strip() or "element"
        count = len(indices)
        for k, idx in enumerate(indices):
            if not (0 <= idx < n):
                continue
            if count == 1:
                labels[idx] = name
            elif count == 2:
                labels[idx] = f"{name} {'front' if k == 0 else 'rear'}"
            else:
                labels[idx] = f"{name} s{k + 1}"
    return labels


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_entries(
    system: "ghostlight.OpticalSystem",
    config: "ghostlight.PointFlareConfig",
    *,
    half_w: float,
    half_h: float,
) -> List[GhostEntry]:
    """The lens's renderable ghosts, numbered, unmetered.

    ``config`` supplies ``min_ghost_intensity`` / ``ghost_normalize`` so the
    list matches the pairs the display render itself will keep. Returns an
    empty list (never raises) when the lens can't be enumerated.
    """
    labels = build_surface_labels(system)

    def _label(idx: int) -> str:
        return labels[idx] if 0 <= idx < len(labels) else f"surface {idx}"

    try:
        pairs, _boosts = ghostlight.filter_ghost_pairs(
            system, float(half_w), float(half_h), config
        )
    except Exception:
        _log.exception("enumerate_entries: filter_ghost_pairs failed")
        return []

    return [
        GhostEntry(
            number=i + 1,
            surf_a=int(p.surf_a), surf_b=int(p.surf_b),
            label_a=_label(int(p.surf_a)), label_b=_label(int(p.surf_b)),
        )
        for i, p in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Rough whole-flare pass
# ---------------------------------------------------------------------------


def render_rough_survey(
    lens: "ghostlight.OpticalSystem",
    calib,
    config: "ghostlight.PointFlareConfig",
    *,
    width: int,
    height: int,
    want_peaks: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[Tuple[int, int], float]]:
    """Render the whole flare coarsely; return ``(hwc, per-pair peaks)``.

    ``hwc`` is the combined ghost layer — every pair at once, which is what the
    panel meters its viewer exposure against. With ``want_peaks`` the render
    also runs ``aov_mode = PER_PAIR``, so the renderer emits one
    ``ghost_s{A}_s{B}_{r,g,b}`` triplet per active pair — exactly the per-ghost
    image the panel would paint at that scrubber position — and each layer's
    peak comes back in the dict. Without it the dict is empty and the render
    is a single cheap pass.

    ``config`` is mutated (aov + ghost_filter fields) and should be a
    throwaway. Runs on the caller's thread and blocks on the GPU — call it off
    the GUI thread. Returns ``(None, {})`` on failure rather than raising,
    which leaves the exposure alone and every entry unmetered (so uncullable).
    """
    config.aov_mode = (
        ghostlight.GhostAovMode.PER_PAIR if want_peaks else ghostlight.GhostAovMode.NONE
    )
    config.aov_max_pairs = -1
    # A pair-selection overlay would restrict this to the selected ghost — the
    # whole point of the rough pass is that it sees every ghost at once.
    config.ghost_filter = ghostlight.GhostFilter()
    try:
        if calib is not None:
            out = lens.render_point_flare(int(width), int(height), config,
                                          calib=calib)
        else:
            out = lens.render_point_flare(int(width), int(height), config)
    except Exception:
        _log.exception("render_rough_survey: render failed")
        return None, {}

    try:
        hwc = ghostlight._arrays.ghost_to_hwc(out)
    except Exception:
        _log.exception("render_rough_survey: ghost layer unavailable")
        hwc = None

    peaks: Dict[Tuple[int, int], float] = {}
    for key in out:
        if not (key.startswith("ghost_s") and key.endswith("_r")):
            continue
        pair = _parse_aov_key(key)
        if pair is None:
            continue
        value = 0.0
        for channel in ("_r", "_g", "_b"):
            plane = out.get(key[:-2] + channel)
            if plane is None:
                continue
            arr = np.asarray(plane)
            if arr.size:
                value = max(value, float(arr.max()))
        peaks[pair] = value
    return hwc, peaks


def _parse_aov_key(key: str) -> Optional[Tuple[int, int]]:
    """``"ghost_s3_s8_r"`` -> ``(3, 8)``; ``None`` if it doesn't parse."""
    try:
        a_txt, b_txt = key[len("ghost_s"):-len("_r")].split("_s", 1)
        return int(a_txt), int(b_txt)
    except Exception:
        return None


def apply_peaks(
    entries: Sequence[GhostEntry],
    peaks: Dict[Tuple[int, int], float],
) -> List[GhostEntry]:
    """Fold measured peaks into ``entries``, normalising to the brightest.

    Entries with no measurement stay unmetered (and so uncullable). A pair the
    renderer dropped from the AOV set — the ``cull_dead_pairs`` accelerator
    removes pairs whose rays never reach the sensor — is reported as a metered
    zero, because "the renderer decided this ghost lands nowhere" is a
    measurement, and precisely the case the cull exists for.
    """
    if not entries:
        return []
    if not peaks:
        return list(entries)
    top = max(peaks.values(), default=0.0)
    out: List[GhostEntry] = []
    for e in entries:
        peak = peaks.get(e.pair, 0.0)
        rel = (peak / top) if top > 0.0 else 0.0
        out.append(replace(e, peak=peak, rel=rel, metered=True))
    return out


# ---------------------------------------------------------------------------
# Culling
# ---------------------------------------------------------------------------


def visible_ghosts(
    entries: Sequence[GhostEntry],
    *,
    cull: bool,
    rel_threshold: float = DEFAULT_CULL_REL,
    sort_by_brightness: bool = False,
) -> List[GhostEntry]:
    """The subset of ``entries`` the scrubber should offer, in slider order.

    With ``cull`` off every entry survives. With it on, an entry survives when
    its peak is at least ``rel_threshold`` of the brightest ghost's — plus any
    entry that hasn't been metered, which is never hidden on absent evidence.
    If the threshold would empty the list the brightest entry is kept, so the
    panel always has something to render and the slider is never degenerate.

    ``sort_by_brightness`` orders the survivors brightest-first instead of by
    ghost number. The sort is stable, so before any metering has landed (every
    entry at ``rel`` 0.0) it is a no-op and the list stays in survey order
    rather than scrambling into an arbitrary one.
    """
    if not entries:
        return []
    if not cull:
        kept = list(entries)
    else:
        thr = max(CULL_REL_MIN, min(CULL_REL_MAX, float(rel_threshold)))
        kept = [e for e in entries if (not e.metered) or e.rel >= thr]
        if not kept:
            kept = [max(entries, key=lambda e: e.rel)]
    if sort_by_brightness:
        kept.sort(key=lambda e: e.rel, reverse=True)
    return kept


def find_by_number(
    entries: Sequence[GhostEntry], number: Optional[int]
) -> Optional[int]:
    """Index of the entry carrying ``number``, or ``None``."""
    if number is None:
        return None
    for i, e in enumerate(entries):
        if e.number == number:
            return i
    return None
