"""Reusable element-level mutations for the optical editor.

These helpers are written to be reusable from anywhere — a toolbar button,
a context menu, a viewport radial menu, a keyboard shortcut. They wrap their
mutation in :meth:`Project.edit` so undo / redo and dirty tracking come for
free, and they return the newly-created element so the caller can update
selection if it wants to.

Element types we know how to construct:

* singlet         — one BK7 lens, two glass surfaces + one material
* doublet         — cemented BK7 + SF5, three surfaces + two materials
* triplet         — cemented BK7 + SF5 + BK7, four surfaces + three materials
* aperture stop   — one stop surface, no materials

Each builder picks reasonable default radii / thicknesses / apertures so the
new element actually traces; the user is expected to edit values afterwards.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from typing import List, Optional, TYPE_CHECKING

import ghostlight
from ghostlight.writer import build_optical_system_doc

from ..material_catalogue import get_catalogue
from ..project import Project, VariableBounds

if TYPE_CHECKING:
    from .anamorphic_dialog import AnamorphicSpec


# Default catalogue keys used by the element builders. These resolve through
# the bundled material catalogue (Schott N-BK7 + SF5 — both Sellmeier in the
# shipped JSON, with the same nd/Vd the legacy hardcoded fallback used).
DEFAULT_GLASS_KEY = "Schott_N-BK7"
DEFAULT_FLINT_KEY = "Schott_SF5"

# Fallback Abbe pairs used only when the catalogue lookup fails (headless
# tests with no bundled JSON on disk, partial bootstrap, etc.). Keyed by the
# same namespaced strings the builders pass in.
_GLASS_DEFAULTS: dict[str, tuple[float, float]] = {
    "Schott_N-BK7": (1.5168, 64.17),
    "Schott_SF5":   (1.6727, 32.21),
}


def _resolve_glass(glass: str) -> tuple[Optional["CatalogueMaterial"], float, float]:  # noqa: F821
    """Return ``(catalogue_material, n_d, V_d)`` for ``glass``.

    Catalogue-first: prefer the bundled material's dispersion (which may be
    Sellmeier — its nd/Vd come from the source SPECS via ui.nd/ui.vd).
    Falls back to the hardcoded Abbe pair in :data:`_GLASS_DEFAULTS` when
    the catalogue can't resolve the key (test environments without the
    bundled JSON, unknown glass strings).
    """
    cat = get_catalogue()
    cm = cat.by_key(glass)
    if cm is not None and cm.nd is not None and cm.vd is not None:
        return cm, float(cm.nd), float(cm.vd)
    n_d, v_d = _GLASS_DEFAULTS.get(glass, _GLASS_DEFAULTS[DEFAULT_GLASS_KEY])
    return None, n_d, v_d

_GLASS_THICKNESS_DEFAULT = 10.0
_CEMENT_THICKNESS_DEFAULT = 10.0
_AIR_THICKNESS_DEFAULT = 10.0


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _make_glass_surface(
    *,
    radius: float,
    thickness: float,
    glass: str,
    semi_aperture: float,
) -> ghostlight.Surface:
    _cm, n_d, v_d = _resolve_glass(glass)
    s = ghostlight.Surface()
    s.radius = float(radius)
    s.thickness = float(thickness)
    s.ior = n_d
    s.abbe_v = v_d
    s.semi_aperture = float(semi_aperture)
    # Surface.disp_model stays ABBE: the live tracer always reads from
    # ior/abbe_v even when the on-disk catalogue is Sellmeier — the next
    # reload re-derives ior/abbe_v from the catalogue's Sellmeier via
    # the C++ sellmeier_to_abbe backfill, so we don't have to wire a
    # Sellmeier path into the bind-time surface.
    s.disp_model = ghostlight.DispersionModel.ABBE
    return s


def _make_air_surface(
    *,
    radius: float,
    thickness: float,
    semi_aperture: float,
) -> ghostlight.Surface:
    s = ghostlight.Surface()
    s.radius = float(radius)
    s.thickness = float(thickness)
    s.ior = 1.0
    s.abbe_v = 0.0
    s.semi_aperture = float(semi_aperture)
    s.disp_model = ghostlight.DispersionModel.AIR
    return s


def _make_stop_surface(*, thickness: float, semi_aperture: float) -> ghostlight.Surface:
    s = ghostlight.Surface()
    s.radius = 0.0
    s.thickness = float(thickness)
    s.ior = 1.0
    s.abbe_v = 0.0
    s.semi_aperture = float(semi_aperture)
    s.is_stop = True
    s.disp_model = ghostlight.DispersionModel.AIR
    return s


def _ensure_glass_in_catalogue(system: ghostlight.OpticalSystem, glass: str) -> None:
    """Register ``glass`` in ``system._raw_glass_catalogue`` if absent.

    The undo path snapshots the system to a tempfile and re-loads it; the
    C++ loader rejects any material whose ``glass`` name isn't defined in
    ``glass_catalogue``. So every glass we drop in must have at least a
    minimal Abbe entry in the catalogue, even on a programmatically-built
    project that never saw a ``.lens`` file.

    Catalogue-aware: when ``glass`` resolves to a bundled material we
    install that material's full ``lens_catalogue_entry`` (display name,
    catalogue_ref, full Sellmeier or Abbe dispersion). Unknown keys still
    fall back to the hardcoded Abbe pair so legacy non-catalogue strings
    keep working.
    """
    if not glass:
        return
    catalogue = system._raw_glass_catalogue
    if glass in catalogue:
        return
    cm, n_d, v_d = _resolve_glass(glass)
    if cm is not None:
        catalogue[glass] = cm.lens_catalogue_entry()
        return
    catalogue[glass] = {
        "name": glass,
        "dispersion": {"model": "abbe", "nd": float(n_d), "Vd": float(v_d)},
    }


def _ensure_element_list(system: ghostlight.OpticalSystem) -> list:
    """Return the live ``system._elements`` list, initialising it if absent.

    A fresh ``ghostlight.OpticalSystem()`` starts with ``_elements = None`` — the
    grouping only exists after a load reparses it from disk. Live editing
    needs a real list, so promote ``None`` to ``[]`` the first time someone
    adds an element to an empty system.
    """
    if system._elements is None:
        system._elements = []
    return system._elements


def _insertion_index(
    system: ghostlight.OpticalSystem,
    after: Optional[ghostlight.Element],
) -> tuple[int, int]:
    """Resolve ``(element_index, surface_index)`` for the insertion point.

    ``after`` is the element the new one should be inserted **after**; when
    ``None`` (default), the new element goes at the **front** of the chain
    (index 0, furthest from the sensor). Front-insertion preserves every
    existing surface's absolute z: ``OpticalSystem.finalize()`` walks the
    chain backward from sensor = 0, so prepending only extends the chain
    further into −z without disturbing what was already laid down. End-
    insertion would shift every existing surface by the new element's
    extent, which is what the user wants to avoid.
    """
    elements = _ensure_element_list(system)
    if after is None or after not in elements:
        return 0, 0
    el_index = elements.index(after) + 1
    last_surface_index = max(after.resolve_surfaces(system))
    return el_index, last_surface_index + 1


def _insert_surfaces(
    system: ghostlight.OpticalSystem,
    insert_at: int,
    surfaces: List[ghostlight.Surface],
    surface_uuids: List[str],
) -> None:
    """Insert N surfaces + parallel UUIDs + aperture-image slots into ``system``.

    Both ``system.surface_ids`` and ``system.aperture_images`` are parallel
    to ``system.surfaces`` — they all have to grow at the same index or the
    indices stop lining up.
    """
    if len(surfaces) != len(surface_uuids):
        raise ValueError("surfaces and surface_uuids must have the same length")
    for offset, (surf, uuid_str) in enumerate(zip(surfaces, surface_uuids)):
        i = insert_at + offset
        system.insert_surface(i, surf, uuid_str)


def _delete_surfaces_at(system: ghostlight.OpticalSystem, indices: List[int]) -> None:
    """Delete the surfaces at ``indices`` (and their parallel-vector entries)."""
    for i in sorted(set(indices), reverse=True):
        system.erase_surface(i)


def _sync_element_positions(system: ghostlight.OpticalSystem) -> None:
    """Update each :class:`ghostlight.Element`'s ``position`` to match its first
    surface's z after ``finalize()`` (re-)laid the chain.

    The writer's JSON snapshot pulls ``position.z`` straight out of
    ``el.position`` — it doesn't introspect surface vertices. So if we
    insert / delete surfaces without also updating element positions,
    every newly-added element keeps its dataclass-default ``position =
    (0, 0, 0)`` and the snapshot encodes the whole chain stacked at
    z = 0. On undo, the loader patches every inter-element gap thickness
    from those stale positions (``next.z - last_surf_z_nominal``), which
    collapses into nonsense (negative thicknesses for any element with
    internal extent). Calling this after every ``finalize()`` keeps the
    in-memory position field consistent with the actual surface layout
    so snapshots round-trip correctly.

    ``x`` and ``y`` are preserved — those come from authored decenter /
    pivot transforms and aren't derived from surface positions.
    """
    if system._elements is None:
        return
    for el in system._elements:
        try:
            indices = el.resolve_surfaces(system)
        except KeyError:
            continue
        if not indices:
            continue
        first_z = float(system.surfaces[indices[0]].z)
        x, y, _z = el.position
        el.position = (x, y, first_z)


# ---------------------------------------------------------------------------
# Add — singlet / doublet / triplet / aperture stop
# ---------------------------------------------------------------------------


def _add_glass_element(
    project: Project,
    *,
    label: str,
    name: str,
    glasses: List[str],
    radii: List[float],
    thicknesses: List[float],
    semi_apertures: List[float],
    after: Optional[ghostlight.Element],
) -> ghostlight.Element:
    """Build a GLASS element with ``len(glasses)`` materials and
    ``len(glasses) + 1`` surfaces, insert it after ``after``."""
    n_glass = len(glasses)
    n_surfaces = n_glass + 1
    if not (len(radii) == len(thicknesses) == len(semi_apertures) == n_surfaces):
        raise ValueError(
            "_add_glass_element: radii/thicknesses/semi_apertures must each "
            f"have {n_surfaces} entries (matching {n_glass} glasses + 1)"
        )

    surface_uuids = [_new_uuid() for _ in range(n_surfaces)]
    surfaces: list[ghostlight.Surface] = []
    for i, glass in enumerate(glasses):
        surfaces.append(_make_glass_surface(
            radius=radii[i], thickness=thicknesses[i],
            glass=glass, semi_aperture=semi_apertures[i],
        ))
    # Trailing air interface — owns the inter-element gap as its thickness.
    surfaces.append(_make_air_surface(
        radius=radii[-1], thickness=thicknesses[-1],
        semi_aperture=semi_apertures[-1],
    ))

    new_element = ghostlight.Element(
        name=name,
        surface_ids=list(surface_uuids),
        material_glasses=list(glasses),
        kind=ghostlight.ElementKind.GLASS,
        element_id=_new_uuid(),
    )

    with project.edit(label):
        system = project.system
        for glass in glasses:
            _ensure_glass_in_catalogue(system, glass)
        el_idx, surf_idx = _insertion_index(system, after)
        _insert_surfaces(system, surf_idx, surfaces, surface_uuids)
        _ensure_element_list(system).insert(el_idx, new_element)
        system.finalize()
        _sync_element_positions(system)

    return new_element


def add_singlet(
    project: Project,
    *,
    after: Optional[ghostlight.Element] = None,
) -> ghostlight.Element:
    """Insert a single-glass lens (2 surfaces, 1 material)."""
    return _add_glass_element(
        project,
        label="Add Singlet",
        name="Singlet",
        glasses=[DEFAULT_GLASS_KEY],
        radii=[0.0, 0.0],
        thicknesses=[_GLASS_THICKNESS_DEFAULT, _AIR_THICKNESS_DEFAULT],
        semi_apertures=[20.0, 20.0],
        after=after,
    )


def add_doublet(
    project: Project,
    *,
    after: Optional[ghostlight.Element] = None,
) -> ghostlight.Element:
    """Insert a cemented doublet (3 surfaces, 2 materials)."""
    return _add_glass_element(
        project,
        label="Add Doublet",
        name="Doublet",
        glasses=[DEFAULT_GLASS_KEY, DEFAULT_FLINT_KEY],
        radii=[0.0, 0.0, 0.0],
        thicknesses=[
            _GLASS_THICKNESS_DEFAULT,
            _CEMENT_THICKNESS_DEFAULT,
            _AIR_THICKNESS_DEFAULT,
        ],
        semi_apertures=[20.0, 20.0, 20.0],
        after=after,
    )


def add_triplet(
    project: Project,
    *,
    after: Optional[ghostlight.Element] = None,
) -> ghostlight.Element:
    """Insert a cemented triplet (4 surfaces, 3 materials)."""
    return _add_glass_element(
        project,
        label="Add Triplet",
        name="Triplet",
        glasses=[DEFAULT_GLASS_KEY, DEFAULT_FLINT_KEY, DEFAULT_GLASS_KEY],
        radii=[50.0, -50.0, -100.0, 100.0],
        thicknesses=[
            _GLASS_THICKNESS_DEFAULT,
            _CEMENT_THICKNESS_DEFAULT,
            _CEMENT_THICKNESS_DEFAULT,
            _AIR_THICKNESS_DEFAULT,
        ],
        semi_apertures=[15.0, 15.0, 14.0, 14.0],
        after=after,
    )


# ---------------------------------------------------------------------------
# Anamorphic front block — inserts N cylindrical elements at the front of
# the system with an initial thin-lens guess and returns a merit function
# the optimizer can converge toward the requested squeeze ratio.
# ---------------------------------------------------------------------------


# Per-element internal geometry defaults used when seeding the block. These
# are just starting points — the optimizer is free to move air gaps within
# the bounds we declare below. Kept small so the block doesn't dominate the
# system's overall length before the optimizer touches it.
_ANAM_GLASS_THICKNESS = 10.0
_ANAM_CEMENT_THICKNESS = 8.0
_ANAM_INTER_ELEMENT_GAP = 8.0
_ANAM_SEMI_APERTURE = 25.0

# Variable-flag bounds. Radii can flip sign (a Galilean seed sometimes
# converges to a Keplerian-like final in the far tail of the space), so
# bounds are symmetric around zero. Air gaps get a tighter positive-only
# clamp because a negative or absurdly-long gap makes no physical sense.
_ANAM_RADIUS_LO = -500.0
_ANAM_RADIUS_HI = 500.0
_ANAM_AIR_LO = 1.0
_ANAM_AIR_HI = 100.0

# EFL_Y goal weight scales inversely with the base EFL so its residual
# (weight * (efl - base_efl)) reads on the same order as the SQUEEZE_RATIO
# residual (weight 1.0, target ~ 2). Without this the mm-scale EFL_Y
# residual would dominate least_squares and the squeeze goal would starve.
_ANAM_EFL_Y_WEIGHT_SCALE = 1.0
_ANAM_SQUEEZE_WEIGHT = 1.0


def _initial_focal_lengths(
    *,
    num_elements: int,
    squeeze_factor: float,
    topology: str,
) -> list[float]:
    """Thin-lens seed focal lengths (mm) for each cyl element in the block.

    Two-element derivation (in the squeeze axis):

    * Galilean expander with angular magnification ``M_ang = 1/N``:
      ``f_1 = -d/(N-1)``, ``f_2 = N·d/(N-1)`` — front is negative, rear
      positive, no intermediate image.
    * Keplerian expander with ``|M_ang| = 1/N`` (image inverted on the
      squeeze axis): ``f_1 = d/(N+1)``, ``f_2 = N·d/(N+1)`` — both
      positive, intermediate image between them.

    For ``num_elements > 2`` the "positive rear" is split into
    ``num_elements - 1`` equal-power lenses (thin-lens power addition:
    ``n`` lenses of ``f`` each combine to focal length ``f/n``, so each
    individual focal length is ``(num_elements - 1) × f_rear``).

    Rough block length assumption: ``d ≈ num_elements × 20 mm`` (roughly
    the total glass + gap extent per element as seeded). The optimizer
    later refines every gap and radius within its declared bounds, so
    perfect thin-lens fidelity here isn't required — just avoiding a
    wrong-sign initial guess that would sit on the far side of a ridge
    in the residual landscape.
    """
    if num_elements < 2:
        raise ValueError("Anamorphic block needs at least 2 elements")
    d = float(num_elements) * 20.0
    N = float(squeeze_factor)
    if topology == "galilean":
        # (N-1) can never be exactly zero because the dialog clamps
        # squeeze_factor >= 1.05, but the max() is a cheap safety.
        denom = max(N - 1.0, 0.05)
        f_neg = -d / denom
        f_pos_total = N * d / denom
        f_pos_each = f_pos_total * float(num_elements - 1)
        return [f_neg] + [f_pos_each] * (num_elements - 1)
    if topology == "keplerian":
        denom = N + 1.0
        f_1 = d / denom
        f_2_total = N * d / denom
        f_2_each = f_2_total * float(num_elements - 1)
        return [f_1] + [f_2_each] * (num_elements - 1)
    raise ValueError(f"Unknown anamorphic topology: {topology!r}")


def _cyl_singlet_radii(f: float, glass: str) -> list[float]:
    """Equiconvex or equiconcave plano-cyl seed radii (in cyl axis) for a
    thin single-glass cyl lens of focal length ``f`` (mm).

    Uses ``1/f = 2(n-1)/R`` — solves to ``R = 2(n-1)|f|``, with signs
    chosen for equiconvex (``f > 0``) or equiconcave (``f < 0``). Returns
    two radii: front and back.
    """
    _cm, n_d, _v_d = _resolve_glass(glass)
    R = 2.0 * (n_d - 1.0) * abs(f)
    if f < 0.0:
        return [-R, R]
    return [R, -R]


def _cyl_doublet_radii(f: float, crown: str, flint: str) -> list[float]:
    """Cemented-doublet seed radii (in cyl axis) for a two-glass cyl
    element of total focal length ``f`` (mm).

    Achromat split by Abbe numbers:

    * Crown focal length ``f_c = f·(V_c - V_f)/V_c``
    * Flint focal length ``f_f = -f·(V_c - V_f)/V_f``

    The crown is seeded as equiconvex/equiconcave; the cemented interface
    (R2) inherits the crown's back curvature. R3 is then chosen so the
    flint's contribution ``(n_f - 1)(1/R2 - 1/R3)`` hits ``1/f_f``.

    Returns three radii: front, cement, back.
    """
    _cm_c, n_c, V_c = _resolve_glass(crown)
    _cm_f, n_f, V_f = _resolve_glass(flint)
    dV = V_c - V_f
    if abs(dV) < 1e-6 or abs(f) < 1e-6:
        # Degenerate glass pair (or zero target focal length) — fall back
        # to a mild non-degenerate shape so the optimizer has slope to
        # walk on rather than a Jacobian-zero seed.
        R = 2.0 * (n_c - 1.0) * max(abs(f), 20.0)
        sign = 1.0 if f >= 0.0 else -1.0
        return [sign * R, -sign * R, sign * R]

    f_c = f * dV / V_c
    f_f = -f * dV / V_f

    R1 = 2.0 * (n_c - 1.0) * abs(f_c)
    if f_c < 0.0:
        R1 = -R1
    R2 = -R1  # crown as equiconvex/equiconcave → cement at same |R|, opposite sign
    inv_r3 = 1.0 / R2 - 1.0 / ((n_f - 1.0) * f_f)
    if abs(inv_r3) < 1e-9:
        R3 = -R2
    else:
        R3 = 1.0 / inv_r3
    # Clamp far-out values before the optimizer sees them so the initial
    # residual is finite; the variable bounds later let the optimizer walk
    # beyond this clamp if needed.
    R3 = max(min(R3, 500.0), -500.0)
    return [R1, R2, R3]


def _make_cyl_glass_surface(
    *,
    radius: float,
    thickness: float,
    glass: str,
    semi_aperture: float,
    cyl_axis: int,
) -> ghostlight.Surface:
    s = _make_glass_surface(
        radius=radius,
        thickness=thickness,
        glass=glass,
        semi_aperture=semi_aperture,
    )
    s.form = int(ghostlight.SurfaceForm.CYLINDRICAL)
    s.cyl_axis = int(cyl_axis)
    return s


def _make_cyl_air_surface(
    *,
    radius: float,
    thickness: float,
    semi_aperture: float,
    cyl_axis: int,
) -> ghostlight.Surface:
    s = _make_air_surface(
        radius=radius,
        thickness=thickness,
        semi_aperture=semi_aperture,
    )
    s.form = int(ghostlight.SurfaceForm.CYLINDRICAL)
    s.cyl_axis = int(cyl_axis)
    return s


def _build_cyl_element(
    *,
    name: str,
    glasses: list[str],
    radii: list[float],
    cyl_axis: int,
    glass_thicknesses: list[float],
    air_gap: float,
) -> tuple[ghostlight.Element, list[ghostlight.Surface], list[str]]:
    """Build a single cylindrical element (surfaces + Element) ready for
    insertion. Does NOT touch the system — the caller inserts inside its
    own edit() transaction.

    ``radii`` has ``len(glasses) + 1`` entries. ``glass_thicknesses`` is
    one thickness per glass surface (front + any interior cement layers).
    The trailing air surface owns the ``air_gap`` — that's what becomes
    the gap-to-next-element and what we later flag as a variable.
    """
    n_glass = len(glasses)
    n_surfaces = n_glass + 1
    if len(radii) != n_surfaces:
        raise ValueError(
            f"_build_cyl_element: radii must have {n_surfaces} entries "
            f"(got {len(radii)})"
        )
    if len(glass_thicknesses) != n_glass:
        raise ValueError(
            f"_build_cyl_element: glass_thicknesses must have {n_glass} entries "
            f"(got {len(glass_thicknesses)})"
        )

    surface_uuids = [_new_uuid() for _ in range(n_surfaces)]
    surfaces: list[ghostlight.Surface] = []
    for i, glass in enumerate(glasses):
        surfaces.append(_make_cyl_glass_surface(
            radius=radii[i],
            thickness=glass_thicknesses[i],
            glass=glass,
            semi_aperture=_ANAM_SEMI_APERTURE,
            cyl_axis=cyl_axis,
        ))
    surfaces.append(_make_cyl_air_surface(
        radius=radii[-1],
        thickness=air_gap,
        semi_aperture=_ANAM_SEMI_APERTURE,
        cyl_axis=cyl_axis,
    ))
    element = ghostlight.Element(
        name=name,
        surface_ids=list(surface_uuids),
        material_glasses=list(glasses),
        kind=ghostlight.ElementKind.GLASS,
        element_id=_new_uuid(),
    )
    return element, surfaces, surface_uuids


def add_anamorphic_front_block(
    project: Project,
    *,
    spec: "AnamorphicSpec",
) -> tuple[list[ghostlight.Element], object]:
    """Insert an N-element cylindrical block in front of the base lens,
    flag its radii + air gaps as optimization variables, and return the
    inserted elements plus a fresh :class:`MeritFunction` targeting the
    requested squeeze ratio.

    The caller is expected to hand the returned merit function to an
    :class:`OptimizationRun` and open :class:`OptimizationPreviewDialog`
    so the user can watch the optimizer converge and accept/reject the
    result. If the user rejects, the lens keeps the initial-guess block
    (the block is inserted inside ``project.edit(...)``, so a single
    Ctrl+Z removes it — variable flags don't participate in undo but the
    surfaces they point at will be gone, so ``collect_variables`` will
    silently skip them on the next run).
    """
    # Local imports to keep module-level import graph small. Both
    # optimization_panel.data and lens_metrics import from ghostlight and
    # from project — neither imports back into element_actions, so this
    # is safe against circulars.
    from .. import lens_metrics as lm
    from ..optimization_panel.data import GoalEntry, GoalKind, MeritFunction

    system = project.system

    # Sample the base lens's Y-axis EFL BEFORE we insert anything. That's
    # the number the merit function pins efl_y to, so the base spherical
    # design stays effectively unchanged in the un-squeezed axis after
    # the optimizer settles.
    base_efl_y = lm._effective_focal_length_on_axis(system, "y")
    if base_efl_y is None or not math.isfinite(base_efl_y):
        # Empty or degenerate base — the wizard still runs, but we default
        # the EFL_Y target to a sensible cinema length so the merit
        # function has a target to walk toward instead of NaN.
        base_efl_y = 50.0

    n = int(spec.num_elements)
    focal_lengths = _initial_focal_lengths(
        num_elements=n,
        squeeze_factor=float(spec.squeeze_factor),
        topology=str(spec.topology),
    )

    # Resolve per-element type + per-element axis up front so the loop
    # body reads cleanly. Per-element axis is uniform except optionally
    # for element 0 (the very front).
    def _element_type_for(i: int) -> str:
        if spec.per_element_types is not None and i < len(spec.per_element_types):
            return spec.per_element_types[i]
        return str(spec.element_type)

    def _axis_for(i: int) -> int:
        if i == 0 and spec.front_axis_override is not None:
            return int(spec.front_axis_override)
        return int(spec.block_axis)

    new_elements: list[ghostlight.Element] = []
    with project.edit("Add Anamorphic Front Block"):
        # Every glass we're about to use has to be in the raw catalogue
        # before finalize() runs, or the undo-round-trip loader rejects
        # the snapshot on the next replay.
        _ensure_glass_in_catalogue(system, spec.crown_glass)
        needs_flint = any(
            _element_type_for(i) == "doublet" for i in range(n)
        )
        if needs_flint:
            _ensure_glass_in_catalogue(system, spec.flint_glass)

        for i in range(n):
            element_type = _element_type_for(i)
            axis = _axis_for(i)
            if element_type == "doublet":
                glasses = [spec.crown_glass, spec.flint_glass]
                radii = _cyl_doublet_radii(
                    focal_lengths[i], spec.crown_glass, spec.flint_glass,
                )
                glass_thicknesses = [_ANAM_GLASS_THICKNESS, _ANAM_CEMENT_THICKNESS]
            else:
                glasses = [spec.crown_glass]
                radii = _cyl_singlet_radii(focal_lengths[i], spec.crown_glass)
                glass_thicknesses = [_ANAM_GLASS_THICKNESS]

            # Last element in the block owns the gap to the base lens;
            # inner elements use the inter-element default.
            air_gap = (
                float(spec.front_gap_mm)
                if i == n - 1
                else _ANAM_INTER_ELEMENT_GAP
            )

            element, surfaces, uuids = _build_cyl_element(
                name=f"Anamorphic Cyl {i + 1}",
                glasses=glasses,
                radii=radii,
                cyl_axis=axis,
                glass_thicknesses=glass_thicknesses,
                air_gap=air_gap,
            )

            # Insert AFTER the previously-inserted new element so the
            # block preserves its own front-to-back order (element 0 is
            # the frontmost). Passing ``after=None`` would prepend every
            # element to index 0 and reverse the block.
            after = new_elements[-1] if new_elements else None
            el_idx, surf_idx = _insertion_index(system, after)
            _insert_surfaces(system, surf_idx, surfaces, uuids)
            _ensure_element_list(system).insert(el_idx, element)
            new_elements.append(element)

        system.finalize()
        _sync_element_positions(system)

    # ---- Variable flags -------------------------------------------------
    # Applied AFTER the edit txn — variable flags don't participate in
    # undo, and the flag setter emits its own signal so listeners
    # (viewport stripe repaint, unflag-all button gating) update live.
    radius_bounds = VariableBounds(lo=_ANAM_RADIUS_LO, hi=_ANAM_RADIUS_HI)
    air_bounds = VariableBounds(lo=_ANAM_AIR_LO, hi=_ANAM_AIR_HI)
    for element in new_elements:
        surface_uuids = list(element.surface_ids)
        for uuid_str in surface_uuids:
            project.set_variable_flag(uuid_str, "radius", radius_bounds)
        # Last surface owns the trailing air gap; flag its thickness so
        # the optimizer can retune inter-element spacing.
        if surface_uuids:
            project.set_variable_flag(
                surface_uuids[-1], "thickness", air_bounds,
            )

    # ---- Merit function -------------------------------------------------
    mf = MeritFunction.make(
        name=f"Anamorphic {spec.squeeze_factor:g}× front block",
    )
    mf.goals.append(GoalEntry.make(
        kind=GoalKind.SQUEEZE_RATIO,
        target=float(spec.squeeze_factor),
        weight=_ANAM_SQUEEZE_WEIGHT,
    ))
    mf.goals.append(GoalEntry.make(
        kind=GoalKind.EFL_Y,
        target=float(base_efl_y),
        # 1 / base_efl_y so a fractional deviation reads on the same
        # residual scale as the squeeze ratio's absolute deviation.
        weight=_ANAM_EFL_Y_WEIGHT_SCALE / max(abs(base_efl_y), 1.0),
    ))
    project.merit_functions.append(mf)
    project.mark_merit_functions_replaced()

    return new_elements, mf


def add_aperture_stop(
    project: Project,
    *,
    after: Optional[ghostlight.Element] = None,
) -> ghostlight.Element:
    """Insert an aperture-stop element (1 stop surface, no materials)."""
    surface_uuid = _new_uuid()
    surface = _make_stop_surface(
        thickness=_AIR_THICKNESS_DEFAULT,
        semi_aperture=12.5,
    )
    new_element = ghostlight.Element(
        name="Aperture Stop",
        surface_ids=[surface_uuid],
        material_glasses=[],
        kind=ghostlight.ElementKind.STOP,
        element_id=_new_uuid(),
    )

    with project.edit("Add Aperture Stop"):
        system = project.system
        el_idx, surf_idx = _insertion_index(system, after)
        _insert_surfaces(system, surf_idx, [surface], [surface_uuid])
        _ensure_element_list(system).insert(el_idx, new_element)
        system.finalize()
        _sync_element_positions(system)

    return new_element


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def remove_element(project: Project, element: ghostlight.Element) -> bool:
    """Remove ``element`` and its surfaces from the system.

    Returns ``True`` when the element was found and removed, ``False`` if
    it wasn't part of the system (e.g. stale selection after a reload). The
    operation is a single undo entry; surface indices are computed before
    deletion so the caller doesn't have to.
    """
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False

    try:
        surface_indices = element.resolve_surfaces(system)
    except KeyError:
        surface_indices = []

    with project.edit(f"Remove {element.name or 'Element'}"):
        # Re-fetch in case anything shifted between the early read and the
        # txn body — the edit context itself doesn't mutate, so it's safe.
        elements.remove(element)
        _delete_surfaces_at(system, surface_indices)
        if system.surfaces:
            system.finalize()
            _sync_element_positions(system)
    return True


# ---------------------------------------------------------------------------
# Move (drag-and-drop reorder)
# ---------------------------------------------------------------------------


def _relay_element_positions(
    entries: list,
    thickness_by_uuid: dict[str, float],
    *,
    relative: bool = False,
) -> None:
    """Rewrite each element's ``transform.position.z`` in ``entries`` so the
    C++ loader's inter-element gap patching reproduces the snapshotted
    surface thicknesses in ``thickness_by_uuid``.

    Loader formula (paraphrased from ``optical_system.cpp``)::

        last_surf_z_nominal[k] = element[k].pos.z
                               + sum(thickness[s] for s in element[k][:-1])
        surfaces[last_idx_of_k].thickness = element[k+1].pos.z
                                          - last_surf_z_nominal[k]

    Re-arranging, element ``k+1`` must sit ``delta_k`` further along than
    element ``k``, where ``delta_k = internal_extent[k] + T[k]`` (element
    ``k``'s full axial extent) and ``T[k]`` is the desired thickness for
    element[k]'s last surface. We leave ``entries[0]`` alone — the loader
    rebases the whole chain to sensor=0 anyway, so the first element is the
    (informational) anchor.

    ``relative=False`` writes absolute positions (``pos.z[k+1] = pos.z[k] +
    delta_k``), used by move / flip / merge. ``relative=True`` instead writes
    each element's z as the delta from its predecessor and tags it
    ``position_mode = "relative_to_preceding"`` — the same geometry, stored
    as a relative chain so the elements read as placed relative to one
    another (and cascade when an upstream thickness is edited). The writer
    re-derives on-disk deltas from resolved positions, so this round-trips.
    """
    if len(entries) < 2:
        return
    for k in range(len(entries) - 1):
        cur = entries[k]
        cur_surfaces = cur["surfaces"]
        # Internal extent = sum of thicknesses of every surface except the
        # last (those values come from disk verbatim; only the last is patched).
        internal_extent = sum(
            float(s.get("thickness", 0.0)) for s in cur_surfaces[:-1]
        )
        last_surf_uuid = cur_surfaces[-1]["id"]
        desired_thickness = thickness_by_uuid.get(
            last_surf_uuid,
            float(cur_surfaces[-1].get("thickness", 0.0)),
        )
        delta = internal_extent + desired_thickness
        next_pos = entries[k + 1]["transform"]["position"]
        if relative:
            next_pos["z"] = delta
            next_pos["mode"] = "relative_to_preceding"
        else:
            cur_pos_z = float(cur["transform"]["position"].get("z", 0.0))
            next_pos["z"] = cur_pos_z + delta


def move_element(project: Project, src_index: int, dst_index: int) -> bool:
    """Reorder ``system.elements`` by moving entry ``src_index`` to ``dst_index``.

    Semantics match Qt's :meth:`QAbstractItemModel.dropMimeData` convention:
    ``dst_index`` is the row in the **current** ordering before any removal,
    so ``dst_index == len(elements)`` appends. Returns ``True`` when the
    list actually changed, ``False`` on no-op (out-of-range src, dropping
    immediately before or after itself).

    Implementation note — the move goes through the JSON writer + loader
    (same path the undo system uses) rather than bind_vector ``insert`` /
    ``del`` shuffling. Pybind11 :class:`Surface` wrappers are index-bound,
    not value-bound: deleting a slot silently shifts other wrappers'
    underlying storage, so direct rearrangement loses data. The JSON
    round-trip rebuilds surfaces from a doc that we mutate in pure Python,
    which is safe.
    """
    system = project.system
    elements = _ensure_element_list(system)
    n = len(elements)
    if not (0 <= src_index < n):
        return False
    if dst_index < 0 or dst_index > n:
        dst_index = n
    # Dropping back at the same spot or immediately after itself is a no-op.
    if dst_index == src_index or dst_index == src_index + 1:
        return False

    new_el_index = dst_index if dst_index < src_index else dst_index - 1
    moved = elements[src_index]
    label = f"Move {moved.name or 'Element'}"
    selected_id = (
        project.selected_element.element_id
        if project.selected_element is not None else None
    )

    # Snapshot every surface's thickness BEFORE the reorder. The loader
    # patches each non-final element's last surface's thickness from the
    # element-position gap (``next_element.z - last_surf_z_nominal``), so
    # if we just shuffle the JSON entries without touching positions, those
    # patched thicknesses become whatever the OLD positions happen to imply
    # in the NEW order — typically nonsense (negative, overlapping). The
    # user's contract is "relative pos-z values must not change on drag",
    # so we re-derive element positions below to make the patcher produce
    # exactly the snapshotted thicknesses.
    thickness_by_uuid: dict[str, float] = {
        system.surface_ids[i]: float(system.surfaces[i].thickness)
        for i in range(len(system.surfaces))
    }

    with project.edit(label):
        doc = build_optical_system_doc(
            system=system,
            metadata=system._raw_metadata,
            glass_catalogue=system._raw_glass_catalogue,
        )
        entry = doc["optical_system"].pop(src_index)
        doc["optical_system"].insert(new_el_index, entry)

        _relay_element_positions(doc["optical_system"], thickness_by_uuid)
        _commit_doc(project, doc, selected_id)
    return True


def flip_element(project: Project, element: ghostlight.Element) -> bool:
    """Reverse ``element`` in place (its surfaces, materials, and radii).

    Models physically rotating the element 180° around its axial centre:

    * Surface order reverses (front ↔ back).
    * Materials reverse (the cement layers between surfaces flip too).
    * Each surface's ``radius`` is negated (a convex-front-from-the-front
      becomes a convex-back-from-the-back).
    * Internal thicknesses (the N-1 gaps between the element's own
      surfaces) reverse. The trailing thickness (the air gap to the next
      element, or the back focal distance if it's the last) is left alone
      so surrounding elements and the sensor don't move.

    Returns ``True`` if the flip happened, ``False`` if ``element`` isn't
    part of the project's system or is a single-surface element (aperture
    stops have nothing to reverse).
    """
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    if len(element.surface_ids) < 2:
        return False

    el_index = elements.index(element)
    label = f"Flip {element.name or 'Element'}"
    selected_id = (
        project.selected_element.element_id
        if project.selected_element is not None else None
    )

    with project.edit(label):
        doc = build_optical_system_doc(
            system=system,
            metadata=system._raw_metadata,
            glass_catalogue=system._raw_glass_catalogue,
        )
        _flip_element_entry(doc["optical_system"][el_index])

        # Rebuild thickness_by_uuid from the (mutated) doc — every surface
        # UUID now maps to its POST-FLIP thickness. The relay then uses
        # those values to keep neighbouring elements anchored: previous
        # element's last-surface thickness stays put because the flipped
        # element's pos.z is unchanged, and the next element's pos.z is
        # recomputed from the flipped element's (unchanged) trailing
        # thickness.
        thickness_by_uuid = {
            s["id"]: float(s.get("thickness", 0.0))
            for el_entry in doc["optical_system"]
            for s in el_entry["surfaces"]
        }
        _relay_element_positions(doc["optical_system"], thickness_by_uuid)
        _commit_doc(project, doc, selected_id)
    return True


def _flip_element_entry(entry: dict) -> None:
    """In-place 180° flip of an element entry — keeps every vertex in place.

    Contract: no surface in the element (and no surface in neighbouring
    elements) moves in axial space. We do this by **decoupling surface
    identity from surface position**: each position keeps its original
    ``thickness`` value (which is what fixes the next surface's z), while
    the surface IDs / forms / radii / modifiers are taken from the
    reversed-order source. Radii negate (each surface is now hit from its
    other side) and materials reverse (so material[0] is the front chunk
    in the new orientation).

    Consequence: glass-chunk thicknesses **swap** for asymmetric elements
    (in a doublet with 5.3 mm BK7 + 2.8 mm SF5, after flip you get 5.3 mm
    SF5 + 2.8 mm BK7). A true physical 180° flip can preserve either
    vertex positions OR glass-chunk volumes, not both; the user's
    "don't move anything" contract picks vertex positions.
    """
    surfaces = entry.get("surfaces") or []
    if len(surfaces) < 2:
        return
    materials = entry.get("materials") or []

    # Per-position thicknesses pin the vertices in place — never reorder
    # these. Identity / form / modifiers come from the reversed-order
    # source so light hits what used to be the back first.
    position_thicknesses = [float(s.get("thickness", 0.0)) for s in surfaces]
    n = len(surfaces)

    new_surfaces: list[dict] = []
    for i in range(n):
        source = dict(surfaces[n - 1 - i])
        source["thickness"] = position_thicknesses[i]
        form = source.get("form")
        if isinstance(form, dict) and "radius" in form:
            new_form = dict(form)
            new_form["radius"] = -float(form["radius"])
            source["form"] = new_form
        new_surfaces.append(source)

    entry["surfaces"] = new_surfaces
    entry["materials"] = list(reversed(materials))


def set_element_muted(
    project: Project, element: ghostlight.Element, muted: bool,
) -> bool:
    """Toggle the mute state of every surface in ``element``.

    Muted surfaces are transparent in the raytracer — the ray passes through
    unbent, the surface contributes no Fresnel weight, and no ghost pair
    includes the surface. Geometry / position stay intact so the viewport
    still draws the element. STOP elements refuse muting (the pupil
    definition is load-bearing regardless of muting decisions); the caller
    receives a False return and can grey out the toggle.

    Wrapped in :meth:`Project.edit` so the change participates in undo /
    redo and dirty tracking. Aborts the edit when no surface actually
    changed state, so toggling an already-muted element doesn't push an
    empty undo entry.
    """
    system = project.system
    desired = bool(muted)
    if element.kind == ghostlight.ElementKind.STOP and desired:
        return False
    label = "Mute Element" if desired else "Unmute Element"
    changed_ref = {"flag": False}
    with project.edit(label) as txn:
        changed = element.set_muted(system, desired)
        if not changed:
            txn.abort()
            return False
        changed_ref["flag"] = True
    return bool(changed_ref["flag"])


def any_element_muted(project: Project) -> bool:
    """Whether at least one element in the system is currently muted.

    Used by the toolbar to gate the "Un-Mute All" button: cheaper than
    counting, and short-circuits on the first hit.
    """
    system = project.system
    return any(el.is_muted(system) for el in system.elements)


def unmute_all_elements(project: Project) -> int:
    """Unmute every muted element in the system in one compound edit.

    Returns the count of elements that actually flipped state. Zero on
    no-op (nothing was muted). STOP elements can never be muted, so
    they're naturally skipped by :meth:`Element.is_muted`. Batched under
    ``begin_compound`` / ``end_compound`` so undo restores everything in
    a single Ctrl+Z, matching the single toolbar click.
    """
    system = project.system
    victims = [el for el in system.elements if el.is_muted(system)]
    if not victims:
        return 0
    project.begin_compound("Unmute All")
    try:
        changed = 0
        for el in victims:
            if el.set_muted(system, False):
                changed += 1
    finally:
        project.end_compound()
    return changed


def is_element_ghost_solo(project: Project, element: ghostlight.Element) -> bool:
    """True iff every surface of the element is currently ghost-solo'd.

    Mixed state (some solo'd, some not) returns False so a subsequent
    toggle fills in the missing ones rather than clearing existing state.
    """
    solo = project.ghost_solo_surface_uuids
    if not solo:
        return False
    uuids = list(element.surface_ids)
    if not uuids:
        return False
    return all(u in solo for u in uuids)


def set_element_ghost_solo(
    project: Project, element: ghostlight.Element, solo: bool,
) -> bool:
    """Toggle ghost-solo across every surface of ``element``.

    STOP elements are refused — soloing the aperture surface doesn't fit
    the "which ghosts does this surface bounce" mental model the feature
    is designed for. Returns True when at least one surface flipped state.
    Ghost-solo is a render-time view filter, so this does not push undo.
    """
    if element.kind == ghostlight.ElementKind.STOP:
        return False
    return project.set_surfaces_ghost_solo(element.surface_ids, solo)


def _commit_doc(project: Project, doc: dict, selected_id: str | None) -> None:
    """Write ``doc`` to a temp .lens, reload the project's system from it,
    and re-resolve selection by ``selected_id`` so drag-drop / flip-driven
    rebuilds don't blow away the user's selection."""
    system = project.system
    fd, tmp_path = tempfile.mkstemp(suffix=".lens")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        system.reload(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # After reload, ``system.elements`` holds FRESH Element instances — the
    # original wrappers (including ``project.selected_element``) are orphaned.
    if selected_id:
        for new_el in system.elements:
            if new_el.element_id == selected_id:
                project.set_selected_element(new_el)
                break


# ---------------------------------------------------------------------------
# Append a whole lens file onto the front / back of the current system
# ---------------------------------------------------------------------------


def _regen_ids(entries: list) -> list[str]:
    """Assign fresh UUIDs to every element and surface in ``entries`` and
    return the new element ids in order.

    Surfaces are referenced positionally inside an element (and materials
    are a positional ``[{glass}]`` list), so regenerating ids is
    self-contained bookkeeping — it only prevents UUID collisions when the
    incoming lens shares ids with the current system (appending a file onto
    itself, or the same file twice). Colliding surface UUIDs would corrupt
    ``Element.resolve_surfaces``' uuid→index map on reload.
    """
    new_element_ids: list[str] = []
    for entry in entries:
        new_eid = _new_uuid()
        entry["id"] = new_eid
        new_element_ids.append(new_eid)
        for surf in entry["surfaces"]:
            surf["id"] = _new_uuid()
    return new_element_ids


def append_lens_from_file(
    project: Project,
    path: str,
    *,
    to_front: bool,
) -> List[ghostlight.Element]:
    """Load the lens at ``path`` and splice its elements onto the current
    system — the object side (``to_front=True``, furthest from the sensor)
    or the image side (``to_front=False``, closest to the sensor).

    Both lenses keep their exact internal spacing; the junction air gap is
    the front group's existing trailing (back-focal) thickness, editable
    afterward like any surface thickness. The combined chain is written with
    ``relative_to_preceding`` element positions (first element the absolute
    anchor) so the elements read as placed relative to one another — matching
    how a lens stack is authored — and cascade correctly when an upstream
    thickness is edited. The geometry is identical to an absolute layout;
    only the on-disk representation differs.

    Goes through the JSON writer/loader (the same path move / flip / merge
    use), so coatings, aspheres, apertures, stops and materials are all
    preserved. The incoming file's pivots are dropped.

    Returns the freshly-loaded ``Element`` wrappers for the appended elements
    (reload rebuilds every wrapper) so the caller can select / expand them.
    Raises ``ValueError`` if the file has no elements to append.
    """
    system = project.system
    incoming = ghostlight.OpticalSystem.load(path)
    if not incoming.elements:
        raise ValueError(
            f"{os.path.basename(path)} has no optical elements to append."
        )

    inc_doc = build_optical_system_doc(
        system=incoming,
        metadata=incoming._raw_metadata,
        glass_catalogue=incoming._raw_glass_catalogue,
    )
    cur_doc = build_optical_system_doc(
        system=system,
        metadata=system._raw_metadata,
        glass_catalogue=system._raw_glass_catalogue,
    )

    inc_entries = inc_doc["optical_system"]
    new_element_ids = _regen_ids(inc_entries)

    # Front append prepends the incoming chain (extends into −z, existing
    # surfaces keep their absolute z); back append inserts it just before the
    # sensor (the whole existing system shifts deeper into −z to make room).
    if to_front:
        merged = inc_entries + cur_doc["optical_system"]
    else:
        merged = cur_doc["optical_system"] + inc_entries
    cur_doc["optical_system"] = merged

    # Merge glass catalogues — the reload's C++ loader rejects any material
    # glass missing from the catalogue.
    merged_cat = dict(cur_doc.get("glass_catalogue") or {})
    merged_cat.update(inc_doc.get("glass_catalogue") or {})
    cur_doc["glass_catalogue"] = merged_cat

    # Both documents came through the loader, so both already carry the one
    # format version.
    cur_doc["version"] = ghostlight.lens_format_version()

    # Re-chain the whole combined system as relative-to-preceding deltas
    # (first element the absolute anchor). thickness_by_uuid carries each
    # surface's on-disk thickness so the junction gap becomes the front
    # group's trailing thickness (natural concatenation).
    thickness_by_uuid = {
        surf["id"]: float(surf.get("thickness", 0.0))
        for entry in merged
        for surf in entry["surfaces"]
    }
    merged[0]["transform"]["position"]["mode"] = "absolute"
    _relay_element_positions(merged, thickness_by_uuid, relative=True)

    first_new_id = new_element_ids[0] if new_element_ids else None
    label = "Import Lens → Front" if to_front else "Import Lens → Back"
    with project.edit(label):
        _commit_doc(project, cur_doc, first_new_id)

    new_ids = set(new_element_ids)
    return [el for el in project.system.elements if el.element_id in new_ids]


# ---------------------------------------------------------------------------
# Convert singlet → cemented doublet
# ---------------------------------------------------------------------------


def convert_to_doublet(project: Project, element: ghostlight.Element) -> bool:
    """Convert a 2-surface singlet into a cemented doublet in place.

    Splits the singlet's glass body in half along its axis and inserts a
    flat (radius 0) cemented interface, appending an SF5 flint half at the
    back — the same crown+flint pairing :func:`add_doublet` seeds. The two
    outer radii are preserved and the total internal glass extent is
    unchanged, so neither the element's outer surfaces nor anything
    downstream moves. Returns ``False`` for anything that isn't a plain
    single-glass singlet (stops, doublets, dummies).

    Live surface-list insertion (no doc round-trip), so the caller's
    ``element`` wrapper survives — but the element is no longer a singlet,
    so a viewport popup keyed on singlet-ness should close afterwards.
    """
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    if element.kind != ghostlight.ElementKind.GLASS:
        return False
    if len(element.surface_ids) != 2 or len(element.material_glasses) != 1:
        return False

    indices = element.resolve_surfaces(system)
    i0 = indices[0]
    front = system.surfaces[i0]
    t = float(front.thickness)
    mid_uuid = _new_uuid()

    with project.edit("Convert to Doublet"):
        _ensure_glass_in_catalogue(system, DEFAULT_FLINT_KEY)
        # Front half keeps the original glass; the new flat interface owns
        # the SF5 back half. Halving the front thickness and giving the new
        # surface the other half keeps the back surface's z exactly where it
        # was (t/2 + t/2 == t).
        front.thickness = t / 2.0
        mid = _make_glass_surface(
            radius=0.0,
            thickness=t / 2.0,
            glass=DEFAULT_FLINT_KEY,
            semi_aperture=float(front.semi_aperture),
        )
        _insert_surfaces(system, i0 + 1, [mid], [mid_uuid])
        # Parallel bookkeeping on the element: the mid surface sits between
        # the element's two existing surfaces, and its glass fills the
        # mid→back chunk (material index 1).
        element.surface_ids.insert(1, mid_uuid)
        element.material_glasses.append(DEFAULT_FLINT_KEY)
        system.finalize()
        _sync_element_positions(system)
    return True


# ---------------------------------------------------------------------------
# Merge adjacent elements — cement two glass elements into one n-let
# ---------------------------------------------------------------------------


def _merge_adjacent(project: Project, front_idx: int, rear_idx: int) -> bool:
    """Cement ``elements[front_idx]`` and ``elements[rear_idx]`` into one.

    The two facing surfaces (the front element's trailing surface and the
    air gap between the elements) are dropped; the rear element's front
    surface becomes the shared cemented interface. The interface keeps the
    front lens's back curvature when it's actually curved (a cemented joint
    reads as the front lens's back face), otherwise it keeps the rear
    lens's front curvature. Materials concatenate, so two singlets fuse
    into a doublet, a singlet+doublet into a triplet, and so on.

    Positions follow the same contract as :func:`flip_element`: the merged
    element's trailing gap is preserved so everything downstream stays
    anchored; the closed air gap is absorbed by the front of the chain
    sliding toward the sensor. Doc round-trip (surfaces are rebuilt on
    reload), so the caller's wrappers are orphaned afterwards.
    """
    system = project.system
    elements = _ensure_element_list(system)
    front = elements[front_idx]
    rear = elements[rear_idx]
    front_indices = front.resolve_surfaces(system)
    front_back_radius = float(system.surfaces[front_indices[-1]].radius)
    label = f"Merge {front.name or 'Element'} + {rear.name or 'Element'}"
    selected_id = front.element_id

    with project.edit(label):
        doc = build_optical_system_doc(
            system=system,
            metadata=system._raw_metadata,
            glass_catalogue=system._raw_glass_catalogue,
        )
        entries = doc["optical_system"]
        ea = entries[front_idx]
        eb = entries[rear_idx]

        # The cemented interface takes the rear element's front-surface
        # identity (UUID, aperture, its own glass-chunk thickness) but the
        # front lens's back curvature when that face is curved.
        fused = dict(eb["surfaces"][0])
        if abs(front_back_radius) > 1e-9:
            new_form = dict(fused.get("form") or {"type": "sphere"})
            new_form["radius"] = front_back_radius
            fused["form"] = new_form

        ea["surfaces"] = ea["surfaces"][:-1] + [fused] + eb["surfaces"][1:]
        ea["materials"] = (
            list(ea.get("materials") or []) + list(eb.get("materials") or [])
        )
        ea["name"] = f"{front.name or 'Element'} + {rear.name or 'Element'}"
        del entries[rear_idx]

        # Rebuild thickness map from the mutated doc (surface UUIDs → their
        # post-merge thicknesses) so the relay keeps neighbours anchored.
        thickness_by_uuid = {
            s["id"]: float(s.get("thickness", 0.0))
            for el_entry in entries
            for s in el_entry["surfaces"]
        }
        _relay_element_positions(entries, thickness_by_uuid)
        _commit_doc(project, doc, selected_id)
    return True


def merge_with_next(project: Project, element: ghostlight.Element) -> bool:
    """Cement ``element`` with the element immediately behind it (toward the
    sensor). Returns ``False`` when there is no next element, or either
    element isn't a plain glass element (stops / dummies can't cement)."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    idx = elements.index(element)
    if idx + 1 >= len(elements):
        return False
    rear = elements[idx + 1]
    if element.kind != ghostlight.ElementKind.GLASS or rear.kind != ghostlight.ElementKind.GLASS:
        return False
    return _merge_adjacent(project, idx, idx + 1)


def merge_with_previous(project: Project, element: ghostlight.Element) -> bool:
    """Cement ``element`` with the element immediately in front of it (toward
    the object). Returns ``False`` when there is no previous element, or
    either element isn't a plain glass element."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    idx = elements.index(element)
    if idx == 0:
        return False
    front = elements[idx - 1]
    if element.kind != ghostlight.ElementKind.GLASS or front.kind != ghostlight.ElementKind.GLASS:
        return False
    return _merge_adjacent(project, idx - 1, idx)


def can_merge_with_next(project: Project, element: ghostlight.Element) -> bool:
    """Whether :func:`merge_with_next` would succeed — drives popup enablement."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    idx = elements.index(element)
    if idx + 1 >= len(elements):
        return False
    return (
        element.kind == ghostlight.ElementKind.GLASS
        and elements[idx + 1].kind == ghostlight.ElementKind.GLASS
    )


def can_merge_with_previous(project: Project, element: ghostlight.Element) -> bool:
    """Whether :func:`merge_with_previous` would succeed."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    idx = elements.index(element)
    if idx == 0:
        return False
    return (
        element.kind == ghostlight.ElementKind.GLASS
        and elements[idx - 1].kind == ghostlight.ElementKind.GLASS
    )


# ---------------------------------------------------------------------------
# Move whole element along the axis
# ---------------------------------------------------------------------------


def move_element_z(project: Project, element: ghostlight.Element, new_z: float) -> bool:
    """Translate every surface of ``element`` so its first surface lands at
    absolute ``new_z``, leaving all other surfaces exactly where they are.

    The element's trailing gap absorbs the shift toward the sensor and the
    gap in front of it absorbs the shift toward the object; for the
    frontmost element (no surface in front) only the trailing gap changes
    and the whole front of the chain extends. Wrapped in a single undo
    entry. Returns ``False`` on a no-op (element gone, zero move)."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    try:
        indices = element.resolve_surfaces(system)
    except KeyError:
        return False
    if not indices:
        return False
    i0 = indices[0]
    i_last = indices[-1]
    z0 = float(system.surfaces[i0].z)
    delta = float(new_z) - z0
    if abs(delta) < 1e-12:
        return False

    with project.edit("Move Element Z"):
        # finalize() lays z backward from the sensor (z=0): shrinking the
        # element's trailing gap by ``delta`` slides the element +delta
        # toward the sensor; growing the gap in front by ``delta`` pins
        # everything ahead of it back where it was.
        s_last = system.surfaces[i_last]
        s_last.thickness = float(s_last.thickness) - delta
        if i0 > 0:
            s_before = system.surfaces[i0 - 1]
            s_before.thickness = float(s_before.thickness) + delta
        system.finalize()
        _sync_element_positions(system)
    return True


# ---------------------------------------------------------------------------
# Aperture — additive offset applied to every surface of an element
# ---------------------------------------------------------------------------


def offset_element_aperture(
    project: Project, element: ghostlight.Element, first_surface_semi_aperture: float,
) -> bool:
    """Shift every surface's ``semi_aperture`` by the delta needed to bring
    the element's first surface to ``first_surface_semi_aperture``.

    Additive semantics: each surface keeps its own relative aperture; only
    the common offset changes, so a scrub row anchored on the first
    surface's value moves the whole element's clear apertures together.
    Clamped to stay positive. Returns ``False`` on a no-op."""
    system = project.system
    elements = _ensure_element_list(system)
    if element not in elements:
        return False
    try:
        indices = element.resolve_surfaces(system)
    except KeyError:
        return False
    if not indices:
        return False
    current_first = float(system.surfaces[indices[0]].semi_aperture)
    delta = float(first_surface_semi_aperture) - current_first
    if abs(delta) < 1e-12:
        return False

    with project.edit("Scrub Aperture"):
        for si in indices:
            surf = system.surfaces[si]
            surf.semi_aperture = max(0.01, float(surf.semi_aperture) + delta)
    return True


# ---------------------------------------------------------------------------
# Focus (power) and bend (shape factor) — thick-singlet lensmaker math
# ---------------------------------------------------------------------------

# Below these thresholds a radius reads as flat (curvature 0) and a
# curvature reads as flat (radius 0). Keeping the flat convention as
# ``radius == 0`` matches the C++ core (FORM_SPHERE, "radius 0 = flat").
_FLAT_RADIUS_EPS = 1e-9
_FLAT_CURV_EPS = 1e-12
# Shape factor is undefined when the two curvatures are equal (flat-flat
# plate, or any equal-curvature meniscus); below this the shape reads None.
_SHAPE_EPS = 1e-12


def singlet_curvatures(r1: float, r2: float) -> tuple[float, float]:
    """Return ``(c1, c2)`` curvatures for radii ``r1, r2`` (flat → 0)."""
    c1 = 0.0 if abs(r1) < _FLAT_RADIUS_EPS else 1.0 / r1
    c2 = 0.0 if abs(r2) < _FLAT_RADIUS_EPS else 1.0 / r2
    return c1, c2


def singlet_power(n: float, t: float, r1: float, r2: float) -> float:
    """Thick-lens power ``φ`` (1/mm) of a singlet: glass index ``n``, centre
    thickness ``t`` (mm), surface radii ``r1, r2`` (mm).

    ``φ = (n-1)(c1 - c2) + (n-1)²·t·c1·c2 / n`` — the standard lensmaker
    equation with the codebase's radius sign convention (R>0 when the
    centre of curvature is on the sensor side)."""
    c1, c2 = singlet_curvatures(r1, r2)
    return (n - 1.0) * (c1 - c2) + (n - 1.0) ** 2 * t * c1 * c2 / n


def singlet_shape(r1: float, r2: float) -> Optional[float]:
    """Coddington shape factor ``X = (c1 + c2)/(c1 - c2)`` for radii
    ``r1, r2``. ``None`` when the curvatures are equal (X undefined)."""
    c1, c2 = singlet_curvatures(r1, r2)
    d = c1 - c2
    if abs(d) < _SHAPE_EPS:
        return None
    return (c1 + c2) / d


def solve_singlet_radii(
    n: float, t: float, power: float, shape_x: float, current_d: float = 0.0,
) -> Optional[tuple[float, float]]:
    """Solve ``(r1, r2)`` for a thick singlet of index ``n`` / thickness
    ``t`` at target power ``power`` (1/mm) and Coddington shape ``shape_x``.

    With ``d = c1 - c2`` and ``c1 = d(X+1)/2``, ``c2 = d(X-1)/2`` the
    lensmaker equation becomes ``A·d² + B·d - φ = 0`` where
    ``A = (n-1)²·t·(X²-1)/(4n)`` and ``B = n-1``. When ``A ≈ 0`` (thin
    lens, or ``|X| == 1`` where one surface is flat) it degenerates to the
    linear ``d = φ/(n-1)``. Two quadratic roots are disambiguated by
    picking the one nearest ``current_d`` so a scrub drag stays on one
    branch instead of jumping. Returns ``None`` when the target power is
    unreachable at this shape (negative discriminant) — the caller drops
    the scrub step, which reads as natural clamping."""
    x = float(shape_x)
    a = (n - 1.0) ** 2 * t * (x * x - 1.0) / (4.0 * n)
    b = n - 1.0
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            return None
        d = power / b
    else:
        disc = b * b + 4.0 * a * power
        if disc < 0.0:
            return None
        sq = math.sqrt(disc)
        d_plus = (-b + sq) / (2.0 * a)
        d_minus = (-b - sq) / (2.0 * a)
        d = (
            d_plus
            if abs(d_plus - current_d) <= abs(d_minus - current_d)
            else d_minus
        )
    c1 = d * (x + 1.0) / 2.0
    c2 = d * (x - 1.0) / 2.0
    r1 = 0.0 if abs(c1) < _FLAT_CURV_EPS else 1.0 / c1
    r2 = 0.0 if abs(c2) < _FLAT_CURV_EPS else 1.0 / c2
    return r1, r2


def _singlet_surfaces(
    system: ghostlight.OpticalSystem, element: ghostlight.Element,
) -> Optional[tuple[ghostlight.Surface, ghostlight.Surface]]:
    """Return ``(front, back)`` surfaces if ``element`` is a 2-surface glass
    singlet, else ``None``."""
    if element.kind != ghostlight.ElementKind.GLASS or len(element.surface_ids) != 2:
        return None
    try:
        idx = element.resolve_surfaces(system)
    except KeyError:
        return None
    if len(idx) != 2:
        return None
    return system.surfaces[idx[0]], system.surfaces[idx[1]]


def element_power(system: ghostlight.OpticalSystem, element: ghostlight.Element) -> Optional[float]:
    """Thick-lens power (1/mm) of a singlet ``element``, or ``None`` if it
    isn't a 2-surface glass singlet."""
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return None
    s1, s2 = pair
    return singlet_power(
        float(s1.ior), float(s1.thickness), float(s1.radius), float(s2.radius)
    )


def element_shape_factor(
    system: ghostlight.OpticalSystem, element: ghostlight.Element,
) -> Optional[float]:
    """Coddington shape factor of a singlet ``element``, or ``None`` if it
    isn't a singlet or its curvatures are equal (shape undefined)."""
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return None
    s1, s2 = pair
    return singlet_shape(float(s1.radius), float(s2.radius))


# Bend clamp: keep the front curvature inside a sane range (|R1| >= 2mm) so a
# runaway scrub can't collapse the surface to a near-point.
_MAX_BEND_CURVATURE = 0.5
# Below this curvature (|R| > 1 km) the front surface reads as flat, so a
# bend back down to zero snaps cleanly to radius 0.
_BEND_FLAT_CURVATURE = 1e-6


def element_front_curvature(
    system: ghostlight.OpticalSystem, element: ghostlight.Element,
) -> Optional[float]:
    """Front-surface curvature (1/mm) of a singlet ``element`` (0 for a flat
    front), or ``None`` if it isn't a singlet.

    Unlike :func:`element_shape_factor`, this is defined for a flat-flat
    singlet — it's the handle the viewport Bend row scrubs so a flat lens
    can still be bent (there's no shape factor to hold there)."""
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return None
    s1, _s2 = pair
    r1 = float(s1.radius)
    return 0.0 if abs(r1) < _FLAT_RADIUS_EPS else 1.0 / r1


def set_element_bend(
    project: Project, element: ghostlight.Element, front_curvature: float,
) -> bool:
    """Bend a singlet by setting its front-surface curvature to
    ``front_curvature``, leaving the back surface unchanged.

    Focal length is free to change: bending reshapes the front rather than
    holding power constant. Works for any singlet including a flat-flat
    plate: scrubbing the curvature up from zero turns a flat lens into a
    plano-convex/-concave one. Returns ``False`` when the element isn't a
    singlet."""
    system = project.system
    if element not in _ensure_element_list(system):
        return False
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return False
    s1, _s2 = pair
    c1 = max(-_MAX_BEND_CURVATURE, min(_MAX_BEND_CURVATURE, float(front_curvature)))
    r1 = 0.0 if abs(c1) < _BEND_FLAT_CURVATURE else 1.0 / c1
    with project.edit("Scrub Bend"):
        s1.radius = r1
    return True


def set_element_power(
    project: Project, element: ghostlight.Element, power_per_mm: float,
) -> bool:
    """Rescale a singlet's radii to hit target power ``power_per_mm`` (1/mm)
    while holding its Coddington shape factor constant.

    Singlet-only. A flat-flat plate (shape undefined) is focused as an
    equibiconvex/-concave lens (shape 0). Returns ``False`` when the
    element isn't a singlet or the target power is unreachable."""
    system = project.system
    if element not in _ensure_element_list(system):
        return False
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return False
    s1, s2 = pair
    n = float(s1.ior)
    t = float(s1.thickness)
    c1, c2 = singlet_curvatures(float(s1.radius), float(s2.radius))
    current_d = c1 - c2
    x = singlet_shape(float(s1.radius), float(s2.radius))
    if x is None:
        x = 0.0
    solved = solve_singlet_radii(n, t, float(power_per_mm), x, current_d)
    if solved is None:
        return False
    r1, r2 = solved
    with project.edit("Scrub Focus"):
        s1.radius = r1
        s2.radius = r2
    return True


def set_element_shape(
    project: Project, element: ghostlight.Element, shape_x: float,
) -> bool:
    """Reshape a singlet to Coddington shape factor ``shape_x`` while holding
    its power constant (a "bend"). Singlet-only. Returns ``False`` when the
    element isn't a singlet or the reshape is unreachable."""
    system = project.system
    if element not in _ensure_element_list(system):
        return False
    pair = _singlet_surfaces(system, element)
    if pair is None:
        return False
    s1, s2 = pair
    n = float(s1.ior)
    t = float(s1.thickness)
    c1, c2 = singlet_curvatures(float(s1.radius), float(s2.radius))
    current_d = c1 - c2
    phi = singlet_power(n, t, float(s1.radius), float(s2.radius))
    solved = solve_singlet_radii(n, t, phi, float(shape_x), current_d)
    if solved is None:
        return False
    r1, r2 = solved
    with project.edit("Scrub Bend"):
        s1.radius = r1
        s2.radius = r2
    return True
