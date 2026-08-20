"""Reusable surface-level mutations for the optical editor.

The toolbar wires its Form select to :func:`set_surface_form`; any other
caller can use the same entry point. Each helper wraps its
mutation in :meth:`Project.edit` so it participates in undo / redo.
"""
from __future__ import annotations

from typing import Iterable, List

import ghostlight

from ..project import Project


_FORM_LABELS = {
    int(ghostlight.SurfaceForm.SPHERE):      "Sphere",
    int(ghostlight.SurfaceForm.ASPHERE):     "Asphere",
    int(ghostlight.SurfaceForm.CYLINDRICAL): "Cylindrical",
}


def available_forms() -> List[tuple[int, str]]:
    """Return ``[(form_int, display_label), ...]`` for the Form select."""
    out: list[tuple[int, str]] = []
    for member in ghostlight.SurfaceForm.__members__.values():
        out.append((int(member), _FORM_LABELS.get(int(member), member.name.title())))
    return out


def form_label(form_int: int) -> str:
    return _FORM_LABELS.get(int(form_int), str(form_int))


def set_surface_form(project: Project, surface_index: int, form_int: int) -> bool:
    """Set ``system.surfaces[surface_index].form`` to ``form_int``.

    Returns ``True`` when the form actually changed, ``False`` on a no-op
    (out-of-range index, same form already set). The undo entry is labelled
    ``"Set Form"`` to match the in-tree Name-column edit path.
    """
    system = project.system
    if not (0 <= surface_index < len(system.surfaces)):
        return False
    surf = system.surfaces[surface_index]
    new_form = int(form_int)
    if new_form == int(surf.form):
        return False

    with project.edit("Set Form"):
        surf.form = new_form
    return True


def set_surface_ghost_solo(
    project: Project, surface_index: int, solo: bool,
) -> bool:
    """Toggle the ghost-solo flag for the surface at ``surface_index``.

    Ghost-solo is a render-time view filter (not a lens mutation), so this
    does NOT participate in undo / redo — Ctrl+Z would surprise the user
    by undoing a debugging-view toggle. The Project stores solo'd surface
    UUIDs and emits :attr:`Project.ghostSoloChanged`; render panels build
    an :class:`ghostlight.GhostFilter` from this state at dispatch time.

    Returns True when the flag actually changed (caller can skip a redraw
    when it didn't). False also covers out-of-range indices and surfaces
    with no UUID.
    """
    system = project.system
    if not (0 <= surface_index < len(system.surfaces)):
        return False
    uuid = str(system.surface_ids[surface_index])
    return project.set_surface_ghost_solo(uuid, bool(solo))


def set_surface_radius(project: Project, surface_index: int, value: float) -> bool:
    """Set the sphere radius of the surface at ``surface_index``.

    Only spherical surfaces have a single editable radius, so aspheres /
    cylindricals and stop surfaces are refused (the caller greys the row).
    Radii don't move vertices in this model, so no ``finalize`` is needed.
    Returns ``True`` when the radius actually changed."""
    system = project.system
    if not (0 <= surface_index < len(system.surfaces)):
        return False
    surf = system.surfaces[surface_index]
    if bool(surf.is_stop):
        return False
    if int(surf.form) != int(ghostlight.SurfaceForm.SPHERE):
        return False
    new_value = float(value)
    if new_value == float(surf.radius):
        return False

    with project.edit("Set Radius"):
        surf.radius = new_value
    return True


def set_surface_thickness(project: Project, surface_index: int, value: float) -> bool:
    """Set the thickness (gap to the next surface) at ``surface_index``.

    Mirrors the tree's relative pos-z write: sets the thickness then calls
    ``system.finalize()`` to re-derive the whole z-chain from the sensor
    back. Returns ``True`` when the thickness actually changed."""
    system = project.system
    if not (0 <= surface_index < len(system.surfaces)):
        return False
    surf = system.surfaces[surface_index]
    new_value = float(value)
    if new_value == float(surf.thickness):
        return False

    with project.edit("Set Thickness"):
        surf.thickness = new_value
        system.finalize()
    return True


def compute_ghost_filter(project: Project) -> "ghostlight.GhostFilter":
    """Build an :class:`ghostlight.GhostFilter` from the project's solo'd surfaces.

    Semantics: if no surfaces are solo'd, the filter is mode ALL (no-op).
    Otherwise mode INCLUDE with the union of ghost pairs in which any
    solo'd surface participates — i.e. multi-solo is "show me ghosts
    where surface A *or* B is involved", not "where both are involved".

    Surfaces solo'd by UUID that aren't currently in the system are
    skipped (the Project also prunes dead UUIDs after each mutation,
    but this is defensive in case the prune hasn't run yet). Muted
    surfaces participate in the index translation but produce no pairs
    on the C++ side, so soloing a muted surface still yields an empty
    filter (which is correct — there's nothing to show).
    """
    solo = project.ghost_solo_surface_uuids
    gf = ghostlight.GhostFilter()
    if not solo:
        gf.mode = ghostlight.GhostFilter.Mode.ALL
        return gf
    system = project.system
    uuid_to_idx = {u: i for i, u in enumerate(system.surface_ids)}
    solo_indices = {uuid_to_idx[u] for u in solo if u in uuid_to_idx}
    if not solo_indices:
        gf.mode = ghostlight.GhostFilter.Mode.ALL
        return gf
    # `enumerate_ghost_pairs` already filters inactive surfaces, so soloing
    # a muted surface naturally yields nothing for that surface — the
    # filter union below picks up whatever pairs still survive.
    pairs: list[tuple[int, int]] = []
    for p in ghostlight.enumerate_ghost_pairs(system):
        if p.surf_a in solo_indices or p.surf_b in solo_indices:
            pairs.append((p.surf_a, p.surf_b))
    gf.mode = ghostlight.GhostFilter.Mode.INCLUDE
    gf.pairs = pairs
    return gf
