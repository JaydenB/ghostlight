"""Element grouping + pivot rig reconstructed from .lens files.

The C++ :class:`OpticalSystem` holds a flat list of surfaces.  The .lens
file format groups them into ``element`` entries with names, transforms,
and materials, and adds a top-level ``pivots`` array that translates /
rotates groups of elements around a pivot point.  Both layers are dropped
by the C++ loader after they have been baked into surface decenter / z /
rot; this module re-parses the JSON so editors can display, mutate, and
write them back.

Each :class:`Element` references its constituent surfaces by UUID; the
parallel ``OpticalSystem.surface_ids`` vector lets
:meth:`Element.resolve_surfaces` translate to integer indices when needed.

Usage::

    import ghostlight

    system = ghostlight.OpticalSystem.load("doublet.lens")
    for el in system.elements:
        indices = el.resolve_surfaces(system)
        print(el.name, el.kind, indices)
    for piv in system.pivots:
        print(piv.name, piv.offset_position)
"""

from __future__ import annotations

import enum
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._ghostlight import _OpticalSystem


class ElementKind(enum.IntEnum):
    # Derived from the surfaces, not serialized: a single-surface group with
    # is_stop becomes STOP, everything else GLASS.
    GLASS = 0
    STOP = 1


@dataclass
class Element:
    """A named group of one or more surfaces that share materials + a transform.

    Mirrors a single ``element`` entry from the .lens ``optical_system``
    array.  Surface membership is by UUID (``surface_ids``); call
    :meth:`resolve_surfaces` against a populated optical system to get
    integer indices into ``system.surfaces``.

    ``position`` and ``rotation_euler_deg`` are the **resolved-absolute**
    pre-pivot pose (i.e. after ``position.mode`` resolution but before any
    pivot composition) — that's what gets written back to disk and what
    editors should mutate when the user grabs a single element directly.
    The pivot rig's baked pose lives on the surfaces themselves via
    ``decenter_x/y``, ``z``, and ``rot``.

    ``pivot`` is this element's centre of rotation in its local frame, relative
    to the first surface vertex. A zero pivot rotates about that vertex. This is
    independent of the group-level :class:`Pivot` rig below.
    """

    name: str
    surface_ids: list[str]
    material_glasses: list[str] = field(default_factory=list)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    element_id: str = ""
    kind: ElementKind = ElementKind.GLASS
    # `position.mode` from the source file — purely a round-trip hint for
    # editors that want to preserve the authoring mode.  The loader always
    # flattens to absolute, so this never affects runtime behaviour.
    position_mode: str = "absolute"

    def resolve_surfaces(self, system: "_OpticalSystem") -> list[int]:
        """Return surface indices into ``system.surfaces`` in trace order.

        Raises :class:`KeyError` if any UUID in ``surface_ids`` is missing
        from ``system.surface_ids``.
        """
        ids = list(system.surface_ids)
        lookup = {uuid: i for i, uuid in enumerate(ids)}
        out: list[int] = []
        for uuid in self.surface_ids:
            if uuid not in lookup:
                raise KeyError(
                    f"surface UUID {uuid!r} not found in OpticalSystem.surface_ids "
                    f"(element {self.name!r})"
                )
            out.append(lookup[uuid])
        return out

    def is_muted(self, system: "_OpticalSystem") -> bool:
        """True iff every surface of this element has ``is_active`` False.

        Stale surface UUIDs are treated as active — an Element with
        broken references is not considered muted.
        """
        try:
            indices = self.resolve_surfaces(system)
        except KeyError:
            return False
        if not indices:
            return False
        return all(not system.surfaces[i].is_active for i in indices)

    def set_muted(self, system: "_OpticalSystem", muted: bool) -> bool:
        """Set ``is_active`` across all surfaces of this element.

        Returns True when at least one surface changed state. STOP
        elements refuse muting so the pupil definition can't drop out.
        """
        if self.kind == ElementKind.STOP and muted:
            return False
        try:
            indices = self.resolve_surfaces(system)
        except KeyError:
            return False
        new_active = not bool(muted)
        changed = False
        for idx in indices:
            surf = system.surfaces[idx]
            if surf.is_active != new_active:
                surf.is_active = new_active
                changed = True
        return changed

    @classmethod
    def stop(
        cls,
        *,
        surface_id: str,
        name: str = "Aperture Stop",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        element_id: str = "",
    ) -> "Element":
        """Build a stop element with one surface and no glass materials."""
        return cls(
            name=name,
            surface_ids=[surface_id],
            material_glasses=[],
            position=position,
            element_id=element_id,
            kind=ElementKind.STOP,
        )

    @classmethod
    def from_lens_file(cls, path: str | os.PathLike) -> list["Element"]:
        """Re-parse a .lens JSON file and recover element groupings.

        Pair with :meth:`ghostlight.OpticalSystem.load(path)`; the surface
        UUIDs returned here match the ``surface_ids`` parallel vector on
        the loaded :class:`OpticalSystem`.
        """
        elements, _pivots = load_elements_and_pivots(path)
        return elements

    @classmethod
    def elements_from_uuids(
        cls,
        system: "_OpticalSystem",
        groupings: list[list[str]],
        *,
        names: list[str] | None = None,
    ) -> list["Element"]:
        """Build elements from a populated system + a list-of-lists of UUIDs.

        Use this for live-design workflows that never hit disk.  Materials
        are inferred from each grouping's IOR transitions:  a non-air gap
        between two surfaces inside the group becomes one entry in
        ``material_glasses`` (set to ``""`` since the glass name isn't
        known here — caller may fill in afterwards).  A single-surface
        group with ``is_stop=True`` becomes :attr:`ElementKind.STOP`.
        """
        ids = list(system.surface_ids)
        lookup = {uuid: i for i, uuid in enumerate(ids)}
        out: list[Element] = []
        for gi, uuids in enumerate(groupings):
            indices = [lookup[u] for u in uuids]
            surfaces = [system.surfaces[i] for i in indices]
            is_stop_group = (
                len(surfaces) == 1
                and getattr(surfaces[0], "is_stop", False)
            )
            kind = ElementKind.STOP if is_stop_group else ElementKind.GLASS
            mats: list[str] = [] if is_stop_group else [""] * max(0, len(surfaces) - 1)
            name = names[gi] if names is not None and gi < len(names) else f"element_{gi}"
            out.append(cls(
                name=name,
                surface_ids=list(uuids),
                material_glasses=mats,
                kind=kind,
            ))
        return out


@dataclass
class ExposedAttr:
    """One artist-facing pivot attribute (UI slider definition).

    ``attr`` is a dotted path into the pivot's offset block; valid values
    are the six entries of the schema's ``exposed_attr.attr`` enum
    (``offset.position.{x,y,z}``, ``offset.rotation.{tilt_x,tilt_y,roll}``).
    """
    name: str
    attr: str
    min: float = 0.0
    max: float = 0.0


_VALID_PIVOT_ATTRS = {
    "offset.position.x",
    "offset.position.y",
    "offset.position.z",
    "offset.rotation.tilt_x",
    "offset.rotation.tilt_y",
    "offset.rotation.roll",
}


@dataclass
class Pivot:
    """Post-process rig entry: translation + rotation on a set of elements.

    Mirrors one entry from the top-level ``pivots`` array.  The C++ loader
    bakes pivot transforms into surface poses at load time; this dataclass
    exists so editors and the round-trip writer can read, mutate, and
    re-emit pivots.

    ``offset_rotation`` is ``(tilt_x, tilt_y, roll)`` in degrees, matching
    the file format.  The pivot point is resolved at load time when
    ``pivot_point_mode == "centroid"``; the on-disk ``pivot_point.{x,y,z}``
    fields are only used when ``mode == "manual"``.
    """

    pivot_id: str
    name: str
    element_ids: list[str]
    pivot_point_mode: str = "centroid"
    pivot_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    exposed: list[ExposedAttr] = field(default_factory=list)

    @classmethod
    def from_lens_file(cls, path: str | os.PathLike) -> list["Pivot"]:
        """Re-parse a .lens JSON file and recover pivot definitions."""
        _elements, pivots = load_elements_and_pivots(path)
        return pivots

    # ------------------------------------------------------------------
    # Attribute paths
    # ------------------------------------------------------------------

    def get_attr(self, attr_path: str) -> float:
        """Read one of the ``exposed_attr`` paths off this pivot."""
        if attr_path not in _VALID_PIVOT_ATTRS:
            raise ValueError(
                f"invalid pivot attr path {attr_path!r}; "
                f"expected one of {sorted(_VALID_PIVOT_ATTRS)}"
            )
        _block, _field, idx = _split_attr_path(attr_path)
        if _block == "position":
            return float(self.offset_position[idx])
        return float(self.offset_rotation[idx])

    def set_attr(self, attr_path: str, value: float) -> None:
        """Write one of the ``exposed_attr`` paths on this pivot.

        Mutating the dataclass alone does not retrace anything — it just
        edits the artist-facing state.  Persist the change by calling
        :meth:`ghostlight.OpticalSystem.save` and re-loading (or use
        :meth:`ghostlight.OpticalSystem.reload` after writing).
        """
        if attr_path not in _VALID_PIVOT_ATTRS:
            raise ValueError(
                f"invalid pivot attr path {attr_path!r}; "
                f"expected one of {sorted(_VALID_PIVOT_ATTRS)}"
            )
        _block, _field, idx = _split_attr_path(attr_path)
        v = float(value)
        if _block == "position":
            tup = list(self.offset_position)
            tup[idx] = v
            self.offset_position = tuple(tup)  # type: ignore[assignment]
        else:
            tup = list(self.offset_rotation)
            tup[idx] = v
            self.offset_rotation = tuple(tup)  # type: ignore[assignment]


def _split_attr_path(attr_path: str) -> tuple[str, str, int]:
    """Map an ``offset.position.x`` style path to (block, field, axis_idx)."""
    # All valid paths are 3 components; the first is always "offset".
    parts = attr_path.split(".")
    block = parts[1]                # "position" | "rotation"
    field_name = parts[2]           # x|y|z | tilt_x|tilt_y|roll
    if block == "position":
        axis = {"x": 0, "y": 1, "z": 2}[field_name]
    else:
        axis = {"tilt_x": 0, "tilt_y": 1, "roll": 2}[field_name]
    return block, field_name, axis


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_elements_and_pivots(
    path: str | os.PathLike,
) -> tuple[list[Element], list[Pivot]]:
    """Parse a .lens file and return ``(elements, pivots)``.

    Element positions are resolved to absolute (``position.mode``
    ``relative_to_preceding`` is collapsed to absolute z = prev_z + z; a
    first-element relative entry emits a warning and is treated as
    absolute).  Pivot definitions are returned verbatim (not baked).
    """
    data = _parse_lens_json(path)

    elements: list[Element] = []
    prev_resolved_z: float | None = None
    surface_offset = 0
    for entry in data.get("optical_system", []):
        if entry.get("type") != "element":
            continue
        el = _element_from_entry(entry, prev_resolved_z, path, surface_offset)
        elements.append(el)
        prev_resolved_z = el.position[2]
        surface_offset += len(el.surface_ids)

    pivots: list[Pivot] = []
    for jp in data.get("pivots", []) or []:
        pivots.append(_pivot_from_entry(jp))

    return elements, pivots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_lens_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(os.fspath(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _element_from_entry(
    entry: dict[str, Any],
    prev_resolved_z: float | None,
    source_path: Any,
    surface_offset: int = 0,
) -> Element:
    name = str(entry.get("name", "element"))
    element_id = str(entry.get("id", ""))
    transform = entry.get("transform") or {}
    raw_position_block = transform.get("position") or {}
    position_mode = str(raw_position_block.get("mode", "absolute"))

    x = float(raw_position_block.get("x", 0.0))
    y = float(raw_position_block.get("y", 0.0))
    z = float(raw_position_block.get("z", 0.0))

    if position_mode == "relative_to_preceding":
        if prev_resolved_z is None:
            warnings.warn(
                f"{source_path}: first element {name!r} has "
                f"position.mode='relative_to_preceding'; treating as absolute.",
                stacklevel=3,
            )
        else:
            z = prev_resolved_z + z
    elif position_mode != "absolute":
        warnings.warn(
            f"{source_path}: element {name!r} has unknown "
            f"position.mode={position_mode!r}; treating as absolute.",
            stacklevel=3,
        )
        position_mode = "absolute"

    rotation = _parse_rotation(transform.get("rotation"))
    pivot = _parse_pivot(transform.get("pivot"))

    materials = entry.get("materials") or []
    material_glasses = [str(m.get("glass", "")) for m in materials]

    surfaces = entry.get("surfaces") or []
    # Blank ids must be filled in with the SAME synthetic value the C++ loader
    # uses (see flatten_element), or the two parses disagree about which
    # surface is which. Identity is by UUID; duplicates collapse in the
    # uuid -> index map that resolve_surfaces() builds.
    surface_ids = [
        str(s.get("id") or "") or f"auto-surface-{surface_offset + i}"
        for i, s in enumerate(surfaces)
    ]

    is_stop = (
        len(surfaces) == 1
        and not material_glasses
        and bool(surfaces[0].get("is_stop", False))
    )
    kind = ElementKind.STOP if is_stop else ElementKind.GLASS

    return Element(
        name=name,
        surface_ids=surface_ids,
        material_glasses=material_glasses,
        position=(x, y, z),
        rotation_euler_deg=rotation,
        pivot=pivot,
        element_id=element_id,
        kind=kind,
        position_mode=position_mode,
    )


def _pivot_from_entry(entry: dict[str, Any]) -> Pivot:
    pivot_id = str(entry.get("id", ""))
    name     = str(entry.get("name", ""))
    element_ids = [str(eid) for eid in entry.get("elements", []) or []]

    pp = entry.get("pivot_point") or {}
    pivot_point_mode = str(pp.get("mode", "centroid"))
    pivot_point = (
        float(pp.get("x", 0.0)),
        float(pp.get("y", 0.0)),
        float(pp.get("z", 0.0)),
    )

    off = entry.get("offset") or {}
    op = off.get("position") or {}
    offset_position = (
        float(op.get("x", 0.0)),
        float(op.get("y", 0.0)),
        float(op.get("z", 0.0)),
    )
    orot = off.get("rotation") or {}
    offset_rotation = (
        float(orot.get("tilt_x", 0.0)),
        float(orot.get("tilt_y", 0.0)),
        float(orot.get("roll", 0.0)),
    )

    exposed: list[ExposedAttr] = []
    for je in entry.get("exposed", []) or []:
        exposed.append(ExposedAttr(
            name=str(je.get("name", "")),
            attr=str(je.get("attr", "")),
            min=float(je.get("min", 0.0)),
            max=float(je.get("max", 0.0)),
        ))

    return Pivot(
        pivot_id=pivot_id,
        name=name,
        element_ids=element_ids,
        pivot_point_mode=pivot_point_mode,
        pivot_point=pivot_point,
        offset_position=offset_position,
        offset_rotation=offset_rotation,
        exposed=exposed,
    )


def _parse_pivot(p: Any) -> tuple[float, float, float]:
    """Accept ``{x, y, z}``; anything else is the all-zero default."""
    if not isinstance(p, dict):
        return (0.0, 0.0, 0.0)
    return (
        float(p.get("x", 0.0)),
        float(p.get("y", 0.0)),
        float(p.get("z", 0.0)),
    )


def _parse_rotation(r: Any) -> tuple[float, float, float]:
    """Parse the canonical ``{tilt_x, tilt_y, roll}`` rotation block."""
    if not isinstance(r, dict):
        return (0.0, 0.0, 0.0)
    return (
        float(r.get("tilt_x", 0.0)),
        float(r.get("tilt_y", 0.0)),
        float(r.get("roll", 0.0)),
    )
