"""Round-trip writer for ``ghostlight-optical`` JSON lens files.

The C++ loader bakes pivots into surface poses and drops the element /
pivot grouping after parsing.  Editors mutate the live :class:`Element` /
:class:`Pivot` dataclasses on the Python side and call
:meth:`ghostlight.OpticalSystem.save`, which delegates here.

The writer composes a JSON document from:

- the in-memory element list (`system.elements`) — supplies element UUIDs,
  names, transforms, materials, and per-element surface UUIDs
- the in-memory pivot list (`system.pivots`) — supplies pivot UUIDs,
  membership, pivot points, offsets, and exposed attributes
- the C++ ``Surface`` array (`system.surfaces`) — supplies the actual
  surface geometry (radius, form, thickness, coating, aperture shape, ...);
  surfaces are keyed by UUID via the parallel ``system.surface_ids``
- the stashed metadata + glass catalogue captured at load time, so we
  don't lose provenance round-tripping
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

from ._ghostlight import CoatingModel, LENS_FORMAT_MAJOR, LENS_FORMAT_MINOR

_CM_SIMPLE           = int(CoatingModel.SIMPLE)
_CM_SPECTRAL         = int(CoatingModel.SPECTRAL)
_CM_ANGULAR          = int(CoatingModel.ANGULAR)
_CM_SPECTRAL_ANGULAR = int(CoatingModel.SPECTRAL_ANGULAR)
_CM_ATTENUATOR_GAUSS = int(CoatingModel.ATTENUATOR_GAUSS)
_CM_ARTIST           = int(CoatingModel.ARTIST)

if TYPE_CHECKING:
    from .optical_system import OpticalSystem


def lens_format_version() -> dict[str, int]:
    """Return the version block for a ``.lens`` file."""
    return {"major": LENS_FORMAT_MAJOR, "minor": LENS_FORMAT_MINOR}


def build_optical_system_doc(
    *,
    system: "OpticalSystem",
    metadata: dict[str, Any] | None = None,
    glass_catalogue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the in-memory ``ghostlight-optical`` JSON document for ``system``.

    Same payload :func:`write_optical_system` writes to disk, returned as a
    dict instead of serialized. Used both by the file writer and by callers
    that need inexpensive in-memory snapshots.
    """
    doc: dict[str, Any] = {
        "format":  "ghostlight-optical",
        "version": {"major": LENS_FORMAT_MAJOR, "minor": LENS_FORMAT_MINOR},
    }

    # Canonical documents always contain every top-level section.
    if metadata is None:
        metadata = {}
        if system.name:
            metadata["name"] = system.name
        fl = float(getattr(system, "focal_length", 0.0) or 0.0)
        if fl:
            metadata["focal_length_mm"] = fl
    doc["metadata"] = dict(metadata)

    doc["glass_catalogue"] = dict(glass_catalogue or {})

    # Walk the chain tracking each element's resolved-absolute z so a
    # ``relative_to_preceding`` element can be re-emitted as a delta from
    # its predecessor (see ``_element_to_json``).
    optical_system: list[dict[str, Any]] = []
    prev_abs_z: float | None = None
    for el in system.elements:
        optical_system.append(_element_to_json(el, system, prev_abs_z))
        prev_abs_z = float(el.position[2])
    doc["optical_system"] = optical_system
    doc["pivots"] = [_pivot_to_json(p) for p in system.pivots]
    return doc


def write_optical_system(
    path: str | os.PathLike,
    *,
    system: "OpticalSystem",
    metadata: dict[str, Any] | None = None,
    glass_catalogue: dict[str, Any] | None = None,
) -> None:
    """Emit ``system`` as a ``ghostlight-optical`` JSON file.

    The flat surface array on the C++ side is reassembled into elements
    via the element list's surface UUIDs.  Pivots are written verbatim
    from the in-memory dataclasses.
    """
    doc = build_optical_system_doc(
        system=system, metadata=metadata, glass_catalogue=glass_catalogue,
    )
    target = os.fspath(path)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Element / surface serialisation
# ---------------------------------------------------------------------------

def _element_to_json(el, system, prev_abs_z: float | None = None) -> dict[str, Any]:
    # ``el.position`` is always the RESOLVED-ABSOLUTE vertex z — the loader
    # collapses ``relative_to_preceding`` to absolute at parse time but keeps
    # the authored mode. To round-trip a relative element we must re-derive
    # its on-disk z as the delta from the preceding element's absolute z;
    # emitting the stored absolute value under a "relative" mode would make
    # the loader re-add the previous z and double-count. A leading element
    # tagged relative has no predecessor, so it falls back to absolute —
    # matching the loader, which treats a first relative entry as absolute.
    abs_z = float(el.position[2])
    if el.position_mode == "relative_to_preceding" and prev_abs_z is not None:
        z_out = abs_z - prev_abs_z
    else:
        z_out = abs_z
    transform: dict[str, Any] = {
        "position": {
            "mode": el.position_mode,
            "x":    float(el.position[0]),
            "y":    float(el.position[1]),
            "z":    z_out,
        },
    }
    rot = el.rotation_euler_deg
    if any(rot):
        transform["rotation"] = {
            "tilt_x": float(rot[0]),
            "tilt_y": float(rot[1]),
            "roll":   float(rot[2]),
        }
    # An omitted pivot rotates about the element's front vertex.
    pivot = getattr(el, "pivot", (0.0, 0.0, 0.0))
    if any(pivot):
        transform["pivot"] = {
            "x": float(pivot[0]),
            "y": float(pivot[1]),
            "z": float(pivot[2]),
        }

    surface_indices = el.resolve_surfaces(system)

    surfaces_json: list[dict[str, Any]] = []
    for i, surf_idx in enumerate(surface_indices):
        s = system.surfaces[surf_idx]
        is_last = i == len(surface_indices) - 1

        entry: dict[str, Any] = {
            "id":            el.surface_ids[i],
            "semi_aperture": float(s.semi_aperture),
            "form":          _form_to_json(s),
        }
        if bool(s.is_stop):
            entry["is_stop"] = True
        if not bool(s.is_active):
            entry["is_active"] = False
        # `thickness` is on every surface (including the last — that's the
        # back focal distance for the final element), but for non-final
        # elements the loader patches the last surface's thickness from
        # inter-element nominal z so the on-disk value isn't authoritative.
        # Emitting it preserves the artist's intent for the in-element gaps;
        # the writer leaves the loader's later patching to do its work.
        entry["thickness"] = float(s.thickness)

        mods = _surface_modifiers_to_json(s, system, surf_idx)
        if mods:
            entry["modifiers"] = mods

        surfaces_json.append(entry)

    materials_json = [{"glass": g} for g in el.material_glasses]

    out: dict[str, Any] = {
        "type":      "element",
        "id":        el.element_id,
        "name":      el.name,
        "transform": transform,
        "surfaces":  surfaces_json,
        "materials": materials_json,
    }
    return out


def _form_to_json(surface) -> dict[str, Any]:
    """Serialise a ``Surface.form`` (+ associated fields) to JSON."""
    # Use the int values rather than the enum names to avoid pulling the
    # full ``ghostlight`` module in here.  Mapping mirrors SurfaceForm in
    # optical_system.h: 0=sphere, 1=asphere, 2=cylindrical.
    form_id = int(getattr(surface, "form", 0))
    if form_id == 0:  # sphere
        return {"type": "sphere", "radius": float(surface.radius)}
    if form_id == 1:  # asphere
        terms_arr = getattr(surface, "asphere_terms", None)
        terms: list[float] = []
        if terms_arr is not None:
            terms = [float(t) for t in list(terms_arr)]
        out: dict[str, Any] = {
            "type":           "asphere",
            "radius":         float(surface.radius),
            "conic_constant": float(getattr(surface, "conic_k", 0.0)),
        }
        if terms:
            out["terms"] = terms
        return out
    if form_id == 2:  # cylindrical
        axis_idx = int(getattr(surface, "cyl_axis", 0))
        axis = "y" if axis_idx == 1 else "x"
        return {"type": "cylindrical", "radius": float(surface.radius), "axis": axis}
    return {"type": "sphere", "radius": float(surface.radius)}


def _coating_modifier_to_json(surface, system, surf_idx) -> dict[str, Any] | None:
    """Serialise ``surface.coating`` to its JSON modifier (or None if uncoated).

    Every model accepted by the C++ parser must be emitted to preserve a
    serialization round trip.
    Table contents live off the Surface POD and are fetched through the
    ``system`` accessors (get_coating_table / get_coating_sa_table /
    get_coating_layers) keyed by the flat surface index.
    """
    coating = getattr(surface, "coating", None)
    if coating is None:
        return None
    model = int(getattr(coating, "model", 0))

    if model == _CM_SIMPLE:
        ar_layers = int(getattr(coating, "ar_layers", 0))
        if ar_layers > 0:
            return {"type": "coating", "model": "simple", "ar_layers": ar_layers}
        return None

    if model == _CM_ARTIST:
        return {
            "type": "coating",
            "model": "artist",
            "tint": [float(coating.tint_r), float(coating.tint_g),
                     float(coating.tint_b)],
            "strength": float(coating.tint_strength),
        }

    if model in (_CM_SPECTRAL, _CM_ANGULAR):
        table = system.get_coating_table(surf_idx)
        if table.shape[0] == 0:
            return None
        key = "lambda_nm" if model == _CM_SPECTRAL else "angle_deg"
        entry: dict[str, Any] = {
            "type": "coating",
            "model": "spectral" if model == _CM_SPECTRAL else "angular",
            "data": [
                {key: float(k), "r": float(r)} for k, r in table
            ],
        }
        if bool(coating.out_of_range_discard):
            entry["out_of_range"] = "discard"
        if model == _CM_ANGULAR and abs(float(coating.angle_ref_ior) - 1.0) > 1e-9:
            entry["angle_ref_ior"] = float(coating.angle_ref_ior)
        return entry

    if model == _CM_SPECTRAL_ANGULAR:  # possibly baked from a layer stack
        layers = system.get_coating_layers(surf_idx)
        if layers:
            # Emit the layer-stack INTENT, not the baked table — the loader
            # re-bakes on load, keeping the stack editable.
            return {
                "type": "coating",
                "model": "layers",
                "layers": [
                    {
                        "material":     str(layer["material"]),
                        "thickness_nm": float(layer["thickness_nm"]),
                        "nk_table": [
                            {"lambda_um": float(lu), "n": float(n), "k": float(k)}
                            for lu, n, k in layer["nk_table"]
                        ],
                    }
                    for layer in layers
                ],
            }
        wl, ang, r = system.get_coating_sa_table(surf_idx)
        if r.size == 0:
            return None
        entry = {
            "type": "coating",
            "model": "spectral_angular",
            "wavelengths_nm": [float(v) for v in wl],
            "angles_deg":     [float(v) for v in ang],
            "r": [[float(v) for v in row] for row in r],
        }
        if bool(coating.out_of_range_discard):
            entry["out_of_range"] = "discard"
        if abs(float(coating.angle_ref_ior) - 1.0) > 1e-9:
            entry["angle_ref_ior"] = float(coating.angle_ref_ior)
        return entry

    if model == _CM_ATTENUATOR_GAUSS:
        return {
            "type": "coating",
            "model": "attenuator_gaussian",
            "sigma":                  float(coating.gauss_sigma),
            "attenuation_background": float(coating.gauss_background),
            "attenuation_peak":       float(coating.gauss_peak),
            "decenter_x":             float(coating.gauss_decenter_x),
            "decenter_y":             float(coating.gauss_decenter_y),
        }

    return None


def _surface_modifiers_to_json(surface, system, surf_idx) -> list[dict[str, Any]]:
    """Serialise coating + aperture modifiers attached to ``surface``."""
    mods: list[dict[str, Any]] = []

    coating_mod = _coating_modifier_to_json(surface, system, surf_idx)
    if coating_mod is not None:
        mods.append(coating_mod)

    aperture_shape = int(getattr(surface, "aperture_shape", 0))
    aspect         = float(getattr(surface, "aperture_aspect", 1.0))
    if aperture_shape == 1:  # polygon
        blades   = int(getattr(surface, "aperture_blades", 0))
        rot_rad  = float(getattr(surface, "aperture_rotation_rad", 0.0))
        rot_deg  = rot_rad * (180.0 / 3.14159265358979323846)
        entry: dict[str, Any] = {
            "type":   "aperture",
            "shape":  "polygon",
            "blades": blades,
        }
        if abs(rot_deg) > 1e-9:
            entry["rotation_deg"] = rot_deg
        if abs(aspect - 1.0) > 1e-9:
            entry["aperture_aspect"] = aspect
        # Omit zero-valued blade parameters from the canonical form.
        for key, value in (
            ("curvature", float(getattr(surface, "aperture_curvature", 0.0))),
            ("twist",     float(getattr(surface, "aperture_twist", 0.0))),
        ):
            if abs(value) > 1e-9:
                entry[key] = value
        for key, rad in (
            ("notch_deg",       float(getattr(surface, "aperture_notch_rad", 0.0))),
            ("notch_angle_deg", float(getattr(surface, "aperture_notch_angle_rad", 0.0))),
        ):
            deg = rad * (180.0 / 3.14159265358979323846)
            if abs(deg) > 1e-9:
                entry[key] = deg
        mods.append(entry)
    elif aperture_shape == 2:  # image
        semi_d = float(getattr(surface, "aperture_semi_diameter", 0.0))
        # Image paths live in the array parallel to Surface storage.
        image_path = ""
        images = getattr(system, "aperture_images", None)
        if images is not None and surf_idx < len(images):
            image_path = str(getattr(images[surf_idx], "source_path", "") or "")
        if semi_d > 0:
            entry = {
                "type":   "aperture",
                "shape":  "image",
                "image_path":    image_path,
                "semi_diameter": semi_d,
            }
            if abs(aspect - 1.0) > 1e-9:
                entry["aperture_aspect"] = aspect
            mods.append(entry)
    elif abs(aspect - 1.0) > 1e-9:
        # Circular default, but with a non-unit aspect that needs preserving.
        mods.append({
            "type":            "aperture",
            "shape":           "circular",
            "aperture_aspect": aspect,
        })

    return mods


# ---------------------------------------------------------------------------
# Pivot serialisation
# ---------------------------------------------------------------------------

def _pivot_to_json(p) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id":       p.pivot_id,
        "name":     p.name,
        "elements": list(p.element_ids),
        "pivot_point": {
            "mode": p.pivot_point_mode,
            "x":    float(p.pivot_point[0]),
            "y":    float(p.pivot_point[1]),
            "z":    float(p.pivot_point[2]),
        },
        "offset": {
            "position": {
                "x": float(p.offset_position[0]),
                "y": float(p.offset_position[1]),
                "z": float(p.offset_position[2]),
            },
            "rotation": {
                "tilt_x": float(p.offset_rotation[0]),
                "tilt_y": float(p.offset_rotation[1]),
                "roll":   float(p.offset_rotation[2]),
            },
        },
    }
    if p.exposed:
        out["exposed"] = [
            {
                "name": e.name,
                "attr": e.attr,
                "min":  float(e.min),
                "max":  float(e.max),
            }
            for e in p.exposed
        ]
    return out
