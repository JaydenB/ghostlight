"""Reusable coating mutations for the optical-design editor.

Mirrors :mod:`surface_actions`: a pure predicate + a pure apply helper, plus
undo-wrapped variants for the surface right-click menu.

``apply_coating_payload_to_system`` is the single place that turns a JSON
coating-modifier dict (catalogue preset payload / writer output) into calls on
the ``OpticalSystem.set_coating_*`` accessors. It is shared by:

* the coating-row **Preset** picker writer (already running inside the model's
  ``project.edit(...)`` transaction — it calls the pure helper directly), and
* the surface **Coating** context submenu (which owns its own transaction via
  :func:`apply_coating_preset`).

Keeping both on one apply path means every coating tier round-trips through the
same, test-covered code.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import ghostlight


def surface_has_coating(surface) -> bool:
    """True when a surface carries an applied coating.

    A default / fresh :class:`ghostlight.Surface` is ``SIMPLE`` with ``ar_layers == 0``
    — i.e. uncoated, bare Fresnel. The optical-design editor shows a coating
    child row only when this returns True, so plain spherical surfaces stay
    leaf rows until the user applies a coating.
    """
    c = surface.coating
    return int(c.model) != int(ghostlight.CoatingModel.SIMPLE) or int(c.ar_layers) > 0


def apply_coating_payload_to_system(system, surface_index: int, mod: dict) -> bool:
    """Apply a JSON coating-modifier ``mod`` to ``system.surfaces[surface_index]``.

    ``mod`` is one entry of a surface's ``modifiers`` array (the shape the
    ``.lens`` writer emits / the C++ parser reads / a catalogue preset stores).
    The caller owns the undo transaction. Returns True on success, False on a
    bad index or malformed payload (leaving the surface untouched-ish — the
    coating is first cleared, so a partial failure lands on "uncoated").
    """
    if not isinstance(mod, dict):
        return False
    if not (0 <= surface_index < len(system.surfaces)):
        return False

    surf = system.surfaces[surface_index]
    system.clear_coating(surface_index)
    model = mod.get("model")
    try:
        # `layers` is discriminated by an explicit model, like every other
        # tier; a bare `layers` array with no model is also accepted.
        if model == "layers" or (model is None and "layers" in mod):
            layers = [
                {"material": str(ly.get("material", "")),
                 "thickness_nm": float(ly["thickness_nm"]),
                 "nk_table": np.array(
                     [[e["lambda_um"], e.get("n", 1.0), e.get("k", 0.0)]
                      for e in ly["nk_table"]], dtype=np.float32)}
                for ly in mod["layers"]
            ]
            system.set_coating_layers(surface_index, layers)
        elif model == "simple":
            # Default matches the C++ parser (0 = uncoated); this read 1,
            # so an ar_layers-less preset applied a coating the loader would
            # not have.
            surf.coating.ar_layers = int(mod.get("ar_layers", 0))
        elif model == "artist":
            c = surf.coating
            c.model = ghostlight.CoatingModel.ARTIST
            tint = mod.get("tint", [1.0, 1.0, 1.0])
            c.tint_r, c.tint_g, c.tint_b = (float(tint[0]), float(tint[1]),
                                            float(tint[2]))
            c.tint_strength = float(mod.get("strength", 0.04))
        elif model == "spectral":
            data = [[float(e["lambda_nm"]), float(e["r"])] for e in mod["data"]]
            system.set_coating_spectral_table(
                surface_index, np.array(data, dtype=np.float32),
                out_of_range_discard=(mod.get("out_of_range") == "discard"))
        elif model == "angular":
            data = [[float(e["angle_deg"]), float(e["r"])] for e in mod["data"]]
            system.set_coating_angular_table(
                surface_index, np.array(data, dtype=np.float32),
                angle_ref_ior=float(mod.get("angle_ref_ior", 1.0)),
                out_of_range_discard=(mod.get("out_of_range") == "discard"))
        elif model == "spectral_angular":
            system.set_coating_sa_table(
                surface_index,
                np.array(mod["wavelengths_nm"], dtype=np.float32),
                np.array(mod["angles_deg"], dtype=np.float32),
                np.array(mod["r"], dtype=np.float32),
                angle_ref_ior=float(mod.get("angle_ref_ior", 1.0)),
                out_of_range_discard=(mod.get("out_of_range") == "discard"))
        elif model == "attenuator_gaussian":
            c = surf.coating
            c.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
            c.gauss_sigma = float(mod.get("sigma", 0.0))  # matches the C++ parser
            c.gauss_background = float(mod.get("attenuation_background", 0.0))
            c.gauss_peak = float(mod.get("attenuation_peak", 0.0))
            c.gauss_decenter_x = float(mod.get("decenter_x", 0.0))
            c.gauss_decenter_y = float(mod.get("decenter_y", 0.0))
        else:
            return False
    except Exception:  # noqa: BLE001 — malformed preset shouldn't crash the UI
        return False
    return True


def apply_coating_preset(
    project, surface_index: int, mod: dict, *, label: str = "Apply Coating",
) -> bool:
    """Undo-wrapped coating-preset application for the surface context menu.

    Returns True when applied (and an undo entry pushed), False on no-op /
    failure (transaction aborted, nothing pushed).
    """
    with project.edit(label) as txn:
        if not apply_coating_payload_to_system(project.system, surface_index, mod):
            txn.abort()
            return False
    return True


def remove_coating(project, surface_index: int) -> bool:
    """Clear a surface's coating (back to uncoated), undo-wrapped.

    No-op (returns False, nothing pushed) when the surface is already
    uncoated or the index is out of range.
    """
    system = project.system
    if not (0 <= surface_index < len(system.surfaces)):
        return False
    if not surface_has_coating(system.surfaces[surface_index]):
        return False
    with project.edit("Remove Coating"):
        system.clear_coating(surface_index)
    return True
