"""Read tunable design variables off an :class:`ghostlight.OpticalSystem`.

Two flag kinds coexist here:

* :func:`collect_variables` — continuous surface attributes flagged in
  the Optical Design Editor (radius, thickness, …). Each yields a
  :class:`VariableRef` the scipy inner loop sweeps.
* :func:`collect_material_flags` — discrete material-substitution flags
  set via the ODE Material row's right-click submenu. Each yields a
  :class:`MaterialFlagRef`; the hammer orchestrator iterates candidates
  from the vendor's catalogue and installs each via
  :func:`install_candidate_glass`.

Keeping both readers here means the optimizer has one import boundary
and one place to gate "no variables at all" short-circuits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import ghostlight


# The attributes we're prepared to tune. Flat names match :class:`ghostlight.Surface`
# fields exactly; a dotted name (e.g. ``coating.tint_strength``) walks into a
# nested object on the surface — pybind ``def_readwrite`` members return a
# live reference, so ``setattr`` on the nested object mutates the surface.
SurfaceAttr = Literal[
    "radius",
    "thickness",
    "semi_aperture",
    "conic_k",
    "coating.tint_strength",
]


def _resolve_attr(surface, attr: str):
    """Return ``(owner_object, leaf_name)`` for a flat or dotted attr path.

    ``"radius"`` → ``(surface, "radius")``;
    ``"coating.tint_strength"`` → ``(surface.coating, "tint_strength")``.
    """
    parts = attr.split(".")
    owner = surface
    for name in parts[:-1]:
        owner = getattr(owner, name)
    return owner, parts[-1]


@dataclass(frozen=True)
class VariableRef:
    """One tunable scalar on one surface.

    ``lo`` / ``hi`` are inclusive bounds. ``None`` on either side means
    "unbounded on that side" — the optimizer translates that into ±inf
    when handing off to scipy.optimize.least_squares.

    ``attr`` may be a flat surface field (``radius``) or a dotted path into
    a nested object (``coating.tint_strength``).
    """
    surface_index: int
    attr: str
    lo: Optional[float] = None
    hi: Optional[float] = None

    def read(self, system: ghostlight.OpticalSystem) -> float:
        owner, leaf = _resolve_attr(system.surfaces[self.surface_index], self.attr)
        return float(getattr(owner, leaf))

    def write(self, system: ghostlight.OpticalSystem, value: float) -> None:
        owner, leaf = _resolve_attr(system.surfaces[self.surface_index], self.attr)
        setattr(owner, leaf, float(value))


def collect_variables(project) -> list[VariableRef]:
    """Return every surface attribute the project has flagged as variable.

    The flag map is stored on ``Project`` (keyed by surface UUID) rather
    than as attributes on ``ghostlight.Surface`` — that lets flags survive the
    system-clone round-trip the optimizer uses (surfaces get rebuilt
    fresh in the clone), and matches how ``ghost_solo`` / ``merit_functions``
    are held.

    Empty list means "no variables declared" — the optimizer short-circuits
    to a ``no_variables`` RunResult so the user is prompted to flag some
    attributes in the Optical Design Editor.
    """
    try:
        system = project.system
        surface_ids = list(system.surface_ids)
    except Exception:
        return []
    flag_map = project.all_variable_flags()
    if not flag_map:
        return []
    out: list[VariableRef] = []
    for si, uuid in enumerate(surface_ids):
        attrs = flag_map.get(uuid)
        if not attrs:
            continue
        for attr, bounds in attrs.items():
            lo = getattr(bounds, "lo", None)
            hi = getattr(bounds, "hi", None)
            out.append(VariableRef(
                surface_index=si,
                attr=str(attr),
                lo=None if lo is None else float(lo),
                hi=None if hi is None else float(hi),
            ))
    return out


# ---------------------------------------------------------------------------
# Bounds packing — used by the optimizer to hand variables to scipy
# ---------------------------------------------------------------------------


def pack_bounds(
    variables: list[VariableRef], system: ghostlight.OpticalSystem,
) -> tuple[list[float], list[float], list[float]]:
    """Return (x0, lb, ub) lists ready for scipy.optimize.least_squares.

    Unbounded sides become ``±math.inf`` — least_squares accepts that
    with the ``trf`` method. Any variable currently outside its declared
    bounds gets clamped into ``x0`` so least_squares doesn't refuse the
    initial guess; this is harmless because the variable still receives
    the optimizer's next proposed step.
    """
    x0: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    for v in variables:
        lo = -math.inf if v.lo is None else float(v.lo)
        hi = math.inf if v.hi is None else float(v.hi)
        if lo > hi:
            # Defensive — a malformed flag pair shouldn't crash the run.
            lo, hi = hi, lo
        cur = v.read(system)
        if cur < lo:
            cur = lo
        elif cur > hi:
            cur = hi
        x0.append(cur)
        lb.append(lo)
        ub.append(hi)
    return x0, lb, ub


def apply_variables(
    variables: list[VariableRef],
    system: ghostlight.OpticalSystem,
    x: list[float],
) -> None:
    """Write ``x`` back onto the system's surfaces.

    The caller is responsible for calling ``system.finalize()`` afterwards
    if any tunable affects z-chain layout (e.g. thickness changes).
    """
    if len(x) != len(variables):
        raise ValueError(
            f"apply_variables: got {len(x)} values for {len(variables)} variables"
        )
    for v, val in zip(variables, x):
        v.write(system, val)


# ---------------------------------------------------------------------------
# Curvature-space parameterization for radius variables
# ---------------------------------------------------------------------------
#
# Optimizing on radius R directly is painful because the R-space
# neighborhood of a "flat" surface has a singularity — going smoothly
# from R = +50 (weak convex) to R = -50 (weak concave) requires
# traversing R = ∞ / R = 0 (ghostlight's flat sentinel). scipy's
# finite-difference Jacobian sees this as a huge magnitude near flat and
# routinely drives the optimizer into hemispheres or across-zero flips.
#
# Curvature C = 1/R (with the C = 0 ↔ R = 0 identity preserved for the
# flat sentinel) is the natural parameter: going from C = +0.02 through
# C = 0 to C = -0.02 is a smooth line through the flat state. Real
# optical design tools (Zemax, CODE V) all optimize on curvature; we do
# the same but keep the storage on R so the UI stays radius-centric.
#
# ``|C| < 1.0`` (equivalently ``|R| > 1 mm``) is the default bound. Any
# tighter would be a hemisphere and the aperture penalty would gate it.


# ``|C|`` default upper bound — chosen so ``|R|`` stays above 1 mm.
CURVATURE_BOUND_ABS: float = 1.0


def radius_to_curvature(R: float) -> float:
    """R → C. Flat (R = 0) maps to C = 0 by ghostlight's convention."""
    if R == 0.0:
        return 0.0
    return 1.0 / R


def curvature_to_radius(C: float) -> float:
    """C → R. C = 0 maps to R = 0 (flat)."""
    if C == 0.0:
        return 0.0
    return 1.0 / C


def _pack_radius_curvature(
    v: VariableRef,
    R_current: float,
    *,
    preserve_sign: bool,
) -> tuple[float, float, float]:
    """Pack one radius variable into scipy's curvature space.

    Returns ``(C0, C_lo, C_hi)``. Bounds default to ``±CURVATURE_BOUND_ABS``;
    user-supplied R bounds are honoured by inverting them into C space
    (with sign handling — inverting a positive interval flips the
    interval endpoints in C). ``preserve_sign`` narrows the C interval
    to the same half-line as the current curvature, so scipy can flatten
    but not cross zero.
    """
    C0 = radius_to_curvature(R_current)
    C_lo = -CURVATURE_BOUND_ABS
    C_hi = CURVATURE_BOUND_ABS

    # Honour user R bounds when present. R bounds map to inverted C
    # bounds on their side of zero: a positive R interval [R_a, R_b] with
    # 0 < R_a < R_b becomes the C interval [1/R_b, 1/R_a]. Bounds that
    # straddle zero we ignore — the user can't meaningfully bound a
    # variable both across and through the flat state.
    v_lo = v.lo if v.lo is not None else None
    v_hi = v.hi if v.hi is not None else None
    if v_lo is not None and v_hi is not None and v_lo * v_hi > 0.0:
        # Same-sign R interval → invert cleanly.
        c_a = 1.0 / v_hi
        c_b = 1.0 / v_lo
        if c_a > c_b:
            c_a, c_b = c_b, c_a
        C_lo = max(C_lo, c_a)
        C_hi = min(C_hi, c_b)

    if preserve_sign and R_current != 0.0:
        if R_current > 0.0:
            # C > 0 half-line.
            C_lo = max(C_lo, 1e-6)
        else:
            C_hi = min(C_hi, -1e-6)

    # Clamp the initial value into the resolved bounds.
    if C0 < C_lo:
        C0 = C_lo
    elif C0 > C_hi:
        C0 = C_hi
    return C0, C_lo, C_hi


def pack_bounds_scipy(
    variables: list[VariableRef],
    system: ghostlight.OpticalSystem,
    *,
    preserve_radius_signs: bool = False,
) -> tuple[list[float], list[float], list[float], list[bool]]:
    """Pack variables into scipy's optimization space.

    Same shape as :func:`pack_bounds` but with an extra ``is_curvature``
    parallel list marking which entries scipy sees as curvature (needing
    the C→R conversion at apply time). Non-radius variables pass through
    untouched.
    """
    x0: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    is_curvature: list[bool] = []
    for v in variables:
        if v.attr == "radius":
            R_current = v.read(system)
            c0, clo, chi = _pack_radius_curvature(
                v, R_current, preserve_sign=preserve_radius_signs,
            )
            x0.append(c0)
            lb.append(clo)
            ub.append(chi)
            is_curvature.append(True)
            continue
        # Non-radius: standard R-space handling.
        lo = -math.inf if v.lo is None else float(v.lo)
        hi = math.inf if v.hi is None else float(v.hi)
        if lo > hi:
            lo, hi = hi, lo
        cur = v.read(system)
        if cur < lo:
            cur = lo
        elif cur > hi:
            cur = hi
        x0.append(cur)
        lb.append(lo)
        ub.append(hi)
        is_curvature.append(False)
    return x0, lb, ub, is_curvature


def apply_variables_scipy(
    variables: list[VariableRef],
    system: ghostlight.OpticalSystem,
    x: list[float],
    is_curvature: list[bool],
) -> None:
    """Write scipy-space ``x`` back onto the system.

    Curvature entries get inverted to radius before writing; non-radius
    entries pass through. Called from :meth:`_Worker._run_scipy` on
    every residuals evaluation.
    """
    if len(x) != len(variables) or len(is_curvature) != len(variables):
        raise ValueError(
            "apply_variables_scipy: length mismatch between x, variables, "
            "and is_curvature parallel lists"
        )
    for v, val, is_c in zip(variables, x, is_curvature):
        if is_c:
            v.write(system, curvature_to_radius(float(val)))
        else:
            v.write(system, float(val))


# ---------------------------------------------------------------------------
# Material-substitution flags — hammer path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialFlagRef:
    """One material-substitution target.

    ``element_id`` identifies the ``ghostlight.Element`` (stable across reloads
    via ``element.element_id``). ``material_index`` is the index into
    that element's ``material_glasses`` list. ``spec`` carries the
    vendor and optional nd/Vd search-region bounds — the hammer resolves
    ``spec`` to a concrete candidate list at each round via
    :func:`ghostlight_designer.material_substitution.candidates_within_spec`.
    ``current_key`` is the glass key that was present when the run
    started — restored if the hammer decides no candidate improves on
    the baseline for this flag.
    """
    element_id: str
    material_index: int
    spec: object              # SubstitutionSpec — kept object-typed to avoid Project import cycle
    current_key: str

    def surface_index(self, system: ghostlight.OpticalSystem) -> int:
        """Global-surface-index of the surface whose IOR this flag governs.

        Material ``j`` sits between surface ``j`` and surface ``j+1`` of
        the owning element; the IOR the tracer reads is on
        ``element.surface_ids[j]`` (see :func:`_refresh_surfaces_for_key`
        in row_schemas.py).
        """
        for el in system.elements:
            if getattr(el, "element_id", None) != self.element_id:
                continue
            if not (0 <= self.material_index < len(el.surface_ids)):
                return -1
            uuid = el.surface_ids[self.material_index]
            try:
                return list(system.surface_ids).index(uuid)
            except ValueError:
                return -1
        return -1


def collect_material_flags(project) -> list[MaterialFlagRef]:
    """Yield one :class:`MaterialFlagRef` per material-substitution flag.

    Empty list means "no substitutions declared" — the optimizer takes
    the standard scipy path. Any flag with an empty vendor is skipped
    (user opened the submenu but didn't finish picking); this is how
    ``SubstitutionSpec()`` — the default when a caller uses
    :meth:`Project.toggle_material_flag` without a ``default_spec`` —
    stays inert.
    """
    try:
        elements = list(project.system.elements)
    except Exception:
        return []
    flag_map = project.all_material_flags()
    if not flag_map:
        return []
    out: list[MaterialFlagRef] = []
    for el in elements:
        eid = getattr(el, "element_id", None)
        if not eid:
            continue
        mats = flag_map.get(str(eid))
        if not mats:
            continue
        glasses = list(el.material_glasses)
        for mi in sorted(mats.keys()):
            if not (0 <= mi < len(glasses)):
                continue
            spec = mats[mi]
            if not getattr(spec, "vendor", "") :
                # Empty-vendor spec — user hasn't finished picking; skip.
                continue
            out.append(MaterialFlagRef(
                element_id=str(eid),
                material_index=int(mi),
                spec=spec,
                current_key=str(glasses[mi]),
            ))
    return out


def install_candidate_glass(
    system: ghostlight.OpticalSystem,
    ref: MaterialFlagRef,
    glass,
) -> None:
    """Swap ``ref``'s material to ``glass`` on ``system``.

    Three writes must land, mirroring
    :func:`ghostlight_designer.optical_editor.row_schemas._refresh_surfaces_for_key`
    but without pulling the Qt-heavy row_schemas module:

    1. Insert the glass's dispersion entry into
       ``system._raw_glass_catalogue`` (if it isn't already there),
       so the C++ writer can round-trip the swap.
    2. Overwrite ``element.material_glasses[material_index]`` with the
       glass's catalogue key.
    3. Push the glass's ``nd`` / ``Vd`` onto the corresponding surface's
       ``ior`` / ``abbe_v`` / ``disp_model`` (Abbe model), so the tracer
       actually uses the new dispersion on the next ``finalize`` +
       ray trace.

    ``glass`` must be a :class:`CatalogueMaterial` with ``nd`` and ``vd``
    populated — :func:`candidates_for_vendor` already guarantees that.
    """
    catalogue_key = str(glass.key)
    nd = float(glass.nd)
    vd = float(glass.vd)

    # 1) project catalogue entry — ensure the C++ loader can find it
    #    on the next round-trip.
    raw_cat = getattr(system, "_raw_glass_catalogue", None)
    if raw_cat is not None and catalogue_key not in raw_cat:
        raw_cat[catalogue_key] = glass.lens_catalogue_entry()

    # 2) element.material_glasses[i] — locate the element.
    element = None
    for el in system.elements:
        if getattr(el, "element_id", None) == ref.element_id:
            element = el
            break
    if element is None:
        return
    glasses = element.material_glasses
    if not (0 <= ref.material_index < len(glasses)):
        return
    glasses[ref.material_index] = catalogue_key

    # 3) surface.ior / abbe_v / disp_model — locate the surface. Material
    #    j sits between surface j and surface j+1 of the element's
    #    surface list; surface[j]'s ior governs the medium to its right.
    if not (0 <= ref.material_index < len(element.surface_ids)):
        return
    surface_uuid = element.surface_ids[ref.material_index]
    try:
        global_idx = list(system.surface_ids).index(surface_uuid)
    except ValueError:
        return
    if not (0 <= global_idx < len(system.surfaces)):
        return
    surf = system.surfaces[global_idx]
    surf.ior = nd
    surf.abbe_v = vd
    surf.disp_model = ghostlight.DispersionModel.ABBE
