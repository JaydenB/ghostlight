"""Row schemas for the optical-editor tree.

Each row in the tree is one of a fixed set of ``NodeKind``s. A per-kind
``RowSchema`` declares the row's NAME-column label plus an ordered list of
``Slot``s, where each slot owns its getter, writer, editor type, and undo
label. The model and delegate dispatch by ``slot_at(node, column)`` rather
than an isinstance ladder.

Slot positions are canonicalized through ``CANONICAL_COLUMNS`` so the
global header strip aligns with the most common row type for each column
(Surface in practice). Slots whose key does not match a canonical column
fill in from the next free trailing column.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import ghostlight

from .nodes import (
    MaterialNode,
    NodeKind,
    SurfaceNode,
    TreeNode,
)


class SlotEditor(enum.Enum):
    TEXT = "text"
    FLOAT = "float"
    INT_SPINBOX = "int_spinbox"      # QSpinBox over an int range, options['min'/'max']
    ENUM_COMBO = "enum_combo"        # combo over an IntEnum, options['enum']
    STRING_COMBO = "string_combo"    # combo over a dynamic str list, options['choices']
    TEXT_PICKER = "text_picker"      # QLineEdit + persistent picker button
    READONLY = "readonly"
    # In-cell push button, no editor widget at all. The delegate paints
    # options['glyph'] and routes clicks to a handler registered by key in
    # options['button'] — used for view-state toggles (the Off Axis ">>>"
    # reveal) that must not create a document edit or an undo entry.
    BUTTON = "button"


# The asphere row schema carries this many trailing coefficient slots so the
# editor can populate up to A18. Read from the binding rather than hand-copied
# from optical_system.h, so the two cannot drift.
MAX_ASPHERE_TERMS = ghostlight.MAX_ASPHERE_TERMS


@dataclass
class SlotContext:
    """Bundle passed to every slot getter/writer.

    ``project`` is included so slots (material catalogue lookups,
    cross-element references) can reach beyond the raw ``OpticalSystem``
    without widening the schema API again.
    """

    node: TreeNode
    system: Any
    project: Any
    pos_z_mode: str = "absolute"


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a ``Slot.write`` call.

    ``requires_reset`` is True when the tree structure may change (e.g.
    a surface form switch adding/removing a child node), prompting the
    model to ``beginResetModel`` rather than emit a narrow ``dataChanged``.

    ``broadcast_column`` names a canonical column key whose cells across
    every row need refreshing — used today by Pos Z relative-mode writes
    that call ``system.finalize()`` and shift z on surfaces other than
    the one edited. Per-cell broadcast (rather than model reset) keeps
    an active value-scrubber drag alive.
    """

    changed: bool
    requires_reset: bool = False
    broadcast_column: str | None = None


_NOOP = WriteResult(False)


_UndoLabel = str | Callable[["SlotContext"], str]


def _always_editable(_ctx: "SlotContext") -> bool:
    return True


@dataclass(frozen=True)
class Slot:
    key: str
    label: str
    editor: SlotEditor
    get: Callable[["SlotContext"], Any]
    write: Callable[["SlotContext", Any], WriteResult]
    undo_label: _UndoLabel
    fmt: Optional[Callable[[Any], str]] = None
    display: Optional[Callable[["SlotContext"], str]] = None
    editable: Callable[["SlotContext"], bool] = _always_editable
    options: Mapping[str, Any] = field(default_factory=dict)

    def resolve_display(self, ctx: "SlotContext") -> str:
        if self.display is not None:
            return self.display(ctx)
        value = self.get(ctx)
        if value is None:
            return ""
        if self.fmt is not None:
            return self.fmt(value)
        return str(value)

    def resolve_undo_label(self, ctx: "SlotContext") -> str:
        if callable(self.undo_label):
            return self.undo_label(ctx)
        return self.undo_label

    def is_editable(self, ctx: "SlotContext") -> bool:
        return self.editable(ctx)


@dataclass(frozen=True)
class RowSchema:
    name_label: Callable[["SlotContext"], str]
    slots: tuple[Slot, ...] = ()
    # Optional "pack instead of pin" hook. When set, the row ignores
    # ``CANONICAL_COLUMNS`` and lays the slots this callable returns for a
    # given node into consecutive columns starting at ``_PACK_FIRST_COLUMN``.
    #
    # Rows whose live slot set varies with a discriminator want this: pinning
    # mutually-exclusive slots to canonical columns leaves a run of blank
    # cells between the ones that ARE live — a coating row would strand its
    # Preset picker three empty columns right of AR Layers. ``slots`` still
    # declares every slot the row can ever show — the callable picks the
    # subset this node actually renders.
    packed_slots: Optional[Callable[[TreeNode], tuple[Slot, ...]]] = None


CANONICAL_COLUMNS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (0, "Name", ()),
    (1, "Identifier",   ("identifier", "material_designer")),
    (2, "Radius",       ("radius", "material_name", "element_efl")),
    (3, "Pos Z",        ("pos_z", "material_nd")),
    (4, "Aperture Rad", ("aperture_rad", "material_vd")),
    # ---- Off-axis block ------------------------------------------------
    # Element placement beyond the optical axis. Column 5 holds the ">>>"
    # reveal toggle and is always visible; 6..13 are hidden by the view until
    # some element turns it on (see ``OFF_AXIS_COLUMN_KEYS`` and
    # ``body.OpticalEditorBody._apply_off_axis_visibility``).
    #
    # Headers use the short names optical designers expect at a layout table;
    # on disk the same values are position.x/y, rotation.tilt_x/
    # tilt_y/roll, and pivot.x/y/z. Rot X -> tilt_x, Rot Y -> tilt_y,
    # Rot Z -> roll.
    #
    # Rot Z looks inert on a spherical element — rolling a body of revolution
    # about its own axis changes nothing, optically or visually. It is a real
    # control on everything that isn't rotationally symmetric: cylindrical
    # surfaces, anamorphic stretch, and polygon apertures all turn with it.
    (5,  "Off Axis", ("off_axis",)),
    (6,  "Pos X",    ("pos_x",)),
    (7,  "Pos Y",    ("pos_y",)),
    (8,  "Rot X",    ("rot_x",)),
    (9,  "Rot Y",    ("rot_y",)),
    (10, "Rot Z",    ("rot_z",)),
    (11, "Pivot X",  ("pivot_x",)),
    (12, "Pivot Y",  ("pivot_y",)),
    (13, "Pivot Z",  ("pivot_z",)),
)

# Slot keys of the collapsible off-axis block, in column order. The toggle in
# column 5 is NOT part of this — it stays visible so there's always something
# to click.
OFF_AXIS_COLUMN_KEYS: tuple[str, ...] = (
    "pos_x", "pos_y", "rot_x", "rot_y", "rot_z", "pivot_x", "pivot_y", "pivot_z",
)

# Default pixel widths per canonical column. Lives here rather than in
# ``body.py`` so the column table and its presentation stay in one file — the
# view re-applies these whenever the header is (re)built, which matters because
# revealing a hidden column would otherwise leave it at Qt's default width.
COLUMN_WIDTHS: dict[int, int] = {
    0: 200,   # Name
    1: 160,   # Identifier
    2: 110,   # Radius
    3: 110,   # Pos Z
    4: 110,   # Aperture Rad
    5: 70,    # Off Axis (just the ">>>" button)
    6: 90, 7: 90,           # Pos X / Y
    8: 90, 9: 90, 10: 90,   # Rot X / Y / Z
    11: 90, 12: 90, 13: 90,  # Pivot X / Y / Z
}


def _canonical_column_for(slot_key: str) -> int | None:
    for col, _label, keys in CANONICAL_COLUMNS:
        if slot_key in keys:
            return col
    return None


# First column a packed row may occupy — column 0 is always Name.
_PACK_FIRST_COLUMN = 1


def _slot_columns(schema: RowSchema, node: TreeNode) -> dict[int, Slot]:
    """Resolve this schema's slots into ``{column_index: Slot}``."""
    if schema.packed_slots is not None:
        return {
            _PACK_FIRST_COLUMN + i: slot
            for i, slot in enumerate(schema.packed_slots(node))
        }
    by_col: dict[int, Slot] = {}
    used: set[int] = set()
    bespoke: list[Slot] = []
    for slot in schema.slots:
        col = _canonical_column_for(slot.key)
        if col is None or col in used:
            bespoke.append(slot)
            continue
        by_col[col] = slot
        used.add(col)
    next_col = len(CANONICAL_COLUMNS)
    for slot in bespoke:
        by_col[next_col] = slot
        next_col += 1
    return by_col


def _asphere_term_header(offset: int) -> str:
    """Asphere coefficient label for the ``offset``-th trailing column.

    Coefficients are even-power: offset 0 → A4 (r^4), 1 → A6 (r^6), ...
    """
    return f"A{2 * (offset + 2)}"


def header_text(column: int) -> str:
    """Column header text.

    Trailing asphere-coefficient columns return ``""``; their slots paint
    "A4" / "A6" / ... as an in-cell label only when active, so an
    inactive trailing column reads as empty space rather than a header
    promising a value that isn't there.
    """
    for col, label, _keys in CANONICAL_COLUMNS:
        if col == column:
            return label
    return ""


def canonical_column_for(slot_key: str) -> int | None:
    """Public lookup: column index for a known canonical slot key, or None."""
    return _canonical_column_for(slot_key)


def name_label(ctx: SlotContext) -> str:
    schema = SCHEMAS.get(ctx.node.kind)
    return "" if schema is None else schema.name_label(ctx)


def slot_at(node: TreeNode, column: int) -> Slot | None:
    schema = SCHEMAS.get(node.kind)
    if schema is None:
        return None
    return _slot_columns(schema, node).get(column)


def max_asphere_terms_in_system(system) -> int:
    """Largest ``n_asphere_terms`` across every asphere surface in ``system``.

    Drives the optical-editor's dynamic column count: trailing columns
    A4, A6, ... exist only while at least one asphere surface needs them.
    Returns 0 when ``system`` is None, has no surfaces, or has no asphere
    surfaces with terms populated.
    """
    if system is None:
        return 0
    surfaces = getattr(system, "surfaces", None)
    if not surfaces:
        return 0
    asphere_form = int(ghostlight.SurfaceForm.ASPHERE)
    max_n = 0
    for s in surfaces:
        if int(getattr(s, "form", 0)) != asphere_form:
            continue
        n = int(getattr(s, "n_asphere_terms", 0))
        if n > max_n:
            max_n = n
    return max_n


def column_count(system=None) -> int:
    """Total columns = canonical width + active trailing columns.

    Trailing columns carry the asphere coefficients (A4, A6, ... — live as
    soon as some asphere surface raises ``n_asphere_terms``). Falls back to
    ``MAX_ASPHERE_TERMS`` trailing columns when no system is supplied —
    covers tests + builders that want the full schema width.

    Floored at ``_MAX_PACKED_WIDTH`` so a packed row (see
    ``RowSchema.packed_slots``) can never be clipped by the canonical width.
    Today every packed row fits inside the canonical strip, so this floor
    only matters if one later grows another live slot.
    """
    canonical = len(CANONICAL_COLUMNS)
    if system is None:
        return max(canonical + MAX_ASPHERE_TERMS, _MAX_PACKED_WIDTH)
    trailing = max_asphere_terms_in_system(system)
    return max(canonical + trailing, _MAX_PACKED_WIDTH)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ELEMENT_KIND_LABELS = {
    "GLASS": "Lens",
    "STOP": "Stop",
}


def _element_type_label(element) -> str:
    return _ELEMENT_KIND_LABELS.get(element.kind.name, element.kind.name.title())


def _cyl_axis_name(axis_int: int) -> str:
    try:
        return ghostlight.CylinderAxis(int(axis_int)).name
    except (ValueError, TypeError):
        return f"Unknown({axis_int})"


_APERTURE_SHAPE_LABELS = {
    int(ghostlight.ApertureShape.CIRCLE):  "Circle",
    int(ghostlight.ApertureShape.POLYGON): "Polygon",
    int(ghostlight.ApertureShape.IMAGE):   "Image",
}


def _aperture_shape_label(shape_int: int) -> str:
    return _APERTURE_SHAPE_LABELS.get(int(shape_int), f"Unknown({shape_int})")


def _surface_of(node: TreeNode, system) -> Any:
    si = getattr(node, "surface_index", -1)
    if 0 <= si < len(system.surfaces):
        return system.surfaces[si]
    return None


def _fmt_float4(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------

def _element_get_identifier(ctx: SlotContext) -> str:
    return ctx.node.element.name


def _element_write_identifier(ctx: SlotContext, value: Any) -> WriteResult:
    el = ctx.node.element
    new = str(value)
    if new == el.name:
        return _NOOP
    el.name = new
    return WriteResult(True)


def _element_name_label(ctx: SlotContext) -> str:
    return _element_type_label(ctx.node.element)


def _element_efl_get(ctx: SlotContext) -> Optional[float]:
    """Paraxial Effective Focal Length of the element, in mm.

    Returns ``None`` for non-GLASS elements (aperture stops, dummies) and
    for elements whose surfaces don't resolve (mid-edit, structurally
    broken). Uses the standard refraction + translation ABCD product in
    reduced (y, n·u) coordinates so EFL = -1 / C in air-to-air.
    """
    el = ctx.node.element
    if el.kind != ghostlight.ElementKind.GLASS:
        return None
    system = ctx.system
    uuid_to_idx = {uuid: i for i, uuid in enumerate(system.surface_ids)}
    surfaces = []
    for uuid in el.surface_ids:
        idx = uuid_to_idx.get(uuid)
        if idx is None or not (0 <= idx < len(system.surfaces)):
            return None
        surfaces.append(system.surfaces[idx])
    if len(surfaces) < 2:
        return None
    # n[i] is the medium just to the left of surface i (n[0]=1 = air
    # before front, n[N]=1 = air after back). Each internal surface's
    # ``ior`` field carries the medium to its RIGHT — that's the same
    # value the C++ tracer reads, so we never drift from raytracer state.
    n = [1.0]
    for s in surfaces[:-1]:
        n.append(float(s.ior))
    n.append(1.0)

    A, B, C, D = 1.0, 0.0, 0.0, 1.0
    for i, surf in enumerate(surfaces):
        R = float(surf.radius)
        # Plane / flagged-infinity surfaces contribute zero power.
        if R == 0.0 or not math.isfinite(R):
            P = 0.0
        else:
            P = (n[i + 1] - n[i]) / R
        # Left-multiply by refraction matrix [[1, 0], [-P, 1]].
        C = -P * A + C
        D = -P * B + D
        # Translation to the next surface (skip after the back surface).
        if i < len(surfaces) - 1:
            t = float(surf.thickness) / n[i + 1]
            A = A + t * C
            B = B + t * D

    if C == 0.0 or not math.isfinite(C):
        return None
    return -1.0 / C


def _fmt_efl(value: Any) -> str:
    try:
        return f"{float(value):.3f} mm"
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Element off-axis placement — Pos X/Y, Rot X/Y/Z, Pivot X/Y/Z
#
# These edit the ``ghostlight.Element`` dataclass (``position``, ``rotation_euler_deg``,
# ``pivot``), which is what the writer serialises and therefore what survives
# save / undo. The C++ loader bakes that pose down into per-surface
# ``decenter_x/y`` / ``z`` / ``rot`` — but only at load time, so a write here
# would move nothing on screen until the next reload. ``ghostlight.bake_system_poses``
# re-runs that bake in Python against the live system.
#
# Why not just reload after each edit: ``OpticalSystem.reload()`` builds fresh
# Element instances, which invalidates the tree's node wrappers and forces a
# model reset — that kills an in-flight value-scrubber drag on the very first
# step. The Python bake keeps the drag alive; the authoritative C++ bake still
# runs on the next save / undo, and ``test_element_pose_bake.py`` holds the two
# to the same answer.
#
# Note there is no ``variable_attr`` on any of these. That option names an
# ``ghostlight.Surface`` attribute for the optimizer to tune; element pose isn't one,
# and setting it would paint a variable stripe on a cell the optimizer can't
# actually drive.
# ---------------------------------------------------------------------------

# (slot key) -> (Element attribute, tuple index)
_ELEMENT_POSE_FIELDS: dict[str, tuple[str, int]] = {
    "pos_x":   ("position", 0),
    "pos_y":   ("position", 1),
    "rot_x":   ("rotation_euler_deg", 0),
    "rot_y":   ("rotation_euler_deg", 1),
    "rot_z":   ("rotation_euler_deg", 2),
    "pivot_x": ("pivot", 0),
    "pivot_y": ("pivot", 1),
    "pivot_z": ("pivot", 2),
}


def element_pose_value(element, slot_key: str) -> float:
    """Read one off-axis component off an ``ghostlight.Element``.

    Shared with the model, which needs to know whether a row holds any
    non-zero pose value without going through a SlotContext.
    """
    attr, idx = _ELEMENT_POSE_FIELDS[slot_key]
    return float(getattr(element, attr, (0.0, 0.0, 0.0))[idx])


def element_has_off_axis_value(element) -> bool:
    """True when any off-axis component of ``element`` is non-zero.

    Drives the "auto-enable and lock open" rule: an element carrying real
    decenter or tilt must never be able to hide it.
    """
    return any(
        element_pose_value(element, key) != 0.0 for key in OFF_AXIS_COLUMN_KEYS
    )


def _element_pose_get_factory(slot_key: str):
    def _get(ctx: SlotContext) -> Optional[float]:
        element = getattr(ctx.node, "element", None)
        if element is None:
            return None
        return element_pose_value(element, slot_key)
    return _get


def _element_pose_write_factory(slot_key: str):
    attr, idx = _ELEMENT_POSE_FIELDS[slot_key]

    def _write(ctx: SlotContext, value: Any) -> WriteResult:
        element = getattr(ctx.node, "element", None)
        if element is None:
            return _NOOP
        try:
            new_value = float(value)
        except (TypeError, ValueError):
            return _NOOP
        current = tuple(getattr(element, attr, (0.0, 0.0, 0.0)))
        if new_value == float(current[idx]):
            return _NOOP
        updated = list(current)
        updated[idx] = new_value
        setattr(element, attr, tuple(updated))

        # Push the authored pose down onto the surfaces so the viewport and
        # any render reflect it immediately. A failure here means the system
        # is mid-edit / structurally broken; the authored value still stands
        # and the next load re-bakes it correctly.
        try:
            ghostlight.bake_system_poses(ctx.system)
        except Exception:  # pragma: no cover - defensive
            pass
        # Every surface can move (a tilt on element 1 shifts nothing downstream,
        # but a pivot or decenter re-poses that element's whole run), so the
        # Pos Z column is rebroadcast rather than reset — a model reset would
        # tear down an active scrubber drag.
        return WriteResult(True, broadcast_column="pos_z")

    return _write


def _element_pose_slots() -> tuple[Slot, ...]:
    """One FLOAT slot per off-axis component, in canonical column order."""
    out: list[Slot] = []
    for key in OFF_AXIS_COLUMN_KEYS:
        is_angle = key.startswith("rot_")
        label = dict(zip(OFF_AXIS_COLUMN_KEYS, (
            "Pos X", "Pos Y", "Rot X", "Rot Y", "Rot Z",
            "Pivot X", "Pivot Y", "Pivot Z",
        )))[key]
        out.append(Slot(
            key=key,
            label=label,
            editor=SlotEditor.FLOAT,
            get=_element_pose_get_factory(key),
            write=_element_pose_write_factory(key),
            # Undo labels name the user-facing column, not the on-disk field —
            # "Set Rot Y" is what the user just did; "Set tilt_y" isn't.
            undo_label=f"Set {label}",
            fmt=(lambda v: f"{float(v):.3f}°") if is_angle else _fmt_float4,
            options={
                "decimals": 3 if is_angle else 4,
                "scrubbable": True,
            },
        ))
    return tuple(out)


ELEMENT_SCHEMA = RowSchema(
    name_label=_element_name_label,
    slots=(
        Slot(
            key="identifier",
            label="Identifier",
            editor=SlotEditor.TEXT,
            get=_element_get_identifier,
            write=_element_write_identifier,
            undo_label="Rename Element",
        ),
        Slot(
            key="element_efl",
            label="EFL",
            editor=SlotEditor.READONLY,
            get=_element_efl_get,
            write=lambda _ctx, _value: _NOOP,
            undo_label="",
            fmt=_fmt_efl,
            editable=lambda _ctx: False,
        ),
        # The ">>>" reveal. Handled entirely by the delegate + model: clicking
        # it is a view-state change, not a document edit, so it must not go
        # through setData / project.edit and must not push an undo entry.
        Slot(
            key="off_axis",
            label="Off Axis",
            editor=SlotEditor.BUTTON,
            get=lambda _ctx: None,
            write=lambda _ctx, _value: _NOOP,
            undo_label="",
            editable=lambda _ctx: False,
            options={"button": "off_axis", "glyph": ">>>", "scrubbable": False},
        ),
        *_element_pose_slots(),
    ),
)


# ---------------------------------------------------------------------------
# Material — Designer / Name / nd / Vd
#
# Designer is derived from the current glass key's catalogue entry (no
# separate storage). The Custom designer sentinel marks materials whose
# nd/Vd are user-entered into the project's bundled glass_catalogue dict
# (``system._raw_glass_catalogue``); the tracer reads from that dict at
# every load, so Custom materials round-trip through .lens files.
#
# Editability rule:
#   * Vendor-catalogue glasses (MaterialCatalogue resolves them, directly
#     or via display-name fallback) — nd/Vd are READ-ONLY. Editing a
#     vendor glass would silently diverge from authoritative data.
#   * Project-local glasses (Custom_-prefix OR keys MaterialCatalogue
#     doesn't know) — nd/Vd are EDITABLE. Edits update both the project's
#     bundled catalogue dict AND every Surface that uses the glass.
# ---------------------------------------------------------------------------

import math
import secrets

CUSTOM_DESIGNER = "Custom"
_CUSTOM_KEY_PREFIX = "Custom_"

# Fraunhofer line wavelengths (μm) for Sellmeier evaluation
_LAMBDA_D_UM = 0.5875618
_LAMBDA_F_UM = 0.4861327
_LAMBDA_C_UM = 0.6562725

# Sensible default nd/Vd for a fresh Custom material — typical crown glass
# (matches Schott_N-BK7) so the surface IORs don't snap to a degenerate
# value on switch.
_DEFAULT_CUSTOM_ND = 1.5168
_DEFAULT_CUSTOM_VD = 64.17


def _material_glass_key(node: MaterialNode) -> str:
    if 0 <= node.material_index < len(node.element.material_glasses):
        return node.element.material_glasses[node.material_index]
    return ""


def _material_set_glass_key(node: MaterialNode, key: str) -> bool:
    if not (0 <= node.material_index < len(node.element.material_glasses)):
        return False
    if node.element.material_glasses[node.material_index] == key:
        return False
    node.element.material_glasses[node.material_index] = key
    return True


def _material_catalogue():
    # Lazy import so headless code paths (tests that don't load Qt
    # resources) don't pull the catalogue singleton unnecessarily.
    from ..material_catalogue import get_catalogue
    return get_catalogue()


def _material_catalogue_entry(ctx: SlotContext):
    """Resolve the current glass key to a ``CatalogueMaterial``, or None.

    Direct lookup first. Falls back to a display-name match across all
    vendors so legacy lens files (bare names like ``"N-BK7"``, pre-vendor
    re-link) still surface in the UI without requiring the user to run
    the catalogue re-link tool first. Schott wins on display-name ties
    since most legacy fixtures came from Schott data.
    """
    key = _material_glass_key(ctx.node)
    if not key or key.startswith(_CUSTOM_KEY_PREFIX):
        return None
    cat = _material_catalogue()
    direct = cat.by_key(key)
    if direct is not None:
        return direct
    matches = [m for m in cat.all() if m.display_name == key]
    if not matches:
        return None
    matches.sort(key=lambda m: (0 if m.source_vendor == "Schott" else 1,
                                  m.source_vendor))
    return matches[0]


def _material_vendors() -> list[str]:
    """Distinct ``source_vendor``s in the catalogue, sorted, plus the
    ``CUSTOM_DESIGNER`` sentinel that opens nd/Vd direct entry."""
    seen = {m.source_vendor for m in _material_catalogue().all() if m.source_vendor}
    return sorted(seen) + [CUSTOM_DESIGNER]


# -- Dispersion helpers --

def _sellmeier_n(disp: dict, lambda_um: float) -> Optional[float]:
    try:
        B = list(disp["B"])
        C = list(disp["C"])
    except (KeyError, TypeError):
        return None
    if not B or len(B) != len(C):
        return None
    lam2 = lambda_um * lambda_um
    s = 0.0
    for b, c in zip(B, C):
        denom = lam2 - float(c)
        if denom == 0.0:
            return None
        s += float(b) * lam2 / denom
    n2 = 1.0 + s
    if n2 <= 0.0:
        return None
    return math.sqrt(n2)


def _dispersion_nd_vd(disp: Any) -> tuple[Optional[float], Optional[float]]:
    """Read (nd, Vd) from a dispersion dict (Abbe direct or Sellmeier-computed)."""
    if not isinstance(disp, dict):
        return (None, None)
    model = str(disp.get("model", "")).lower()
    if model == "abbe":
        nd_raw = disp.get("nd")
        # On-disk files have used both "Vd" (canonical) and "vd" (legacy);
        # accept either rather than show empty for the wrong-case spelling.
        vd_raw = disp.get("Vd", disp.get("vd"))
        try:
            return (
                float(nd_raw) if nd_raw is not None else None,
                float(vd_raw) if vd_raw is not None else None,
            )
        except (TypeError, ValueError):
            return (None, None)
    if model == "sellmeier":
        nd = _sellmeier_n(disp, _LAMBDA_D_UM)
        nF = _sellmeier_n(disp, _LAMBDA_F_UM)
        nC = _sellmeier_n(disp, _LAMBDA_C_UM)
        if nd is None or nF is None or nC is None:
            return (None, None)
        if nF == nC:
            return (nd, None)
        return (nd, (nd - 1.0) / (nF - nC))
    return (None, None)


def _system_glass_entry(system, key: str) -> Optional[dict]:
    if not key:
        return None
    return getattr(system, "_raw_glass_catalogue", {}).get(key)


def _current_nd_vd(ctx: SlotContext) -> tuple[Optional[float], Optional[float]]:
    """Resolve (nd, Vd) for the current material.

    MaterialCatalogue first (its ``ui.nd``/``ui.vd`` are already curated
    per glass). Falls back to evaluating the project's bundled
    ``glass_catalogue`` dispersion so legacy / custom / unknown glasses
    still show useful values in the UI.
    """
    entry = _material_catalogue_entry(ctx)
    if entry is not None and entry.nd is not None and entry.vd is not None:
        return (float(entry.nd), float(entry.vd))
    key = _material_glass_key(ctx.node)
    sys_entry = _system_glass_entry(ctx.system, key)
    if sys_entry is None:
        return (None, None)
    return _dispersion_nd_vd(sys_entry.get("dispersion"))


def _is_dispersion_editable(ctx: SlotContext) -> bool:
    """Project-local materials accept inline nd/Vd edits — that's
    anything MaterialCatalogue doesn't resolve, whether Custom-prefix
    keys or legacy / unmatched names."""
    if not _material_glass_key(ctx.node):
        return False
    return _material_catalogue_entry(ctx) is None


# -- Custom-material persistence --

def _generate_custom_key() -> str:
    return _CUSTOM_KEY_PREFIX + secrets.token_hex(4)


def _ensure_abbe_entry(system, key: str, nd: float, vd: float) -> None:
    """Insert or overwrite an Abbe-mode glass_catalogue entry."""
    catalogue = getattr(system, "_raw_glass_catalogue", None)
    if catalogue is None:
        return
    entry = catalogue.get(key) or {}
    entry["name"] = entry.get("name") or key
    entry["dispersion"] = {
        "model": "abbe",
        "nd": float(nd),
        "Vd": float(vd),
    }
    catalogue[key] = entry


def _refresh_surfaces_for_key(system, key: str, nd: float, vd: float) -> None:
    """Push new nd/Vd onto every Surface whose owning element references
    ``key`` at the matching material index. Material ``j`` in an element
    sits between surface ``j`` and surface ``j+1``, so ``surface[j].ior``
    is the medium just to the right of surface ``j`` — that's the slot
    the material's nd governs."""
    elements = getattr(system, "_elements", None)
    if not elements:
        return
    surface_ids = list(system.surface_ids)
    id_to_index = {uuid: i for i, uuid in enumerate(surface_ids)}
    for el in elements:
        glasses = el.material_glasses or []
        for mat_idx, glass in enumerate(glasses):
            if glass != key or mat_idx >= len(el.surface_ids):
                continue
            global_idx = id_to_index.get(el.surface_ids[mat_idx])
            if global_idx is None or not (0 <= global_idx < len(system.surfaces)):
                continue
            surf = system.surfaces[global_idx]
            surf.ior = float(nd)
            surf.abbe_v = float(vd)
            surf.disp_model = ghostlight.DispersionModel.ABBE


# -- Designer slot --

def _designer_get(ctx: SlotContext) -> str:
    """Vendor name when MaterialCatalogue resolves the glass; otherwise
    ``"Custom"`` — covers both ``Custom_<hash>`` keys generated by the
    Designer combo AND legacy lens files whose glass names aren't in any
    bundled vendor catalogue. Either way the row is project-local and
    nd/Vd are user-editable."""
    entry = _material_catalogue_entry(ctx)
    return entry.source_vendor if entry is not None else CUSTOM_DESIGNER


def _designer_write(ctx: SlotContext, value: Any) -> WriteResult:
    node: MaterialNode = ctx.node  # type: ignore[assignment]
    new_vendor = str(value)
    if new_vendor == _designer_get(ctx):
        return _NOOP
    if new_vendor == CUSTOM_DESIGNER:
        # Seed the Custom entry from the current nd/Vd so the user doesn't
        # see the surface IORs jump on switch. Defaults to a crown glass
        # when the current material doesn't resolve to anything.
        current_nd, current_vd = _current_nd_vd(ctx)
        nd = current_nd if current_nd is not None else _DEFAULT_CUSTOM_ND
        vd = current_vd if current_vd is not None else _DEFAULT_CUSTOM_VD
        key = _generate_custom_key()
        _ensure_abbe_entry(ctx.system, key, nd, vd)
        if not _material_set_glass_key(node, key):
            return _NOOP
        _refresh_surfaces_for_key(ctx.system, key, nd, vd)
        return WriteResult(True)
    # Vendor switch — snap to the first glass of the new vendor so the
    # row never sits in a wrong-vendor / wrong-name limbo. Refresh surface
    # IORs with the new vendor's nd/Vd.
    catalogue = _material_catalogue()
    matches = catalogue.search(vendor=new_vendor)
    if not matches:
        return _NOOP
    chosen = matches[0]
    if not _material_set_glass_key(node, chosen.key):
        return _NOOP
    # Mirror the existing element_actions._ensure_glass_in_catalogue
    # pattern: any glass written into a material slot must also live in
    # the project's bundled catalogue, or the C++ loader will reject the
    # next round-trip through the undo snapshot.
    _ensure_glass_in_project_catalogue(ctx.system, chosen)
    if chosen.nd is not None and chosen.vd is not None:
        _refresh_surfaces_for_key(
            ctx.system, chosen.key, float(chosen.nd), float(chosen.vd)
        )
    return WriteResult(True)


def _ensure_glass_in_project_catalogue(system, cm) -> None:
    catalogue = getattr(system, "_raw_glass_catalogue", None)
    if catalogue is None or cm.key in catalogue:
        return
    catalogue[cm.key] = cm.lens_catalogue_entry()


# -- Name slot (picker) --

def _name_get(ctx: SlotContext) -> str:
    entry = _material_catalogue_entry(ctx)
    return entry.display_name if entry is not None else ""


def _name_write(ctx: SlotContext, value: Any) -> WriteResult:
    # The picker dialog writes the catalogue *key* (e.g. ``Schott_N-BK7``)
    # — not the display name. Free-text writes through the QLineEdit also
    # arrive here; we treat the input as a catalogue key.
    node: MaterialNode = ctx.node  # type: ignore[assignment]
    new = str(value)
    if not _material_set_glass_key(node, new):
        return _NOOP
    # If the new key resolves to a vendor catalogue glass, also bundle
    # its dispersion into the project and refresh surface IORs — the
    # tracer reads from per-surface ior/abbe_v, not from the catalogue
    # name, so the picker selection needs to land everywhere.
    cm = _material_catalogue().by_key(new)
    if cm is not None:
        _ensure_glass_in_project_catalogue(ctx.system, cm)
        if cm.nd is not None and cm.vd is not None:
            _refresh_surfaces_for_key(
                ctx.system, new, float(cm.nd), float(cm.vd)
            )
    return WriteResult(True)


def _name_editable(ctx: SlotContext) -> bool:
    # Name picker is only meaningful when a vendor catalogue resolves
    # the material — that's the same condition as "not Custom".
    return _material_catalogue_entry(ctx) is not None


# -- nd / Vd slots --

def _nd_get(ctx: SlotContext) -> Optional[float]:
    # nd/Vd are only surfaced when the user can act on them — i.e. for
    # Custom + project-local glasses. Vendor-catalogue materials hide
    # these cells entirely: showing read-only canonical numbers next to
    # editable rows reads as visual noise and invites misclicks.
    if not _is_dispersion_editable(ctx):
        return None
    nd, _vd = _current_nd_vd(ctx)
    return nd


def _vd_get(ctx: SlotContext) -> Optional[float]:
    if not _is_dispersion_editable(ctx):
        return None
    _nd, vd = _current_nd_vd(ctx)
    return vd


def _nd_write(ctx: SlotContext, value: Any) -> WriteResult:
    if not _is_dispersion_editable(ctx):
        return _NOOP
    key = _material_glass_key(ctx.node)
    if not key:
        return _NOOP
    try:
        new_nd = float(value)
    except (TypeError, ValueError):
        return _NOOP
    current_nd, current_vd = _current_nd_vd(ctx)
    if current_nd is not None and new_nd == current_nd:
        return _NOOP
    vd = current_vd if current_vd is not None else _DEFAULT_CUSTOM_VD
    _ensure_abbe_entry(ctx.system, key, new_nd, vd)
    _refresh_surfaces_for_key(ctx.system, key, new_nd, vd)
    return WriteResult(True)


def _vd_write(ctx: SlotContext, value: Any) -> WriteResult:
    if not _is_dispersion_editable(ctx):
        return _NOOP
    key = _material_glass_key(ctx.node)
    if not key:
        return _NOOP
    try:
        new_vd = float(value)
    except (TypeError, ValueError):
        return _NOOP
    current_nd, current_vd = _current_nd_vd(ctx)
    if current_vd is not None and new_vd == current_vd:
        return _NOOP
    nd = current_nd if current_nd is not None else _DEFAULT_CUSTOM_ND
    _ensure_abbe_entry(ctx.system, key, nd, new_vd)
    _refresh_surfaces_for_key(ctx.system, key, nd, new_vd)
    return WriteResult(True)


MATERIAL_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Material",
    slots=(
        Slot(
            key="material_designer",
            label="Designer",
            editor=SlotEditor.STRING_COMBO,
            get=_designer_get,
            write=_designer_write,
            undo_label="Set Glass Designer",
            options={"choices": _material_vendors},
        ),
        Slot(
            key="material_name",
            label="Name",
            editor=SlotEditor.TEXT_PICKER,
            get=_name_get,
            write=_name_write,
            undo_label="Set Glass",
            editable=_name_editable,
            options={"picker": "material_glass"},
        ),
        Slot(
            key="material_nd",
            label="nd",
            editor=SlotEditor.FLOAT,
            get=_nd_get,
            write=_nd_write,
            undo_label="Set nd",
            fmt=lambda v: f"{float(v):.5f}",
            editable=_is_dispersion_editable,
            options={"decimals": 5, "scrubbable": True},
        ),
        Slot(
            key="material_vd",
            label="Vd",
            editor=SlotEditor.FLOAT,
            get=_vd_get,
            write=_vd_write,
            undo_label="Set Vd",
            fmt=lambda v: f"{float(v):.2f}",
            editable=_is_dispersion_editable,
            options={"decimals": 2, "scrubbable": True},
        ),
    ),
)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

def _surface_get_radius(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None or int(surf.form) != int(ghostlight.SurfaceForm.SPHERE):
        return None
    if surf.is_stop:
        # Aperture-stop surfaces have a nominal radius value but it's
        # geometrically meaningless. Hide it so the user doesn't think
        # they should tune it. Pos Z and Aperture Rad stay populated —
        # those *are* meaningful for the stop.
        return None
    return float(surf.radius)


def _surface_radius_editable(ctx: SlotContext) -> bool:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return False
    return not surf.is_stop


def _surface_write_radius(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None or int(surf.form) != int(ghostlight.SurfaceForm.SPHERE):
        return _NOOP
    if surf.is_stop:
        return _NOOP
    try:
        new_r = float(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_r == surf.radius:
        return _NOOP
    surf.radius = new_r
    return WriteResult(True)


def _surface_get_pos_z(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return None
    if ctx.pos_z_mode == "relative":
        return float(surf.thickness)
    return float(surf.z)


def _surface_display_pos_z(ctx: SlotContext) -> str:
    value = _surface_get_pos_z(ctx)
    if value is None:
        return ""
    if ctx.pos_z_mode == "relative":
        return f"{value:.4f} (rel)"
    return f"{value:.4f}"


def _surface_write_pos_z(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_value = float(value)
    except (TypeError, ValueError):
        return _NOOP
    if ctx.pos_z_mode == "relative":
        if new_value == surf.thickness:
            return _NOOP
        surf.thickness = new_value
        # surface.z is what the raytracer + viewport read; rederive the
        # whole chain from thicknesses so the edit is visible. Other
        # surface rows' z values shift too — flag the model to broadcast
        # dataChanged for the whole Pos Z column.
        ctx.system.finalize()
        return WriteResult(True, broadcast_column="pos_z")
    if new_value == surf.z:
        return _NOOP
    surf.z = new_value
    return WriteResult(True)


def _surface_pos_z_undo(ctx: SlotContext) -> str:
    return "Set Thickness" if ctx.pos_z_mode == "relative" else "Set Pos Z"


def _surface_get_aperture(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else float(surf.semi_aperture)


def _surface_write_aperture(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_a = float(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_a == surf.semi_aperture:
        return _NOOP
    surf.semi_aperture = new_a
    return WriteResult(True)


def _surface_name_label(ctx: SlotContext) -> str:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        node: SurfaceNode = ctx.node  # type: ignore[assignment]
        return f"<missing surface {node.surface_uuid[:8]}>"
    return "Surface"


SURFACE_SCHEMA = RowSchema(
    name_label=_surface_name_label,
    slots=(
        Slot(
            key="radius",
            label="Radius",
            editor=SlotEditor.FLOAT,
            get=_surface_get_radius,
            write=_surface_write_radius,
            undo_label="Set Radius",
            fmt=_fmt_float4,
            editable=_surface_radius_editable,
            # ``variable_attr`` marks this cell as flag-able as an
            # optimizer design variable — the value is the ``ghostlight.Surface``
            # attribute name the optimizer will tune. Read by the delegate
            # to paint the amber left-stripe on flagged cells and by the
            # right-click menu to Toggle Variable / Edit Bounds.
            options={"decimals": 4, "scrubbable": True, "variable_attr": "radius"},
        ),
        Slot(
            key="pos_z",
            label="Pos Z",
            editor=SlotEditor.FLOAT,
            get=_surface_get_pos_z,
            write=_surface_write_pos_z,
            undo_label=_surface_pos_z_undo,
            display=_surface_display_pos_z,
            # Relative-mode writes call ``system.finalize()`` which shifts
            # z on every other surface; the model rebroadcasts this column.
            # ``variable_attr = "thickness"`` regardless of the cell's
            # display mode — physically the optimizer must vary thickness
            # (surface spacing) so ``finalize()`` can rebuild the z chain;
            # varying absolute z on one surface would break the layout.
            options={
                "decimals": 4,
                "scrubbable": True,
                "broadcast_column": "pos_z",
                "variable_attr": "thickness",
            },
        ),
        Slot(
            key="aperture_rad",
            label="Aperture Rad",
            editor=SlotEditor.FLOAT,
            get=_surface_get_aperture,
            write=_surface_write_aperture,
            undo_label="Set Aperture Rad",
            fmt=_fmt_float4,
            options={"decimals": 4, "scrubbable": True},
        ),
    ),
)


# ---------------------------------------------------------------------------
# Asphere Form
# ---------------------------------------------------------------------------

def _form_get_radius(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else float(surf.radius)


def _form_write_radius(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_r = float(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_r == surf.radius:
        return _NOOP
    surf.radius = new_r
    return WriteResult(True)


def _asphere_get_n_terms(ctx: SlotContext) -> Optional[int]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else int(surf.n_asphere_terms)


def _asphere_write_n_terms(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_n = int(value)
    except (TypeError, ValueError):
        return _NOOP
    new_n = max(0, min(MAX_ASPHERE_TERMS, new_n))
    old_n = int(surf.n_asphere_terms)
    if new_n == old_n:
        return _NOOP
    # Rewrite the full active range. Newly-activated slots default to zero
    # so the user starts from a known state; trailing values beyond the
    # new count stay in the C++ array but are no longer exposed. The
    # binding's array setter sets n_asphere_terms from arr.size().
    current = list(surf.asphere_terms)
    new_terms = (current + [0.0] * MAX_ASPHERE_TERMS)[:new_n]
    surf.asphere_terms = new_terms
    # Tree column count is derived from max n_asphere_terms across the
    # system, so a count change can shift the tree's column geometry.
    # Request a model reset so the view re-queries columnCount + headers.
    return WriteResult(True, requires_reset=True)


def _asphere_get_conic(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else float(surf.conic_k)


def _asphere_write_conic(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_k = float(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_k == float(surf.conic_k):
        return _NOOP
    surf.conic_k = new_k
    return WriteResult(True)


def _asphere_term_get_factory(term_index: int):
    def _get(ctx: SlotContext) -> Optional[float]:
        surf = _surface_of(ctx.node, ctx.system)
        if surf is None:
            return None
        if term_index >= int(surf.n_asphere_terms):
            return None
        try:
            return float(surf.asphere_terms[term_index])
        except (IndexError, TypeError):
            return None
    return _get


def _asphere_term_write_factory(term_index: int):
    def _write(ctx: SlotContext, value: Any) -> WriteResult:
        surf = _surface_of(ctx.node, ctx.system)
        if surf is None:
            return _NOOP
        if term_index >= int(surf.n_asphere_terms):
            return _NOOP
        try:
            new_v = float(value)
        except (TypeError, ValueError):
            return _NOOP
        current = list(surf.asphere_terms)
        if term_index >= len(current):
            return _NOOP
        if current[term_index] == new_v:
            return _NOOP
        current[term_index] = new_v
        surf.asphere_terms = current
        return WriteResult(True)
    return _write


def _asphere_term_editable_factory(term_index: int):
    def _editable(ctx: SlotContext) -> bool:
        surf = _surface_of(ctx.node, ctx.system)
        if surf is None:
            return False
        return term_index < int(surf.n_asphere_terms)
    return _editable


def _asphere_term_slots() -> tuple[Slot, ...]:
    """One float slot per potential asphere coefficient.

    All ``MAX_ASPHERE_TERMS`` slots are declared so a row may surface any
    coefficient that ``n_asphere_terms`` activates. Per-slot ``get`` /
    ``editable`` gating hides + locks the cells whose index is beyond the
    current count, so the row reads as having "only the active ones"
    even though every slot exists structurally.
    """
    slots: list[Slot] = []
    for i in range(MAX_ASPHERE_TERMS):
        label = _asphere_term_header(i)
        slots.append(
            Slot(
                key=f"asphere_{label.lower()}",
                label=label,
                editor=SlotEditor.FLOAT,
                get=_asphere_term_get_factory(i),
                write=_asphere_term_write_factory(i),
                undo_label=f"Set {label}",
                fmt=lambda v: f"{float(v):.6g}",
                editable=_asphere_term_editable_factory(i),
                options={"decimals": 8, "scrubbable": True},
            )
        )
    return tuple(slots)


ASPHERE_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Asphere Form",
    slots=(
        Slot(
            key="identifier",
            label="Terms",
            editor=SlotEditor.INT_SPINBOX,
            get=_asphere_get_n_terms,
            write=_asphere_write_n_terms,
            undo_label="Set Asphere Term Count",
            fmt=lambda v: f"{int(v)}",
            # Term count drives a model reset (column count is derived
            # from it); scrubbing would tear down the active scrubber on
            # every step. Click the spinbox arrows instead.
            options={"min": 0, "max": MAX_ASPHERE_TERMS, "scrubbable": False},
        ),
        Slot(
            key="radius",
            label="Radius",
            editor=SlotEditor.FLOAT,
            get=_form_get_radius,
            write=_form_write_radius,
            undo_label="Set Radius",
            fmt=_fmt_float4,
            # Asphere form radius is the same physical ``ghostlight.Surface.radius``
            # the top-level Surface row shows; flag as variable via the
            # same attr name so the optimizer sees one variable per surface
            # regardless of which row the user clicked.
            options={"decimals": 4, "scrubbable": True, "variable_attr": "radius"},
        ),
        Slot(
            # Re-use the canonical Pos Z column for the conic constant K
            # — the in-cell label paints "K" since slot.label differs
            # from the strip header. Surface-row Pos Z broadcasts walk
            # element children only (not surface grandchildren), so the
            # K cell here doesn't collide with that broadcast.
            key="pos_z",
            label="K",
            editor=SlotEditor.FLOAT,
            get=_asphere_get_conic,
            write=_asphere_write_conic,
            undo_label="Set Conic Constant",
            fmt=lambda v: f"{float(v):.6g}",
            # Deliberately NO variable_attr — the optimizer's variable set
            # covers radius and thickness only, so a flagged conic constant
            # would never be driven.
            options={"decimals": 8, "scrubbable": True},
        ),
        *_asphere_term_slots(),
    ),
)


# ---------------------------------------------------------------------------
# Cylindrical Form
# ---------------------------------------------------------------------------

def _cyl_get_axis(ctx: SlotContext) -> Optional[int]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else int(surf.cyl_axis)


def _cyl_write_axis(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_axis = int(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_axis == int(surf.cyl_axis):
        return _NOOP
    surf.cyl_axis = new_axis
    return WriteResult(True)


CYLINDRICAL_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Cylindrical Form",
    slots=(
        Slot(
            key="identifier",
            label="Axis",
            editor=SlotEditor.ENUM_COMBO,
            get=_cyl_get_axis,
            write=_cyl_write_axis,
            undo_label="Set Cylinder Axis",
            fmt=lambda v: _cyl_axis_name(int(v)),
            options={"enum": ghostlight.CylinderAxis},
        ),
        Slot(
            key="radius",
            label="Radius",
            editor=SlotEditor.FLOAT,
            get=_form_get_radius,
            write=_form_write_radius,
            undo_label="Set Radius",
            fmt=_fmt_float4,
            options={"decimals": 4, "scrubbable": True, "variable_attr": "radius"},
        ),
    ),
)


# ---------------------------------------------------------------------------
# Aperture Form
#
# Child row attached to every aperture-stop surface. The C++ Surface struct
# carries aperture_shape (CIRCLE / POLYGON / IMAGE) plus shape-specific
# scalars (blades, rotation, aspect, semi_diameter). The row exposes them
# via the canonical column strip:
#
#   * Identifier  → shape combo (discriminator)
#   * Radius      → aperture_aspect (always editable — bounding ellipse X/Y)
#   * Pos Z       → aperture_blades (POLYGON-only)
#   * Aperture Rad→ aperture_rotation_deg (POLYGON-only; stored as radians
#                                          on the surface, edited as deg)
#
# CIRCLE / IMAGE leave the blade + rotation cells blank — get() returns
# None and editable() returns False, so the delegate paints nothing.
# A shape switch can flip cell editability, so writes return
# requires_reset=True to refresh flags + display across the row.
# ---------------------------------------------------------------------------

# Blade-count range for the in-cell spinbox. Floor at 3 because the
# geometry tessellator silently falls back to an ellipse below that
# (a "2-sided polygon" isn't a closed shape); ceiling at 32 because real
# iris diaphragms top out far below — keeps the value-scrubber's largest
# sensitivity (100) from instantly running off the end.
_MIN_APERTURE_BLADES = 3
_MAX_APERTURE_BLADES = 32

# Default blade count seeded when the user first flips a stop to POLYGON
# with `aperture_blades` still at the C++ default of 0. Six matches the
# most common iris geometry and lands the user in a state that renders.
_DEFAULT_POLYGON_BLADES = 6

# Lazy import deferred — math is already imported above for the material
# Sellmeier helpers, so degrees-to-radians conversions reuse it.
_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi


def _aperture_shape_of(ctx: SlotContext) -> Optional[int]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else int(surf.aperture_shape)


def _aperture_get_shape(ctx: SlotContext) -> Optional[int]:
    return _aperture_shape_of(ctx)


def _aperture_write_shape(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_shape = int(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_shape == int(surf.aperture_shape):
        return _NOOP
    surf.aperture_shape = new_shape
    # POLYGON requires blades >= 3 to render — otherwise the tessellator
    # silently falls back to an ellipse and the user sees a circle even
    # though the discriminator says polygon. Seed a sensible default the
    # first time the user switches into polygon mode; leave any existing
    # in-range value alone so re-toggling away and back doesn't clobber
    # the user's choice.
    if (
        new_shape == int(ghostlight.ApertureShape.POLYGON)
        and int(surf.aperture_blades) < _MIN_APERTURE_BLADES
    ):
        surf.aperture_blades = _DEFAULT_POLYGON_BLADES
    # Polygon-only cells (blades / rotation) become editable or not based on
    # the new shape; tear down + rebuild the row so flags and display refresh
    # in lockstep with the discriminator.
    return WriteResult(True, requires_reset=True)


def _aperture_get_aspect(ctx: SlotContext) -> Optional[float]:
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else float(surf.aperture_aspect)


def _aperture_write_aspect(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_aspect = float(value)
    except (TypeError, ValueError):
        return _NOOP
    # aspect = 0 collapses the bounding ellipse to a line; the geometry
    # tessellator also guards against this via ``or 1.0``. Treat as no-op
    # rather than letting the surface silently switch to "circle" math.
    if new_aspect == 0.0:
        return _NOOP
    if new_aspect == float(surf.aperture_aspect):
        return _NOOP
    surf.aperture_aspect = new_aspect
    return WriteResult(True)


def _is_polygon_shape(ctx: SlotContext) -> bool:
    shape = _aperture_shape_of(ctx)
    return shape is not None and shape == int(ghostlight.ApertureShape.POLYGON)


def _aperture_get_blades(ctx: SlotContext) -> Optional[int]:
    if not _is_polygon_shape(ctx):
        return None
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else int(surf.aperture_blades)


def _aperture_write_blades(ctx: SlotContext, value: Any) -> WriteResult:
    if not _is_polygon_shape(ctx):
        return _NOOP
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_blades = int(value)
    except (TypeError, ValueError):
        return _NOOP
    # Polygons need at least 3 blades to form a closed shape; the
    # tessellator silently falls back to an ellipse below that. Clamp
    # here so the model state matches what the user sees.
    new_blades = max(_MIN_APERTURE_BLADES, min(_MAX_APERTURE_BLADES, new_blades))
    if new_blades == int(surf.aperture_blades):
        return _NOOP
    surf.aperture_blades = new_blades
    return WriteResult(True)


def _aperture_get_rotation_deg(ctx: SlotContext) -> Optional[float]:
    if not _is_polygon_shape(ctx):
        return None
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return None
    return float(surf.aperture_rotation_rad) * _RAD_TO_DEG


def _aperture_write_rotation_deg(ctx: SlotContext, value: Any) -> WriteResult:
    if not _is_polygon_shape(ctx):
        return _NOOP
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_deg = float(value)
    except (TypeError, ValueError):
        return _NOOP
    new_rad = new_deg * _DEG_TO_RAD
    if new_rad == float(surf.aperture_rotation_rad):
        return _NOOP
    surf.aperture_rotation_rad = new_rad
    return WriteResult(True)


def _aperture_polygon_editable(ctx: SlotContext) -> bool:
    return _is_polygon_shape(ctx)


APERTURE_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Aperture Form",
    slots=(
        Slot(
            key="identifier",
            label="Shape",
            editor=SlotEditor.ENUM_COMBO,
            get=_aperture_get_shape,
            write=_aperture_write_shape,
            undo_label="Set Aperture Shape",
            fmt=lambda v: _aperture_shape_label(int(v)),
            # Combo populates from ghostlight.ApertureShape enum members. Shape
            # changes flip column editability, so the write returns
            # requires_reset=True and tears down any active scrubber.
            # IMAGE is excluded from the dropdown: the writer's image-
            # aperture round-trip is still open (no image_path on the
            # Surface struct itself) so picking it from the UI would
            # leave the user in a state that doesn't save back to disk.
            # The fmt callback still resolves IMAGE for any legacy data
            # loaded with that shape already set.
            options={
                "enum": ghostlight.ApertureShape,
                "exclude": (int(ghostlight.ApertureShape.IMAGE),),
                "scrubbable": False,
            },
        ),
        Slot(
            key="radius",
            label="Aspect",
            editor=SlotEditor.FLOAT,
            get=_aperture_get_aspect,
            write=_aperture_write_aspect,
            undo_label="Set Aperture Aspect",
            fmt=lambda v: f"{float(v):.4f}",
            options={"decimals": 4, "scrubbable": True},
        ),
        Slot(
            # Reuse the canonical Pos Z column for the polygon blade count.
            # In-cell label paints "Blades" since slot.label differs from
            # the strip header. CIRCLE / IMAGE shapes return None from get,
            # so the cell renders blank and is uneditable — no surface
            # broadcast collision with the surface row's Pos Z (the parent
            # row's broadcast walks element children only, not surface
            # grandchildren).
            key="pos_z",
            label="Blades",
            editor=SlotEditor.INT_SPINBOX,
            get=_aperture_get_blades,
            write=_aperture_write_blades,
            undo_label="Set Aperture Blades",
            fmt=lambda v: f"{int(v)}",
            editable=_aperture_polygon_editable,
            options={
                "min": _MIN_APERTURE_BLADES,
                "max": _MAX_APERTURE_BLADES,
                "scrubbable": True,
            },
        ),
        Slot(
            # Polygon rotation lives in the Aperture Rad column. Stored in
            # radians on the Surface struct; surfaced as degrees in the UI
            # since the .lens file format and human convention both use
            # degrees for blade rotation.
            key="aperture_rad",
            label="Rotation°",
            editor=SlotEditor.FLOAT,
            get=_aperture_get_rotation_deg,
            write=_aperture_write_rotation_deg,
            undo_label="Set Aperture Rotation",
            fmt=lambda v: f"{float(v):.2f}",
            editable=_aperture_polygon_editable,
            options={"decimals": 2, "scrubbable": True},
        ),
    ),
)


# ---------------------------------------------------------------------------
# Blade Shape
#
# Second child row on a bladed stop, holding the four controls that deform the
# blade edge away from a straight chord: curvature, twist, notch and notch
# angle. Its own row rather than more cells on Aperture Form, because the
# aperture row's four canonical columns are already shape / aspect / blades /
# rotation, and these four only exist while the shape is POLYGON — nodes.py
# doesn't build the row otherwise.
#
# Units follow each control's own convention rather than the storage format:
# curvature and twist are stored as fractions but authored as percentages
# (-100..100 %), notch and notch angle are stored in radians and authored in
# degrees, as rotation already is. The degrees are NOMINAL — the profile
# applies them as a fraction of the blade's sector, not as literal angles.
# ---------------------------------------------------------------------------

_BLADE_PCT_RANGE = 100.0
_BLADE_NOTCH_DEG = 45.0


def _blade_surface(ctx: SlotContext):
    """The stop surface this row edits, or None if it isn't a bladed stop."""
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None or int(surf.aperture_shape) != int(ghostlight.ApertureShape.POLYGON):
        return None
    return surf


def _blade_getter(attr: str, scale: float):
    def get(ctx: SlotContext) -> Optional[float]:
        surf = _blade_surface(ctx)
        return None if surf is None else float(getattr(surf, attr)) * scale

    return get


def _blade_writer(attr: str, scale: float, limit_lo: float, limit_hi: float):
    """Write an authored control, clamping in UI units before converting back.

    Clamping here rather than only in C++ keeps the cell showing what the model
    actually holds: the loader clamps too, but a designer edit never goes
    through the loader.
    """

    def write(ctx: SlotContext, value: Any) -> WriteResult:
        surf = _blade_surface(ctx)
        if surf is None:
            return _NOOP
        try:
            new_ui = float(value)
        except (TypeError, ValueError):
            return _NOOP
        new_ui = max(limit_lo, min(limit_hi, new_ui))
        new_val = new_ui / scale
        if new_val == float(getattr(surf, attr)):
            return _NOOP
        setattr(surf, attr, new_val)
        return WriteResult(True)

    return write


def _blade_slot(name: str, unit: str, attr: str, scale: float,
                lo: float, hi: float, decimals: int) -> Slot:
    return Slot(
        key=f"blade_{name.lower().replace(' ', '_')}",
        label=f"{name}{unit}",
        editor=SlotEditor.FLOAT,
        get=_blade_getter(attr, scale),
        write=_blade_writer(attr, scale, lo, hi),
        undo_label=f"Set Blade {name}",
        fmt=lambda v, d=decimals: f"{float(v):.{d}f}",
        options={"decimals": decimals, "scrubbable": True},
    )


_BLADE_SHAPE_SLOTS: tuple[Slot, ...] = (
    _blade_slot("Curvature", "%", "aperture_curvature",
                _BLADE_PCT_RANGE, -_BLADE_PCT_RANGE, _BLADE_PCT_RANGE, 1),
    _blade_slot("Twist", "%", "aperture_twist",
                _BLADE_PCT_RANGE, -_BLADE_PCT_RANGE, _BLADE_PCT_RANGE, 1),
    _blade_slot("Notch", "°", "aperture_notch_rad",
                _RAD_TO_DEG, -_BLADE_NOTCH_DEG, _BLADE_NOTCH_DEG, 2),
    _blade_slot("Notch Angle", "°", "aperture_notch_angle_rad",
                _RAD_TO_DEG, 0.0, _BLADE_NOTCH_DEG, 2),
)


BLADE_SHAPE_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Blade Shape",
    slots=_BLADE_SHAPE_SLOTS,
    # Packed rather than pinned: none of these four map to a canonical column,
    # and pinning bespoke keys would push them past the off-axis block.
    packed_slots=lambda _node: _BLADE_SHAPE_SLOTS,
)


# ---------------------------------------------------------------------------
# Coating Form
#
# Child row attached to every surface. The C++ Surface carries a Coating POD
# (model discriminator + ar_layers + artist tint/strength + attenuator
# scalars) plus, for table-backed models, side data owned by the system
# (spectral / angular / spectral-angular tables, TMM layer stacks) reached
# through the OpticalSystem.get/set_coating_* accessors.
#
# Unlike every other row, the coating row does NOT pin its slots to canonical
# columns — its per-model slots are mutually exclusive, so pinning left a run
# of blank cells (a Simple coating showed Model, AR Layers, then three empty
# columns, then Preset). It sets ``RowSchema.packed_slots`` instead and lays
# only the live slots into consecutive columns:
#
#   Simple        → Model | AR Layers | Preset
#   Artist        → Model | Strength  | Tint | Preset
#   table-backed  → Model | Data      | Preset
#     (Spectral / Angular / Spectral×Angular / Layer Stack / Attenuator —
#      "Data" opens a modal editor and shows a summary in-cell)
#
# The in-cell label (painted whenever a slot's label differs from the strip
# header, which is always true here) is what keeps a packed column readable.
# Widest row is Artist at 4 slots, so a coating row never reaches past the
# canonical strip — no trailing columns are reserved for it.
#
# A model switch changes which slots are live, so the Model write returns
# requires_reset=True; that rebuild is also what refreshes the packed layout,
# since ``CoatingFormNode.coating_ui_model`` is baked by ``build_tree``. All
# writes ride the model's project.edit(...) wrapper for undo; the table
# pickers apply their payload via the system accessors INSIDE that
# transaction, so undo round-trips through the writer.
# ---------------------------------------------------------------------------


class CoatingModelUI(enum.IntEnum):
    """Designer-facing coating model choices.

    Distinct from :class:`ghostlight.CoatingModel` because ``LAYER_STACK`` and
    ``SPECTRAL_ANGULAR`` share the same runtime model (a baked λ×angle table)
    but are authored differently — a layer stack keeps editable layer specs,
    a raw table does not. ``_coating_ui_model_of`` disambiguates them by
    whether the surface has stored layer specs.
    """

    SIMPLE = 0
    ARTIST = 1
    SPECTRAL = 2
    ANGULAR = 3
    SPECTRAL_ANGULAR = 4
    LAYER_STACK = 5
    ATTENUATOR = 6


_COATING_UI_LABELS = {
    CoatingModelUI.SIMPLE: "Simple (AR)",
    CoatingModelUI.ARTIST: "Artist",
    CoatingModelUI.SPECTRAL: "Spectral",
    CoatingModelUI.ANGULAR: "Angular",
    CoatingModelUI.SPECTRAL_ANGULAR: "Spectral×Angular",
    CoatingModelUI.LAYER_STACK: "Layer Stack",
    CoatingModelUI.ATTENUATOR: "Attenuator",
}

_MAX_AR_LAYERS = 8

# Default MgF2 quarter-wave n,k table (non-absorbing) seeded when the user
# first switches a surface to a layer stack. λ in μm.
_DEFAULT_MGF2_NK = [
    [0.40, 1.384, 0.0],
    [0.55, 1.380, 0.0],
    [0.70, 1.377, 0.0],
]


def coating_ui_model_for_surface(system, surface_index: int) -> Optional[int]:
    """UI-facing coating model for ``system.surfaces[surface_index]``.

    Public because ``nodes.build_tree`` bakes it onto every
    ``CoatingFormNode`` — the packed column layout is a pure function of the
    node, so ``slot_at`` keeps its node-only signature.
    """
    surfaces = getattr(system, "surfaces", None) if system is not None else None
    if not surfaces or not (0 <= surface_index < len(surfaces)):
        return None
    surf = surfaces[surface_index]
    model = int(surf.coating.model)
    if model == int(ghostlight.CoatingModel.ARTIST):
        return int(CoatingModelUI.ARTIST)
    if model == int(ghostlight.CoatingModel.SPECTRAL):
        return int(CoatingModelUI.SPECTRAL)
    if model == int(ghostlight.CoatingModel.ANGULAR):
        return int(CoatingModelUI.ANGULAR)
    if model == int(ghostlight.CoatingModel.ATTENUATOR_GAUSS):
        return int(CoatingModelUI.ATTENUATOR)
    if model == int(ghostlight.CoatingModel.SPECTRAL_ANGULAR):
        try:
            has_layers = bool(system.get_coating_layers(surface_index))
        except Exception:
            has_layers = False
        return int(CoatingModelUI.LAYER_STACK if has_layers
                   else CoatingModelUI.SPECTRAL_ANGULAR)
    return int(CoatingModelUI.SIMPLE)


def _coating_ui_model_of(ctx: SlotContext) -> Optional[int]:
    return coating_ui_model_for_surface(
        ctx.system, getattr(ctx.node, "surface_index", -1)
    )


def _coating_model_label(ui_int: int) -> str:
    try:
        return _COATING_UI_LABELS[CoatingModelUI(int(ui_int))]
    except (ValueError, KeyError):
        return f"Unknown({ui_int})"


def _coating_seed_model(system, si: int, ui_model: int) -> None:
    """Convert surface ``si``'s coating to ``ui_model`` with sane defaults.

    Runs inside the model's project.edit() transaction (called from the
    Model combo write), so every mutation is captured for undo.
    """
    import numpy as np

    surf = system.surfaces[si]
    ui = CoatingModelUI(int(ui_model))

    # Always start from a clean uncoated state so stale tables/layers don't
    # leak across a model switch.
    system.clear_coating(si)

    if ui == CoatingModelUI.SIMPLE:
        surf.coating.ar_layers = 1
    elif ui == CoatingModelUI.ARTIST:
        c = surf.coating
        c.model = ghostlight.CoatingModel.ARTIST
        c.tint_r = c.tint_g = c.tint_b = 1.0
        c.tint_strength = 0.04
    elif ui == CoatingModelUI.SPECTRAL:
        system.set_coating_spectral_table(
            si, np.array([[400.0, 0.02], [700.0, 0.02]], dtype=np.float32))
    elif ui == CoatingModelUI.ANGULAR:
        system.set_coating_angular_table(
            si, np.array([[0.0, 0.02], [80.0, 0.10]], dtype=np.float32),
            angle_ref_ior=1.0)
    elif ui == CoatingModelUI.SPECTRAL_ANGULAR:
        wl = np.array([400.0, 700.0], dtype=np.float32)
        ang = np.array([0.0, 80.0], dtype=np.float32)
        r = np.array([[0.02, 0.10], [0.02, 0.10]], dtype=np.float32)
        system.set_coating_sa_table(si, wl, ang, r, angle_ref_ior=1.0)
    elif ui == CoatingModelUI.LAYER_STACK:
        nk = np.array(_DEFAULT_MGF2_NK, dtype=np.float32)
        system.set_coating_layers(
            si, [{"material": "MgF2",
                  "thickness_nm": 550.0 / (4.0 * 1.38),
                  "nk_table": nk}])
    elif ui == CoatingModelUI.ATTENUATOR:
        c = surf.coating
        c.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
        c.gauss_sigma = max(float(surf.semi_aperture) * 0.5, 1.0)
        c.gauss_background = 0.2
        c.gauss_peak = 0.8
        c.gauss_decenter_x = 0.0
        c.gauss_decenter_y = 0.0


def _coating_write_model(ctx: SlotContext, value: Any) -> WriteResult:
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        new_ui = int(value)
    except (TypeError, ValueError):
        return _NOOP
    if new_ui == _coating_ui_model_of(ctx):
        return _NOOP
    si = getattr(ctx.node, "surface_index", -1)
    if si < 0:
        return _NOOP
    _coating_seed_model(ctx.system, si, new_ui)
    # Which per-model cells (layers / strength / tint / data) are live flips
    # with the discriminator, so rebuild the row.
    return WriteResult(True, requires_reset=True)


# ---- Simple (ar_layers) ----

def _coating_get_ar_layers(ctx: SlotContext) -> Optional[int]:
    if _coating_ui_model_of(ctx) != int(CoatingModelUI.SIMPLE):
        return None
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else int(surf.coating.ar_layers)


def _coating_simple_editable(ctx: SlotContext) -> bool:
    return _coating_ui_model_of(ctx) == int(CoatingModelUI.SIMPLE)


def _coating_write_ar_layers(ctx: SlotContext, value: Any) -> WriteResult:
    if not _coating_simple_editable(ctx):
        return _NOOP
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        n = int(value)
    except (TypeError, ValueError):
        return _NOOP
    n = max(0, min(_MAX_AR_LAYERS, n))
    if n == int(surf.coating.ar_layers):
        return _NOOP
    surf.coating.ar_layers = n
    return WriteResult(True)


# ---- Artist strength / tint ----

def _coating_artist_editable(ctx: SlotContext) -> bool:
    return _coating_ui_model_of(ctx) == int(CoatingModelUI.ARTIST)


def _coating_get_strength(ctx: SlotContext) -> Optional[float]:
    if not _coating_artist_editable(ctx):
        return None
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else float(surf.coating.tint_strength)


def _coating_write_strength(ctx: SlotContext, value: Any) -> WriteResult:
    if not _coating_artist_editable(ctx):
        return _NOOP
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    try:
        s = float(value)
    except (TypeError, ValueError):
        return _NOOP
    s = max(0.0, min(1.0, s))
    if s == float(surf.coating.tint_strength):
        return _NOOP
    surf.coating.tint_strength = s
    return WriteResult(True)


def _coating_tint_hex(surf) -> str:
    def q(v):
        return max(0, min(255, int(round(float(v) * 255.0))))
    c = surf.coating
    return f"#{q(c.tint_r):02X}{q(c.tint_g):02X}{q(c.tint_b):02X}"


def _coating_get_tint(ctx: SlotContext) -> Optional[str]:
    if not _coating_artist_editable(ctx):
        return None
    surf = _surface_of(ctx.node, ctx.system)
    return None if surf is None else _coating_tint_hex(surf)


def _coating_write_tint(ctx: SlotContext, value: Any) -> WriteResult:
    if not _coating_artist_editable(ctx):
        return _NOOP
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return _NOOP
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        return _NOOP
    try:
        r = int(text[0:2], 16) / 255.0
        g = int(text[2:4], 16) / 255.0
        b = int(text[4:6], 16) / 255.0
    except ValueError:
        return _NOOP
    c = surf.coating
    if (r, g, b) == (float(c.tint_r), float(c.tint_g), float(c.tint_b)):
        return _NOOP
    c.tint_r, c.tint_g, c.tint_b = r, g, b
    return WriteResult(True)


# ---- Data picker (tables / layer stack / attenuator) ----

_COATING_DATA_MODELS = frozenset({
    int(CoatingModelUI.SPECTRAL),
    int(CoatingModelUI.ANGULAR),
    int(CoatingModelUI.SPECTRAL_ANGULAR),
    int(CoatingModelUI.LAYER_STACK),
    int(CoatingModelUI.ATTENUATOR),
})


def _coating_data_editable(ctx: SlotContext) -> bool:
    return _coating_ui_model_of(ctx) in _COATING_DATA_MODELS


def _coating_get_data_summary(ctx: SlotContext) -> Optional[str]:
    ui = _coating_ui_model_of(ctx)
    if ui not in _COATING_DATA_MODELS:
        return None
    surf = _surface_of(ctx.node, ctx.system)
    if surf is None:
        return None
    si = getattr(ctx.node, "surface_index", -1)
    try:
        if ui == int(CoatingModelUI.SPECTRAL):
            n = int(surf.coating.table_count)
            return f"{n}-pt λ table"
        if ui == int(CoatingModelUI.ANGULAR):
            n = int(surf.coating.table_count)
            return f"{n}-pt angle table"
        if ui == int(CoatingModelUI.SPECTRAL_ANGULAR):
            return f"{int(surf.coating.sa_n_wavelengths)}×" \
                   f"{int(surf.coating.sa_n_angles)} table"
        if ui == int(CoatingModelUI.LAYER_STACK):
            n = len(ctx.system.get_coating_layers(si))
            return f"{n} layer{'s' if n != 1 else ''}"
        if ui == int(CoatingModelUI.ATTENUATOR):
            return f"σ={float(surf.coating.gauss_sigma):.1f}"
    except Exception:
        return "…"
    return None


def _coating_write_data(ctx: SlotContext, value: Any) -> WriteResult:
    """Apply a JSON payload produced by the coating-data dialog.

    The dialog returns its result as a JSON string (the delegate coerces
    picker returns via ``str()``); we decode + apply it here, inside the
    model's project.edit() transaction, so the edit is undoable and
    round-trips through the writer.
    """
    import json
    import numpy as np

    if not _coating_data_editable(ctx):
        return _NOOP
    si = getattr(ctx.node, "surface_index", -1)
    if si < 0:
        return _NOOP
    try:
        payload = json.loads(str(value))
    except (ValueError, TypeError):
        return _NOOP
    kind = payload.get("kind")
    system = ctx.system
    surf = system.surfaces[si]
    try:
        if kind == "spectral":
            system.set_coating_spectral_table(
                si, np.array(payload["data"], dtype=np.float32),
                out_of_range_discard=bool(payload.get("out_of_range_discard", False)))
        elif kind == "angular":
            system.set_coating_angular_table(
                si, np.array(payload["data"], dtype=np.float32),
                angle_ref_ior=float(payload.get("angle_ref_ior", 1.0)),
                out_of_range_discard=bool(payload.get("out_of_range_discard", False)))
        elif kind == "spectral_angular":
            system.set_coating_sa_table(
                si,
                np.array(payload["wavelengths"], dtype=np.float32),
                np.array(payload["angles"], dtype=np.float32),
                np.array(payload["r"], dtype=np.float32),
                angle_ref_ior=float(payload.get("angle_ref_ior", 1.0)),
                out_of_range_discard=bool(payload.get("out_of_range_discard", False)))
        elif kind == "layers":
            layers = [
                {"material": str(ly.get("material", "")),
                 "thickness_nm": float(ly["thickness_nm"]),
                 "nk_table": np.array(ly["nk_table"], dtype=np.float32)}
                for ly in payload["layers"]
            ]
            system.set_coating_layers(si, layers)
        elif kind == "attenuator":
            c = surf.coating
            c.model = ghostlight.CoatingModel.ATTENUATOR_GAUSS
            c.gauss_sigma = float(payload.get("sigma", c.gauss_sigma))
            c.gauss_background = float(payload.get("background", c.gauss_background))
            c.gauss_peak = float(payload.get("peak", c.gauss_peak))
            c.gauss_decenter_x = float(payload.get("decenter_x", 0.0))
            c.gauss_decenter_y = float(payload.get("decenter_y", 0.0))
        else:
            return _NOOP
    except Exception:
        return _NOOP
    return WriteResult(True)


# ---- Preset picker (coating catalogue) ----

def _coating_write_preset(ctx: SlotContext, value: Any) -> WriteResult:
    """Apply a coating-catalogue preset (a JSON coating-modifier payload).

    The preset picker returns the modifier dict as a JSON string; the shared
    ``coating_actions.apply_coating_payload_to_system`` applies it (the same
    path the surface right-click menu uses). We run inside the model's
    project.edit() transaction, so no extra undo wrapping here.
    """
    import json

    from . import coating_actions

    si = getattr(ctx.node, "surface_index", -1)
    if si < 0:
        return _NOOP
    try:
        mod = json.loads(str(value))
    except (ValueError, TypeError):
        return _NOOP
    if not coating_actions.apply_coating_payload_to_system(ctx.system, si, mod):
        return _NOOP
    return WriteResult(True, requires_reset=True)


def _coating_preset_display(ctx: SlotContext) -> str:
    # Presets are apply-and-forget templates (no live link stored on the
    # surface), so the cell always reads as an action affordance.
    return "Apply…"


# Slots are named so the per-model groups below can reference them; the row
# packs a group into consecutive columns instead of pinning by key. Each
# slot keeps its own get/editable model gate anyway — a slot that isn't in
# the active group is unreachable through the tree, but ``setData`` can still
# arrive on a stale index, and the guards are what make that a no-op.
_COATING_MODEL_SLOT = Slot(
    key="coating_model",
    label="Model",
    editor=SlotEditor.ENUM_COMBO,
    get=_coating_ui_model_of,
    write=_coating_write_model,
    undo_label="Set Coating Model",
    fmt=lambda v: _coating_model_label(int(v)),
    options={"enum": CoatingModelUI, "scrubbable": False},
)

_COATING_AR_LAYERS_SLOT = Slot(
    key="coating_ar_layers",
    label="AR Layers",
    editor=SlotEditor.INT_SPINBOX,
    get=_coating_get_ar_layers,
    write=_coating_write_ar_layers,
    undo_label="Set AR Layers",
    fmt=lambda v: f"{int(v)}",
    editable=_coating_simple_editable,
    options={"min": 0, "max": _MAX_AR_LAYERS, "scrubbable": False},
)

_COATING_STRENGTH_SLOT = Slot(
    key="coating_strength",
    label="Strength",
    editor=SlotEditor.FLOAT,
    get=_coating_get_strength,
    write=_coating_write_strength,
    undo_label="Set Coating Strength",
    fmt=lambda v: f"{float(v):.4f}",
    editable=_coating_artist_editable,
    options={"decimals": 4, "scrubbable": True,
             "variable_attr": "coating.tint_strength"},
)

_COATING_TINT_SLOT = Slot(
    key="coating_tint",
    label="Tint",
    editor=SlotEditor.TEXT_PICKER,
    get=_coating_get_tint,
    write=_coating_write_tint,
    undo_label="Set Coating Tint",
    editable=_coating_artist_editable,
    options={"picker": "coating_tint", "color_swatch": True},
)

_COATING_DATA_SLOT = Slot(
    key="coating_data",
    label="Data",
    editor=SlotEditor.TEXT_PICKER,
    get=_coating_get_data_summary,
    write=_coating_write_data,
    undo_label="Edit Coating Data",
    editable=_coating_data_editable,
    options={"picker": "coating_data"},
)

_COATING_PRESET_SLOT = Slot(
    key="coating_preset",
    label="Preset",
    editor=SlotEditor.TEXT_PICKER,
    get=lambda _ctx: None,
    display=_coating_preset_display,
    write=_coating_write_preset,
    undo_label="Apply Coating Preset",
    options={"picker": "coating_preset"},
)

# Every table-backed model authors its values through the one Data picker,
# so they all share a group; Simple and Artist get their own scalar cells.
_COATING_DATA_GROUP = (
    _COATING_MODEL_SLOT, _COATING_DATA_SLOT, _COATING_PRESET_SLOT,
)

_COATING_SLOT_GROUPS: dict[int, tuple[Slot, ...]] = {
    int(CoatingModelUI.SIMPLE): (
        _COATING_MODEL_SLOT, _COATING_AR_LAYERS_SLOT, _COATING_PRESET_SLOT,
    ),
    int(CoatingModelUI.ARTIST): (
        _COATING_MODEL_SLOT, _COATING_STRENGTH_SLOT, _COATING_TINT_SLOT,
        _COATING_PRESET_SLOT,
    ),
}


def _coating_packed_slots(node: TreeNode) -> tuple[Slot, ...]:
    """Live slots for this coating row, left to right.

    ``coating_ui_model`` is baked onto the node by ``nodes.build_tree``.
    A node built before that field existed (or one whose surface didn't
    resolve) falls back to the table-backed group — Model + Preset stay
    reachable either way, so the row is never a dead end.
    """
    ui = getattr(node, "coating_ui_model", None)
    if ui is None:
        return _COATING_DATA_GROUP
    return _COATING_SLOT_GROUPS.get(int(ui), _COATING_DATA_GROUP)


COATING_SCHEMA = RowSchema(
    name_label=lambda _ctx: "Coating",
    slots=(
        _COATING_MODEL_SLOT,
        _COATING_AR_LAYERS_SLOT,
        _COATING_STRENGTH_SLOT,
        _COATING_TINT_SLOT,
        _COATING_DATA_SLOT,
        _COATING_PRESET_SLOT,
    ),
    packed_slots=_coating_packed_slots,
)


# Widest packed row, in columns (Name + that row's live slots). ``column_count``
# floors the tree width here so a packed row can never be clipped.
_MAX_PACKED_WIDTH = _PACK_FIRST_COLUMN + max(
    len(group)
    for group in (*_COATING_SLOT_GROUPS.values(), _COATING_DATA_GROUP,
                  _BLADE_SHAPE_SLOTS)
)


SCHEMAS: dict[NodeKind, RowSchema] = {
    NodeKind.ELEMENT: ELEMENT_SCHEMA,
    NodeKind.MATERIAL: MATERIAL_SCHEMA,
    NodeKind.SURFACE: SURFACE_SCHEMA,
    NodeKind.ASPHERE_FORM: ASPHERE_SCHEMA,
    NodeKind.CYLINDRICAL_FORM: CYLINDRICAL_SCHEMA,
    NodeKind.APERTURE_FORM: APERTURE_SCHEMA,
    NodeKind.BLADE_SHAPE_FORM: BLADE_SHAPE_SCHEMA,
    NodeKind.COATING_FORM: COATING_SCHEMA,
}


# ---------------------------------------------------------------------------
# Public Material conversion APIs
#
# Driven by the right-click "Convert to" submenu on Material rows. Both
# wrap the mutation in ``project.edit(...)`` so undo restores the prior
# key / catalogue entry / surface IORs atomically.
# ---------------------------------------------------------------------------

# Tolerance for "close enough" vendor match against the current Custom
# nd/Vd. Loosened one step beyond Zemax six-digit glass-code precision
# (~5e-4 nd, ~5e-2 Vd) so a hand-typed glass that lost its last decimal
# in transcription still snaps to its catalogue twin. Matches the
# ``_LOOSE_*`` band used by ``migrate_to_catalogue`` for the same job.
ND_MATCH_TOLERANCE = 1.0e-3
VD_MATCH_TOLERANCE = 1.0e-1


def material_designers() -> list[str]:
    """Vendor strings available in the bundled MaterialCatalogue (no Custom).

    Used by the body's context menu to populate 'Convert to <vendor>'
    entries. Custom is offered as a separate action since the conversion
    code path differs (synth a Custom_<hash> entry instead of resolving a
    catalogue match).
    """
    seen = {m.source_vendor for m in _material_catalogue().all() if m.source_vendor}
    return sorted(seen)


def convert_material_to_vendor(
    project,
    node: MaterialNode,
    vendor: str,
    *,
    nd_tolerance: float = ND_MATCH_TOLERANCE,
    vd_tolerance: float = VD_MATCH_TOLERANCE,
) -> tuple[bool, str]:
    """Switch ``material_glasses[i]`` to the closest ``vendor`` glass by
    (nd, Vd) proximity.

    Returns ``(True, "")`` on success, or ``(False, message)`` if no
    glass is within tolerance — the caller surfaces ``message`` to the
    user; the material stays unchanged. Conversion is wrapped in
    ``project.edit(...)`` for undo. The previous Custom catalogue entry
    is left orphaned in ``system._raw_glass_catalogue`` (harmless; it
    just costs a few bytes in the next save).
    """
    system = project.system
    ctx = SlotContext(node=node, system=system, project=project)
    current_nd, current_vd = _current_nd_vd(ctx)
    if current_nd is None or current_vd is None:
        return (False, "Current material has no nd/Vd to match against.")
    matches = _material_catalogue().search(vendor=vendor)
    if not matches:
        return (False, f"No glasses loaded for designer {vendor!r}.")
    best = None
    best_score = float("inf")
    for m in matches:
        if m.nd is None or m.vd is None:
            continue
        d_nd = abs(float(m.nd) - current_nd)
        d_vd = abs(float(m.vd) - current_vd)
        if d_nd > nd_tolerance or d_vd > vd_tolerance:
            continue
        # Normalized score so nd and Vd contribute equally; tightest
        # match wins.
        score = d_nd / nd_tolerance + d_vd / vd_tolerance
        if score < best_score:
            best = m
            best_score = score
    if best is None:
        return (
            False,
            (
                f"No {vendor} glass found within tolerance of "
                f"nd={current_nd:.5f}, Vd={current_vd:.2f}. "
                "Material left as-is."
            ),
        )
    with project.edit(f"Convert Material to {vendor}"):
        if not _material_set_glass_key(node, best.key):
            return (False, "Failed to update material reference.")
        _ensure_glass_in_project_catalogue(system, best)
        if best.nd is not None and best.vd is not None:
            _refresh_surfaces_for_key(
                system, best.key, float(best.nd), float(best.vd)
            )
    return (True, "")


def convert_material_to_custom(project, node: MaterialNode) -> bool:
    """Convert any vendor-resolved material to a fresh project-local
    Custom entry. Seeded with the current nd/Vd so Surface IORs don't
    snap on switch. Returns False if the material can't be located
    (mid-edit, race, stale node). Idempotent: if the material is already
    Custom (i.e. has no MaterialCatalogue match), nothing changes."""
    system = project.system
    ctx = SlotContext(node=node, system=system, project=project)
    if _material_catalogue_entry(ctx) is None:
        # Already Custom-like — no-op rather than churn the key for no
        # behavioral change.
        return False
    current_nd, current_vd = _current_nd_vd(ctx)
    nd = current_nd if current_nd is not None else _DEFAULT_CUSTOM_ND
    vd = current_vd if current_vd is not None else _DEFAULT_CUSTOM_VD
    key = _generate_custom_key()
    with project.edit("Convert Material to Custom"):
        _ensure_abbe_entry(system, key, nd, vd)
        if not _material_set_glass_key(node, key):
            return False
        _refresh_surfaces_for_key(system, key, nd, vd)
    return True
