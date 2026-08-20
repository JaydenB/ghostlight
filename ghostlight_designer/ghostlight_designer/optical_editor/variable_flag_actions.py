"""Right-click / menu handlers for surface variable flags.

Split out of :mod:`body` so:

* The per-cell "Toggle Variable" / "Edit Bounds…" flow and the bulk
  "Flag All Radii / Thicknesses / Clear" flow live next to each other
  rather than scattered across the 700-line body.
* The bulk actions are re-usable from anywhere (a radial menu,
  a keyboard shortcut, the optimization panel's own toolbar) — same
  contract every other ``*_actions`` module in this package follows.

All mutations go through :class:`Project` methods that emit
:sig:`variableFlagChanged` / :sig:`variableFlagsReplaced`; the tree
delegate repaints from those signals.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QModelIndex
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QWidget


from ..project import Project, VariableBounds
from .delegates import SlotRole
from .nodes import surface_uuid_for
from .row_schemas import Slot
from .variable_bounds_dialog import edit_variable_bounds


# ---------------------------------------------------------------------------
# Per-cell state resolution
# ---------------------------------------------------------------------------


def variable_attr_at(index: QModelIndex) -> str:
    """Return the ``variable_attr`` for the cell at ``index``, or ``""``.

    A cell is flag-able iff its ``Slot.options`` carries
    ``variable_attr``. See :mod:`row_schemas` for which slots opt in.
    """
    if not index.isValid():
        return ""
    slot = index.data(SlotRole)
    if not isinstance(slot, Slot):
        return ""
    return str(slot.options.get("variable_attr", "") or "")


def surface_uuid_at(index: QModelIndex) -> str:
    if not index.isValid():
        return ""
    return surface_uuid_for(index.internalPointer())


# ---------------------------------------------------------------------------
# Per-cell menu builder — invoked from body._on_tree_context_menu
# ---------------------------------------------------------------------------


def populate_cell_menu(
    menu: QMenu,
    project: Project,
    index: QModelIndex,
    *,
    dialog_parent: QWidget,
) -> bool:
    """Add "Toggle Variable" / "Edit Bounds…" if ``index`` is flag-able.

    Returns True when at least one action was appended (caller uses this
    to decide whether to draw a separator before the next section).
    """
    attr = variable_attr_at(index)
    if not attr:
        return False
    uuid = surface_uuid_at(index)
    if not uuid:
        return False

    is_flagged = project.is_variable_flagged(uuid, attr)
    label_summary = _cell_label(index, attr)

    toggle_action = QAction(
        "Unflag Variable" if is_flagged else "Flag as Variable",
        menu,
    )
    toggle_action.triggered.connect(
        lambda _checked=False:
            project.toggle_variable_flag(uuid, attr)
    )
    menu.addAction(toggle_action)

    if is_flagged:
        bounds_action = QAction("Edit Variable Bounds…", menu)
        bounds_action.triggered.connect(
            lambda _checked=False:
                _open_bounds_dialog(project, uuid, attr, label_summary,
                                    dialog_parent)
        )
        menu.addAction(bounds_action)
    return True


def _cell_label(index: QModelIndex, attr: str) -> str:
    """Human-friendly summary line for the bounds dialog header.

    ``"Surface 3 — Radius"`` when we can resolve the surface index;
    falls back to ``"— <attr>"`` when we can't.
    """
    node = index.internalPointer() if index.isValid() else None
    surface_index: Optional[int] = None
    if node is not None:
        si = getattr(node, "surface_index", None)
        if isinstance(si, int) and si >= 0:
            surface_index = si
        else:
            parent = getattr(node, "parent", None)
            if parent is not None:
                psi = getattr(parent, "surface_index", None)
                if isinstance(psi, int) and psi >= 0:
                    surface_index = psi
    slot = index.data(SlotRole) if index.isValid() else None
    label = str(getattr(slot, "label", "")) if isinstance(slot, Slot) else ""
    if not label:
        label = attr
    if surface_index is None:
        return label
    return f"Surface {surface_index} — {label}"


def _open_bounds_dialog(
    project: Project,
    uuid: str,
    attr: str,
    title_summary: str,
    parent: QWidget,
) -> None:
    current = project.variable_bounds(uuid, attr) or VariableBounds()
    outcome = edit_variable_bounds(
        parent,
        title_summary=title_summary,
        current_enabled=True,
        current_bounds=current,
    )
    if outcome is None:
        return
    enabled, new_bounds = outcome
    if enabled:
        project.set_variable_flag(uuid, attr, new_bounds)
    else:
        project.clear_variable_flag(uuid, attr)


# ---------------------------------------------------------------------------
# Bulk operations — surface across every context menu
# ---------------------------------------------------------------------------


def _collect_uuid_pairs_by_attr(
    project: Project, attr: str,
) -> list[tuple[str, str]]:
    """Yield ``(uuid, attr)`` pairs for every surface in the system for
    which ``attr`` is a sensible flag target.

    * ``radius`` — skips ``is_stop`` surfaces (their radius is
      geometrically meaningless).
    * ``thickness`` — every surface.

    Returned list order matches ``system.surface_ids`` for stability
    when the user re-runs the bulk action.
    """
    pairs: list[tuple[str, str]] = []
    system = project.system
    try:
        uuids = list(system.surface_ids)
        surfaces = system.surfaces
    except Exception:
        return pairs
    for uuid, surf in zip(uuids, surfaces):
        if attr == "radius" and bool(getattr(surf, "is_stop", False)):
            continue
        pairs.append((uuid, attr))
    return pairs


def flag_all_radii(project: Project) -> int:
    """Flag every non-stop surface's radius as variable.

    Returns the number of surfaces newly flagged. Existing flags with
    user-set bounds are preserved (:meth:`Project.bulk_set_variable_flags`
    skips already-flagged entries)."""
    pairs = _collect_uuid_pairs_by_attr(project, "radius")
    before = sum(len(v) for v in project.all_variable_flags().values())
    project.bulk_set_variable_flags(pairs)
    after = sum(len(v) for v in project.all_variable_flags().values())
    return max(0, after - before)


def flag_all_thicknesses(project: Project) -> int:
    pairs = _collect_uuid_pairs_by_attr(project, "thickness")
    before = sum(len(v) for v in project.all_variable_flags().values())
    project.bulk_set_variable_flags(pairs)
    after = sum(len(v) for v in project.all_variable_flags().values())
    return max(0, after - before)


def clear_all_variable_flags(project: Project) -> bool:
    return project.clear_all_variable_flags()


def populate_bulk_menu(menu: QMenu, project: Project) -> None:
    """Append a "Variable Flags" submenu with the bulk actions.

    Appearing in every context menu is deliberate — the user should
    never have to hunt for the empty-area click to reach these; and the
    surface/element menus feel like the right place to reach them too.
    """
    sub = menu.addMenu("Variable Flags")

    act_radii = QAction("Flag All Radii", sub)
    act_radii.setToolTip(
        "Mark every non-stop surface's radius as an optimization variable."
    )
    act_radii.triggered.connect(
        lambda _checked=False: flag_all_radii(project)
    )
    sub.addAction(act_radii)

    act_thick = QAction("Flag All Thicknesses", sub)
    act_thick.setToolTip(
        "Mark every surface's thickness (spacing) as an optimization variable."
    )
    act_thick.triggered.connect(
        lambda _checked=False: flag_all_thicknesses(project)
    )
    sub.addAction(act_thick)

    sub.addSeparator()

    act_clear = QAction("Clear All Variable Flags", sub)
    act_clear.setToolTip("Unflag every currently-variable attribute.")
    # Disable when there's nothing to clear so the menu shows the state.
    act_clear.setEnabled(bool(project.all_variable_flags()))
    act_clear.triggered.connect(
        lambda _checked=False: clear_all_variable_flags(project)
    )
    sub.addAction(act_clear)
