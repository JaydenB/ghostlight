"""Curated ordering for the panel-type menus.

Both the top-level **Windows** menu (``main_window``) and each panel's
**Panel ▸** change-type menu (``panel.Panel``) list the registered
:class:`PanelType`\ s. The order is curated centrally here so the two menus
stay in lockstep: a fixed top-level sequence with two named submenus. Rules:

* ids in :data:`HIDDEN_TYPE_IDS` are dropped from the menus but remain
  registered, so saved layouts / serialized panels referencing them still
  resolve;
* any registered type not named in :data:`MENU_LAYOUT` still appears —
  appended alphabetically as a top-level leaf — so a newly added panel is
  never silently hidden;
* ids named here that are not registered are skipped harmlessly.

Type ids are kept as plain string literals to avoid importing every panel
subpackage into the panel-system core.
"""
from __future__ import annotations

from typing import List, Tuple, Union

from .types import PanelType, PanelTypeRegistry

# Panel types that stay registered — a saved workspace can still restore them —
# but are not offered in the menus.
HIDDEN_TYPE_IDS: frozenset = frozenset()

# The exact menu order, top to bottom. Each item is either a leaf
# ``("type", <id>)`` or a submenu ``("group", <title>, [<id>, ...])``.
MENU_LAYOUT: List[tuple] = [
    ("type", "viewport"),
    ("type", "optical_editor"),
    ("type", "optimization"),
    ("group", "Rendering", ["sourceflare", "ghost_explorer", "psf"]),
    (
        "group",
        "Evaluations",
        ["evaluation_spot_diagram", "evaluation_field_diagrams", "evaluation_seidel"],
    ),
    ("type", "system_setup"),
]

# A resolved entry is either a leaf PanelType or a titled group of them.
LeafEntry = Tuple[str, PanelType]
GroupEntry = Tuple[str, str, List[PanelType]]
Entry = Union[LeafEntry, GroupEntry]


def ordered_menu_entries(registry: PanelTypeRegistry) -> List[Entry]:
    """Resolve :data:`MENU_LAYOUT` against ``registry``.

    Returns an ordered list where each entry is ``("type", PanelType)`` for a
    top-level leaf or ``("group", title, [PanelType, ...])`` for a submenu.
    Hidden and unregistered ids are omitted; unlisted registered types are
    appended alphabetically as leaves.
    """
    placed: set = set(HIDDEN_TYPE_IDS)
    entries: List[Entry] = []
    for item in MENU_LAYOUT:
        if item[0] == "type":
            tid = item[1]
            placed.add(tid)
            t = registry.get(tid)
            if t is not None:
                entries.append(("type", t))
        else:  # "group"
            _, title, ids = item
            group: List[PanelType] = []
            for tid in ids:
                placed.add(tid)
                t = registry.get(tid)
                if t is not None:
                    group.append(t)
            if group:
                entries.append(("group", title, group))

    leftovers = [t for t in registry.all() if t.id not in placed]
    for t in sorted(leftovers, key=lambda x: x.display_name.casefold()):
        entries.append(("type", t))
    return entries
