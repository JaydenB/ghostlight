"""Tree node hierarchy for the optimization panel.

    Root
    ├── VariablesHeaderNode                   (top; only present when >=1 flag)
    │   ├── VariableEntryNode                 (Surface 0 · radius)
    │   ├── VariableEntryNode                 (Surface 3 · thickness)
    │   └── ...
    ├── MeritFunctionNode
    │   ├── GoalNode
    │   ├── GoalNode
    │   └── ...
    └── MeritFunctionNode
        └── ...

The Variables section is read-only-add: users flag attributes in the
Optical Design Editor; this panel only displays them and offers a Remove
affordance (Delete key + right-click "Unflag"). Adding here would
duplicate the ODE's flag UI without adding value.

Nodes carry references back into project-owned state by index / uuid.
We re-resolve on every read so mutations don't strand a stale pointer;
the indices are cheap to recompute when the model rebuilds.
"""
from __future__ import annotations

from typing import List, Optional


class TreeNode:
    __slots__ = ("parent", "children")

    def __init__(self) -> None:
        self.parent: Optional["TreeNode"] = None
        self.children: List["TreeNode"] = []

    def add(self, child: "TreeNode") -> "TreeNode":
        child.parent = self
        self.children.append(child)
        return child

    def row(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.children.index(self)


class RootNode(TreeNode):
    pass


class MeritFunctionNode(TreeNode):
    """One merit function row. ``mf_index`` is the index into
    ``project.merit_functions``."""

    __slots__ = ("mf_index",)

    def __init__(self, mf_index: int) -> None:
        super().__init__()
        self.mf_index = mf_index


class GoalNode(TreeNode):
    """One goal row inside a merit function. ``goal_index`` indexes
    ``project.merit_functions[mf_index].goals``."""

    __slots__ = ("mf_index", "goal_index")

    def __init__(self, mf_index: int, goal_index: int) -> None:
        super().__init__()
        self.mf_index = mf_index
        self.goal_index = goal_index


class VariablesHeaderNode(TreeNode):
    """Top-level header for the "Variables" section.

    Displays the count of currently-flagged variables in its Name cell;
    all other cells are empty. Only appears when at least one variable
    is flagged so a clean project doesn't show an empty header.

    "Variables" is the umbrella name — both surface-attribute flags
    (radius, thickness, …) and material-substitution flags live under
    this header so the user sees one unified list of "what will move on
    Run" instead of hunting between two sections.
    """
    __slots__ = ()


class VariableEntryNode(TreeNode):
    """One row per flagged (surface, attr) pair.

    Keyed by ``surface_uuid`` (stable across system reloads via
    :meth:`Project.set_variable_flag`) plus ``attr`` (``"radius"`` /
    ``"thickness"`` / …). ``surface_index`` is a cached snapshot at
    build time — resolve via the project on read if the tree might
    have gone stale.
    """
    __slots__ = ("surface_uuid", "attr", "surface_index")

    def __init__(
        self, surface_uuid: str, attr: str, surface_index: int,
    ) -> None:
        super().__init__()
        self.surface_uuid = surface_uuid
        self.attr = attr
        self.surface_index = surface_index


class MaterialSubstitutionEntryNode(TreeNode):
    """One row per flagged (element, material_index) pair.

    Keyed by ``element_id`` (stable across system reloads via
    :attr:`ghostlight.Element.element_id`) plus ``material_index`` (int into
    the element's ``material_glasses`` list). ``element_index`` is a
    cached snapshot at build time; resolve via the project on read
    if the tree might have gone stale.
    """
    __slots__ = ("element_id", "material_index", "element_index")

    def __init__(
        self,
        element_id: str,
        material_index: int,
        element_index: int,
    ) -> None:
        super().__init__()
        self.element_id = element_id
        self.material_index = int(material_index)
        self.element_index = int(element_index)


def build_tree(project) -> RootNode:
    """Build the optimization-panel tree from a live ``Project``.

    Signature evolved from ``build_tree(merit_functions)`` to take the
    whole project so the Variables section can read flags off it. The
    model's callers all had a project handy anyway.
    """
    root = RootNode()

    # Variables section first — most-recently-set is presented near the
    # top so the user sees "what's tunable right now" before scrolling
    # through merit functions. Header hidden when both flag maps are empty.
    surface_ids = _safe_surface_ids(project)
    flag_map = project.all_variable_flags() if surface_ids else {}
    material_flag_map = project.all_material_flags()
    if flag_map or material_flag_map:
        header = VariablesHeaderNode()
        root.add(header)
        # Emit entries in system-surface order so a lens with radii
        # flagged on surfaces 0, 2, 3 always renders in that order,
        # not in dict-iteration order.
        for si, uuid in enumerate(surface_ids):
            attrs = flag_map.get(uuid)
            if not attrs:
                continue
            # Within a surface, sort attr names for a stable readout.
            for attr in sorted(attrs.keys()):
                header.add(VariableEntryNode(uuid, attr, si))
        # Material entries after all surface entries: they operate on
        # elements rather than surfaces so they read as a distinct
        # group — but under the same "Variables" umbrella so the count
        # header reflects the full set of things that will move.
        if material_flag_map:
            try:
                elements = list(project.system.elements)
            except Exception:
                elements = []
            for ei, el in enumerate(elements):
                eid = getattr(el, "element_id", "") or ""
                if not eid:
                    continue
                mats = material_flag_map.get(eid)
                if not mats:
                    continue
                for mi in sorted(mats.keys()):
                    header.add(MaterialSubstitutionEntryNode(eid, mi, ei))

    for mi, mf in enumerate(project.merit_functions):
        mf_node = MeritFunctionNode(mi)
        root.add(mf_node)
        for gi, _ in enumerate(mf.goals):
            mf_node.add(GoalNode(mi, gi))
    return root


def _safe_surface_ids(project) -> list:
    """Read ``project.system.surface_ids`` defensively.

    Returns ``[]`` when the C++ wrapper is unhappy (a half-loaded state
    during startup, a system with zero surfaces, a test double without
    the attribute). Keeps ``build_tree`` from crashing when a Project
    isn't fully populated yet."""
    try:
        return list(project.system.surface_ids)
    except Exception:
        return []
