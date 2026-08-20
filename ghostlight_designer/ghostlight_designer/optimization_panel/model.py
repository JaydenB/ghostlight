"""Tree model for the optimization panel.

Same shape as :mod:`ghostlight_designer.system_setup.model` — a static
:class:`QAbstractItemModel` that rebuilds whenever the structure changes
(merit functions added/removed, goals added/removed) and emits per-cell
``dataChanged`` for value-only edits.

Editable cells:

* Merit function row:
    - NAME column: MF.name (text)
* Goal row:
    - NAME column: GoalEntry.name (text, optional comment)
    - TARGET column: GoalEntry.target (float)
    - WEIGHT column: GoalEntry.weight (float)
    - Trailing param columns: goal-kind-specific (combo or text)

Everything else (Type, Value, Residual) is read-only display.

The model writes through to ``project.merit_functions`` and emits
``mark_merit_functions_modified`` / ``mark_merit_functions_replaced`` so
the panel and every other observer stay in sync.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt

from ..project import Project
from .columns import Column, base_column_count, header_text
from .data import GoalEntry, MeritFunction
from .goals.base import display_name_for, param_schema_for
from .nodes import (
    GoalNode,
    MaterialSubstitutionEntryNode,
    MeritFunctionNode,
    RootNode,
    TreeNode,
    VariableEntryNode,
    VariablesHeaderNode,
    build_tree,
)


# Sentinel returned by data() for columns this row doesn't populate.
_EMPTY = ""


def _format_bound(value, unbounded_text: str) -> str:
    """Render a bound for the Target / Weight cells of a variable row.

    ``None`` → ``unbounded_text`` (either ``"-∞"`` or ``"+∞"``); numbers
    render at the same precision as goal targets so the two columns
    line up visually.
    """
    if value is None:
        return unbounded_text
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return unbounded_text


class OptimizationTreeModel(QAbstractItemModel):
    def __init__(self, project: Project, parent=None) -> None:
        super().__init__(parent)
        self._project = project
        self._root: RootNode = RootNode()
        self._max_param_cols: int = 0
        project.meritFunctionsReplaced.connect(self._on_replaced)
        project.meritFunctionsChanged.connect(self._on_modified)
        # Variables section — its header / rows appear or disappear as the
        # user flags things in the ODE. Any add / remove is structural
        # (new row appears / disappears), so we route through the same
        # rebuild path merit-function structure changes use.
        project.variableFlagChanged.connect(self._on_replaced)
        project.variableFlagsReplaced.connect(self._on_replaced)
        # Material-substitution flags share the Variables header — any
        # add / remove is structural for the tree, so we route through
        # the same rebuild path.
        project.materialFlagChanged.connect(self._on_replaced)
        project.materialFlagsReplaced.connect(self._on_replaced)
        # A lens edit can shift a surface's index (element re-order) or
        # change a variable's underlying value — keep the display fresh.
        project.systemModified.connect(self._on_modified)
        project.systemReplaced.connect(self._on_replaced)
        self._rebuild()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        self.beginResetModel()
        try:
            self._root = build_tree(self._project)
            self._max_param_cols = self._compute_max_param_cols()
        finally:
            self.endResetModel()

    def _compute_max_param_cols(self) -> int:
        n = 0
        for mf in self._project.merit_functions:
            for g in mf.goals:
                n = max(n, len(param_schema_for(g.kind)))
        return n

    def _on_replaced(self) -> None:
        self._rebuild()

    def _on_modified(self) -> None:
        # Structural changes (adding / removing a goal) go through
        # mark_merit_functions_replaced. modified-only events come from
        # cell edits / cached_value updates; emit a wide dataChanged so
        # display columns refresh without resetting expansion state.
        if not self._root.children:
            return
        top = self.index(0, 0, QModelIndex())
        bot = self.index(
            len(self._root.children) - 1,
            self.columnCount() - 1,
            QModelIndex(),
        )
        # Walk into each MF to cover its goal rows too.
        self.dataChanged.emit(top, bot, [Qt.DisplayRole, Qt.EditRole])
        for mi in range(len(self._root.children)):
            mf_idx = self.index(mi, 0, QModelIndex())
            n_goals = self.rowCount(mf_idx)
            if n_goals > 0:
                g_top = self.index(0, 0, mf_idx)
                g_bot = self.index(n_goals - 1, self.columnCount() - 1, mf_idx)
                self.dataChanged.emit(g_top, g_bot, [Qt.DisplayRole, Qt.EditRole])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node(self, index: QModelIndex) -> TreeNode:
        if not index.isValid():
            return self._root
        ptr = index.internalPointer()
        return ptr if ptr is not None else self._root

    def _mf_for(self, node: TreeNode) -> Optional[MeritFunction]:
        idx = getattr(node, "mf_index", None)
        if idx is None:
            return None
        mfs = self._project.merit_functions
        if 0 <= idx < len(mfs):
            return mfs[idx]
        return None

    def _goal_for(self, node: TreeNode) -> Optional[GoalEntry]:
        if not isinstance(node, GoalNode):
            return None
        mf = self._mf_for(node)
        if mf is None:
            return None
        if 0 <= node.goal_index < len(mf.goals):
            return mf.goals[node.goal_index]
        return None

    # ------------------------------------------------------------------
    # QAbstractItemModel
    # ------------------------------------------------------------------

    def columnCount(self, _parent: QModelIndex = QModelIndex()) -> int:
        return base_column_count() + self._max_param_cols

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._node(parent)
        return len(node.children)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = self._node(parent)
        if row < 0 or row >= len(parent_node.children):
            return QModelIndex()
        return self.createIndex(row, column, parent_node.children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        node = self._node(index)
        parent_node = node.parent
        if parent_node is None or parent_node is self._root:
            return QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation != Qt.Horizontal or role != Qt.DisplayRole:
            return None
        if section < base_column_count():
            return header_text(section)
        # Param columns are labelled per-row in the cell text rather than
        # in a fixed header — different goals expose different knobs.
        return "Params"

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        node = self._node(index)
        col = index.column()

        if isinstance(
            node,
            (VariablesHeaderNode, VariableEntryNode, MaterialSubstitutionEntryNode),
        ):
            # Read-only across the board — the ODE is the single write seam
            # for flag state. Removal is offered by the panel's context
            # menu + Delete-key handler, not by making cells editable.
            return base

        if isinstance(node, MeritFunctionNode):
            if col == Column.NAME:
                return base | Qt.ItemIsEditable
            return base

        if isinstance(node, GoalNode):
            if col == Column.NAME:
                return base | Qt.ItemIsEditable
            if col in (Column.TARGET, Column.WEIGHT):
                return base | Qt.ItemIsEditable
            # Trailing param columns
            param_idx = col - base_column_count()
            if param_idx >= 0:
                goal = self._goal_for(node)
                if goal is not None:
                    schema = param_schema_for(goal.kind)
                    if 0 <= param_idx < len(schema):
                        return base | Qt.ItemIsEditable
            return base

        return base

    # ------------------------------------------------------------------
    # data / setData
    # ------------------------------------------------------------------

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        node = self._node(index)
        col = index.column()

        if isinstance(node, VariablesHeaderNode):
            return self._variables_header_data(node, col, role)
        if isinstance(node, VariableEntryNode):
            return self._variable_entry_data(node, col, role)
        if isinstance(node, MaterialSubstitutionEntryNode):
            return self._material_entry_data(node, col, role)
        if isinstance(node, MeritFunctionNode):
            return self._mf_data(node, col, role)
        if isinstance(node, GoalNode):
            return self._goal_data(node, col, role)
        return None

    # ------------------------------------------------------------------
    # Variables section — header + per-entry data
    # ------------------------------------------------------------------

    def _variables_header_data(
        self, node: VariablesHeaderNode, col: int, role: int,
    ):
        if role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None
        if col == Column.NAME:
            n = len(node.children)
            return f"Variables ({n})" if n else "Variables"
        if col == Column.TYPE:
            summary = self._material_candidate_summary()
            if role == Qt.DisplayRole and summary:
                return f"~{summary['candidates']} candidates"
            if role == Qt.ToolTipRole:
                base = (
                    "Attributes flagged as optimization variables in the "
                    "Optical Design Editor. Add / edit bounds there; use "
                    "Delete or the right-click menu to unflag here."
                )
                if summary:
                    base += (
                        f"\n\nMaterial substitution: {summary['flags']} "
                        f"flag(s) with {summary['candidates']} catalogue "
                        "candidates in total — the catalogue-hammer "
                        "optimizer will run a scipy inner loop for each "
                        "candidate. Expect the run to take longer than a "
                        "standard optimization."
                    )
                return base
        return _EMPTY

    def _material_candidate_summary(self):
        """Return ``{flags, candidates}`` for the Variables header display.

        ``None`` when no material flags are set — the standard scipy
        path applies and no extra hint is needed. Candidate counts are
        drawn from the bundled catalogue so a stale catalogue-load
        doesn't sink the header display.
        """
        try:
            flag_map = self._project.all_material_flags()
        except Exception:
            return None
        if not flag_map:
            return None
        try:
            from ..material_catalogue import get_catalogue
            from ..material_substitution import candidates_within_spec
        except Exception:
            return None
        cat = get_catalogue()
        n_flags = 0
        n_cand = 0
        for mats in flag_map.values():
            for spec in mats.values():
                n_flags += 1
                try:
                    n_cand += len(candidates_within_spec(cat, spec))
                except Exception:
                    continue
        return {"flags": n_flags, "candidates": n_cand}

    def _variable_entry_data(
        self, node: VariableEntryNode, col: int, role: int,
    ):
        if role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None

        bounds = self._project.variable_bounds(node.surface_uuid, node.attr)
        if col == Column.NAME:
            si = self._resolve_surface_index(node)
            surface_label = f"Surface {si}" if si is not None else "Surface ?"
            return f"{surface_label} · {node.attr}"
        if col == Column.TYPE:
            return "Variable"
        if col == Column.TARGET:
            return _format_bound(bounds.lo if bounds else None, "-∞")
        if col == Column.WEIGHT:
            return _format_bound(bounds.hi if bounds else None, "+∞")
        if col == Column.VALUE:
            v = self._read_surface_attr(node)
            if v is None:
                return _EMPTY
            return f"{v:.4g}" if role in (Qt.DisplayRole, Qt.ToolTipRole) else v
        return _EMPTY

    def _resolve_surface_index(self, node: VariableEntryNode):
        """Look up the surface's current index by UUID. Falls back to the
        cached ``surface_index`` from build time — mostly right, but a
        rebuild sits between element re-orders and this read anyway."""
        try:
            return list(self._project.system.surface_ids).index(
                node.surface_uuid
            )
        except (ValueError, AttributeError):
            si = node.surface_index
            return si if si >= 0 else None

    def _read_surface_attr(self, node: VariableEntryNode):
        """Live value of the flagged attribute on the current lens."""
        try:
            si = self._resolve_surface_index(node)
            if si is None:
                return None
            return float(getattr(self._project.system.surfaces[si], node.attr))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Material-substitution entries
    # ------------------------------------------------------------------

    def _material_entry_data(
        self, node: MaterialSubstitutionEntryNode, col: int, role: int,
    ):
        if role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None

        spec = self._project.material_flag_spec(
            node.element_id, node.material_index,
        )
        vendor = str(getattr(spec, "vendor", "") or "")
        if col == Column.NAME:
            ei = self._resolve_element_index(node)
            el_label = f"Element {ei}" if ei is not None else "Element ?"
            return f"{el_label} · material {node.material_index}"
        if col == Column.TYPE:
            return f"Substitute · {vendor}" if vendor else "Substitute"
        if col == Column.VALUE:
            # Live glass label ("Schott N-BK7") — mirrors what the ODE
            # Name column shows so the user can cross-reference at a
            # glance between panels.
            return self._current_material_label(node)
        return _EMPTY

    def _resolve_element_index(
        self, node: MaterialSubstitutionEntryNode,
    ):
        try:
            for i, el in enumerate(self._project.system.elements):
                if getattr(el, "element_id", None) == node.element_id:
                    return i
        except Exception:
            pass
        ei = node.element_index
        return ei if ei >= 0 else None

    def _current_material_label(
        self, node: MaterialSubstitutionEntryNode,
    ) -> str:
        try:
            for el in self._project.system.elements:
                if getattr(el, "element_id", None) != node.element_id:
                    continue
                glasses = el.material_glasses
                if 0 <= node.material_index < len(glasses):
                    return str(glasses[node.material_index])
                return ""
        except Exception:
            return ""
        return ""

    def _mf_data(self, node: MeritFunctionNode, col: int, role: int):
        mf = self._mf_for(node)
        if mf is None:
            return None

        if role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None

        if col == Column.NAME:
            return mf.name
        if col == Column.TYPE:
            status = mf.last_status or ""
            return f"MF · {status}" if status else "Merit Function"
        if col == Column.VALUE:
            if mf.last_total is None:
                return _EMPTY
            return f"{mf.last_total:.4g}" if role == Qt.DisplayRole else float(mf.last_total)
        if col == Column.RESIDUAL and role == Qt.ToolTipRole:
            return mf.last_message or ""
        return _EMPTY

    def _goal_data(self, node: GoalNode, col: int, role: int):
        goal = self._goal_for(node)
        if goal is None:
            return None

        if role not in (Qt.DisplayRole, Qt.EditRole, Qt.ToolTipRole):
            return None

        if col == Column.NAME:
            kind_label = display_name_for(goal.kind)
            if goal.name:
                return f"{kind_label} — {goal.name}" if role == Qt.DisplayRole else goal.name
            return kind_label if role == Qt.DisplayRole else ""
        if col == Column.TYPE:
            return display_name_for(goal.kind)
        if col == Column.TARGET:
            return (
                f"{goal.target:.6g}"
                if role in (Qt.DisplayRole, Qt.ToolTipRole)
                else float(goal.target)
            )
        if col == Column.WEIGHT:
            return (
                f"{goal.weight:.4g}"
                if role in (Qt.DisplayRole, Qt.ToolTipRole)
                else float(goal.weight)
            )
        if col == Column.VALUE:
            if goal.cached_value is None:
                return _EMPTY
            return (
                f"{goal.cached_value:.6g}"
                if role in (Qt.DisplayRole, Qt.ToolTipRole)
                else float(goal.cached_value)
            )
        if col == Column.RESIDUAL:
            if goal.cached_residual is None:
                return _EMPTY
            return (
                f"{goal.cached_residual:.6g}"
                if role in (Qt.DisplayRole, Qt.ToolTipRole)
                else float(goal.cached_residual)
            )

        # Trailing param columns: render "label: value" for the goal's
        # param at index ``col - base_column_count()``.
        param_idx = col - base_column_count()
        schema = param_schema_for(goal.kind)
        if 0 <= param_idx < len(schema):
            pdef = schema[param_idx]
            value = goal.params.get(pdef.name, pdef.default)
            if role == Qt.EditRole:
                return value
            return f"{pdef.label}: {value}"
        return _EMPTY

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if not index.isValid():
            return False
        node = self._node(index)
        col = index.column()

        if isinstance(node, MeritFunctionNode):
            return self._set_mf_data(node, col, value, role)
        if isinstance(node, GoalNode):
            return self._set_goal_data(node, col, value, role)
        return False

    def _set_mf_data(self, node: MeritFunctionNode, col: int, value, role: int) -> bool:
        mf = self._mf_for(node)
        if mf is None:
            return False

        if role != Qt.EditRole:
            return False

        if col == Column.NAME:
            new_name = str(value).strip() or mf.name
            if new_name == mf.name:
                return False
            mf.name = new_name
            self._project.mark_merit_functions_modified()
            return True
        return False

    def _set_goal_data(self, node: GoalNode, col: int, value, role: int) -> bool:
        goal = self._goal_for(node)
        if goal is None:
            return False

        if role != Qt.EditRole:
            return False

        if col == Column.NAME:
            new_name = str(value).strip()
            if new_name == goal.name:
                return False
            goal.name = new_name
            self._project.mark_merit_functions_modified()
            return True
        if col == Column.TARGET:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if v == goal.target:
                return False
            goal.target = v
            self._project.mark_merit_functions_modified()
            return True
        if col == Column.WEIGHT:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if v < 0.0:
                # Negative weights are mathematically valid but confusing
                # in a UI; clamp at zero rather than silently rejecting.
                v = 0.0
            if v == goal.weight:
                return False
            goal.weight = v
            self._project.mark_merit_functions_modified()
            return True

        # Trailing param columns
        param_idx = col - base_column_count()
        schema = param_schema_for(goal.kind)
        if 0 <= param_idx < len(schema):
            pdef = schema[param_idx]
            try:
                # Store the value as-supplied — the evaluator normalises.
                # Combo editors hand us str / int already in the right shape.
                if pdef.kind in ("wavelength_pick", "field_pick"):
                    new_value: Any = value
                elif pdef.kind == "surface_pick":
                    new_value = int(value)
                else:
                    new_value = value
            except (TypeError, ValueError):
                return False
            if goal.params.get(pdef.name) == new_value:
                return False
            goal.params[pdef.name] = new_value
            self._project.mark_merit_functions_modified()
            return True
        return False

    # ------------------------------------------------------------------
    # Mutation helpers — used by the body's toolbar / menu actions
    # ------------------------------------------------------------------

    def add_merit_function(self, mf: MeritFunction) -> None:
        self._project.merit_functions.append(mf)
        self._project.mark_merit_functions_replaced()

    def remove_merit_function(self, mf_index: int) -> None:
        mfs = self._project.merit_functions
        if not (0 <= mf_index < len(mfs)):
            return
        del mfs[mf_index]
        self._project.mark_merit_functions_replaced()

    def add_goal(self, mf_index: int, goal: GoalEntry) -> None:
        mfs = self._project.merit_functions
        if not (0 <= mf_index < len(mfs)):
            return
        mfs[mf_index].goals.append(goal)
        self._project.mark_merit_functions_replaced()

    def remove_goal(self, mf_index: int, goal_index: int) -> None:
        mfs = self._project.merit_functions
        if not (0 <= mf_index < len(mfs)):
            return
        goals = mfs[mf_index].goals
        if not (0 <= goal_index < len(goals)):
            return
        del goals[goal_index]
        self._project.mark_merit_functions_replaced()
