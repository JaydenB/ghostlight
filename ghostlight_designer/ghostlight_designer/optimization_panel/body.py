"""Body widget for the optimization panel.

Tree on top, toolbar on bottom (top, actually — same convention the ODE
uses). Toolbar emits intent signals; the body resolves the user's
selection into a target merit function / goal and dispatches to the
model + the optimizer.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QMenu

from ..project import Project
from ..value_scrubber import attach_value_scrubber
from .columns import Column, base_column_count
from .data import GoalEntry, GoalKind, MeritFunction
from .delegates import OptimizationCellDelegate
from .goals.base import default_params_for, default_target_for
from .model import OptimizationTreeModel
from .nodes import (
    GoalNode,
    MaterialSubstitutionEntryNode,
    MeritFunctionNode,
    VariableEntryNode,
    VariablesHeaderNode,
)
from .optimizer import OptimizationRun
from .preview_dialog import OptimizationPreviewDialog
from .presets import PRESETS
from .toolbar import OptimizationToolbar

_log = logging.getLogger("ghostlight_designer.optimization_panel.body")


class _ComboClickToEdit(QObject):
    """Single-click to edit + open combo dropdown.

    Mirrors :class:`ghostlight_designer.optical_editor.body._ComboClickToEdit`
    so the optimization panel behaves the same way the ODE does. Without
    this filter, combo cells require double-click to enter edit mode and
    then a third click to open the dropdown — the same three-click flow the
    ODE filter removes there.

    The delegate's ``uses_combo(index)`` predicate decides which cells
    qualify; non-combo cells keep the standard double-click behaviour so
    drag-to-scrub still works on numeric cells.
    """

    def __init__(self, tree: QTreeView) -> None:
        super().__init__(tree)
        self._tree = tree
        tree.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):  # type: ignore[override]
        tree = self._tree
        try:
            viewport = tree.viewport()
        except RuntimeError:
            return False
        if obj is not viewport:
            return False
        if ev.type() != QEvent.MouseButtonPress:
            return False
        if ev.button() != Qt.LeftButton:
            return False
        index = tree.indexAt(ev.position().toPoint())
        if not index.isValid():
            return False
        delegate = tree.itemDelegate(index)
        if not hasattr(delegate, "uses_combo"):
            return False
        if not delegate.uses_combo(index):
            return False
        sel = tree.selectionModel()
        sel.setCurrentIndex(
            index,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        tree.edit(index)
        return True


class _NonUndoingProjectShim:
    """Wrap a :class:`Project` so the value scrubber's compound-undo
    calls are no-ops.

    The scrubber uses ``project.begin_compound`` / ``end_compound`` to
    coalesce a drag's per-pixel edits into a single undo entry. That
    snapshots the **OpticalSystem**, but goal target / weight edits
    don't change the system — so the entries would be empty undo slots
    that confuse the user (an undo that visibly does nothing).

    Merit functions don't participate in undo/redo, matching SystemSetup.
    Swallow the compound calls
    so drag-scrubs don't pollute the undo stack.
    """

    __slots__ = ("_project",)

    def __init__(self, project: Project) -> None:
        self._project = project

    def begin_compound(self, _label: str) -> None:
        pass

    def end_compound(self) -> None:
        pass

    # Pass-throughs for anything else the scrubber might read from the
    # project. Today it only calls begin_compound / end_compound, but a
    # a scrubber tweak could need more.
    def __getattr__(self, name: str):
        return getattr(self._project, name)


def _is_scrubbable(index: QModelIndex) -> bool:
    """Gate Ctrl+MMB to numeric cells the user is allowed to drag.

    Target / Weight columns on a goal row qualify. Everything else
    (Name, Type, Value, Residual, params) does not — Value / Residual
    are display-only, Name + Type are textual, params are combos.
    """
    if not index.isValid():
        return False
    model = index.model()
    if model is None or not (model.flags(index) & Qt.ItemIsEditable):
        return False
    col = index.column()
    # Only goal rows have scrubbable scalars. MF NAME (the one editable
    # MF cell) is text, not a number.
    if col not in (int(Column.TARGET), int(Column.WEIGHT)):
        return False
    node = index.internalPointer()
    if not isinstance(node, GoalNode):
        return False
    val = model.data(index, Qt.EditRole)
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _scrub_compound_label(index: QModelIndex) -> str:
    """Undo-entry label for the drag — unused because the shim no-ops,
    but the scrubber API requires it."""
    col = index.column()
    if col == int(Column.TARGET):
        return "Scrub Target"
    if col == int(Column.WEIGHT):
        return "Scrub Weight"
    return "Scrub"


class OptimizationPanelBody(QWidget):
    def __init__(self, project: Project, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project = project

        self.model = OptimizationTreeModel(project, self)

        self.tree = QTreeView(self)
        self.tree.setModel(self.model)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )

        header = self.tree.header()
        header.setStretchLastSection(False)
        for c in range(self.model.columnCount()):
            header.setSectionResizeMode(c, QHeaderView.Interactive)
        self.tree.setColumnWidth(int(Column.NAME), 220)
        self.tree.setColumnWidth(int(Column.TYPE), 140)
        self.tree.setColumnWidth(int(Column.TARGET), 90)
        self.tree.setColumnWidth(int(Column.WEIGHT), 80)
        self.tree.setColumnWidth(int(Column.VALUE), 110)
        self.tree.setColumnWidth(int(Column.RESIDUAL), 110)

        self._delegate = OptimizationCellDelegate(project, self.tree)
        self.tree.setItemDelegate(self._delegate)

        # Single-click on combo cells enters edit mode AND pops the
        # dropdown — same UX as the ODE.  Numeric cells keep the
        # standard double-click trigger so drag-to-scrub still works.
        self._combo_click_filter = _ComboClickToEdit(self.tree)

        # Ctrl+MMB scrubber on Target / Weight cells. The project shim
        # swallows the compound-undo calls because merit functions
        # aren't on the undo stack — see _NonUndoingProjectShim.
        self._scrub_project = _NonUndoingProjectShim(project)
        self._scrub_trigger = attach_value_scrubber(
            self.tree,
            self._scrub_project,
            is_scrubbable=_is_scrubbable,
            compound_label=_scrub_compound_label,
        )

        self.toolbar = OptimizationToolbar(self)
        self.toolbar.addMeritFunctionRequested.connect(self._on_add_mf)
        self.toolbar.addGoalRequested.connect(self._on_add_goal)
        self.toolbar.removeRequested.connect(self._on_remove)
        self.toolbar.runSelectedRequested.connect(self._on_run_selected)
        self.toolbar.runAllRequested.connect(self._on_run_all)

        # Delete key on a selected row acts as the primary remove
        # affordance for variables — matches how many tree UIs behave and
        # is what the user reaches for after ``select row`` far more
        # often than the toolbar button. Also works for goal / MF rows
        # via the same ``_on_remove`` dispatch.
        remove_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.tree)
        remove_shortcut.setContext(Qt.WidgetShortcut)
        remove_shortcut.activated.connect(self._on_remove)

        # Right-click on a variable row → single "Unflag Variable" action.
        # Nothing to add here for goal / MF rows; those already have the
        # toolbar + menu bar. We keep this menu tight so the panel stays
        # feeling different from the ODE (where right-click is heavier).
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.tree)

        # Expand new merit-function subtrees automatically so the user
        # sees what was just added without an extra click.
        self.model.modelReset.connect(self._expand_all_mfs)
        self._expand_all_mfs()

        # Track any in-flight preview dialogs so Run All can wait for the
        # previous one to close before launching the next. We use a simple
        # queue rather than chaining via signals to keep the flow obvious.
        self._run_queue: list[MeritFunction] = []
        self._active_dialog: Optional[OptimizationPreviewDialog] = None

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand_all_mfs(self) -> None:
        # Also expands the Variables header row when one is present —
        # the user probably wants to see the flagged variables at a glance
        # rather than having to click the disclosure triangle every launch.
        root_count = self.model.rowCount(QModelIndex())
        for r in range(root_count):
            idx = self.model.index(r, 0, QModelIndex())
            self.tree.setExpanded(idx, True)

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click dispatch.

        Variable rows get a single "Unflag Variable" action — the panel
        deliberately doesn't offer bounds editing here (the ODE owns
        that surface). Other row types get no menu today; if the user
        wants MF-level or goal-level actions they've got the toolbar +
        menu bar.
        """
        index = self.tree.indexAt(pos)
        if not index.isValid():
            return
        node = index.internalPointer()
        menu = QMenu(self.tree)
        if isinstance(node, VariableEntryNode):
            act = QAction("Unflag Variable", menu)
            act.triggered.connect(
                lambda _checked=False, n=node:
                    self._project.clear_variable_flag(n.surface_uuid, n.attr)
            )
            menu.addAction(act)
        elif isinstance(node, MaterialSubstitutionEntryNode):
            act = QAction("Unflag Material Substitution", menu)
            act.triggered.connect(
                lambda _checked=False, n=node:
                    self._project.clear_material_flag(
                        n.element_id, n.material_index,
                    )
            )
            menu.addAction(act)
        elif isinstance(node, MeritFunctionNode):
            self._populate_mf_menu(menu, node)
            if not menu.actions():
                return
        else:
            return
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _populate_mf_menu(self, menu, node: MeritFunctionNode) -> None:
        """Append per-MF actions to the tree's right-click menu.

        Today: only the "Preserve Radius Signs" toggle. Others (e.g. a
        per-MF max_iters editor) can slot in here without touching the
        dispatch in :meth:`_on_tree_context_menu`.
        """
        mfs = self._project.merit_functions
        if not (0 <= node.mf_index < len(mfs)):
            return
        mf = mfs[node.mf_index]
        act = QAction("Preserve Radius Signs", menu)
        act.setCheckable(True)
        act.setChecked(bool(getattr(mf, "preserve_radius_signs", False)))
        act.setToolTip(
            "When on, each flagged radius stays on its starting side of "
            "flat (positive or negative). scipy can flatten a surface but "
            "won't flip it through zero. Useful for keeping the element "
            "topology intact during a tuning pass."
        )
        act.triggered.connect(
            lambda checked, m=mf: self._on_toggle_preserve_signs(m, checked)
        )
        menu.addAction(act)

    def _on_toggle_preserve_signs(self, mf, checked: bool) -> None:
        if bool(mf.preserve_radius_signs) == bool(checked):
            return
        mf.preserve_radius_signs = bool(checked)
        # Merit-function settings edit — matches the modified signal
        # tree cell edits use.
        self._project.mark_merit_functions_modified()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _current_node(self):
        idx = self.tree.currentIndex()
        return idx.internalPointer() if idx.isValid() else None

    def _current_mf_index(self) -> Optional[int]:
        node = self._current_node()
        if isinstance(node, MeritFunctionNode):
            return node.mf_index
        if isinstance(node, GoalNode):
            return node.mf_index
        # No selection: target the last MF if there's only one or zero.
        mfs = self._project.merit_functions
        if len(mfs) == 1:
            return 0
        return None

    def _current_goal_indices(self) -> Optional[tuple[int, int]]:
        node = self._current_node()
        if isinstance(node, GoalNode):
            return (node.mf_index, node.goal_index)
        return None

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------

    def _on_add_mf(self, preset_label: str) -> None:
        builder = None
        for label, b in PRESETS:
            if label == preset_label:
                builder = b
                break
        if builder is None:
            return
        try:
            mf: MeritFunction = builder(self._project.system)
        except Exception:
            _log.exception("Preset %r builder raised", preset_label)
            return
        self.model.add_merit_function(mf)

    def _on_add_goal(self, kind_value: str) -> None:
        mf_idx = self._current_mf_index()
        if mf_idx is None:
            QMessageBox.information(
                self,
                "Add Goal",
                "Select a merit function first.",
            )
            return
        try:
            kind = GoalKind(kind_value)
        except ValueError:
            return
        goal = GoalEntry.make(
            kind=kind,
            target=default_target_for(kind),
            params=dict(default_params_for(kind)),
        )
        self.model.add_goal(mf_idx, goal)

    def _on_remove(self) -> None:
        node = self._current_node()
        if isinstance(node, VariableEntryNode):
            self._project.clear_variable_flag(node.surface_uuid, node.attr)
            return
        if isinstance(node, MaterialSubstitutionEntryNode):
            self._project.clear_material_flag(
                node.element_id, node.material_index,
            )
            return
        if isinstance(node, GoalNode):
            self.model.remove_goal(node.mf_index, node.goal_index)
            return
        if isinstance(node, MeritFunctionNode):
            mfs = self._project.merit_functions
            if not (0 <= node.mf_index < len(mfs)):
                return
            mf = mfs[node.mf_index]
            answer = QMessageBox.question(
                self,
                "Remove Merit Function",
                f"Remove {mf.name!r}? This cannot be undone (merit "
                "functions don't participate in Edit › Undo).",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self.model.remove_merit_function(node.mf_index)

    def _on_run_selected(self) -> None:
        mf_idx = self._current_mf_index()
        if mf_idx is None:
            QMessageBox.information(
                self,
                "Run Optimization",
                "Select a merit function to run.",
            )
            return
        mfs = self._project.merit_functions
        if not (0 <= mf_idx < len(mfs)):
            return
        self._launch_dialog(mfs[mf_idx])

    def _on_run_all(self) -> None:
        if self._active_dialog is not None:
            QMessageBox.information(
                self,
                "Run All",
                "An optimization preview is already open.",
            )
            return
        enabled = [mf for mf in self._project.merit_functions if mf.enabled]
        if not enabled:
            QMessageBox.information(
                self,
                "Run All",
                "No enabled merit functions to run.",
            )
            return
        self._run_queue = list(enabled)
        self._drain_run_queue()

    def _drain_run_queue(self) -> None:
        if self._active_dialog is not None:
            return
        if not self._run_queue:
            return
        next_mf = self._run_queue.pop(0)
        self._launch_dialog(next_mf)

    def _launch_dialog(self, mf: MeritFunction) -> None:
        if self._active_dialog is not None:
            QMessageBox.information(
                self,
                "Run Optimization",
                "An optimization preview is already open.",
            )
            return
        try:
            run = OptimizationRun(
                self._project, mf, self._project.system_setup, self,
            )
        except Exception as exc:
            _log.exception("Failed to construct OptimizationRun")
            QMessageBox.warning(
                self,
                "Run Optimization",
                f"Could not start optimisation: {exc}",
            )
            return
        dlg = OptimizationPreviewDialog(self._project, mf, run, self)
        self._active_dialog = dlg
        dlg.finished.connect(self._on_dialog_finished)
        dlg.show()
        dlg.start_run()

    def _on_dialog_finished(self, _result_code: int) -> None:
        self._active_dialog = None
        # The dialog has updated the project + mf caches if accepted;
        # nothing more for us to do here. If Run All is in flight,
        # advance the queue.
        if self._run_queue:
            self._drain_run_queue()
