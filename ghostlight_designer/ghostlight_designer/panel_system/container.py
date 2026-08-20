"""Panel container — root widget plus split/remove/serialize tree ops."""
from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ..project import Project
from .layouts import PanelLayoutRegistry, layout_registry as default_layout_registry
from .panel import Panel
from .types import PanelTypeRegistry, registry as default_registry

_log = logging.getLogger("ghostlight_designer.panel_system")

FALLBACK_TYPE_ID = "optical_editor"

# Maps a panel type id to its successor, so a workspace naming a type that no
# longer exists reopens as the panel that supersedes it rather than collapsing
# to the generic fallback.
RETIRED_TYPE_IDS: dict = {}


class PanelRoot(QWidget):
    panelAdded = Signal(object)
    panelRemoved = Signal(object)

    def __init__(
        self,
        project: Project,
        parent: Optional[QWidget] = None,
        registry: Optional[PanelTypeRegistry] = None,
        layout_registry: Optional[PanelLayoutRegistry] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._registry = registry if registry is not None else default_registry
        self._layout_registry = (
            layout_registry if layout_registry is not None else default_layout_registry
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout
        self._root_widget: Optional[QWidget] = None

    @property
    def project(self) -> Project:
        return self._project

    @property
    def root_widget(self) -> Optional[QWidget]:
        return self._root_widget

    def make_panel(self, type_id: str) -> Panel:
        if self._registry.get(type_id) is None:
            successor = RETIRED_TYPE_IDS.get(type_id)
            if successor is not None and self._registry.get(successor) is not None:
                _log.info(
                    "Panel type %r was retired; substituting %r",
                    type_id, successor,
                )
                type_id = successor
        if self._registry.get(type_id) is None:
            _log.warning(
                "Panel type %r not registered; falling back to %r",
                type_id, FALLBACK_TYPE_ID,
            )
            type_id = FALLBACK_TYPE_ID
        return Panel(
            type_id,
            self._project,
            registry=self._registry,
            layout_registry=self._layout_registry,
        )

    def set_root_panel(self, panel: Panel) -> None:
        self._clear_root()
        self._root_widget = panel
        self._layout.addWidget(panel)
        self.panelAdded.emit(panel)

    def build_default_layout(self, spec: dict) -> None:
        """Build a fresh panel tree from a nested split/leaf spec.

        Spec form (same as :meth:`from_dict`'s ``"docked"`` payload but no
        outer ``"version"`` wrapper)::

            leaf:  {"kind": "leaf", "type_id": <str>}
            split: {"kind": "split",
                    "orient": "h" | "v",
                    "sizes":  [int, ...],
                    "children": [<spec>, ...]}

        Sizes are in QSplitter "logical" units; relative weights are what
        matter. ``panelAdded`` fires once per leaf after the tree is
        installed.
        """
        widget = self._node_from_dict(spec)
        self._clear_root()
        self._root_widget = widget
        self._layout.addWidget(widget)
        for p in self.leaves():
            self.panelAdded.emit(p)

    def split_panel(
        self, leaf: Panel, orient: Qt.Orientation, new_type_id: str = "optical_editor"
    ) -> Panel:
        parent = leaf.parentWidget()
        new_panel = self.make_panel(new_type_id)

        if leaf is self._root_widget:
            splitter = QSplitter(orient)
            self._layout.removeWidget(leaf)
            self._root_widget = splitter
            self._layout.addWidget(splitter)
            splitter.addWidget(leaf)
            splitter.addWidget(new_panel)
            self._set_even_sizes(splitter)
        elif isinstance(parent, QSplitter):
            idx = parent.indexOf(leaf)
            sizes = parent.sizes()
            new_splitter = QSplitter(orient)
            # Insert new_splitter then reparent leaf into it directly. No
            # intermediate setParent(None) — that briefly makes the leaf a
            # top-level widget, which destroys/recreates the underlying
            # QWindow for any QOpenGLWidget inside and crashes if the GL
            # context isn't fully initialized yet.
            parent.insertWidget(idx, new_splitter)
            new_splitter.addWidget(leaf)
            new_splitter.addWidget(new_panel)
            if sizes and len(sizes) == parent.count():
                parent.setSizes(sizes)
            self._set_even_sizes(new_splitter)
        else:
            raise RuntimeError(
                f"split_panel: leaf {leaf!r} has unexpected parent {parent!r}"
            )
        self.panelAdded.emit(new_panel)
        return new_panel

    def remove_panel(self, leaf: Panel) -> None:
        if leaf is self._root_widget:
            self._layout.removeWidget(leaf)
            leaf.setParent(None)
            self._root_widget = None
            self.panelRemoved.emit(leaf)
            return
        parent = leaf.parentWidget()
        if parent is None:
            # Already removed (e.g. handler fired twice). No-op.
            return
        if not isinstance(parent, QSplitter):
            raise RuntimeError(
                f"remove_panel: leaf {leaf!r} has unexpected parent {parent!r}"
            )
        leaf.setParent(None)
        self.panelRemoved.emit(leaf)
        self._collapse_if_singleton(parent)

    def leaves(self) -> List[Panel]:
        out: List[Panel] = []
        if self._root_widget is not None:
            self._collect_leaves(self._root_widget, out)
        return out

    def to_dict(self) -> dict:
        if self._root_widget is None:
            return {"version": 1, "docked": None}
        return {"version": 1, "docked": self._node_to_dict(self._root_widget)}

    def from_dict(self, d: dict) -> None:
        self._clear_root()
        docked = d.get("docked")
        if docked is None:
            return
        widget = self._node_from_dict(docked)
        self._root_widget = widget
        self._layout.addWidget(widget)
        for p in self.leaves():
            self.panelAdded.emit(p)

    def _set_even_sizes(self, splitter: QSplitter) -> None:
        n = splitter.count()
        if n == 0:
            return
        for i in range(n):
            splitter.setStretchFactor(i, 1)
        total = max(splitter.width(), splitter.height(), 10_000)
        splitter.setSizes([total // n] * n)

    def _collect_leaves(self, node: QWidget, out: List[Panel]) -> None:
        if isinstance(node, Panel):
            out.append(node)
            return
        if isinstance(node, QSplitter):
            for i in range(node.count()):
                self._collect_leaves(node.widget(i), out)

    def _node_to_dict(self, node: QWidget) -> dict:
        if isinstance(node, Panel):
            t = self._registry.get(node.type_id)
            state = t.serialize_state(node.body) if t is not None and node.body is not None else {}
            return {"kind": "leaf", "type_id": node.type_id, "state": state}
        if isinstance(node, QSplitter):
            return {
                "kind": "split",
                "orient": "h" if node.orientation() == Qt.Horizontal else "v",
                "sizes": list(node.sizes()),
                "children": [self._node_to_dict(node.widget(i)) for i in range(node.count())],
            }
        raise TypeError(f"_node_to_dict: unexpected widget {node!r}")

    def _node_from_dict(self, d: dict) -> QWidget:
        kind = d.get("kind")
        if kind == "leaf":
            type_id = d.get("type_id", FALLBACK_TYPE_ID)
            panel = self.make_panel(type_id)
            t = self._registry.get(panel.type_id)
            state = d.get("state") or {}
            if t is not None and state and panel.body is not None:
                try:
                    t.restore_state(panel.body, state)
                except Exception:
                    _log.exception("restore_state failed for type_id=%r", type_id)
            return panel
        if kind == "split":
            orient = Qt.Horizontal if d.get("orient", "h") == "h" else Qt.Vertical
            splitter = QSplitter(orient)
            for child in d.get("children", []):
                splitter.addWidget(self._node_from_dict(child))
            sizes = d.get("sizes")
            if sizes and len(sizes) == splitter.count():
                splitter.setSizes(list(sizes))
            return splitter
        raise ValueError(f"_node_from_dict: unknown kind {kind!r}")

    def _collapse_if_singleton(self, splitter: QSplitter) -> None:
        """Prune empty splitters from the tree.

        Single-child splitters are intentionally LEFT INTACT. A QSplitter
        with one child renders identically to that child (no visible
        handle), and reparenting the child out would destroy/recreate the
        QWindow underlying any QOpenGLWidget inside — which crashes in
        PySide6 when the GL widget hasn't fully initialized yet (typical
        immediately after launch). The cosmetic cost of degenerate
        single-child splitters is zero; the safety cost of collapsing them
        is a hard-to-reproduce native crash.
        """
        if splitter.count() > 0:
            return
        parent = splitter.parentWidget()
        if splitter is self._root_widget:
            self._layout.removeWidget(splitter)
            self._root_widget = None
            splitter.deleteLater()
        elif isinstance(parent, QSplitter):
            splitter.setParent(None)
            splitter.deleteLater()
            self._collapse_if_singleton(parent)
        else:
            splitter.deleteLater()

    def _clear_root(self) -> None:
        if self._root_widget is not None:
            self._layout.removeWidget(self._root_widget)
            self._root_widget.deleteLater()
            self._root_widget = None
