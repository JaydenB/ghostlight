"""Drag-scrub rows for the viewport context popup.

A :class:`ScrubRow` is a full-width row (label + live value) that the
popup drives as a horizontal press-drag scrubber — the same interaction
the tree's :mod:`value_scrubber` offers, but bound to plain get/set
callables instead of a ``QModelIndex`` so it works against the raw
optical model from inside the GL viewport.

The popup owns the mouse grab (a ``Qt.Popup`` funnels every mouse event
to the top-level popup, not its children), so it hit-tests which row is
under the cursor and calls :meth:`ScrubRow.begin_scrub` /
:meth:`ScrubRow.update_scrub` / :meth:`ScrubRow.end_scrub`. Each drag
coalesces into a single undo entry via ``Project.begin_compound`` /
``end_compound``; the compound opens lazily on the first value change so
a press-without-drag leaves no empty undo entry behind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

if TYPE_CHECKING:
    from ..project import Project


# Shared popup palette (kept local rather than importing value_scrubber's
# privates; the two popups deliberately look alike).
BG = QColor("#2a2a2a")
BG_HOVER = QColor("#383838")
BG_ARMED = QColor("#3a6ea5")
FG = QColor("#dddddd")
FG_DISABLED = QColor("#6a6a6a")
FG_MUTED = QColor("#9a9a9a")
SEPARATOR = QColor("#444444")
BORDER = QColor("#555555")

# Sensitivity multipliers applied live from modifier keys during a drag.
_FINE_MULT = 0.1     # Shift → finer
_COARSE_MULT = 10.0  # Ctrl → coarser


def _default_fmt(v: float) -> str:
    return f"{v:.3f}"


@dataclass
class ScrubRowSpec:
    """Describes one scrub row.

    ``get`` returns the current value (or ``None`` to disable the row —
    e.g. shape factor of a flat-flat plate). ``set`` applies a new value;
    it is expected to wrap its mutation in ``Project.edit`` so the open
    compound coalesces the drag into one undo entry. ``sensitivity`` is
    value-units per pixel at the default (no-modifier) speed.
    """

    label: str
    get: Callable[[], Optional[float]]
    set: Callable[[float], None]
    fmt: Callable[[float], str] = _default_fmt
    sensitivity: float = 0.1
    undo_label: str = "Scrub"
    # Optional trailing widget (e.g. the Focus row's dpt/mm unit toggle).
    trailing: Optional[QWidget] = field(default=None)


class Separator(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), SEPARATOR)


class ScrubRow(QFrame):
    """A label + live value row scrubbed by horizontal mouse drag."""

    def __init__(
        self,
        spec: ScrubRowSpec,
        project: "Project",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._project = project
        self._hover = False
        self._armed = False
        self._row_enabled = True

        self._dragging = False
        self._compound_open = False
        self._press_value = 0.0
        self._press_global = QPoint()
        self._total_dx = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 5, 14, 5)
        layout.setSpacing(10)
        self._label = QLabel(spec.label, self)
        self._label.setStyleSheet(f"color: {FG.name()};")
        self._value = QLabel("", self)
        self._value.setStyleSheet(f"color: {FG.name()};")
        vfont = QFont(self._value.font())
        vfont.setBold(True)
        self._value.setFont(vfont)
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._label)
        layout.addStretch(1)
        layout.addWidget(self._value)
        if spec.trailing is not None:
            spec.trailing.setParent(self)
            layout.addWidget(spec.trailing)

        self.refresh()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_scrubbable(self) -> bool:
        return self._row_enabled and not self._dragging

    def refresh(self) -> None:
        """Re-read the value from the spec and update the label / enabled
        state. Called on construction and whenever the system changes."""
        try:
            v = self._spec.get()
        except Exception:
            v = None
        self._row_enabled = v is not None
        if v is None:
            self._value.setText("—")
            self._label.setStyleSheet(f"color: {FG_DISABLED.name()};")
            self._value.setStyleSheet(f"color: {FG_DISABLED.name()};")
        else:
            self._value.setText(self._spec.fmt(float(v)))
            fg = FG.name()
            self._label.setStyleSheet(f"color: {fg};")
            self._value.setStyleSheet(f"color: {fg};")

    def set_hover(self, hover: bool) -> None:
        if hover == self._hover:
            return
        self._hover = hover
        self.update()

    # ------------------------------------------------------------------
    # Drag lifecycle (driven by the owning popup)
    # ------------------------------------------------------------------

    def begin_scrub(self, global_pos: QPoint) -> bool:
        """Start a drag anchored at ``global_pos``. Returns False (and does
        nothing) when the row is disabled."""
        if not self._row_enabled:
            return False
        try:
            v = self._spec.get()
        except Exception:
            v = None
        if v is None:
            return False
        self._press_value = float(v)
        self._press_global = QPoint(global_pos)
        self._total_dx = 0
        self._dragging = True
        self._armed = True
        self.update()
        return True

    def update_scrub(self, global_pos: QPoint, modifiers) -> None:
        if not self._dragging:
            return
        dx = global_pos.x() - self._press_global.x()
        if dx == 0:
            return
        self._total_dx += dx
        mult = 1.0
        if modifiers & Qt.ShiftModifier:
            mult = _FINE_MULT
        elif modifiers & Qt.ControlModifier:
            mult = _COARSE_MULT
        new_value = (
            self._press_value
            + self._total_dx * self._spec.sensitivity * mult
        )
        # Open the undo compound lazily so a click without motion doesn't
        # leave an empty entry (and so the first set() lands inside it).
        if not self._compound_open:
            self._project.begin_compound(self._spec.undo_label)
            self._compound_open = True
        try:
            self._spec.set(new_value)
        finally:
            # Re-anchor the cursor so the drag is unbounded (mirrors the
            # tree value-scrubber's infinite-drag warp).
            QCursor.setPos(self._press_global)

    def end_scrub(self) -> None:
        was_dragging = self._dragging
        self._dragging = False
        self._armed = False
        if self._compound_open:
            self._project.end_compound()
            self._compound_open = False
        if was_dragging:
            self.refresh()
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _bg_color(self) -> QColor:
        if self._armed:
            return BG_ARMED
        if self._hover and self._row_enabled:
            return BG_HOVER
        return BG

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color())
