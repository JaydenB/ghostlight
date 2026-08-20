"""Cull-threshold dialog for the ghost explorer.

Modeless and live, like the panel's other popups: the ghost list rebuilds as
the value changes, so the user can watch entries drop out of the scrubber
while dragging rather than guessing a number and closing the dialog.

The threshold is edited as a **percentage of the brightest ghost** rather than
an absolute intensity. Absolute ghost weights span many decades and shift with
the lens's coatings and f-number, so a relative figure is the only one that
means the same thing on two different lenses.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox
from ..render_common import attach_spinbox_scrubber
from .ghost_survey import CULL_REL_MAX, CULL_REL_MIN


class CullThresholdDialog(QDialog):
    """Modeless editor for the ghost cull threshold.

    Signals:
        thresholdChanged(float): the new threshold as a *fraction* in
            ``[CULL_REL_MIN, CULL_REL_MAX]`` — the spinbox shows percent.
    """

    thresholdChanged = Signal(float)

    def __init__(self, current_rel: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ghost Cull Threshold")
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        self._spin = MathDoubleSpinBox(self)
        self._spin.setRange(CULL_REL_MIN * 100.0, CULL_REL_MAX * 100.0)
        self._spin.setDecimals(4)
        self._spin.setSingleStep(0.1)
        self._spin.setSuffix(" %")
        self._spin.setValue(float(current_rel) * 100.0)
        self._spin.setKeyboardTracking(False)
        self._spin.setToolTip(
            "Hide ghosts whose on-sensor flux falls below this percentage of "
            "the brightest ghost's. Ctrl+MMB to scrub."
        )
        self._spin.valueChanged.connect(self._on_value_changed)
        attach_spinbox_scrubber(self._spin, label="Cull threshold (%)")

        info = QLabel(
            "Applies only while “Cull Dim Ghosts” is ticked. Each ghost is "
            "scored by tracing a coarse pupil grid and summing the reflected "
            "weight that lands on the sensor, so a ghost that misses the frame "
            "scores zero. The brightest ghost is always kept.",
            self,
        )
        info.setStyleSheet("color: #888;")
        info.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.close)

        form = QFormLayout()
        form.addRow("Minimum brightness:", self._spin)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(info)
        outer.addWidget(buttons)

    def _on_value_changed(self, percent: float) -> None:
        self.thresholdChanged.emit(float(percent) / 100.0)
