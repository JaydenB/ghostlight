"""Setup dialog for the "Add ▸ Anamorphic Front Block…" action.

Collects the user's intent (squeeze factor, element count, axis, topology,
element type, glasses) and returns an :class:`AnamorphicSpec` describing
the front block to build. The caller in :mod:`.element_actions` uses that
spec to insert cylindrical elements at the front of the system and to
seed a merit function for the optimizer to converge.

The dialog is opinionated: the Basic section shows the four knobs an
artist cares about (squeeze, count, axis, topology). Everything else
(front-element axis override, per-element singlet/doublet, glasses) sits
in the collapsed Advanced section. Defaults produce a classic 2× Galilean
front doublet block with N-BK7 + SF5, block axis X — the canonical
starting point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ghostlight
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from .element_actions import DEFAULT_FLINT_KEY, DEFAULT_GLASS_KEY


# ---------------------------------------------------------------------------
# Spec dataclass returned by the dialog
# ---------------------------------------------------------------------------


@dataclass
class AnamorphicSpec:
    """User's intent for one front-anamorphic build."""

    # Target squeeze ratio (efl_y / efl_x). 2.0 = classic anamorphic.
    squeeze_factor: float = 2.0

    # Number of cylindrical elements in the block (2–4).
    num_elements: int = 2

    # Block-wide cyl axis. Every element in the block uses this axis
    # *unless* ``front_axis_override`` is set. Even then, no element ever
    # mixes axes across its own surfaces: a single element must have one
    # cylinder axis throughout, or its two faces grind against each other.
    block_axis: int = int(ghostlight.CylinderAxis.AXIS_X)

    # "galilean" (front element negative, rear positive — Iscorama style)
    # or "keplerian" (both positive, has intermediate image, X inverted).
    topology: str = "galilean"

    # Block-wide "singlet" or "doublet". Overridden per-element by
    # ``per_element_types`` when the advanced list is populated.
    element_type: str = "doublet"

    # Glass keys resolved through the material catalogue. ``flint_glass``
    # is only used when at least one element in the block is a doublet.
    crown_glass: str = DEFAULT_GLASS_KEY
    flint_glass: str = DEFAULT_FLINT_KEY

    # Air gap between the last cyl element and the base lens (mm). Seeds
    # the trailing-air-surface thickness of the block's last element and
    # is also declared variable for the optimizer.
    front_gap_mm: float = 20.0

    # Advanced: axis for the very front element ONLY (rest of block uses
    # ``block_axis``). ``None`` = same as block; otherwise an int in the
    # ``ghostlight.CylinderAxis`` enum. One-axis-per-element is enforced by the
    # builder — the override is a single axis, never a per-surface split.
    front_axis_override: Optional[int] = None

    # Advanced: per-element type override. ``None`` = block-wide
    # ``element_type`` used for every element. Otherwise a list of length
    # ``num_elements`` where each entry is "singlet" or "doublet".
    per_element_types: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


# Squeeze-axis label → ghostlight.CylinderAxis int. The ghostlight convention
# stores the AXIS OF THE CYLINDER (i.e. the flat direction) as
# ``cyl_axis``, so power lives on the perpendicular axis:
#   * CYL_AXIS_X → cylinder flat along X → curvature/power in Y
#   * CYL_AXIS_Y → cylinder flat along Y → curvature/power in X
# The dialog labels talk about which SENSOR axis gets squeezed, so a
# horizontal (X-axis) squeeze needs power in X, which means
# ``CYL_AXIS_Y``. Getting this wrong silently squeezes the wrong axis.
_AXIS_LABEL_TO_INT = {
    "X (squeeze horizontal)": int(ghostlight.CylinderAxis.AXIS_Y),
    "Y (squeeze vertical)":   int(ghostlight.CylinderAxis.AXIS_X),
}
# Same enum but stored as a plain int here so the "Same as block" sentinel
# (None) doesn't clash with any real enum value.
_ADVANCED_AXIS_LABEL_TO_VALUE: list[tuple[str, Optional[int]]] = [
    ("Same as block",         None),
    ("X (horizontal)",        int(ghostlight.CylinderAxis.AXIS_Y)),
    ("Y (vertical)",          int(ghostlight.CylinderAxis.AXIS_X)),
]

_TOPOLOGY_LABEL_TO_KEY = {
    "Galilean (compact, no intermediate image)": "galilean",
    "Keplerian (longer, X inverted)":            "keplerian",
}

_ELEMENT_TYPE_LABEL_TO_KEY = {
    "Singlets": "singlet",
    "Doublets (chromatically corrected)": "doublet",
}

_PER_ELEMENT_LABELS = ["Singlet", "Doublet"]


class AnamorphicSetupDialog(QDialog):
    """Modal setup dialog for the anamorphic-block wizard.

    Usage::

        dlg = AnamorphicSetupDialog(parent=main_window)
        if dlg.exec() == QDialog.Accepted:
            spec = dlg.spec()
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Anamorphic Front Block")
        self.setModal(True)

        self._build_ui()
        self._on_element_type_changed()
        self._on_num_elements_changed()
        self.resize(460, 520)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QLabel(
            "Build a front cylindrical block onto the existing lens and "
            "optimise it toward the requested squeeze ratio.",
            self,
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #888;")
        outer.addWidget(header)

        # --------------------------------------------------------------
        # Basic section
        # --------------------------------------------------------------
        basic = QGroupBox("Block", self)
        form = QFormLayout(basic)
        form.setLabelAlignment(Qt.AlignRight)

        self._sp_squeeze = MathDoubleSpinBox(basic)
        self._sp_squeeze.setDecimals(2)
        self._sp_squeeze.setRange(1.05, 4.00)
        self._sp_squeeze.setSingleStep(0.05)
        self._sp_squeeze.setValue(2.00)
        self._sp_squeeze.setSuffix("×")
        form.addRow("Squeeze factor:", self._sp_squeeze)

        self._sp_num_elements = MathSpinBox(basic)
        self._sp_num_elements.setRange(2, 4)
        self._sp_num_elements.setValue(2)
        self._sp_num_elements.valueChanged.connect(self._on_num_elements_changed)
        form.addRow("Number of elements:", self._sp_num_elements)

        self._cb_axis = QComboBox(basic)
        for label in _AXIS_LABEL_TO_INT:
            self._cb_axis.addItem(label)
        form.addRow("Block axis:", self._cb_axis)

        self._cb_topology = QComboBox(basic)
        for label in _TOPOLOGY_LABEL_TO_KEY:
            self._cb_topology.addItem(label)
        form.addRow("Front topology:", self._cb_topology)

        self._cb_element_type = QComboBox(basic)
        for label in _ELEMENT_TYPE_LABEL_TO_KEY:
            self._cb_element_type.addItem(label)
        self._cb_element_type.setCurrentIndex(1)  # doublets by default
        self._cb_element_type.currentIndexChanged.connect(
            self._on_element_type_changed
        )
        form.addRow("Element type:", self._cb_element_type)

        self._sp_gap = MathDoubleSpinBox(basic)
        self._sp_gap.setDecimals(2)
        self._sp_gap.setRange(1.0, 200.0)
        self._sp_gap.setSingleStep(1.0)
        self._sp_gap.setValue(20.0)
        self._sp_gap.setSuffix(" mm")
        form.addRow("Gap to base lens:", self._sp_gap)

        outer.addWidget(basic)

        # --------------------------------------------------------------
        # Advanced section (collapsed by default)
        # --------------------------------------------------------------
        self._advanced_group = QGroupBox("Advanced", self)
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        self._advanced_group.toggled.connect(self._on_advanced_toggled)
        adv_layout = QFormLayout(self._advanced_group)
        adv_layout.setLabelAlignment(Qt.AlignRight)

        self._cb_front_axis = QComboBox(self._advanced_group)
        for label, _val in _ADVANCED_AXIS_LABEL_TO_VALUE:
            self._cb_front_axis.addItem(label)
        adv_layout.addRow("Front element axis:", self._cb_front_axis)

        self._cb_crown_glass = QComboBox(self._advanced_group)
        self._cb_crown_glass.setEditable(True)
        self._cb_crown_glass.addItems([DEFAULT_GLASS_KEY, "Schott_N-BK10", "Schott_N-SK16"])
        self._cb_crown_glass.setCurrentText(DEFAULT_GLASS_KEY)
        adv_layout.addRow("Crown glass:", self._cb_crown_glass)

        self._cb_flint_glass = QComboBox(self._advanced_group)
        self._cb_flint_glass.setEditable(True)
        self._cb_flint_glass.addItems([DEFAULT_FLINT_KEY, "Schott_SF2", "Schott_N-SF2"])
        self._cb_flint_glass.setCurrentText(DEFAULT_FLINT_KEY)
        adv_layout.addRow("Flint glass:", self._cb_flint_glass)

        # Per-element type row — rebuilt when num_elements or block-wide
        # element type changes. The container is a QFrame so we can wipe
        # and repopulate its layout without leaking widgets across rebuilds.
        self._per_elem_container = QFrame(self._advanced_group)
        self._per_elem_layout = QHBoxLayout(self._per_elem_container)
        self._per_elem_layout.setContentsMargins(0, 0, 0, 0)
        self._per_elem_layout.setSpacing(4)
        self._per_elem_combos: list[QComboBox] = []
        adv_layout.addRow("Per-element type:", self._per_elem_container)

        self._chk_use_per_elem = QCheckBox(
            "Override per element (otherwise block-wide type is used)",
            self._advanced_group,
        )
        self._chk_use_per_elem.setChecked(False)
        self._chk_use_per_elem.toggled.connect(self._on_use_per_elem_toggled)
        adv_layout.addRow("", self._chk_use_per_elem)

        outer.addWidget(self._advanced_group)

        # --------------------------------------------------------------
        # Standard OK / Cancel
        # --------------------------------------------------------------
        outer.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self,
        )
        buttons.button(QDialogButtonBox.Ok).setText("Build && Optimise")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Initial state: advanced hidden, per-element controls hidden.
        self._sync_advanced_child_enabled()

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_advanced_toggled(self, _checked: bool) -> None:
        # QGroupBox with setCheckable(True) already gates enabled state on
        # its children — we don't need to hide them explicitly.
        self._sync_advanced_child_enabled()

    def _on_element_type_changed(self) -> None:
        is_doublet = self._cb_element_type.currentIndex() == 1
        # Flint combo is only meaningful when the block has at least one
        # doublet element. Toggle its enabled state so the intent is clear.
        self._cb_flint_glass.setEnabled(
            is_doublet or self._chk_use_per_elem.isChecked()
        )
        # Also rebuild per-elem combos so their default reflects the new
        # block-wide type when the override list is untouched.
        self._rebuild_per_elem_combos()

    def _on_num_elements_changed(self) -> None:
        self._rebuild_per_elem_combos()

    def _on_use_per_elem_toggled(self, _checked: bool) -> None:
        self._sync_advanced_child_enabled()
        self._on_element_type_changed()

    def _sync_advanced_child_enabled(self) -> None:
        override_on = (
            self._advanced_group.isChecked()
            and self._chk_use_per_elem.isChecked()
        )
        self._per_elem_container.setEnabled(override_on)

    def _rebuild_per_elem_combos(self) -> None:
        target_n = int(self._sp_num_elements.value())
        default_key = _current_element_type_key(self._cb_element_type)
        default_label = "Doublet" if default_key == "doublet" else "Singlet"

        # Preserve any user-picked values from the previous combo list —
        # if the count grew we extend with defaults; if it shrank we drop
        # the tail values.
        existing_values = [c.currentText() for c in self._per_elem_combos]

        while self._per_elem_layout.count():
            item = self._per_elem_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._per_elem_combos = []

        for i in range(target_n):
            cb = QComboBox(self._per_elem_container)
            cb.addItems(_PER_ELEMENT_LABELS)
            value = existing_values[i] if i < len(existing_values) else default_label
            if value not in _PER_ELEMENT_LABELS:
                value = default_label
            cb.setCurrentText(value)
            self._per_elem_layout.addWidget(QLabel(f"E{i + 1}:", self._per_elem_container))
            self._per_elem_layout.addWidget(cb)
            self._per_elem_combos.append(cb)
        self._per_elem_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def spec(self) -> AnamorphicSpec:
        """Read the current UI state as an :class:`AnamorphicSpec`.

        Safe to call only after :meth:`exec` returned ``Accepted`` (or in
        tests where the caller sets values directly).
        """
        block_axis = _AXIS_LABEL_TO_INT[self._cb_axis.currentText()]

        # Front-axis override sits inside advanced. When advanced is
        # unchecked we ignore the combo's value — the user hasn't opted
        # into that override, so the front matches the block.
        front_axis: Optional[int] = None
        if self._advanced_group.isChecked():
            label = self._cb_front_axis.currentText()
            for lab, val in _ADVANCED_AXIS_LABEL_TO_VALUE:
                if lab == label:
                    front_axis = val
                    break

        # Per-element types only when advanced is on AND the checkbox is
        # ticked. Otherwise all elements use the block-wide type.
        per_element: Optional[list[str]] = None
        if (
            self._advanced_group.isChecked()
            and self._chk_use_per_elem.isChecked()
            and self._per_elem_combos
        ):
            per_element = []
            for cb in self._per_elem_combos:
                text = cb.currentText()
                per_element.append("doublet" if text == "Doublet" else "singlet")

        crown = self._cb_crown_glass.currentText().strip() or DEFAULT_GLASS_KEY
        flint = self._cb_flint_glass.currentText().strip() or DEFAULT_FLINT_KEY

        return AnamorphicSpec(
            squeeze_factor=float(self._sp_squeeze.value()),
            num_elements=int(self._sp_num_elements.value()),
            block_axis=block_axis,
            topology=_TOPOLOGY_LABEL_TO_KEY[self._cb_topology.currentText()],
            element_type=_current_element_type_key(self._cb_element_type),
            crown_glass=crown,
            flint_glass=flint,
            front_gap_mm=float(self._sp_gap.value()),
            front_axis_override=front_axis,
            per_element_types=per_element,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_element_type_key(cb: QComboBox) -> str:
    label = cb.currentText()
    return _ELEMENT_TYPE_LABEL_TO_KEY.get(label, "doublet")
