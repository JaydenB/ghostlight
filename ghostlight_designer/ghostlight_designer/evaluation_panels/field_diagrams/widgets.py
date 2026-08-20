"""Spec-editor widget for the field diagrams panel.

Mirrors :class:`SpotDiagramSpecEditor`'s shape — a compact sidebar
emitting :sig:`specChanged` whenever a control changes — so users move
between the two evaluation panels without re-learning the layout.
"""
from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...math_spinbox import MathDoubleSpinBox, MathSpinBox
from .spec import DEFAULT_WAVELENGTHS_NM, FieldDiagramSpec


def _format_wavelengths(values: Tuple[float, ...]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _parse_wavelengths(text: str) -> Tuple[float, ...]:
    out: List[float] = []
    for tok in text.replace(";", ",").replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if v > 0.0:
            out.append(v)
    return tuple(out)


class FieldDiagramSpecEditor(QWidget):
    """Sidebar widget editing one :class:`FieldDiagramSpec`."""

    specChanged = Signal(object)  # emits FieldDiagramSpec

    def __init__(
        self,
        initial: FieldDiagramSpec,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = initial.clamp()

        self._max_field_spin = MathDoubleSpinBox()
        self._max_field_spin.setRange(0.1, 90.0)
        self._max_field_spin.setDecimals(2)
        self._max_field_spin.setSingleStep(1.0)
        self._max_field_spin.setSuffix(" °")
        self._max_field_spin.setValue(float(self._spec.max_field_deg))
        self._max_field_spin.setToolTip("Largest field angle plotted on the Y axis.")

        self._samples_spin = MathSpinBox()
        self._samples_spin.setRange(2, 200)
        self._samples_spin.setValue(int(self._spec.field_samples))
        self._samples_spin.setToolTip(
            "Number of field-angle samples (inclusive of 0 and max)."
        )

        self._wavelengths_edit = QLineEdit(_format_wavelengths(self._spec.wavelengths_nm))
        self._wavelengths_edit.setToolTip(
            "Comma-separated wavelengths in nm. Astigmatism / distortion "
            "trace at the primary; lateral chromatic uses every wavelength."
        )

        self._primary_spin = MathSpinBox()
        self._primary_spin.setRange(0, max(0, len(self._spec.wavelengths_nm) - 1))
        self._primary_spin.setValue(int(self._spec.primary_wavelength_index))
        self._primary_spin.setToolTip(
            "Index into the wavelengths list that drives astigmatism / distortion."
        )

        self._pupil_spin = MathDoubleSpinBox()
        self._pupil_spin.setRange(0.0, 1000.0)
        self._pupil_spin.setDecimals(3)
        self._pupil_spin.setSingleStep(0.5)
        self._pupil_spin.setSuffix(" mm")
        self._pupil_spin.setSpecialValueText("auto")
        self._pupil_spin.setValue(float(self._spec.pupil_radius_mm))
        self._pupil_spin.setToolTip(
            "Pupil radius used for the sagittal/tangential fans. "
            "0 = auto (from front surface semi-aperture)."
        )

        self._rays_spin = MathSpinBox()
        self._rays_spin.setRange(3, 51)
        self._rays_spin.setValue(int(self._spec.rays_per_fan))
        self._rays_spin.setToolTip("Rays per sagittal/tangential fan.")

        # Per-diagram visibility toggles.
        self._astig_check = QCheckBox("Astigmatism")
        self._astig_check.setChecked(self._spec.show_astigmatism)
        self._dist_check = QCheckBox("Distortion")
        self._dist_check.setChecked(self._spec.show_distortion)
        self._chrom_check = QCheckBox("Lateral chromatic")
        self._chrom_check.setChecked(self._spec.show_lateral_chromatic)

        self._defaults_button = QPushButton("Reset to Defaults")
        self._defaults_button.clicked.connect(self._on_reset_defaults)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(4)
        form.addRow("Max field", self._max_field_spin)
        form.addRow("Field samples", self._samples_spin)
        form.addRow("Wavelengths (nm)", self._wavelengths_edit)
        form.addRow("Primary λ index", self._primary_spin)
        form.addRow("Pupil radius", self._pupil_spin)
        form.addRow("Rays / fan", self._rays_spin)

        visibility = QGroupBox("Show diagrams", self)
        vis_layout = QVBoxLayout(visibility)
        vis_layout.setContentsMargins(4, 4, 4, 4)
        vis_layout.setSpacing(2)
        vis_layout.addWidget(self._astig_check)
        vis_layout.addWidget(self._dist_check)
        vis_layout.addWidget(self._chrom_check)

        group = QGroupBox("Field Diagrams", self)
        gl = QVBoxLayout(group)
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addLayout(form)
        gl.addWidget(visibility)
        gl.addWidget(self._defaults_button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)

        self._wavelengths_edit.editingFinished.connect(self._collect_and_emit)
        self._max_field_spin.valueChanged.connect(self._collect_and_emit)
        self._samples_spin.valueChanged.connect(self._collect_and_emit)
        self._primary_spin.valueChanged.connect(self._collect_and_emit)
        self._pupil_spin.valueChanged.connect(self._collect_and_emit)
        self._rays_spin.valueChanged.connect(self._collect_and_emit)
        self._astig_check.toggled.connect(self._collect_and_emit)
        self._dist_check.toggled.connect(self._collect_and_emit)
        self._chrom_check.toggled.connect(self._collect_and_emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def spec(self) -> FieldDiagramSpec:
        return self._spec

    def set_spec(self, new_spec: FieldDiagramSpec, *, emit: bool = False) -> None:
        new_spec = new_spec.clamp()
        self._spec = new_spec
        blockers = [
            self._max_field_spin, self._samples_spin, self._wavelengths_edit,
            self._primary_spin, self._pupil_spin, self._rays_spin,
            self._astig_check, self._dist_check, self._chrom_check,
        ]
        old = [w.blockSignals(True) for w in blockers]
        try:
            self._max_field_spin.setValue(float(new_spec.max_field_deg))
            self._samples_spin.setValue(int(new_spec.field_samples))
            self._wavelengths_edit.setText(_format_wavelengths(new_spec.wavelengths_nm))
            # Primary-index range tracks the wavelengths list length.
            self._primary_spin.setRange(0, max(0, len(new_spec.wavelengths_nm) - 1))
            self._primary_spin.setValue(int(new_spec.primary_wavelength_index))
            self._pupil_spin.setValue(float(new_spec.pupil_radius_mm))
            self._rays_spin.setValue(int(new_spec.rays_per_fan))
            self._astig_check.setChecked(bool(new_spec.show_astigmatism))
            self._dist_check.setChecked(bool(new_spec.show_distortion))
            self._chrom_check.setChecked(bool(new_spec.show_lateral_chromatic))
        finally:
            for w, b in zip(blockers, old):
                w.blockSignals(b)
        if emit:
            self.specChanged.emit(self._spec)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_and_emit(self, *_args) -> None:
        wavelengths = _parse_wavelengths(self._wavelengths_edit.text()) or DEFAULT_WAVELENGTHS_NM
        # Re-clamp the primary index in case the user shrank the wavelength
        # list to below the previous index.
        max_primary = max(0, len(wavelengths) - 1)
        primary = min(int(self._primary_spin.value()), max_primary)
        if primary != self._primary_spin.value():
            blocked = self._primary_spin.blockSignals(True)
            self._primary_spin.setRange(0, max_primary)
            self._primary_spin.setValue(primary)
            self._primary_spin.blockSignals(blocked)
        elif self._primary_spin.maximum() != max_primary:
            blocked = self._primary_spin.blockSignals(True)
            self._primary_spin.setRange(0, max_primary)
            self._primary_spin.blockSignals(blocked)

        new_spec = FieldDiagramSpec(
            max_field_deg=float(self._max_field_spin.value()),
            field_samples=int(self._samples_spin.value()),
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary,
            pupil_radius_mm=float(self._pupil_spin.value()),
            rays_per_fan=int(self._rays_spin.value()),
            show_astigmatism=bool(self._astig_check.isChecked()),
            show_distortion=bool(self._dist_check.isChecked()),
            show_lateral_chromatic=bool(self._chrom_check.isChecked()),
        ).clamp()
        if new_spec == self._spec:
            return
        self._spec = new_spec
        self.specChanged.emit(self._spec)

    def _on_reset_defaults(self) -> None:
        self.set_spec(FieldDiagramSpec(), emit=True)
