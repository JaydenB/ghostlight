"""Spec-editor widget for the Seidel bar-chart panel.

Sidebar form + per-chart visibility checkboxes + Reset button. Emits
:sig:`specChanged` on any control change.
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
from .spec import DEFAULT_WAVELENGTHS_NM, SeidelSpec


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


class SeidelSpecEditor(QWidget):
    """Sidebar widget editing one :class:`SeidelSpec`."""

    specChanged = Signal(object)  # emits SeidelSpec

    def __init__(
        self,
        initial: SeidelSpec,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = initial.clamp()

        self._field_spin = MathDoubleSpinBox()
        self._field_spin.setRange(0.0, 60.0)
        self._field_spin.setDecimals(2)
        self._field_spin.setSingleStep(0.5)
        self._field_spin.setSuffix(" °")
        self._field_spin.setValue(float(self._spec.field_deg))
        self._field_spin.setToolTip(
            "Field angle for the chief-ray paraxial trace. Distortion "
            "and coma contributions scale with this — pick a value "
            "representative of the lens' working field."
        )

        self._wavelengths_edit = QLineEdit(_format_wavelengths(self._spec.wavelengths_nm))
        self._wavelengths_edit.setToolTip(
            "Comma-separated wavelengths in nm. Monochromatic sums use "
            "the primary; chromatic sums bracket using the lowest and "
            "highest in the list (typically F and C)."
        )

        self._primary_spin = MathSpinBox()
        self._primary_spin.setRange(0, max(0, len(self._spec.wavelengths_nm) - 1))
        self._primary_spin.setValue(int(self._spec.primary_wavelength_index))
        self._primary_spin.setToolTip(
            "Index into the wavelengths list that drives the "
            "monochromatic (S_I…S_V) trace."
        )

        self._pupil_spin = MathDoubleSpinBox()
        self._pupil_spin.setRange(0.0, 1000.0)
        self._pupil_spin.setDecimals(3)
        self._pupil_spin.setSingleStep(0.5)
        self._pupil_spin.setSuffix(" mm")
        self._pupil_spin.setSpecialValueText("auto")
        self._pupil_spin.setValue(float(self._spec.pupil_radius_mm))
        self._pupil_spin.setToolTip(
            "Marginal-ray height at the aperture stop. "
            "0 = auto (from the stop's semi-aperture)."
        )

        self._sph_check = QCheckBox("S_I  spherical")
        self._coma_check = QCheckBox("S_II  coma")
        self._astig_check = QCheckBox("S_III  astigmatism")
        self._petz_check = QCheckBox("S_IV  Petzval")
        self._dist_check = QCheckBox("S_V  distortion")
        self._ax_color_check = QCheckBox("C_I  axial colour")
        self._lat_color_check = QCheckBox("C_II  lateral colour")
        self._sph_check.setChecked(self._spec.show_spherical)
        self._coma_check.setChecked(self._spec.show_coma)
        self._astig_check.setChecked(self._spec.show_astigmatism)
        self._petz_check.setChecked(self._spec.show_petzval)
        self._dist_check.setChecked(self._spec.show_distortion)
        self._ax_color_check.setChecked(self._spec.show_axial_color)
        self._lat_color_check.setChecked(self._spec.show_lateral_color)

        self._defaults_button = QPushButton("Reset to Defaults")
        self._defaults_button.clicked.connect(self._on_reset_defaults)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(4)
        form.addRow("Field", self._field_spin)
        form.addRow("Wavelengths (nm)", self._wavelengths_edit)
        form.addRow("Primary λ index", self._primary_spin)
        form.addRow("Pupil radius", self._pupil_spin)

        visibility = QGroupBox("Show charts", self)
        vis_layout = QVBoxLayout(visibility)
        vis_layout.setContentsMargins(4, 4, 4, 4)
        vis_layout.setSpacing(2)
        vis_layout.addWidget(self._sph_check)
        vis_layout.addWidget(self._coma_check)
        vis_layout.addWidget(self._astig_check)
        vis_layout.addWidget(self._petz_check)
        vis_layout.addWidget(self._dist_check)
        vis_layout.addWidget(self._ax_color_check)
        vis_layout.addWidget(self._lat_color_check)

        group = QGroupBox("Seidel Bar Chart", self)
        gl = QVBoxLayout(group)
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addLayout(form)
        gl.addWidget(visibility)
        gl.addWidget(self._defaults_button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)

        self._field_spin.valueChanged.connect(self._collect_and_emit)
        self._wavelengths_edit.editingFinished.connect(self._collect_and_emit)
        self._primary_spin.valueChanged.connect(self._collect_and_emit)
        self._pupil_spin.valueChanged.connect(self._collect_and_emit)
        for cb in (
            self._sph_check, self._coma_check, self._astig_check,
            self._petz_check, self._dist_check,
            self._ax_color_check, self._lat_color_check,
        ):
            cb.toggled.connect(self._collect_and_emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def spec(self) -> SeidelSpec:
        return self._spec

    def set_spec(self, new_spec: SeidelSpec, *, emit: bool = False) -> None:
        new_spec = new_spec.clamp()
        self._spec = new_spec
        blockers = [
            self._field_spin, self._wavelengths_edit, self._primary_spin,
            self._pupil_spin,
            self._sph_check, self._coma_check, self._astig_check,
            self._petz_check, self._dist_check,
            self._ax_color_check, self._lat_color_check,
        ]
        old = [w.blockSignals(True) for w in blockers]
        try:
            self._field_spin.setValue(float(new_spec.field_deg))
            self._wavelengths_edit.setText(_format_wavelengths(new_spec.wavelengths_nm))
            self._primary_spin.setRange(0, max(0, len(new_spec.wavelengths_nm) - 1))
            self._primary_spin.setValue(int(new_spec.primary_wavelength_index))
            self._pupil_spin.setValue(float(new_spec.pupil_radius_mm))
            self._sph_check.setChecked(bool(new_spec.show_spherical))
            self._coma_check.setChecked(bool(new_spec.show_coma))
            self._astig_check.setChecked(bool(new_spec.show_astigmatism))
            self._petz_check.setChecked(bool(new_spec.show_petzval))
            self._dist_check.setChecked(bool(new_spec.show_distortion))
            self._ax_color_check.setChecked(bool(new_spec.show_axial_color))
            self._lat_color_check.setChecked(bool(new_spec.show_lateral_color))
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

        new_spec = SeidelSpec(
            field_deg=float(self._field_spin.value()),
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary,
            pupil_radius_mm=float(self._pupil_spin.value()),
            show_spherical=bool(self._sph_check.isChecked()),
            show_coma=bool(self._coma_check.isChecked()),
            show_astigmatism=bool(self._astig_check.isChecked()),
            show_petzval=bool(self._petz_check.isChecked()),
            show_distortion=bool(self._dist_check.isChecked()),
            show_axial_color=bool(self._ax_color_check.isChecked()),
            show_lateral_color=bool(self._lat_color_check.isChecked()),
        ).clamp()
        if new_spec == self._spec:
            return
        self._spec = new_spec
        self.specChanged.emit(self._spec)

    def _on_reset_defaults(self) -> None:
        self.set_spec(SeidelSpec(), emit=True)
