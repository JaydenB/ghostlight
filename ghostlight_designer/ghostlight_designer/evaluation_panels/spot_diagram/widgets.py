"""Spec-editor widgets for the spot diagram panel.

Compact sidebar with one form per spec field: wavelengths (comma list),
fields (table of (tx_deg, ty_deg)), rings, fans, pupil radius, defocus
offsets, plot extent. Edits emit :sig:`specChanged` so the panel body
can recompute with the new spec.

Keeping the widget dumb — no project access, no compute. The body owns
the spec, hands it to the widget on construct, and re-renders when the
widget signals a change.
"""
from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...math_spinbox import MathDoubleSpinBox, MathSpinBox
from .spec import (
    DEFAULT_DEFOCUS_OFFSETS_MM,
    DEFAULT_FIELDS_DEG,
    DEFAULT_WAVELENGTHS_NM,
    SpotDiagramSpec,
)


def _format_wavelengths(values: Tuple[float, ...]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _parse_wavelengths(text: str) -> Tuple[float, ...]:
    """Parse a comma- (or whitespace-) separated list of floats.

    Tolerant: empty strings and non-numeric tokens are skipped rather
    than raising — the user is mid-edit half the time.
    """
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


def _format_fields(values: Tuple[Tuple[float, float], ...]) -> str:
    return "; ".join(f"{fx:g},{fy:g}" for (fx, fy) in values)


def _parse_fields(text: str) -> Tuple[Tuple[float, float], ...]:
    """Parse semicolon-separated ``tx,ty`` pairs.

    Same tolerance as wavelengths — empty/bad pairs are skipped.
    """
    out: List[Tuple[float, float]] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            continue
        try:
            tx = float(parts[0])
            ty = float(parts[1])
        except ValueError:
            continue
        out.append((tx, ty))
    return tuple(out)


def _format_defocus(values: Tuple[float, ...]) -> str:
    return ", ".join(f"{v:g}" for v in values)


def _parse_defocus(text: str) -> Tuple[float, ...]:
    out: List[float] = []
    for tok in text.replace(";", ",").replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return tuple(out)


class SpotDiagramSpecEditor(QWidget):
    """Sidebar widget editing one :class:`SpotDiagramSpec`.

    Emits :sig:`specChanged` whenever any field is edited and the result
    is a different spec. The body wires this signal up to its
    ``request_refresh()`` so a wavelength tweak debounces a refresh just
    like a lens edit does.
    """

    specChanged = Signal(object)  # emits SpotDiagramSpec

    def __init__(
        self,
        initial: SpotDiagramSpec,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = initial.clamp()

        self._wavelengths_edit = QLineEdit(_format_wavelengths(self._spec.wavelengths_nm))
        self._wavelengths_edit.setToolTip(
            "Comma-separated wavelengths in nm. Each contributes one "
            "colour-coded point per pupil sample."
        )
        self._fields_edit = QLineEdit(_format_fields(self._spec.fields_deg))
        self._fields_edit.setToolTip(
            "Semicolon-separated tilts ``tx,ty`` in degrees. One spot "
            "subplot per field."
        )
        self._rings_spin = MathSpinBox()
        self._rings_spin.setRange(0, 32)
        self._rings_spin.setValue(int(self._spec.rings))
        self._rings_spin.setToolTip("Number of concentric pupil rings (0 = axial ray only).")

        self._fans_spin = MathSpinBox()
        self._fans_spin.setRange(1, 64)
        self._fans_spin.setValue(int(self._spec.fans))
        self._fans_spin.setToolTip("Number of radial fans per ring.")

        self._pupil_spin = MathDoubleSpinBox()
        self._pupil_spin.setRange(0.0, 1000.0)
        self._pupil_spin.setDecimals(3)
        self._pupil_spin.setSingleStep(0.5)
        self._pupil_spin.setValue(float(self._spec.pupil_radius_mm))
        self._pupil_spin.setSuffix(" mm")
        self._pupil_spin.setSpecialValueText("auto")
        self._pupil_spin.setToolTip(
            "Entrance pupil radius. 0 = auto (use the front surface's "
            "semi-aperture)."
        )

        self._defocus_edit = QLineEdit(_format_defocus(self._spec.defocus_offsets_mm))
        self._defocus_edit.setToolTip(
            "Comma-separated defocus offsets in mm relative to the "
            "Gaussian image plane. One column of subplots per value."
        )

        self._extent_spin = MathDoubleSpinBox()
        # 0 = auto (per-field scale from data). Positive = fixed extent
        # in mm applied to every row.
        self._extent_spin.setRange(0.0, 100.0)
        self._extent_spin.setDecimals(3)
        self._extent_spin.setSingleStep(0.05)
        self._extent_spin.setValue(float(self._spec.plot_half_extent_mm))
        self._extent_spin.setSuffix(" mm")
        self._extent_spin.setSpecialValueText("auto")
        self._extent_spin.setToolTip(
            "Subplot half-width. 0 = auto (per-field, sized to fit the "
            "actual bundle spread). Positive = fixed scale applied to "
            "every field row."
        )

        self._defaults_button = QPushButton("Reset to Defaults")
        self._defaults_button.clicked.connect(self._on_reset_defaults)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(4)
        form.addRow("Wavelengths (nm)", self._wavelengths_edit)
        form.addRow("Fields (tx,ty deg)", self._fields_edit)
        form.addRow("Rings", self._rings_spin)
        form.addRow("Fans", self._fans_spin)
        form.addRow("Pupil radius", self._pupil_spin)
        form.addRow("Defocus (mm)", self._defocus_edit)
        form.addRow("Plot extent", self._extent_spin)

        group = QGroupBox("Spot Diagram", self)
        gl = QVBoxLayout(group)
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addLayout(form)
        gl.addWidget(self._defaults_button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(group)
        outer.addStretch(1)

        # Defer to editingFinished so users mid-typing in a line edit
        # don't recompute on every keystroke. SpinBoxes use valueChanged
        # — their up/down arrows produce one event per step which is
        # already what we want.
        self._wavelengths_edit.editingFinished.connect(self._collect_and_emit)
        self._fields_edit.editingFinished.connect(self._collect_and_emit)
        self._defocus_edit.editingFinished.connect(self._collect_and_emit)
        self._rings_spin.valueChanged.connect(self._collect_and_emit)
        self._fans_spin.valueChanged.connect(self._collect_and_emit)
        self._pupil_spin.valueChanged.connect(self._collect_and_emit)
        self._extent_spin.valueChanged.connect(self._collect_and_emit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def spec(self) -> SpotDiagramSpec:
        return self._spec

    def set_spec(self, new_spec: SpotDiagramSpec, *, emit: bool = False) -> None:
        """Replace the editor's contents with ``new_spec``.

        Used by the body when "Sync from System Setup" copies new values
        in. ``emit=False`` (the default) suppresses the change signal so
        the caller can sequence its own refresh.
        """
        new_spec = new_spec.clamp()
        self._spec = new_spec
        # Avoid re-emitting from the editing-finished signals during the
        # block — Qt edits trigger editingFinished even if focus didn't
        # change.
        blockers = [
            self._wavelengths_edit, self._fields_edit, self._rings_spin,
            self._fans_spin, self._pupil_spin, self._defocus_edit,
            self._extent_spin,
        ]
        old_blocks = [w.blockSignals(True) for w in blockers]
        try:
            self._wavelengths_edit.setText(_format_wavelengths(new_spec.wavelengths_nm))
            self._fields_edit.setText(_format_fields(new_spec.fields_deg))
            self._rings_spin.setValue(int(new_spec.rings))
            self._fans_spin.setValue(int(new_spec.fans))
            self._pupil_spin.setValue(float(new_spec.pupil_radius_mm))
            self._defocus_edit.setText(_format_defocus(new_spec.defocus_offsets_mm))
            self._extent_spin.setValue(float(new_spec.plot_half_extent_mm))
        finally:
            for w, old in zip(blockers, old_blocks):
                w.blockSignals(old)
        if emit:
            self.specChanged.emit(self._spec)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_and_emit(self, *_args) -> None:
        new_spec = SpotDiagramSpec(
            wavelengths_nm=_parse_wavelengths(self._wavelengths_edit.text())
                or DEFAULT_WAVELENGTHS_NM,
            fields_deg=_parse_fields(self._fields_edit.text())
                or DEFAULT_FIELDS_DEG,
            rings=int(self._rings_spin.value()),
            fans=int(self._fans_spin.value()),
            pupil_radius_mm=float(self._pupil_spin.value()),
            defocus_offsets_mm=_parse_defocus(self._defocus_edit.text())
                or DEFAULT_DEFOCUS_OFFSETS_MM,
            plot_half_extent_mm=float(self._extent_spin.value()),
        ).clamp()
        if new_spec == self._spec:
            return
        self._spec = new_spec
        self.specChanged.emit(self._spec)

    def _on_reset_defaults(self) -> None:
        self.set_spec(SpotDiagramSpec(), emit=True)
