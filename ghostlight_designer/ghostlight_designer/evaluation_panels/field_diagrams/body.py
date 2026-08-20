"""Body widget for the ``field_diagrams`` evaluation panel.

Mirrors :class:`SpotDiagramBody`'s shape — sidebar editor + canvas +
status — so users hop between evaluation panels without re-learning
the layout. Compute happens on a worker thread via the inherited
:class:`EvaluationPanelBody` machinery.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...project import Project
from ...settings import AppSettings
from ...system_setup_data import SystemSetup, Wavelength
from ..base import EvaluationPanelBody
from .canvas import FieldDiagramCanvas
from .compute import compute_field_diagrams, FieldDiagramResult
from .spec import DEFAULT_WAVELENGTHS_NM, FieldDiagramSpec
from .widgets import FieldDiagramSpecEditor

_log = logging.getLogger("ghostlight_designer.evaluation_panels.field_diagrams")


# A field-diagrams trace needs at least two lens surfaces.
MIN_SURFACES = 2


class FieldDiagramBody(EvaluationPanelBody):
    """Field-diagrams evaluation panel.

    Spec lives on the body (per-panel custom fields rather than mirroring
    System Setup). The View → Sync from System Setup menu action pulls
    wavelengths and a single largest field tilt from the project's
    first sequence.
    """

    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(project, settings, parent)

        self._spec: FieldDiagramSpec = FieldDiagramSpec()

        self._editor = FieldDiagramSpecEditor(self._spec, self)
        self._editor.specChanged.connect(self._on_spec_changed)
        self._editor.setMinimumWidth(240)
        self._editor.setMaximumWidth(360)

        self._canvas = FieldDiagramCanvas(self)

        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)
        main_row.addWidget(self._editor, 0)
        main_row.addWidget(self._canvas, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(main_row, 1)
        outer.addWidget(self.status_label)

        if not self._lens_eligible():
            self._canvas.clear("Load a lens with at least 2 surfaces.")
            self.set_status("Waiting for a lens.")
        else:
            self.set_status("Idle.")

    # ------------------------------------------------------------------
    # Public API used by the View menu
    # ------------------------------------------------------------------

    @property
    def spec(self) -> FieldDiagramSpec:
        return self._spec

    @property
    def settings_visible(self) -> bool:
        return not self._editor.isHidden()

    def set_settings_visible(self, visible: bool) -> None:
        self._editor.setVisible(bool(visible))

    def reset_spec_to_defaults(self) -> None:
        new_spec = FieldDiagramSpec()
        if new_spec == self._spec:
            self.set_status("Spec already at defaults.")
            return
        self._spec = new_spec
        self._editor.set_spec(new_spec, emit=False)
        self.set_status("Spec reset to defaults.")
        self.force_refresh_now()

    # ------------------------------------------------------------------
    # EvaluationPanelBody hooks
    # ------------------------------------------------------------------

    def compute(self):
        lens = self._project.system
        spec = self._spec
        if not self._lens_eligible_for(lens):
            return None
        return compute_field_diagrams(lens, spec)

    def apply_result(self, result) -> None:
        if result is None:
            self._canvas.clear("Lens has too few surfaces.")
            self.set_status("Lens has too few surfaces.")
            return
        assert isinstance(result, FieldDiagramResult)
        self._canvas.set_result(result)
        # Surface the headline numbers in the status so an artist sees
        # the values change even when the curve shapes are subtle.
        d_pct = result.distortion_pct
        d_pct_finite = d_pct[~_isnan(d_pct)]
        max_d_str = (
            f"{float(d_pct_finite[_argmax_abs(d_pct_finite)]):+.3f}%"
            if d_pct_finite.size else "n/a"
        )
        self.set_status(
            f"max field {result.field_angles_deg[-1]:.1f}°  ·  "
            f"D_max = {max_d_str}  ·  pupil R = {result.pupil_radius_mm:.2f} mm"
        )

    def apply_error(self, exc: BaseException) -> None:
        # Don't clear the canvas — leave the previous successful render
        # in place so the user can compare against the bad edit.
        super().apply_error(exc)

    def sync_from_system_setup(self) -> None:
        """Pull wavelengths and max field angle from the project's setup.

        Field-diagrams needs a *range* (0 → max_field_deg), not the
        discrete list of fields the spot diagram sucks in. We use the
        largest absolute Y tilt across the setup's fields as the max
        — that's the field a defocus / astigmatism / distortion plot
        wants to extend to.
        """
        setup: SystemSetup = self._project.system_setup
        if not setup.sequences:
            self.set_status("System Setup has no sequences to sync from.")
            return
        source = setup.sequences[0].source
        wavelengths = tuple(
            float(w.value_nm) for w in source.wavelengths.wavelengths
            if isinstance(w, Wavelength) and float(w.value_nm) > 0.0
        ) or DEFAULT_WAVELENGTHS_NM
        primary_idx = max(
            0, min(int(source.wavelengths.primary_index), len(wavelengths) - 1)
        )

        # Largest |tilt_y_deg| across the configured fields. Falls back
        # to the spec's current max if no fields are configured.
        max_field = self._spec.max_field_deg
        if source.fields:
            cand = max(
                abs(float(f.tilt_y_deg)) for f in source.fields
            )
            if cand > 0.0:
                max_field = cand

        new_spec = FieldDiagramSpec(
            max_field_deg=max_field,
            field_samples=self._spec.field_samples,
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary_idx,
            pupil_radius_mm=float(source.aperture_radius)
                if source.aperture_radius > 0.0
                else self._spec.pupil_radius_mm,
            rays_per_fan=self._spec.rays_per_fan,
            show_astigmatism=self._spec.show_astigmatism,
            show_distortion=self._spec.show_distortion,
            show_lateral_chromatic=self._spec.show_lateral_chromatic,
        ).clamp()

        if new_spec == self._spec:
            self.set_status("System Setup matches current spec — no changes.")
            return
        self._spec = new_spec
        self._editor.set_spec(new_spec, emit=False)
        self.set_status("Synced from System Setup.")

    # ------------------------------------------------------------------
    # Spec / lens lifecycle
    # ------------------------------------------------------------------

    def _on_spec_changed(self, new_spec: FieldDiagramSpec) -> None:
        if new_spec == self._spec:
            return
        self._spec = new_spec
        self.request_refresh()

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def _lens_eligible(self) -> bool:
        return self._lens_eligible_for(self._project.system)

    @staticmethod
    def _lens_eligible_for(lens) -> bool:
        try:
            return lens is not None and lens.num_surfaces() >= MIN_SURFACES
        except Exception:
            return False


def _isnan(arr):
    import numpy as np
    return np.isnan(arr)


def _argmax_abs(arr):
    import numpy as np
    return int(np.argmax(np.abs(arr)))
