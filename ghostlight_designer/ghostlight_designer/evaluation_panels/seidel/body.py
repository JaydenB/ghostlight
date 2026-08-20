"""Body widget for the Seidel bar-chart evaluation panel.

Sidebar editor + canvas + status, same shape as the other evaluation
panels. Compute runs on a worker thread via the inherited
:class:`EvaluationPanelBody` machinery, even though the paraxial
Seidel calculation is sub-millisecond — keeping the dispatch path
uniform avoids surprises when a panel does heavier work.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...project import Project
from ...settings import AppSettings
from ...system_setup_data import SystemSetup, Wavelength
from ..base import EvaluationPanelBody
from .canvas import SeidelCanvas
from .compute import SeidelResult, compute_seidel
from .spec import DEFAULT_WAVELENGTHS_NM, SeidelSpec
from .widgets import SeidelSpecEditor

_log = logging.getLogger("ghostlight_designer.evaluation_panels.seidel")


# Seidel needs at least one refracting surface plus the image plane.
MIN_SURFACES = 2


class SeidelBody(EvaluationPanelBody):
    """Per-surface Seidel sum panel.

    The Sync menu pulls wavelengths + primary index + pupil radius from
    the project's first sequence, and uses the largest absolute Y tilt
    among the configured fields as the chief-ray angle — same field
    convention as the field-diagrams panel.
    """

    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(project, settings, parent)

        self._spec: SeidelSpec = SeidelSpec()

        self._editor = SeidelSpecEditor(self._spec, self)
        self._editor.specChanged.connect(self._on_spec_changed)
        self._editor.setMinimumWidth(240)
        self._editor.setMaximumWidth(360)

        self._canvas = SeidelCanvas(self)

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
    def spec(self) -> SeidelSpec:
        return self._spec

    @property
    def settings_visible(self) -> bool:
        return not self._editor.isHidden()

    def set_settings_visible(self, visible: bool) -> None:
        self._editor.setVisible(bool(visible))

    def reset_spec_to_defaults(self) -> None:
        new_spec = SeidelSpec()
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
        return compute_seidel(lens, spec)

    def apply_result(self, result) -> None:
        if result is None:
            self._canvas.clear("Lens has too few surfaces.")
            self.set_status("Lens has too few surfaces.")
            return
        assert isinstance(result, SeidelResult)
        self._canvas.set_result(result)
        # Headline status — the spherical + Petzval sums are the two
        # most-watched system totals; surfacing them in the status keeps
        # the user looking at the bottom-of-panel even when the bars are
        # subtle.
        sums = result.sums
        self.set_status(
            f"ΣS_I = {sums['spherical']:+.3e}  ·  "
            f"ΣS_IV = {sums['petzval']:+.3e}  ·  "
            f"H = {result.lagrange_invariant:+.3f}"
        )

    def sync_from_system_setup(self) -> None:
        """Pull wavelengths + primary index + pupil + field from the setup.

        Field comes from the largest |tilt_y_deg| across the configured
        fields — matches the field-diagrams convention.
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

        field_deg = self._spec.field_deg
        if source.fields:
            cand = max(abs(float(f.tilt_y_deg)) for f in source.fields)
            if cand > 0.0:
                field_deg = cand

        new_spec = SeidelSpec(
            field_deg=field_deg,
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary_idx,
            pupil_radius_mm=float(source.aperture_radius)
                if source.aperture_radius > 0.0
                else self._spec.pupil_radius_mm,
            show_spherical=self._spec.show_spherical,
            show_coma=self._spec.show_coma,
            show_astigmatism=self._spec.show_astigmatism,
            show_petzval=self._spec.show_petzval,
            show_distortion=self._spec.show_distortion,
            show_axial_color=self._spec.show_axial_color,
            show_lateral_color=self._spec.show_lateral_color,
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

    def _on_spec_changed(self, new_spec: SeidelSpec) -> None:
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
