"""Body widget for the ``spot_diagram`` evaluation panel.

Layout mirrors the existing render panels: the matplotlib canvas takes
the main area, the status label sits at the bottom, and user-facing
actions are exposed through the panel's View menu (see :mod:`menus`).

The spec-editor sidebar lives to the left of the canvas and can be
hidden/shown from the View menu so a packed workspace doesn't dedicate
screen space to controls the user isn't using.

**Scale lock.** The body keeps a per-field reference half-extent — the
auto-fit value computed on the first successful render — and replays
that scale on every subsequent render. Without this, each render
auto-rescales to fit its own data and a defocus-vs-focused lens edit
looks visually identical (the spot always fills ~77% of the subplot).
The user explicitly reset the lock via View → Auto-Fit Scale Now, and
the lock is also cleared on a new lens load / spec reset / sync.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...project import Project
from ...settings import AppSettings
from ...system_setup_data import SystemSetup, Wavelength
from ..base import EvaluationPanelBody
from .canvas import SpotDiagramCanvas, compute_auto_extents
from .compute import compute_spot_diagram, SpotResult
from .spec import (
    DEFAULT_FIELDS_DEG,
    DEFAULT_WAVELENGTHS_NM,
    SpotDiagramSpec,
)
from .widgets import SpotDiagramSpecEditor

_log = logging.getLogger("ghostlight_designer.evaluation_panels.spot_diagram")


# A spot diagram needs at least two surfaces to produce anything
# meaningful — same threshold the other render panels use.
MIN_SURFACES = 2


class SpotDiagramBody(EvaluationPanelBody):
    """Spot-diagram evaluation panel.

    The spec lives on this body, not on the project — see the package
    docstring for the per-panel-fields rationale. Use the View → Sync
    from System Setup menu action to pull the project's current
    wavelengths and fields into the spec.
    """

    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(project, settings, parent)

        # Spec lives here, owned by the body. The editor is a view onto it.
        self._spec: SpotDiagramSpec = SpotDiagramSpec()

        # Per-field scale lock. Populated from the first successful
        # render's auto-fit; reused on every subsequent render so a
        # lens edit visibly changes the spot SIZE rather than being
        # silently rescaled away. ``None`` = no lock yet → next render
        # auto-fits and captures the lock. Cleared on lens swap /
        # spec reset / sync / explicit "Auto-Fit Scale Now".
        self._reference_extents: Optional[Dict[Tuple[float, float], float]] = None

        self._editor = SpotDiagramSpecEditor(self._spec, self)
        self._editor.specChanged.connect(self._on_spec_changed)
        # The editor is sized for its content; cap the width so the
        # canvas dominates the panel area when both are visible.
        self._editor.setMinimumWidth(240)
        self._editor.setMaximumWidth(360)

        self._canvas = SpotDiagramCanvas(self)

        # Body row: editor on the left (toggleable), canvas on the right.
        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)
        main_row.addWidget(self._editor, 0)
        main_row.addWidget(self._canvas, 1)

        # Outer column: body row above, status label below. Matches the
        # primary / PSF panel layout (canvas-on-top, status-on-bottom).
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
    def spec(self) -> SpotDiagramSpec:
        return self._spec

    @property
    def settings_visible(self) -> bool:
        # ``isHidden()`` reports the explicit visibility *setting* — i.e.
        # whether the widget has been hidden by user action — independent
        # of whether the body itself is currently on screen. Using
        # ``isVisible()`` here would report False during construction
        # (before the panel is first shown) and confuse the menu's
        # checked state, since the user never asked for it to be hidden.
        return not self._editor.isHidden()

    def set_settings_visible(self, visible: bool) -> None:
        """Show/hide the spec-editor sidebar.

        When hidden the editor takes no horizontal space and the canvas
        expands to fill. Spec values are preserved either way — this is
        purely a layout toggle.
        """
        self._editor.setVisible(bool(visible))

    def reset_spec_to_defaults(self) -> None:
        """Replace the current spec with :class:`SpotDiagramSpec`'s defaults.

        Explicit user action — triggers an immediate refresh and drops
        the scale lock since the new spec may have different fields.
        """
        new_spec = SpotDiagramSpec()
        if new_spec == self._spec:
            self.set_status("Spec already at defaults.")
            return
        self._spec = new_spec
        self._editor.set_spec(new_spec, emit=False)
        self._reference_extents = None
        self.set_status("Spec reset to defaults.")
        self.force_refresh_now()

    def auto_fit_scale_now(self) -> None:
        """Drop the locked scale; the next render auto-fits per-field.

        Use this when the user has edited the lens so far from its
        original state that the locked scale no longer makes sense —
        either rays clip out of the subplots, or the locked scale is
        much larger than the new bundle and the spot has collapsed
        into a dot in the middle.
        """
        self._reference_extents = None
        self.set_status("Scale will auto-fit on next render.")
        self.force_refresh_now()

    # ------------------------------------------------------------------
    # EvaluationPanelBody hooks
    # ------------------------------------------------------------------

    def compute(self):
        # Read all inputs upfront — this runs on a worker thread, so
        # we capture the lens and spec here and never touch them later.
        lens = self._project.system
        spec = self._spec
        if not self._lens_eligible_for(lens):
            return None
        return compute_spot_diagram(lens, spec)

    def apply_result(self, result) -> None:
        if result is None:
            self._canvas.clear("Lens has too few surfaces.")
            self.set_status("Lens has too few surfaces.")
            return
        assert isinstance(result, SpotResult)
        # Capture the FIRST successful render's per-field auto-fit as
        # the reference scale. Subsequent renders use these locked
        # values so lens-edit-driven size changes are actually visible
        # — without the lock, each render re-auto-fits and the spot
        # appears to never change.
        if self._reference_extents is None and result.fields:
            self._reference_extents = compute_auto_extents(result)
        self._canvas.set_result(result, reference_extents=self._reference_extents)
        self.set_status(
            f"{len(result.fields)} fields × {len(result.wavelengths_nm)} λ "
            f"@ pupil R={result.pupil_radius_mm:.2f} mm"
        )

    def apply_error(self, exc: BaseException) -> None:
        # Don't blank the canvas — leave the previous successful render
        # in place so the user can compare against the value they just
        # set that broke things. Just surface the message.
        super().apply_error(exc)

    def sync_from_system_setup(self) -> None:
        """Replace this panel's wavelengths and fields with the project's
        System Setup values.

        Pulls from the first sequence's source — same convention the
        viewport ray tracing uses. Pupil radius is mapped from the
        source's aperture_radius. Rings/fans/defocus stay as-is since
        System Setup doesn't carry equivalent knobs.

        Called by :meth:`apply_sync_from_system_setup` (base class),
        which is what the View menu binds to — that helper takes care
        of firing the follow-up refresh on success.
        """
        setup: SystemSetup = self._project.system_setup
        if not setup.sequences:
            self.set_status("System Setup has no sequences to sync from.")
            return
        source = setup.sequences[0].source
        wavelengths = tuple(
            float(w.value_nm) for w in source.wavelengths.wavelengths
            if isinstance(w, Wavelength) and float(w.value_nm) > 0.0
        )
        fields = tuple(
            (float(f.tilt_x_deg), float(f.tilt_y_deg))
            for f in source.fields
        )
        new_spec = SpotDiagramSpec(
            wavelengths_nm=wavelengths or DEFAULT_WAVELENGTHS_NM,
            fields_deg=fields or DEFAULT_FIELDS_DEG,
            rings=self._spec.rings,
            fans=self._spec.fans,
            pupil_radius_mm=float(source.aperture_radius)
                if source.aperture_radius > 0.0
                else self._spec.pupil_radius_mm,
            defocus_offsets_mm=self._spec.defocus_offsets_mm,
            plot_half_extent_mm=self._spec.plot_half_extent_mm,
        ).clamp()
        if new_spec == self._spec:
            self.set_status("System Setup matches current spec — no changes.")
            return
        self._spec = new_spec
        # Update the editor view without re-emitting (the base class
        # apply_sync_from_system_setup will fire force_refresh_now itself).
        self._editor.set_spec(new_spec, emit=False)
        # The new spec likely has different fields, which invalidates
        # the per-field scale lock (keyed by tilt). Drop it so the
        # next render auto-fits to the new bundle sizes.
        self._reference_extents = None
        self.set_status("Synced from System Setup.")

    # ------------------------------------------------------------------
    # Spec change wiring
    # ------------------------------------------------------------------

    def _on_spec_changed(self, new_spec: SpotDiagramSpec) -> None:
        if new_spec == self._spec:
            return
        # A spec edit that adds / removes / re-tilts a field invalidates
        # the per-field scale lock (keyed by tilt). Detect that and
        # drop the lock — other spec edits (wavelengths, pupil, defocus)
        # keep the lock so the user can A/B-compare under a stable scale.
        if tuple(new_spec.fields_deg) != tuple(self._spec.fields_deg):
            self._reference_extents = None
        self._spec = new_spec
        # Spec edits are direct user input — same treatment as a lens
        # edit. Debounce, then recompute.
        self.request_refresh()

    # ------------------------------------------------------------------
    # Lens lifecycle
    # ------------------------------------------------------------------

    def _on_system_replaced(self, system) -> None:
        # New lens = totally different geometry. The old scale lock is
        # almost certainly wrong; drop it so the first render of the
        # new lens auto-fits.
        self._reference_extents = None
        super()._on_system_replaced(system)

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
