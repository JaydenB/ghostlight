"""Body widget for the ``viewport`` panel type."""
from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtWidgets import QVBoxLayout, QWidget

import ghostlight
from ghostlight_viewport import LensViewport, SensorSpec

from ..project import Project
from ..settings import AppSettings
from .. import lens_metrics as lm_mod
from .. import ray_tracing as rt_mod
from .context_popup import ViewportContextPopup

_log = logging.getLogger("ghostlight_designer.viewport_panel")


class ViewportPanelBody(QWidget):
    def __init__(
        self,
        project: Project,
        parent: Optional[QWidget] = None,
        *,
        settings: Optional[AppSettings] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._settings = settings
        self._context_popup: Optional[ViewportContextPopup] = None
        self._viewport = LensViewport(self)

        # Rays default visible per the panel spec; the View menu toggles
        # this and the body decides whether to push new bundles or clear.
        self._show_rays: bool = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._viewport)

        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)
        # Ghost-solo doesn't fire systemModified — reuse the same handler
        # so the viewport's solo-element highlight updates immediately.
        project.ghostSoloChanged.connect(self._on_system_modified)
        project.systemSetupChanged.connect(self._on_system_setup_changed)
        project.selectionChanged.connect(self._on_project_selection_changed)
        project.surfaceSelectionChanged.connect(
            self._on_project_surface_selection_changed
        )
        self._viewport.elementSelected.connect(self._on_viewport_selected)
        self._viewport.surfaceSelected.connect(self._on_viewport_surface_selected)
        self._viewport.contextMenuRequested.connect(
            self._on_context_menu_requested
        )

        self._on_system_replaced(project.system)
        self._on_project_selection_changed(project.selected_element)
        self._on_project_surface_selection_changed(
            project.selected_surface_index
        )

    @property
    def viewport(self) -> LensViewport:
        return self._viewport

    @property
    def show_rays(self) -> bool:
        return self._show_rays

    def set_show_rays(self, visible: bool) -> None:
        """Toggle ray visibility.  Hiding clears the viewport's ray buffer;
        showing rebuilds and pushes a fresh trace immediately."""
        visible = bool(visible)
        if visible == self._show_rays:
            return
        self._show_rays = visible
        if visible:
            self._refresh_rays()
        else:
            self._viewport.clear_trace_results()

    def _resolve_elements(self) -> List[ghostlight.Element]:
        # Use the system's canonical Element instances so the viewport's
        # picked Element is identity-equal to what other panels see via
        # ``project.system.elements`` — required for cross-panel selection
        # sync to work without UUID-matching.
        return list(self._project.system.elements)

    def _resolve_solo_surface_indices(self) -> set[int]:
        """Translate the Project's solo UUIDs to system surface indices.

        Returned as a plain set so it can be passed straight into
        ``LensViewport.set_lens``. UUIDs that no longer resolve (mid-
        edit, stale entry) are silently dropped; the Project also prunes
        them on every system mutation."""
        solo = self._project.ghost_solo_surface_uuids
        if not solo:
            return set()
        uuid_to_idx = {u: i for i, u in enumerate(self._project.system.surface_ids)}
        return {uuid_to_idx[u] for u in solo if u in uuid_to_idx}

    def _resolve_sensor(self) -> Optional[SensorSpec]:
        # The viewport's sensor mirrors the System Setup → Image Sensor
        # block exactly; the lens calibration is no longer consulted. This
        # means picking a Super 35 vs Full Frame preset (or typing custom
        # mm) immediately updates the viewport's sensor rectangle.
        s = self._project.system_setup.sensor
        try:
            return SensorSpec(
                half_w=float(s.width_mm) / 2.0,
                half_h=float(s.height_mm) / 2.0,
            )
        except Exception:
            _log.exception("ViewportPanelBody: failed to build SensorSpec")
            return None

    def _refresh_rays(self) -> None:
        """Rebuild ray bundles from the current project setup + system and
        push them into the viewport.  No-op when rays are hidden."""
        if not self._show_rays:
            return
        try:
            bundles = rt_mod.build_ray_bundles(
                self._project.system, self._project.system_setup
            )
        except Exception:
            _log.exception("ViewportPanelBody: build_ray_bundles failed")
            self._viewport.clear_trace_results()
            return
        if bundles:
            self._viewport.set_trace_results(bundles)
        else:
            self._viewport.clear_trace_results()

    def _refresh_info_bar(self) -> None:
        """Compute EFFL + AFOV and push them to the viewport's info bar.

        Hides the bar (passes empty text) whenever either value can't be
        produced — the user shouldn't see partial metrics for a lens
        that doesn't yet trace cleanly.
        """
        sensor = self._project.system_setup.sensor
        try:
            metrics = lm_mod.compute_metrics(
                self._project.system,
                float(sensor.width_mm),
                float(sensor.height_mm),
            )
        except Exception:
            _log.exception("ViewportPanelBody: compute_metrics failed")
            metrics = None
        if metrics is None:
            self._viewport.set_info_text(None)
            return
        # Working f-number, next to the focal length — the exact value the
        # starburst uses for its physical size.  Per-axis on anamorphics
        # (H shown alongside the vertical value); omitted if unavailable.
        f_part = ""
        if metrics.f_number is not None:
            f_part = f"  f/{metrics.f_number:.1f}"
            if (
                abs(metrics.squeeze - 1.0) > 0.02
                and metrics.f_number_x is not None
            ):
                f_part += f"·{metrics.f_number_x:.1f}H"
        text = (
            f"EFFL {metrics.efl_mm:.2f}mm{f_part}    "
            f"HFOV {metrics.hfov_deg:.1f}° · VFOV {metrics.vfov_deg:.1f}°    "
            f"{metrics.squeeze:.2f}×"
        )
        self._viewport.set_info_text(text)

    def _on_system_replaced(self, system: ghostlight.OpticalSystem) -> None:
        # Only refit the camera on a user-initiated load — undo / redo
        # snapshots through the same code path but the user wants to
        # compare before / after states with the same framing, so refitting
        # there would just snap their view away. Project tags each
        # ``systemReplaced`` with the reason so we can honour that here.
        fit = self._project.last_replacement_kind == "load"
        self._viewport.set_lens(
            system,
            self._resolve_elements(),
            fit_view=fit,
            ghost_solo_surface_indices=self._resolve_solo_surface_indices(),
        )
        sensor = self._resolve_sensor()
        if sensor is not None:
            self._viewport.set_sensor(sensor)
        self._refresh_rays()
        self._refresh_info_bar()

    def _on_system_modified(self) -> None:
        # System changes (radius/aperture edits etc.) don't move the
        # sensor — the sensor is owned by SystemSetup now.
        self._viewport.set_lens(
            self._project.system,
            self._resolve_elements(),
            fit_view=False,
            ghost_solo_surface_indices=self._resolve_solo_surface_indices(),
        )
        self._refresh_rays()
        self._refresh_info_bar()

    def _on_system_setup_changed(self) -> None:
        # SystemSetup now owns the sensor dimensions, so changes here can
        # affect the sensor rectangle as well as the rays. Push both.
        sensor = self._resolve_sensor()
        if sensor is not None:
            self._viewport.set_sensor(sensor)
        self._refresh_rays()
        self._refresh_info_bar()

    def _on_context_menu_requested(self, info: dict) -> None:
        # Sync project selection to the right-clicked target so the tree and
        # highlight agree with the popup, then open the popup at the cursor.
        try:
            if info.get("mode") == "surface":
                self._project.set_selection(
                    info.get("element"), info.get("surface_index")
                )
            else:
                self._project.set_selection(info.get("element"), None)
        except Exception:
            _log.exception("Viewport context: selection sync failed")

        # Close any popup still lingering (defensive — Qt.Popup normally
        # closes the previous one on the outside press).
        if self._context_popup is not None:
            self._context_popup.close()
        popup = ViewportContextPopup(
            self._project, self._settings, self._viewport, info
        )
        self._context_popup = popup
        popup.destroyed.connect(self._on_context_popup_destroyed)
        popup.show()

    def _on_context_popup_destroyed(self, _obj=None) -> None:
        self._context_popup = None

    def _on_viewport_selected(self, element) -> None:
        self._project.set_selected_element(element)

    def _on_viewport_surface_selected(self, surface_index) -> None:
        # The viewport already updated its own element selection alongside
        # this surface pick (if surface mode picked one); the Project routes
        # element via the elementSelected path and surface via here.
        self._project.set_selected_surface_index(surface_index)

    def _on_project_selection_changed(self, element) -> None:
        # Programmatic setter — does not re-emit elementSelected, so no
        # feedback loop with the connection above.
        self._viewport.set_selected_element(element)
        # set_selected_element clears the viewport's surface highlight; if
        # the project still has a surface selected for this element, restore
        # it.  surfaceSelectionChanged will also fire shortly and refresh,
        # but doing it here keeps highlight transitions free of flicker.
        if self._project.selected_surface_index is not None:
            self._viewport.set_selected_surface(
                self._project.selected_surface_index
            )

    def _on_project_surface_selection_changed(self, surface_index) -> None:
        self._viewport.set_selected_surface(surface_index)
