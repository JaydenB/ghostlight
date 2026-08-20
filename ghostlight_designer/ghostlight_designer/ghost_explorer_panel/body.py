"""Body widget for the ``ghost_explorer`` panel type.

A point-source flare render restricted to **one ghost at a time**, with a
slider along the bottom that scrubs through the lens's ghost pairs.

It reuses the source-flare panel's rendering stack — the same
``render_source_flare`` dispatch, exposure / view-transform and
render-settings machinery — via
:func:`ghostlight_designer.sourceflare_panel.body.render_chunk`, so the two panels
can never drift in how they configure a render. Four things differ:

* **One ghost per frame.** The selected ghost pair goes into the render's
  ``GhostFilter`` as a single-entry INCLUDE list, so the GPU only ever traces
  that pair. Nothing is pre-rendered and cached: moving the slider bumps the
  render serial, which aborts the in-flight pass and dispatches the newly
  selected ghost. The project's own ghost-solo state is deliberately ignored
  here — there is one ``GhostFilter`` per render and the scrubber owns it.

* **No additive layers.** The aperture starburst and the veiling glare are
  forced off on every settings change (:meth:`apply_settings`), including the
  quality presets, so what you see is the ghost's geometric contribution
  alone. The render-settings dialog is opened with those two controls locked.

* **A fixed point source.** The source is a point at frame fraction
  (0.75, 0.25) — an off-axis practical, the framing that separates ghosts from
  the source. No position or shape controls, so a ghost's place and shape in
  frame are attributable to the lens rather than to how the source was set up.
  An extended source would convolve every ghost with the same blur, which
  hides the ghost's own footprint rather than revealing it.

* **Exposure metered from the whole flare.** A coarse render of the lens with
  *every* ghost active (:meth:`_maybe_launch_rough`) sets one viewer exposure
  that then holds across the entire scrub. That is the point: a ghost that is
  six stops down from its neighbour reads as six stops down, instead of every
  ghost being normalised to look equally bright. The same pass optionally
  scores each ghost for the cull.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import ghostlight
from ghostlight import source_sampling

from ..math_spinbox import MathDoubleSpinBox
from ..project import Project
from ..settings import AppSettings
from .. import lens_metrics as lm_mod
from ..render_common import (
    DRAFT_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    MIN_SURFACES,
    POLL_INTERVAL_MS,
    ExposureDialog,
    FlareCanvas,
    RenderSettings,
    RenderSettingsDialog,
    attach_spinbox_scrubber,
)
from .. import viewtransform as vt
from ..sourceflare_panel.body import (
    MATTE_HALF_MM_MAX,
    MATTE_HALF_MM_MIN,
    MATTE_Z_MM_MAX,
    MATTE_Z_MM_MIN,
    _MatteBoxState,
    build_flare_config,
    render_chunk,
)
from ..vignette import VignetteController
from . import ghost_survey
from .ghost_survey import GhostEntry

_log = logging.getLogger("ghostlight_designer.ghost_explorer_panel")

# Fixed source position in frame fractions. Off-axis in the upper-right
# quadrant: an on-axis source stacks every ghost on top of the source itself,
# and a fixed position means a ghost's place in frame is a property of the
# lens, not of where a marker happened to be dragged.
SOURCE_X = 0.75
SOURCE_Y = 0.25


class GhostCanvas(FlareCanvas):
    """The shared flare canvas with the source drag removed.

    This panel pins the source, so a drag that silently did nothing would read
    as a broken control. The marker still paints, as a reminder of where the
    source sits relative to the ghost on screen.
    """

    def mousePressEvent(self, event) -> None:  # noqa: D102 - see class docstring
        event.ignore()

    def mouseMoveEvent(self, event) -> None:  # noqa: D102
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: D102
        event.ignore()


class GhostExplorerPanelBody(QWidget):
    def __init__(
        self,
        project: Project,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._app_settings = settings

        # Front-of-lens matte box: not a source-shape control but an occluder
        # that changes which ghosts survive at all, which is squarely this
        # panel's subject. Off by default, so it is a no-op until enabled.
        self._matte: _MatteBoxState = _MatteBoxState()

        # Ghost bookkeeping. ``_ghosts`` is every renderable pair of the current
        # lens; ``_view`` is what the slider offers (``_ghosts`` unless culling
        # is on). ``_selected_number`` is a GhostEntry.number, not an index, so
        # the selection survives a cull toggle or a re-survey.
        self._ghosts: List[GhostEntry] = []
        self._view: List[GhostEntry] = []
        self._selected_number: Optional[int] = None
        # False until the user picks a ghost themselves. Until then the
        # selection stays pinned to the head of the scrubber, so a list that
        # reorders under us — which it does once brightness metering lands and
        # the brightest-first sort kicks in — leaves the panel on the first
        # ghost rather than stranded wherever the seed entry drifted to.
        self._user_selected: bool = False
        # Culling on by default: a real lens carries a long tail of ghosts
        # seven-plus stops down that the scrubber would otherwise make the user
        # step through one by one. Uncheck it in the View menu to see them all.
        self._cull: bool = True
        self._cull_rel: float = ghost_survey.DEFAULT_CULL_REL
        # Scrubber order, brightest-first by default: a left-to-right sweep
        # then walks the ghosts in the order they matter, and the ones worth
        # looking at are the first you land on. Uncheck it in the View menu to
        # get the lens's own surface-pair order, where the slider position and
        # the readout's "Ghost N" agree.
        self._sort_by_brightness: bool = True
        self._cull_dialog: Optional[QWidget] = None
        # Guards the slider's valueChanged while we push state into it.
        self._suppress_ghost_signals: bool = False

        self._exposure_stops: float = 0.0
        self._auto_expose_pending: bool = True
        self._auto_render: bool = True
        # Desqueeze on by default, as in the source-flare panel: an anamorphic
        # ghost is only the right shape once unsqueezed.
        self._desqueeze: bool = True
        self._squeeze_factor: float = 1.0

        self._is_active: bool = False
        self._dirty_pending: bool = False
        # Set whenever the lens / sensor changed under us; the ghost list is
        # rebuilt on the next render request rather than on every signal, so a
        # burst of edits costs one enumeration.
        self._survey_dirty: bool = True

        # High by default. This panel renders one ghost at a time from a point
        # source, so a frame costs a fraction of a full flare render — there is
        # headroom to look at a ghost's real shape rather than a noisy sketch.
        self._settings: RenderSettings = _strip_layers(HIGH_PRESET)
        self._latest_hwc: Optional[np.ndarray] = None
        self._cc_dialog: Optional[ExposureDialog] = None
        self._settings_dialog: Optional[RenderSettingsDialog] = None

        self._lock = threading.Lock()
        self._pending: bool = False
        self._busy: bool = False
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._epoch: int = 0
        self._serial: int = 0

        # Rough whole-flare pass (see _maybe_launch_rough): the exposure meter's
        # reference frame, and — while culling is on — the per-ghost brightness
        # the cull ranks by.
        self._rough_hwc: Optional[np.ndarray] = None
        self._rough_dirty: bool = True
        self._rough_busy: bool = False
        self._rough_results: queue.SimpleQueue = queue.SimpleQueue()

        self._canvas = GhostCanvas(self)
        self._canvas.set_source(SOURCE_X, SOURCE_Y)

        self._vignette = VignetteController(project, self._canvas, parent=self)

        self._status = QLabel("", self)
        self._status.setStyleSheet("color: #aaa; padding: 2px 6px;")

        matte_grid = self._build_matte_grid()
        ghost_block = self._build_ghost_block()

        self._scrubbers = [
            attach_spinbox_scrubber(self._spin_matte_w, label="Matte width (mm)"),
            attach_spinbox_scrubber(self._spin_matte_h, label="Matte height (mm)"),
            attach_spinbox_scrubber(self._spin_matte_z, label="Matte distance (mm)"),
        ]

        matte_sep = QFrame(self)
        matte_sep.setFrameShape(QFrame.HLine)
        matte_sep.setFrameShadow(QFrame.Sunken)
        self._matte_container = QWidget(self)
        matte_layout = QVBoxLayout(self._matte_container)
        matte_layout.setContentsMargins(0, 0, 0, 0)
        matte_layout.setSpacing(0)
        matte_layout.addWidget(matte_sep)
        matte_layout.addLayout(matte_grid)
        self._matte_container.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._canvas, 1)
        outer.addWidget(self._status)
        outer.addWidget(self._matte_container)
        outer.addWidget(ghost_block)

        self._sync_matte_widget_enabled()

        project.systemReplaced.connect(self._on_system_replaced)
        project.systemModified.connect(self._on_system_modified)
        project.systemSetupChanged.connect(self._on_system_setup_changed)
        self._app_settings.autoUpdateChanged.connect(
            self._on_global_auto_update_changed
        )
        self._app_settings.viewTransformChanged.connect(
            self._on_view_transform_changed
        )

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll_results)
        self._timer.start()

        self._refresh_squeeze_factor()
        self._push_squeeze_to_canvas()
        self._refresh_survey()
        self._refresh_status_for_lens()
        self._dirty_pending = True

    # ------------------------------------------------------------------
    # Public API used by the View / Settings menus
    # ------------------------------------------------------------------

    @property
    def auto_render(self) -> bool:
        return self._auto_render

    def set_auto_render(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._auto_render:
            return
        self._auto_render = enabled
        if enabled:
            self.request_render()
        else:
            self._status.setText("Auto-render off")

    def request_render(self) -> None:
        if self._survey_dirty:
            self._refresh_survey()
        if not self._lens_eligible():
            self._canvas.clear_image(self._placeholder_for_lens())
            self._status.setText(self._placeholder_for_lens())
            return
        if self.selected_ghost() is None:
            self._canvas.clear_image("No ghosts in this lens")
            self._status.setText("No ghosts in this lens")
            return
        if not self._is_active:
            self._dirty_pending = True
            return
        self._dirty_pending = False
        self._request()

    @property
    def vignette_overlay(self) -> bool:
        return self._vignette.enabled

    def set_vignette_overlay(self, enabled: bool) -> None:
        self._vignette.set_enabled(bool(enabled))

    @property
    def matte_controls_visible(self) -> bool:
        return not self._matte_container.isHidden()

    def set_matte_controls_visible(self, visible: bool) -> None:
        self._matte_container.setVisible(bool(visible))

    @property
    def desqueeze(self) -> bool:
        return self._desqueeze

    def set_desqueeze(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._desqueeze:
            return
        self._desqueeze = enabled
        if enabled:
            self._refresh_squeeze_factor()
        self._push_squeeze_to_canvas()

    # ------------------------------------------------------------------
    # Ghost culling (View menu)
    # ------------------------------------------------------------------

    @property
    def cull_dim_ghosts(self) -> bool:
        return self._cull

    def set_cull_dim_ghosts(self, enabled: bool) -> None:
        """Hide ghosts too dim to matter from the scrubber.

        Enabling it re-runs the rough pass with per-pair layers on — until that
        measurement lands every ghost counts as unmetered and stays visible, so
        the list only shrinks once there is evidence to shrink it by.
        """
        enabled = bool(enabled)
        if enabled == self._cull:
            return
        self._cull = enabled
        if enabled and not any(e.metered for e in self._ghosts):
            # The rough pass so far ran without per-pair layers; it needs to be
            # redone before the cull has anything to rank.
            self._rough_dirty = True

        self._rebuild_view()
        self._maybe_launch_rough()

    @property
    def cull_threshold(self) -> float:
        return self._cull_rel

    def set_cull_threshold(self, rel: float) -> None:
        rel = max(ghost_survey.CULL_REL_MIN,
                  min(ghost_survey.CULL_REL_MAX, float(rel)))
        if rel == self._cull_rel:
            return
        self._cull_rel = rel
        if self._cull:
            self._rebuild_view()

    @property
    def sort_by_brightness(self) -> bool:
        return self._sort_by_brightness

    def set_sort_by_brightness(self, enabled: bool) -> None:
        """Order the scrubber brightest-to-dimmest instead of by ghost number.

        Only the slider order changes — a ghost keeps its number, so the
        readout still names the same pair and the cull is unaffected. Needs the
        rough pass's measurement to do anything; until that lands the order is
        left alone rather than scrambled.
        """
        enabled = bool(enabled)
        if enabled == self._sort_by_brightness:
            return
        self._sort_by_brightness = enabled
        if enabled and not any(e.metered for e in self._ghosts):
            self._rough_dirty = True
        # No re-render: the selected ghost is unchanged, it just sits at a
        # different slider position.
        self._rebuild_view()
        self._maybe_launch_rough()

    def open_cull_threshold_dialog(self) -> None:
        from .cull_dialog import CullThresholdDialog

        if self._cull_dialog is not None and self._cull_dialog.isVisible():
            self._cull_dialog.raise_()
            self._cull_dialog.activateWindow()
            return
        dlg = CullThresholdDialog(self._cull_rel, parent=self)
        dlg.thresholdChanged.connect(self.set_cull_threshold)
        dlg.finished.connect(self._on_cull_dialog_finished)
        self._cull_dialog = dlg
        dlg.show()

    def _on_cull_dialog_finished(self, _result: int) -> None:
        self._cull_dialog = None

    # ------------------------------------------------------------------
    # Exposure + view transform
    # ------------------------------------------------------------------

    def open_exposure_dialog(self) -> None:
        if self._cc_dialog is not None and self._cc_dialog.isVisible():
            self._cc_dialog.raise_()
            self._cc_dialog.activateWindow()
            return
        dlg = ExposureDialog(self._exposure_stops, parent=self)
        dlg.stopsChanged.connect(self._on_exposure_stops_changed)
        dlg.finished.connect(self._on_exposure_dialog_finished)
        self._cc_dialog = dlg
        dlg.show()

    def _on_exposure_dialog_finished(self, _result: int) -> None:
        self._cc_dialog = None

    def _on_exposure_stops_changed(self, stops: float) -> None:
        self._exposure_stops = float(stops)
        self._redisplay()

    def auto_expose(self) -> None:
        """Meter the viewer exposure from the *whole* flare, not this ghost.

        The reference is the rough all-ghosts pass, so one exposure serves the
        entire scrub and each ghost is displayed at its true brightness
        relative to the others — metering per ghost would normalise them all to
        look the same and throw away the comparison the panel exists to make.
        Falls back to the current frame only when no rough pass has landed.
        """
        meter_src = (self._rough_hwc if self._rough_hwc is not None
                     else self._latest_hwc)
        if meter_src is None:
            return
        stops = vt.meter_auto_stops(meter_src)
        self._exposure_stops = stops
        if self._cc_dialog is not None:
            self._cc_dialog.set_stops(stops)
        self._redisplay()

    def reset_exposure(self) -> None:
        self._exposure_stops = 0.0
        if self._cc_dialog is not None:
            self._cc_dialog.set_stops(0.0)
        self._redisplay()

    def _on_view_transform_changed(self) -> None:
        self._redisplay()

    def _redisplay(self) -> None:
        if self._latest_hwc is None:
            return
        try:
            spec = vt.spec_from_settings(self._app_settings)
            display = vt.apply_view(self._latest_hwc, self._exposure_stops, spec)
        except vt.ViewTransformError as exc:
            self._status.setText(f"View transform error: {exc}")
            return
        self._canvas.set_image(vt.to_qimage(display))
        self._status.setText(self._final_status_text())

    # ------------------------------------------------------------------
    # Render settings + presets
    # ------------------------------------------------------------------

    @property
    def settings(self) -> RenderSettings:
        return self._settings

    def apply_settings(self, new_settings: RenderSettings) -> None:
        """Adopt ``new_settings`` with the two additive layers forced off.

        The stripping happens here rather than at each call site so presets,
        the live dialog, and every other caller all land on a ghost-only
        configuration — this panel's whole point is the geometric ghost.
        """
        new_settings = _strip_layers(new_settings)
        if new_settings == self._settings:
            return
        self._settings = new_settings
        self._latest_hwc = None
        # The sampling accelerators feed into rendered brightness, so both the
        # exposure reference and the cull's ranking are now stale.
        self._rough_dirty = True
        if self._lens_eligible():
            self.request_render()
        self._maybe_launch_rough()

    def apply_preset_draft(self) -> None:
        self.apply_settings(DRAFT_PRESET)

    def apply_preset_mid(self) -> None:
        self.apply_settings(MID_PRESET)

    def apply_preset_high(self) -> None:
        self.apply_settings(HIGH_PRESET)

    def open_render_settings_dialog(self) -> None:
        if self._settings_dialog is not None and self._settings_dialog.isVisible():
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        dlg = RenderSettingsDialog(
            self._settings, parent=self, allow_diffraction_layers=False
        )
        dlg.settingsChanged.connect(self.apply_settings)
        dlg.finished.connect(self._on_settings_dialog_finished)
        self._settings_dialog = dlg
        dlg.show()

    def _on_settings_dialog_finished(self, _result: int) -> None:
        self._settings_dialog = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._is_active = True
        self._vignette.set_active(True)
        if self._dirty_pending:
            self.request_render()
        self._maybe_launch_rough()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._is_active = False
        self._vignette.set_active(False)
        self._dirty_pending = True

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_matte_grid(self) -> QGridLayout:
        """Front-of-lens matte box — the source-flare panel's inline control.

        It clips entrance rays, so it changes which ghosts survive and how
        bright they are; hidden by default via the View menu.
        """
        self._matte_check = QCheckBox("Matte box", self)
        self._matte_check.setChecked(self._matte.enabled)
        self._matte_check.setToolTip(
            "Front-of-lens rectangular flag that clips rays entering the lens, "
            "cutting flare from bright sources just outside the frame — like "
            "closing a matte box's blades in. Off = no occluder (no-op)."
        )
        self._matte_check.toggled.connect(self._on_matte_enabled_changed)

        self._spin_matte_w = MathDoubleSpinBox(self)
        self._spin_matte_w.setRange(MATTE_HALF_MM_MIN, MATTE_HALF_MM_MAX)
        self._spin_matte_w.setDecimals(1)
        self._spin_matte_w.setSingleStep(1.0)
        self._spin_matte_w.setSuffix(" mm")
        self._spin_matte_w.setValue(self._matte.half_w_mm)
        self._spin_matte_w.setKeyboardTracking(False)
        self._spin_matte_w.setToolTip(
            "Horizontal half-aperture of the flag from the axis (mm): the left "
            "and right blades. Smaller = the flag closes in from the sides."
        )
        self._spin_matte_w.valueChanged.connect(self._on_matte_w_changed)

        self._spin_matte_h = MathDoubleSpinBox(self)
        self._spin_matte_h.setRange(MATTE_HALF_MM_MIN, MATTE_HALF_MM_MAX)
        self._spin_matte_h.setDecimals(1)
        self._spin_matte_h.setSingleStep(1.0)
        self._spin_matte_h.setSuffix(" mm")
        self._spin_matte_h.setValue(self._matte.half_h_mm)
        self._spin_matte_h.setKeyboardTracking(False)
        self._spin_matte_h.setToolTip(
            "Vertical half-aperture of the flag from the axis (mm): the top and "
            "bottom blades. Smaller = the flag closes in from top and bottom."
        )
        self._spin_matte_h.valueChanged.connect(self._on_matte_h_changed)

        self._spin_matte_z = MathDoubleSpinBox(self)
        self._spin_matte_z.setRange(MATTE_Z_MM_MIN, MATTE_Z_MM_MAX)
        self._spin_matte_z.setDecimals(1)
        self._spin_matte_z.setSingleStep(5.0)
        self._spin_matte_z.setSuffix(" mm")
        self._spin_matte_z.setValue(self._matte.z_front_mm)
        self._spin_matte_z.setKeyboardTracking(False)
        self._spin_matte_z.setToolTip(
            "Distance of the flag plane ahead of the front element (mm). "
            "Farther out = the same blade opening cuts a wider field angle."
        )
        self._spin_matte_z.valueChanged.connect(self._on_matte_z_changed)

        grid = QGridLayout()
        grid.setContentsMargins(6, 0, 6, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.addWidget(self._matte_check, 0, 0, 1, 4)
        grid.addWidget(QLabel("W", self), 1, 0)
        grid.addWidget(self._spin_matte_w, 1, 1)
        grid.addWidget(QLabel("H", self), 1, 2)
        grid.addWidget(self._spin_matte_h, 1, 3)
        grid.addWidget(QLabel("Dist", self), 2, 0)
        grid.addWidget(self._spin_matte_z, 2, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    def _build_ghost_block(self) -> QWidget:
        """The panel's reason for existing: the ghost readout + scrubber.

        Step buttons flank the slider because a ghost list is a short ordinal
        sequence — dragging a slider to land on ghost 12 of 40 is fiddly, and
        stepping one pair at a time is the actual review workflow.
        """
        self._ghost_readout = QLabel("", self)
        self._ghost_readout.setWordWrap(True)
        self._ghost_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._ghost_readout.setStyleSheet(
            "color: #ddd; padding: 2px 6px; font-weight: 600;"
        )

        self._btn_prev = QToolButton(self)
        self._btn_prev.setText("◀")
        self._btn_prev.setCursor(Qt.PointingHandCursor)
        self._btn_prev.setToolTip("Previous ghost")
        self._btn_prev.clicked.connect(self.select_previous_ghost)

        self._btn_next = QToolButton(self)
        self._btn_next.setText("▶")
        self._btn_next.setCursor(Qt.PointingHandCursor)
        self._btn_next.setToolTip("Next ghost")
        self._btn_next.clicked.connect(self.select_next_ghost)

        self._ghost_slider = QSlider(Qt.Horizontal, self)
        self._ghost_slider.setMinimum(0)
        self._ghost_slider.setMaximum(0)
        self._ghost_slider.setSingleStep(1)
        self._ghost_slider.setPageStep(1)
        self._ghost_slider.setTracking(True)
        self._ghost_slider.setToolTip(
            "Scrub through the lens's ghost pairs. Only the selected ghost is "
            "rendered — moving the slider aborts the in-flight pass and "
            "dispatches the new one."
        )
        self._ghost_slider.valueChanged.connect(self._on_ghost_slider)

        self._ghost_count = QLabel("", self)
        self._ghost_count.setStyleSheet("color: #aaa;")
        self._ghost_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._ghost_count.setMinimumWidth(96)
        self._ghost_count.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 4)
        row.setSpacing(6)
        row.addWidget(self._btn_prev)
        row.addWidget(self._ghost_slider, 1)
        row.addWidget(self._btn_next)
        row.addWidget(self._ghost_count)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        block = QWidget(self)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(sep)
        layout.addWidget(self._ghost_readout)
        layout.addLayout(row)
        return block

    # ------------------------------------------------------------------
    # Ghost selection
    # ------------------------------------------------------------------

    def selected_ghost(self) -> Optional[GhostEntry]:
        idx = ghost_survey.find_by_number(self._view, self._selected_number)
        if idx is None:
            return None
        return self._view[idx]

    @property
    def ghosts(self) -> List[GhostEntry]:
        """Every renderable ghost of the current lens, cull or no cull."""
        return list(self._ghosts)

    @property
    def visible_ghosts(self) -> List[GhostEntry]:
        """The ghosts the scrubber currently offers."""
        return list(self._view)

    def select_ghost_number(self, number: int) -> None:
        """Select the ghost carrying ``number``, if it is currently offered."""
        idx = ghost_survey.find_by_number(self._view, int(number))
        if idx is None:
            return
        self._select_index(idx)

    def select_next_ghost(self) -> None:
        self._step_ghost(+1)

    def select_previous_ghost(self) -> None:
        self._step_ghost(-1)

    def _step_ghost(self, delta: int) -> None:
        if not self._view:
            return
        idx = ghost_survey.find_by_number(self._view, self._selected_number)
        idx = 0 if idx is None else idx + int(delta)
        self._select_index(max(0, min(len(self._view) - 1, idx)))

    def _select_index(self, idx: int) -> None:
        """Select ``idx`` within the current view and re-render.

        The exposure is deliberately left alone: it belongs to the lens (see
        :meth:`auto_expose`), not to the ghost, so switching ghosts changes
        what is on screen and not how it is displayed.
        """
        if not (0 <= idx < len(self._view)):
            return
        self._user_selected = True
        number = self._view[idx].number
        if number == self._selected_number:
            return
        self._selected_number = number
        self._sync_ghost_widgets()
        if self._auto_render:
            self.request_render()

    def _on_ghost_slider(self, value: int) -> None:
        if self._suppress_ghost_signals:
            return
        self._select_index(int(value))

    def _sync_ghost_widgets(self) -> None:
        """Push the current view + selection into the slider and readouts."""
        n = len(self._view)
        idx = ghost_survey.find_by_number(self._view, self._selected_number)
        self._suppress_ghost_signals = True
        try:
            self._ghost_slider.setMaximum(max(0, n - 1))
            self._ghost_slider.setEnabled(n > 1)
            if idx is not None:
                self._ghost_slider.setValue(idx)
            # Tick marks make the discrete pair count legible, but only while
            # they stay countable.
            self._ghost_slider.setTickPosition(
                QSlider.TicksBelow if 0 < n <= 40 else QSlider.NoTicks
            )
            self._ghost_slider.setTickInterval(1)
        finally:
            self._suppress_ghost_signals = False

        self._btn_prev.setEnabled(idx is not None and idx > 0)
        self._btn_next.setEnabled(idx is not None and idx < n - 1)

        entry = self.selected_ghost()
        total = len(self._ghosts)
        if entry is None:
            self._ghost_readout.setText(
                "No ghosts" if total == 0
                else "No ghosts pass the cull threshold"
            )
            self._ghost_count.setText("— / —")
            return
        self._ghost_readout.setText(
            f"Ghost {entry.number} of {total} — reflects off "
            f"{entry.surfaces_text()}   ·   {entry.brightness_text()}"
        )
        pos = (idx + 1) if idx is not None else 0
        if self._cull and n != total:
            self._ghost_count.setText(f"{pos} / {n} shown")
        else:
            self._ghost_count.setText(f"{pos} / {n}")

    # ------------------------------------------------------------------
    # Ghost enumeration
    # ------------------------------------------------------------------

    def _refresh_survey(self) -> None:
        """Re-enumerate the lens's ghosts, preserving the selection.

        Pure CPU: a ghost-pair enumeration plus ghostlight's own pre-filter, so
        it runs inline on the GUI thread. The entries come back *unmetered* —
        brightness arrives later from the rough whole-flare pass.
        """
        self._survey_dirty = False
        entries: List[GhostEntry] = []
        if self._lens_eligible():
            _w, _h, half_w, half_h = self._resolve_render_dims()
            # A default config carries the same min_ghost_intensity /
            # ghost_normalize the render call will use, so the enumeration and
            # the renderer agree on which pairs exist.
            probe_cfg = ghostlight.PointFlareConfig()
            probe_cfg.sensor_half_w = half_w
            probe_cfg.sensor_half_h = half_h
            entries = ghost_survey.enumerate_entries(
                self._project.system, probe_cfg, half_w=half_w, half_h=half_h,
            )
        self._ghosts = entries
        # The lens's ghosts changed, so the exposure reference and any measured
        # brightness are both stale.
        self._rough_dirty = True
        self._rebuild_view()
        self._maybe_launch_rough()

    def _rebuild_view(self) -> None:
        """Recompute the offered subset and keep (or re-seat) the selection."""
        self._view = ghost_survey.visible_ghosts(
            self._ghosts, cull=self._cull, rel_threshold=self._cull_rel,
            sort_by_brightness=self._sort_by_brightness,
        )
        previous = self._selected_number
        if not self._user_selected:
            # Nobody has chosen a ghost yet — follow the head of the list, so
            # the panel opens on the brightest ghost once the sort has one.
            self._selected_number = self._view[0].number if self._view else None
        elif ghost_survey.find_by_number(self._view, previous) is None:
            # The selected ghost is no longer offered (culled away, or the lens
            # changed under us). Land on the nearest surviving number rather
            # than snapping to the first, so a cull toggle keeps the user near
            # the ghost they were looking at.
            if not self._view:
                self._selected_number = None
            elif previous is None:
                self._selected_number = self._view[0].number
            else:
                self._selected_number = min(
                    self._view, key=lambda e: abs(e.number - previous)
                ).number
        self._sync_ghost_widgets()
        if self._selected_number != previous and self._auto_render:
            self.request_render()

    # ------------------------------------------------------------------
    # Rough whole-flare pass (exposure reference + cull ranking)
    # ------------------------------------------------------------------

    def _needs_ghost_peaks(self) -> bool:
        """True when something is consuming per-ghost brightness.

        Per-pair AOV layers are what make the rough pass cost scale with the
        ghost count, so they are only requested when the cull is filtering by
        brightness or the scrubber is ordering by it.
        """
        return self._cull or self._sort_by_brightness

    def _maybe_launch_rough(self) -> None:
        """Kick the coarse all-ghosts render if its result is stale.

        This is the panel's only render of more than one ghost, and it exists
        because the exposure has to describe the flare rather than whichever
        pair happens to be selected. It waits for the display render to finish
        rather than competing with it for the GPU; :meth:`_poll_results`
        retries on the next tick.

        Per-pair layers ride along only when :meth:`_needs_ghost_peaks` says
        someone wants them; otherwise this is a single small render.
        """
        if not self._rough_dirty or self._rough_busy:
            return
        if not self._is_active or not self._lens_eligible() or not self._ghosts:
            return
        with self._lock:
            if self._busy:
                return
        self._rough_busy = True

        settings = replace(
            self._settings,
            width_px=ghost_survey.ROUGH_WIDTH_PX,
            ray_grid=ghost_survey.ROUGH_RAY_GRID,
            spectral_samples=ghost_survey.ROUGH_SPECTRAL,
        ).clamp()
        width, height, half_w, half_h = self._resolve_render_dims(settings)
        lens = self._project.system
        calib = None
        try:
            lens._check_invalidate()
            calib = lens.calibration()
        except Exception:
            _log.exception("GhostExplorerPanelBody: rough calibration failed")
        cfg = build_flare_config(
            SOURCE_X, SOURCE_Y, half_w, half_h, settings, self._matte,
        )
        epoch = self._epoch
        want_peaks = self._needs_ghost_peaks()
        if not self._status.text().startswith("Rendering"):
            self._status.setText("Metering flare…")
        threading.Thread(
            target=self._rough_worker,
            args=(lens, calib, cfg, width, height, want_peaks, epoch),
            daemon=True,
        ).start()

    def _rough_worker(
        self, lens, calib, cfg, width: int, height: int,
        want_peaks: bool, epoch: int,
    ) -> None:
        try:
            hwc, peaks = ghost_survey.render_rough_survey(
                lens, calib, cfg, width=width, height=height,
                want_peaks=want_peaks,
            )
        except Exception:
            _log.exception("GhostExplorerPanelBody: rough pass failed")
            hwc, peaks = None, {}
        finally:
            self._rough_busy = False
        self._rough_results.put((epoch, hwc, peaks))

    def _drain_rough_results(self) -> None:
        """Fold a finished rough pass into the panel state (GUI thread)."""
        landed = False
        for_cull = False
        while not self._rough_results.empty():
            epoch, hwc, peaks = self._rough_results.get()
            # Whatever the outcome, stop retrying: a failed pass leaves the
            # entries unmetered, so nothing is culled on absent evidence.
            self._rough_dirty = False
            if epoch != self._epoch:
                continue
            if hwc is not None:
                self._rough_hwc = hwc
                landed = True
            if peaks:
                self._ghosts = ghost_survey.apply_peaks(self._ghosts, peaks)
                for_cull = True
        if for_cull:
            self._rebuild_view()
        if landed and self._auto_expose_pending:
            self._auto_expose_pending = False
            self.auto_expose()
        if not self._rough_busy and self._status.text().startswith("Metering"):
            self._status.setText(self._final_status_text())

    # ------------------------------------------------------------------
    # Matte-box state
    # ------------------------------------------------------------------

    def _on_matte_enabled_changed(self, checked: bool) -> None:
        self._matte = replace(self._matte, enabled=bool(checked))
        self._sync_matte_widget_enabled()
        # The flag clips entrance rays, so it changes each ghost's throughput —
        # and therefore the exposure reference and the cull ranking.
        self._rough_dirty = True
        if self._auto_render:
            self.request_render()
        self._maybe_launch_rough()

    def _on_matte_w_changed(self, val: float) -> None:
        self._matte = replace(self._matte, half_w_mm=float(val))
        self._on_matte_geometry_changed()

    def _on_matte_h_changed(self, val: float) -> None:
        self._matte = replace(self._matte, half_h_mm=float(val))
        self._on_matte_geometry_changed()

    def _on_matte_z_changed(self, val: float) -> None:
        self._matte = replace(self._matte, z_front_mm=float(val))
        self._on_matte_geometry_changed()

    def _on_matte_geometry_changed(self) -> None:
        if not self._matte.enabled:
            return
        self._rough_dirty = True
        if self._auto_render:
            self.request_render()
        self._maybe_launch_rough()

    def _sync_matte_widget_enabled(self) -> None:
        on = self._matte.enabled
        self._spin_matte_w.setEnabled(on)
        self._spin_matte_h.setEnabled(on)
        self._spin_matte_z.setEnabled(on)

    # ------------------------------------------------------------------
    # Project signal handlers
    # ------------------------------------------------------------------

    def _on_system_replaced(self, _system) -> None:
        self._epoch += 1
        self._exposure_stops = 0.0
        # A different lens can sit many stops away from the old one, so meter
        # the first rough pass it produces rather than leaving the user at 0.
        self._auto_expose_pending = True
        self._latest_hwc = None
        self._rough_hwc = None
        self._vignette.invalidate()
        # A different lens has a different ghost list; drop the selection so
        # the re-survey seats on the head of the new one rather than carrying a
        # stale number (or a stale choice) across from the old lens.
        self._selected_number = None
        self._user_selected = False
        self._survey_dirty = True
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        self._refresh_survey()
        self._refresh_status_for_lens()
        if self._lens_eligible():
            self.request_render()
        else:
            self._canvas.clear_image(self._placeholder_for_lens())

    def _on_system_modified(self) -> None:
        self._vignette.invalidate()
        # Surface geometry, materials and muting all move the ghost list.
        self._survey_dirty = True
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _on_system_setup_changed(self) -> None:
        self._epoch += 1
        self._latest_hwc = None
        self._rough_hwc = None
        self._vignette.invalidate()
        # The sensor size crops the frame, so both the ghost list and the
        # exposure reference move.
        self._survey_dirty = True
        if self._desqueeze:
            self._refresh_squeeze_factor()
            self._push_squeeze_to_canvas()
        if not self._should_auto_render():
            self._dirty_pending = True
            return
        self.request_render()

    def _should_auto_render(self) -> bool:
        return self._auto_render and self._app_settings.auto_update_enabled()

    def _on_global_auto_update_changed(self, enabled: bool) -> None:
        if enabled and self._dirty_pending and self._auto_render:
            self.request_render()

    def _refresh_squeeze_factor(self) -> None:
        sensor = self._project.system_setup.sensor
        try:
            metrics = lm_mod.compute_metrics(
                self._project.system,
                float(sensor.width_mm),
                float(sensor.height_mm),
            )
        except Exception:
            _log.exception("GhostExplorerPanelBody: compute_metrics failed")
            metrics = None
        self._squeeze_factor = float(metrics.squeeze) if metrics is not None else 1.0

    def _push_squeeze_to_canvas(self) -> None:
        self._canvas.set_squeeze(self._squeeze_factor if self._desqueeze else 1.0)

    # ------------------------------------------------------------------
    # Render dispatch
    # ------------------------------------------------------------------

    def _lens_eligible(self) -> bool:
        try:
            return self._project.system.num_surfaces() >= MIN_SURFACES
        except Exception:
            return False

    def _placeholder_for_lens(self) -> str:
        try:
            n = self._project.system.num_surfaces()
        except Exception:
            n = 0
        if n < MIN_SURFACES:
            return f"Add at least {MIN_SURFACES} surfaces to render"
        return "Load a lens to render"

    def _refresh_status_for_lens(self) -> None:
        if self._lens_eligible():
            self._status.setText("Rendering…")
        else:
            self._status.setText(self._placeholder_for_lens())

    def _resolve_render_dims(
        self, settings: Optional[RenderSettings] = None
    ) -> Tuple[int, int, float, float]:
        settings = settings if settings is not None else self._settings
        sensor = self._project.system_setup.sensor
        w_mm = float(sensor.width_mm) if sensor.width_mm > 0 else 24.0
        h_mm = float(sensor.height_mm) if sensor.height_mm > 0 else 16.0
        width_px = int(settings.width_px)
        height_px = max(8, int(round(width_px * (h_mm / w_mm))))
        return width_px, height_px, w_mm * 0.5, h_mm * 0.5

    def _final_status_text(self) -> str:
        w, h, _, _ = self._resolve_render_dims()
        entry = self.selected_ghost()
        ghost = f"ghost {entry.number}" if entry is not None else "no ghost"
        matte = ""
        if self._matte.enabled:
            matte = (f"  matte {self._matte.half_w_mm:.0f}×{self._matte.half_h_mm:.0f}"
                     f"@{self._matte.z_front_mm:.0f}mm")
        return (
            f"{ghost}  point src ({SOURCE_X:.2f}, {SOURCE_Y:.2f})  "
            f"{self._exposure_stops:+.2f} st  {w}×{h}{matte}"
        )

    def _request(self) -> None:
        was_idle = False
        with self._lock:
            was_idle = (not self._busy) and (not self._pending)
            self._pending = True
            self._serial += 1
        if was_idle and self._status.text() != "Rendering…":
            self._status.setText("Rendering…")
        self._maybe_launch()

    def _maybe_launch(self) -> None:
        if not self._is_active or not self._auto_render:
            return
        if self._rough_busy:
            # The rough pass owns the GPU for its (short) duration; the poll
            # timer relaunches us as soon as it lands.
            return
        if not self._lens_eligible():
            return
        entry = self.selected_ghost()
        if entry is None:
            return

        with self._lock:
            if self._busy or not self._pending:
                return
            self._pending = False
            self._busy = True
            serial = self._serial

        width, height, half_w, half_h = self._resolve_render_dims()
        lens = self._project.system
        calib = None
        try:
            lens._check_invalidate()
            calib = lens.calibration()
        except Exception:
            _log.exception("GhostExplorerPanelBody: calibration failed")

        # The scrubber owns the render's GhostFilter: one INCLUDE entry, so the
        # GPU traces this pair and nothing else. (Project-level ghost solo is
        # ignored here — there is only one filter slot per render.)
        ghost_filter = ghostlight.GhostFilter()
        ghost_filter.mode = ghostlight.GhostFilter.Mode.INCLUDE
        ghost_filter.pairs = [(int(entry.surf_a), int(entry.surf_b))]

        settings = self._settings
        matte = self._matte
        epoch = self._epoch
        stops = self._exposure_stops
        try:
            view_spec = vt.spec_from_settings(self._app_settings)
        except vt.ViewTransformError:
            view_spec = None
        threading.Thread(
            target=self._worker,
            args=(
                lens, calib, width, height, half_w, half_h,
                settings, matte, stops, view_spec, epoch, serial, ghost_filter,
            ),
            daemon=True,
        ).start()

    def _worker(
        self,
        lens: "ghostlight.OpticalSystem",
        calib,
        width: int,
        height: int,
        half_w: float,
        half_h: float,
        settings: RenderSettings,
        matte: "_MatteBoxState",
        stops: float,
        view_spec,
        epoch: int,
        serial: int,
        ghost_filter: "ghostlight.GhostFilter",
    ) -> None:
        """Producer thread: one render of one ghost.

        A point source is a single unit-weight sample, so there is nothing to
        accumulate progressively — the frame either lands whole or the user has
        already scrubbed past it, which the serial / epoch check below catches
        before the result is ever painted.
        """
        try:
            hwc, _flare = render_chunk(
                lens, calib, source_sampling.sample_point(), SOURCE_X, SOURCE_Y,
                width, height, half_w, half_h, settings, matte,
                ghost_filter=ghost_filter,
            )
            if serial != self._serial or epoch != self._epoch:
                return  # stale — the user moved on while this was rendering
            qimg = None
            if view_spec is not None:
                try:
                    qimg = vt.to_qimage(vt.apply_view(hwc, stops, view_spec))
                except Exception:
                    _log.exception(
                        "GhostExplorerPanelBody: view transform failed"
                    )
            self._results.put((epoch, qimg, hwc))
        except Exception:
            _log.exception("GhostExplorerPanelBody: render failed")
            self._results.put((epoch, None, None))
        finally:
            with self._lock:
                self._busy = False

    def _poll_results(self) -> None:
        self._drain_rough_results()

        latest_qimg = None
        latest_hwc: Optional[np.ndarray] = None
        had_error = False
        while not self._results.empty():
            epoch, qimg, hwc = self._results.get()
            if epoch != self._epoch:
                continue
            if hwc is None:
                had_error = True
                continue
            latest_qimg = qimg
            latest_hwc = hwc

        if latest_hwc is not None:
            self._latest_hwc = latest_hwc
            if latest_qimg is not None:
                self._canvas.set_image(latest_qimg)
            else:
                self._redisplay()

        with self._lock:
            still_busy = self._busy or self._pending
        if self._rough_busy:
            pass  # the rough pass owns the status line until it lands
        elif still_busy:
            if not self._status.text().startswith("Rendering"):
                self._status.setText("Rendering…")
        elif had_error and latest_hwc is None:
            self._status.setText("Render failed (see log)")
        elif latest_qimg is not None:
            self._status.setText(self._final_status_text())

        self._maybe_launch()
        self._maybe_launch_rough()


def _strip_layers(settings: RenderSettings) -> RenderSettings:
    """``settings`` with the starburst, veiling-glare and gate layers forced off.

    This panel renders one ghost's geometric contribution; the additive layers
    are whole-frame passes that belong to no particular ghost pair, so leaving
    them on would paint the same wash over every scrubber position.
    """
    return replace(settings, starburst=False, veil=False, gate=False).clamp()
