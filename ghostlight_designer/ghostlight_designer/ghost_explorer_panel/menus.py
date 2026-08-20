"""Menus contributed by the ``ghost_explorer`` panel type."""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ..project import Project
from .body import GhostExplorerPanelBody


def build_menus(body: GhostExplorerPanelBody, project: Project) -> List[QMenu]:
    view_menu = QMenu("&View")

    act_auto = QAction("&Auto Render", view_menu)
    act_auto.setCheckable(True)
    act_auto.setChecked(body.auto_render)
    act_auto.toggled.connect(body.set_auto_render)
    view_menu.addAction(act_auto)

    act_render = QAction("&Re-render", view_menu)
    act_render.triggered.connect(body.request_render)
    view_menu.addAction(act_render)

    view_menu.addSeparator()

    # --- the ghost scrubber's own options ---------------------------------
    act_cull = QAction("&Cull Dim Ghosts", view_menu)
    act_cull.setCheckable(True)
    act_cull.setChecked(body.cull_dim_ghosts)
    act_cull.setToolTip(
        "Drop ghosts too dim to see from the scrubber, so it only steps "
        "through pairs that matter. On by default. Brightness comes from the "
        "coarse whole-flare pass, which renders every pair into its own layer; "
        "anything whose peak falls below the threshold — including ghosts the "
        "renderer finds land nowhere on the sensor — is hidden. Ghost numbers "
        "do not change, so a culled list keeps the same names."
    )
    act_cull.toggled.connect(body.set_cull_dim_ghosts)
    view_menu.addAction(act_cull)

    act_cull_thr = QAction("Cull &Threshold…", view_menu)
    act_cull_thr.setToolTip(
        "Set how dim a ghost has to be before the cull hides it, as a "
        "percentage of the brightest ghost in the lens."
    )
    act_cull_thr.triggered.connect(body.open_cull_threshold_dialog)
    view_menu.addAction(act_cull_thr)

    act_sort = QAction("Sort by &Brightness", view_menu)
    act_sort.setCheckable(True)
    act_sort.setChecked(body.sort_by_brightness)
    act_sort.setToolTip(
        "Order the scrubber brightest on the left, dimmest on the right, so a "
        "sweep left to right walks the ghosts in the order they matter. On by "
        "default; uncheck for the lens's own surface-pair order, where the "
        "slider position tracks the ghost number. Only the slider order "
        "changes either way: each ghost keeps its number, and the cull is "
        "unaffected."
    )
    act_sort.toggled.connect(body.set_sort_by_brightness)
    view_menu.addAction(act_sort)

    view_menu.addSeparator()

    act_desqueeze = QAction("&Desqueeze", view_menu)
    act_desqueeze.setCheckable(True)
    act_desqueeze.setChecked(body.desqueeze)
    act_desqueeze.setToolTip(
        "Stretch the displayed image horizontally by the lens's "
        "anamorphic squeeze factor (matches the × value in the viewport "
        "info bar). The render itself is unchanged."
    )
    act_desqueeze.toggled.connect(body.set_desqueeze)
    view_menu.addAction(act_desqueeze)

    act_vignette = QAction("&Vignetting Overlay", view_menu)
    act_vignette.setCheckable(True)
    act_vignette.setChecked(body.vignette_overlay)
    act_vignette.setToolTip(
        "Tint the frame's unreachable regions half-red: sensor areas no "
        "primary ray can reach through the aperture stop and lens rims (the "
        "lens's hard image circle). Diagnostic overlay; does not affect the "
        "render."
    )
    act_vignette.toggled.connect(body.set_vignette_overlay)
    view_menu.addAction(act_vignette)

    act_matte_controls = QAction("&Matte Box Controls", view_menu)
    act_matte_controls.setCheckable(True)
    act_matte_controls.setChecked(body.matte_controls_visible)
    act_matte_controls.setToolTip(
        "Show/hide the inline front-of-lens matte-box controls below the "
        "source shape controls. Layout-only; the matte box's own on/off "
        "state is unaffected."
    )
    act_matte_controls.toggled.connect(body.set_matte_controls_visible)
    view_menu.addAction(act_matte_controls)

    view_menu.addSeparator()

    act_exposure = QAction("&Exposure…", view_menu)
    act_exposure.setToolTip(
        "Adjust the per-panel viewer exposure in stops (applied before the "
        "designer-wide ACES view transform, like Nuke's Viewer gain)."
    )
    act_exposure.triggered.connect(body.open_exposure_dialog)
    view_menu.addAction(act_exposure)

    act_auto_expose = QAction("A&uto-Expose", view_menu)
    act_auto_expose.setToolTip(
        "Meter the viewer exposure from a coarse render of the whole flare — "
        "every ghost at once, not just the selected one. One exposure then "
        "serves the entire scrub, so a ghost eight stops down reads as eight "
        "stops down instead of being normalised to look as bright as the rest."
    )
    act_auto_expose.triggered.connect(body.auto_expose)
    view_menu.addAction(act_auto_expose)

    act_reset_exposure = QAction("Re&set Exposure", view_menu)
    act_reset_exposure.setToolTip("Reset the viewer exposure to 0 stops.")
    act_reset_exposure.triggered.connect(body.reset_exposure)
    view_menu.addAction(act_reset_exposure)

    settings_menu = QMenu("&Settings")

    # Every preset lands with the starburst and veil stripped out (see
    # GhostExplorerPanelBody.apply_settings), so these differ in ghost sampling
    # fidelity and resolution only.
    act_draft = QAction("&Draft", settings_menu)
    act_draft.setToolTip("Fastest preview — coarse ghost sampling, low resolution.")
    act_draft.triggered.connect(body.apply_preset_draft)
    settings_menu.addAction(act_draft)

    act_mid = QAction("&Mid", settings_menu)
    act_mid.setToolTip("Default — ghost sampling between Draft and High.")
    act_mid.triggered.connect(body.apply_preset_mid)
    settings_menu.addAction(act_mid)

    act_high = QAction("&High", settings_menu)
    act_high.setToolTip("Full-quality ghost render — slower but smoother.")
    act_high.triggered.connect(body.apply_preset_high)
    settings_menu.addAction(act_high)

    settings_menu.addSeparator()

    act_custom = QAction("&Custom…", settings_menu)
    act_custom.setToolTip(
        "Edit resolution, ray grid, and spectral count. The starburst and "
        "veiling-glare passes are locked off in this panel. Settings persist "
        "for the lifetime of this panel only."
    )
    act_custom.triggered.connect(body.open_render_settings_dialog)
    settings_menu.addAction(act_custom)

    return [view_menu, settings_menu]
