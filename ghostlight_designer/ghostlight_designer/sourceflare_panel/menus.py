"""Menus contributed by the ``sourceflare`` panel type."""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ..project import Project
from .body import SourceFlarePanelBody


def build_menus(body: SourceFlarePanelBody, project: Project) -> List[QMenu]:
    view_menu = QMenu("&View")

    act_auto = QAction("&Auto Render", view_menu)
    act_auto.setCheckable(True)
    act_auto.setChecked(body.auto_render)
    act_auto.toggled.connect(body.set_auto_render)
    view_menu.addAction(act_auto)

    act_render = QAction("&Re-render", view_menu)
    act_render.triggered.connect(body.request_render)
    view_menu.addAction(act_render)

    act_recenter = QAction("Re&center Source", view_menu)
    act_recenter.triggered.connect(body.recenter_source)
    view_menu.addAction(act_recenter)

    act_render_anim = QAction("Render &Animation…", view_menu)
    act_render_anim.setToolTip(
        "Render an animation of the flare following a motion pattern across "
        "the frame and export it as a GIF, MOV, or JPEG / EXR sequence."
    )
    act_render_anim.triggered.connect(body.open_export_dialog)
    view_menu.addAction(act_render_anim)

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
        "Meter a convenience exposure (in stops) from the most recent render."
    )
    act_auto_expose.triggered.connect(body.auto_expose)
    view_menu.addAction(act_auto_expose)

    act_reset_exposure = QAction("Re&set Exposure", view_menu)
    act_reset_exposure.setToolTip("Reset the viewer exposure to 0 stops.")
    act_reset_exposure.triggered.connect(body.reset_exposure)
    view_menu.addAction(act_reset_exposure)

    settings_menu = QMenu("&Settings")

    # Draft / Mid / High differ ONLY in how well the ghost pass is sampled —
    # the extra whole-frame layers stay off across all three, so stepping up
    # cleans the same picture rather than changing what is in it. High+ is the
    # one that adds them.
    act_draft = QAction("&Draft", settings_menu)
    act_draft.setToolTip(
        "Fastest preview — coarse ghost sampling at low resolution, extra "
        "layers off. For framing a shot."
    )
    act_draft.triggered.connect(body.apply_preset_draft)
    settings_menu.addAction(act_draft)

    act_mid = QAction("&Mid", settings_menu)
    act_mid.setToolTip(
        "Default — ghost sampling between Draft and High, extra layers off."
    )
    act_mid.triggered.connect(body.apply_preset_mid)
    settings_menu.addAction(act_mid)

    act_high = QAction("&High", settings_menu)
    act_high.setToolTip(
        "Full-quality ghost render, plus HURB ghost-edge glow — the physical "
        "diffraction kick each ray takes passing a hard edge. Starburst, "
        "veiling glare and gate flare stay off."
    )
    act_high.triggered.connect(body.apply_preset_high)
    settings_menu.addAction(act_high)

    act_high_plus = QAction("High&+", settings_menu)
    act_high_plus.setToolTip(
        "High, plus the three whole-frame extras: aperture starburst "
        "(exact MDFT engine), veiling glare, and gate flare. The full look — "
        "and the slowest."
    )
    act_high_plus.triggered.connect(body.apply_preset_high_plus)
    settings_menu.addAction(act_high_plus)

    settings_menu.addSeparator()

    act_custom = QAction("&Custom…", settings_menu)
    act_custom.setToolTip(
        "Edit resolution, ray grid, and spectral count. Sample count "
        "lives in the panel's Samples spinbox. Settings persist for "
        "the lifetime of this panel only."
    )
    act_custom.triggered.connect(body.open_render_settings_dialog)
    settings_menu.addAction(act_custom)

    return [view_menu, settings_menu]
