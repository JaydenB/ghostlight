"""Menus contributed by the ``psf`` panel type."""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ..project import Project
from .body import PSFPanelBody


def build_menus(body: PSFPanelBody, project: Project) -> List[QMenu]:
    view_menu = QMenu("&View")

    act_auto = QAction("&Auto Render", view_menu)
    act_auto.setCheckable(True)
    act_auto.setChecked(body.auto_render)
    act_auto.toggled.connect(body.set_auto_render)
    view_menu.addAction(act_auto)

    act_render = QAction("&Re-render", view_menu)
    act_render.setToolTip("Render now, bypassing the edit-settle debounce.")
    act_render.triggered.connect(body.force_render_now)
    view_menu.addAction(act_render)

    view_menu.addSeparator()

    act_mono = QAction("&Monochromatic", view_menu)
    act_mono.setCheckable(True)
    act_mono.setChecked(body.settings.monochromatic)
    act_mono.setToolTip(
        "Trace at a single wavelength (d-line).  Removes lateral-chromatic "
        "spatial separation so each tile reads as one PSF shape."
    )
    act_mono.toggled.connect(body.set_monochromatic)
    view_menu.addAction(act_mono)

    act_norm = QAction("&Per-tile Normalisation", view_menu)
    act_norm.setCheckable(True)
    act_norm.setChecked(body.per_tile_norm)
    act_norm.setToolTip(
        "Normalise each PSF tile's display independently — makes dim off-"
        "axis PSFs visible at the cost of losing the relative brightness "
        "cue across tiles.  Display-only; no GPU re-render."
    )
    act_norm.toggled.connect(body.set_per_tile_norm)
    view_menu.addAction(act_norm)

    act_desqueeze = QAction("&Desqueeze", view_menu)
    act_desqueeze.setCheckable(True)
    act_desqueeze.setChecked(body.desqueeze)
    act_desqueeze.setToolTip(
        "Stretch the entire PSF composite horizontally by the lens's "
        "anamorphic squeeze factor (same × value the viewport info bar "
        "shows). Each tile scales together so you can see how each PSF "
        "would look in the de-squeezed projected image."
    )
    act_desqueeze.toggled.connect(body.set_desqueeze)
    view_menu.addAction(act_desqueeze)

    view_menu.addSeparator()

    act_tone = QAction("&Tone Mapping (diagnostic)…", view_menu)
    act_tone.setToolTip(
        "Edit the display log-gain.  Lower = Zemax-style compact PSFs; "
        "higher = lifts faint chromatic halos into view at the cost of "
        "amplifying ray-hit noise.  Display-only; no GPU re-render.\n"
        "Note: the PSF composite is peak-normalised, so it is intentionally "
        "outside the designer-wide ACES view transform."
    )
    act_tone.triggered.connect(body.open_tone_mapping_dialog)
    view_menu.addAction(act_tone)

    settings_menu = QMenu("&Settings")

    act_low = QAction("&Low", settings_menu)
    act_low.setToolTip("Coarse PSF render — quicker, more visible noise.")
    act_low.triggered.connect(body.apply_preset_low)
    settings_menu.addAction(act_low)

    act_high = QAction("&High", settings_menu)
    act_high.setToolTip("Full-quality PSF render — slower but smoother.")
    act_high.triggered.connect(body.apply_preset_high)
    settings_menu.addAction(act_high)

    settings_menu.addSeparator()

    act_custom = QAction("&Custom…", settings_menu)
    act_custom.setToolTip(
        "Edit grid size, tile dimensions, ray grid, spectral samples, "
        "splat, field fraction, and monochromatic mode. "
        "Settings persist for the lifetime of this panel only."
    )
    act_custom.triggered.connect(body.open_render_settings_dialog)
    settings_menu.addAction(act_custom)

    return [view_menu, settings_menu]
