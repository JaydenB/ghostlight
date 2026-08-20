"""Menus contributed by the ``spot_diagram`` panel type.

Order and grouping mirror the existing render panels
(``sourceflare_panel``, ``ghost_explorer_panel``, ``psf_panel``) so users see
the same affordances in the same places across panels:

1. **Auto-Update** (checkable) — per-panel autocomputation toggle,
   same slot as the render panels' ``Auto Render``.
2. **Refresh** — manual recompute that bypasses the edit-settle
   debounce, same slot as the render panels' ``Re-render``.
3. **Sync from System Setup** — data-source action, in the same slot
   the source-flare panel uses for ``Recenter Source``.
4. *—separator—*
5. **Show Settings Sidebar** (checkable) — display toggle in the slot
   the other panels use for ``Desqueeze`` / ``Correct Distortion`` /
   ``Per-tile Normalisation``.
6. *—separator—*
7. **Reset to Defaults** — one-shot reset, same slot as the render
   panels' ``Reset Color Correction``.
"""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ...project import Project
from .body import SpotDiagramBody


def build_menus(body: SpotDiagramBody, project: Project) -> List[QMenu]:
    view_menu = QMenu("&View")

    # --- Group 1: render-toggle + manual fire + data-refresh ---------
    act_auto = QAction("&Auto-Update", view_menu)
    act_auto.setCheckable(True)
    act_auto.setChecked(body.auto_update_local)
    act_auto.setToolTip(
        "Per-panel auto-update toggle. Lens edits only trigger a recompute "
        "when this AND the global View → Auto-Update Panels are both on."
    )
    act_auto.toggled.connect(body.set_auto_update_local)
    view_menu.addAction(act_auto)

    act_refresh = QAction("&Refresh", view_menu)
    act_refresh.setToolTip(
        "Recompute the spot diagram now, bypassing the edit-settle debounce."
    )
    act_refresh.triggered.connect(body.force_refresh_now)
    view_menu.addAction(act_refresh)

    act_sync = QAction("&Sync from System Setup", view_menu)
    act_sync.setToolTip(
        "Replace this panel's wavelengths, fields, and aperture with the "
        "project's System Setup values. Other spec knobs (rings, fans, "
        "defocus, plot extent) stay as set."
    )
    act_sync.triggered.connect(body.apply_sync_from_system_setup)
    view_menu.addAction(act_sync)

    view_menu.addSeparator()

    # --- Group 2: display toggles ------------------------------------
    act_show_settings = QAction("Show &Settings Sidebar", view_menu)
    act_show_settings.setCheckable(True)
    act_show_settings.setChecked(body.settings_visible)
    act_show_settings.setToolTip(
        "Show or hide the left-hand spec editor. Spec values are preserved "
        "either way — this is purely a layout toggle."
    )
    act_show_settings.toggled.connect(body.set_settings_visible)
    view_menu.addAction(act_show_settings)

    view_menu.addSeparator()

    # --- Group 3: resets ---------------------------------------------
    act_autofit = QAction("&Auto-Fit Scale Now", view_menu)
    act_autofit.setToolTip(
        "Drop the locked plot scale and refit each field row to its "
        "current bundle. Useful when a series of lens edits has pushed "
        "the spot well outside the locked extents — or the locked scale "
        "is now huge compared to a freshly focused spot."
    )
    act_autofit.triggered.connect(body.auto_fit_scale_now)
    view_menu.addAction(act_autofit)

    act_reset = QAction("Reset Spec to &Defaults", view_menu)
    act_reset.setToolTip(
        "Replace the current spec with the panel's default wavelengths, "
        "fields, and sampling. Triggers an immediate recompute."
    )
    act_reset.triggered.connect(body.reset_spec_to_defaults)
    view_menu.addAction(act_reset)

    return [view_menu]
