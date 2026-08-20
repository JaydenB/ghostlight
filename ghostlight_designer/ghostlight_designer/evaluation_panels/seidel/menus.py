"""Menus contributed by the Seidel evaluation panel.

Same order + grouping as :mod:`spot_diagram.menus` and
:mod:`field_diagrams.menus` so users hop between evaluation panels
without re-learning the layout. The Seidel-specific actions (none yet)
would slot in at the bottom of the View menu after the resets group.
"""
from __future__ import annotations

from typing import List

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ...project import Project
from .body import SeidelBody


def build_menus(body: SeidelBody, project: Project) -> List[QMenu]:
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
        "Recompute the Seidel sums now, bypassing the edit-settle debounce."
    )
    act_refresh.triggered.connect(body.force_refresh_now)
    view_menu.addAction(act_refresh)

    act_sync = QAction("&Sync from System Setup", view_menu)
    act_sync.setToolTip(
        "Replace this panel's wavelengths, primary index, pupil radius, "
        "and field angle with the project's System Setup values."
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
    act_reset = QAction("Reset Spec to &Defaults", view_menu)
    act_reset.setToolTip(
        "Replace the current spec with the panel's default field, "
        "wavelengths, and pupil. Triggers an immediate recompute."
    )
    act_reset.triggered.connect(body.reset_spec_to_defaults)
    view_menu.addAction(act_reset)

    return [view_menu]
