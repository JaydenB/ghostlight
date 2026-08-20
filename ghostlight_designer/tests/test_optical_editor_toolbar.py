"""Tests for the optical-editor toolbar widget + body wiring."""
from __future__ import annotations

import pytest

import ghostlight
from PySide6.QtWidgets import QMenu, QMessageBox

from ghostlight_designer.optical_editor import element_actions, surface_actions
from ghostlight_designer.optical_editor.body import OpticalEditorBody
from ghostlight_designer.optical_editor.toolbar import (
    ADD_APERTURE_STOP,
    ADD_DOUBLET,
    ADD_SINGLET,
    OpticalEditorToolbar,
)
from ghostlight_designer.project import Project


# ---------------------------------------------------------------------------
# Toolbar widget (standalone)
# ---------------------------------------------------------------------------


def test_toolbar_remove_disabled_when_no_selection(qapp):
    project = Project()
    toolbar = OpticalEditorToolbar(project)
    try:
        assert toolbar.btn_remove.isEnabled() is False
    finally:
        toolbar.deleteLater()


def test_toolbar_remove_enables_with_selection(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    toolbar = OpticalEditorToolbar(project)
    try:
        project.set_selected_element(project.system.elements[0])
        assert toolbar.btn_remove.isEnabled() is True
    finally:
        toolbar.deleteLater()


def test_toolbar_buttons_use_painted_icons(qapp):
    project = Project()
    toolbar = OpticalEditorToolbar(project)
    try:
        # Glyph icons, not text labels — verify both buttons carry a
        # non-null QIcon and no fallback text.
        assert not toolbar.btn_add.icon().isNull()
        assert not toolbar.btn_remove.icon().isNull()
        assert toolbar.btn_add.text() in ("", None)
        assert toolbar.btn_remove.text() in ("", None)
    finally:
        toolbar.deleteLater()


def test_toolbar_emits_add_signal_for_each_kind(qapp):
    project = Project()
    toolbar = OpticalEditorToolbar(project)
    try:
        received: list[str] = []
        toolbar.addElementRequested.connect(lambda kind: received.append(kind))
        # Triggering the QAction directly bypasses popping the menu (which
        # would require a running event loop with focus).
        actions = toolbar.add_menu.actions()
        kinds = [a.text() for a in actions]
        assert "Singlet" in kinds and "Aperture Stop" in kinds
        for a in actions:
            a.trigger()
        assert ADD_SINGLET in received
        assert ADD_DOUBLET in received
        assert ADD_APERTURE_STOP in received
    finally:
        toolbar.deleteLater()


# ---------------------------------------------------------------------------
# State-clear buttons — Un-Solo All / Un-Mute All
# ---------------------------------------------------------------------------


def test_toolbar_unsolo_and_unmute_disabled_on_clean_state(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    toolbar = OpticalEditorToolbar(project)
    try:
        assert toolbar.btn_unsolo_all.isEnabled() is False
        assert toolbar.btn_unmute_all.isEnabled() is False
    finally:
        toolbar.deleteLater()


def test_toolbar_unsolo_enables_when_surface_soloed(qapp, sample_lens_path):
    """Button-state gating: standalone toolbar refreshes off
    ghostSoloChanged even without a body wired in."""
    project = Project()
    project.load(str(sample_lens_path))
    toolbar = OpticalEditorToolbar(project)
    try:
        assert toolbar.btn_unsolo_all.isEnabled() is False
        surface_actions.set_surface_ghost_solo(project, 0, True)
        assert toolbar.btn_unsolo_all.isEnabled() is True
        # Clearing solo through the same helper flips the button back
        # off — verifies the refresh signal path, without depending on
        # body wiring for the click action.
        project.clear_ghost_solo()
        assert toolbar.btn_unsolo_all.isEnabled() is False
    finally:
        toolbar.deleteLater()


def test_toolbar_unmute_enables_when_element_muted(qapp, sample_lens_path):
    """Button-state gating: standalone toolbar refreshes off
    systemModified when an element flips mute."""
    project = Project()
    project.load(str(sample_lens_path))
    toolbar = OpticalEditorToolbar(project)
    try:
        assert toolbar.btn_unmute_all.isEnabled() is False
        target = next(
            el for el in project.system.elements if el.kind != ghostlight.ElementKind.STOP
        )
        element_actions.set_element_muted(project, target, True)
        assert toolbar.btn_unmute_all.isEnabled() is True
        element_actions.set_element_muted(project, target, False)
        assert toolbar.btn_unmute_all.isEnabled() is False
    finally:
        toolbar.deleteLater()


def test_body_unsolo_all_click_clears_every_surface(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        surface_actions.set_surface_ghost_solo(project, 0, True)
        surface_actions.set_surface_ghost_solo(project, 1, True)
        assert len(project.ghost_solo_surface_uuids) == 2

        body.toolbar.btn_unsolo_all.click()
        assert not project.ghost_solo_surface_uuids
    finally:
        body.deleteLater()


def test_toolbar_unmute_all_batches_into_single_undo(qapp, sample_lens_path):
    """Un-Mute All must land as one undo entry regardless of how many
    elements it unmutes, so a single Ctrl+Z restores the prior state."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        # Mute every element that can be muted.
        for el in project.system.elements:
            if el.kind != ghostlight.ElementKind.STOP:
                element_actions.set_element_muted(project, el, True)
        n_muted_before = sum(
            1 for el in project.system.elements
            if el.is_muted(project.system)
        )
        assert n_muted_before >= 2  # sample has enough non-stop elements

        body.toolbar.btn_unmute_all.click()

        assert element_actions.any_element_muted(project) is False
        assert project.undo_label == "Unmute All"

        project.undo()
        # A single undo restores every muted element in one step.
        after_undo = sum(
            1 for el in project.system.elements
            if el.is_muted(project.system)
        )
        assert after_undo == n_muted_before
    finally:
        body.deleteLater()


def test_toolbar_new_state_buttons_use_painted_icons(qapp):
    project = Project()
    toolbar = OpticalEditorToolbar(project)
    try:
        assert not toolbar.btn_unsolo_all.icon().isNull()
        assert not toolbar.btn_unmute_all.icon().isNull()
        assert toolbar.btn_unsolo_all.text() in ("", None)
        assert toolbar.btn_unmute_all.text() in ("", None)
    finally:
        toolbar.deleteLater()


# ---------------------------------------------------------------------------
# Body wiring — add / remove / set-form end-to-end
# ---------------------------------------------------------------------------


def test_body_add_inserts_at_front_regardless_of_selection(qapp, sample_lens_path):
    """The body must NOT pass the current selection as the insertion
    anchor — adds always land at the front of the chain so existing
    lenses don't shift in absolute z."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        n_before = len(project.system.elements)
        # Select something in the middle of the chain to prove that
        # selection is ignored for placement.
        project.set_selected_element(project.system.elements[1])
        z_before = {
            uuid: project.system.surfaces[i].z
            for i, uuid in enumerate(project.system.surface_ids)
        }

        body._on_add_element_requested(ADD_DOUBLET)

        assert len(project.system.elements) == n_before + 1
        new_el = project.system.elements[0]
        assert project.selected_element is new_el
        assert project.selected_surface_index is None
        # Every existing surface keeps its absolute z.
        for i, uuid in enumerate(project.system.surface_ids):
            if uuid in z_before:
                assert project.system.surfaces[i].z == pytest.approx(z_before[uuid])
    finally:
        body.deleteLater()


def test_body_remove_skipped_when_user_says_no(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        n_before = len(project.system.elements)
        project.set_selected_element(project.system.elements[0])

        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **kw: QMessageBox.No),
        )
        body._on_remove_element_requested()
        assert len(project.system.elements) == n_before
    finally:
        body.deleteLater()


def test_body_remove_runs_when_user_confirms(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        n_before = len(project.system.elements)
        victim = project.system.elements[0]
        project.set_selected_element(victim)

        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **kw: QMessageBox.Yes),
        )
        body._on_remove_element_requested()
        assert len(project.system.elements) == n_before - 1
        assert victim not in project.system.elements
    finally:
        body.deleteLater()


def test_body_empty_area_context_menu_has_add_element(qapp, sample_lens_path):
    """Right-clicking somewhere with no row under the cursor (or on a row
    that doesn't have its own custom menu, like a material row) offers
    Add Element with the four builders plus the two import-from-file items,
    mirroring the toolbar + button."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        menu = QMenu()
        body._populate_add_element_menu(menu)
        # First action is the "Add Element" submenu.
        first = menu.actions()[0]
        assert first.text() == "Add Element"
        sub = first.menu()
        assert sub is not None
        labels = [a.text() for a in sub.actions()]
        # The four fixed builders lead; a separator then the import items.
        assert labels[:4] == ["Singlet", "Doublet", "Triplet", "Aperture Stop"]
        assert "Import Lens → Front (object side)…" in labels
        assert "Import Lens → Back (sensor side)…" in labels

        # Triggering one actually adds — uses the body's add handler so
        # the new element goes to the front and gets selected.
        n_before = len(project.system.elements)
        next(a for a in sub.actions() if a.text() == "Doublet").trigger()
        assert len(project.system.elements) == n_before + 1
        assert project.system.elements[0] is project.selected_element
    finally:
        body.deleteLater()


def test_body_element_and_surface_menus_do_not_offer_add(qapp, sample_lens_path):
    """Custom-context rows (ElementNode, SurfaceNode) must NOT carry the
    Add Element submenu — that menu is the fallback for rows without
    their own context."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        el = project.system.elements[0]
        el_menu = QMenu()
        body._populate_element_menu(el_menu, el)
        assert "Add Element" not in [a.text() for a in el_menu.actions()]

        from ghostlight_designer.optical_editor.nodes import SurfaceNode
        si = el.resolve_surfaces(project.system)[0]
        uuid = project.system.surface_ids[si]
        surf_node = SurfaceNode(uuid, si, el, parent=None)
        surf_menu = QMenu()
        body._populate_surface_menu(surf_menu, surf_node)
        assert "Add Element" not in [a.text() for a in surf_menu.actions()]
    finally:
        body.deleteLater()


def test_body_element_context_menu_flip_runs_for_multisurface(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        doublet = next(
            el for el in project.system.elements if el.name == "Front Doublet"
        )
        uuids_before = list(doublet.surface_ids)

        menu = QMenu()
        body._populate_element_menu(menu, doublet)
        flip_action = next(a for a in menu.actions() if a.text() == "Flip")
        assert flip_action.isEnabled() is True
        flip_action.trigger()

        doublet_after = next(
            el for el in project.system.elements if el.name == "Front Doublet"
        )
        assert list(doublet_after.surface_ids) == list(reversed(uuids_before))
    finally:
        body.deleteLater()


def test_body_element_context_menu_flip_disabled_for_aperture_stop(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        stop = next(
            el for el in project.system.elements if el.name == "Aperture Stop"
        )
        menu = QMenu()
        body._populate_element_menu(menu, stop)
        flip_action = next(a for a in menu.actions() if a.text() == "Flip")
        assert flip_action.isEnabled() is False
    finally:
        body.deleteLater()


def test_body_element_context_menu_has_remove(qapp, sample_lens_path, monkeypatch):
    """Right-clicking an element row offers a Remove option that goes through
    the same confirmation dialog as the toolbar button."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        victim = project.system.elements[0]
        n_before = len(project.system.elements)

        menu = QMenu()
        body._populate_element_menu(menu, victim)
        actions = [a.text() for a in menu.actions()]
        assert "Remove" in actions

        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **kw: QMessageBox.Yes),
        )
        remove_action = next(a for a in menu.actions() if a.text() == "Remove")
        remove_action.trigger()

        assert len(project.system.elements) == n_before - 1
        assert victim not in project.system.elements
    finally:
        body.deleteLater()


def test_body_element_context_menu_remove_respects_no(qapp, sample_lens_path, monkeypatch):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        victim = project.system.elements[0]
        n_before = len(project.system.elements)

        menu = QMenu()
        body._populate_element_menu(menu, victim)
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **kw: QMessageBox.No),
        )
        next(a for a in menu.actions() if a.text() == "Remove").trigger()
        assert len(project.system.elements) == n_before
    finally:
        body.deleteLater()


def test_body_form_submenu_lists_all_forms_and_writes_through(
    qapp, sample_lens_path,
):
    """Right-click on a surface row → Form submenu lists every SurfaceForm
    with the current one ticked; triggering a different one mutates."""
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        el = project.system.elements[0]
        si = el.resolve_surfaces(project.system)[0]
        current = int(project.system.surfaces[si].form)

        menu = QMenu()
        body._populate_form_submenu(menu, si)
        # First action is the "Swap Form" submenu.
        form_action = menu.actions()[0]
        assert form_action.text() == "Swap Form"
        form_menu = form_action.menu()
        assert form_menu is not None

        form_actions = form_menu.actions()
        assert len(form_actions) == len(ghostlight.SurfaceForm.__members__)
        checked = [a for a in form_actions if a.isChecked()]
        assert len(checked) == 1
        # The ticked entry is the surface's current form.
        # (Match by display label since that's what's exposed on the action.)
        from ghostlight_designer.optical_editor import surface_actions
        assert checked[0].text() == surface_actions.form_label(current)

        # Trigger a different form and verify it took.
        target = next(
            int(m) for m in ghostlight.SurfaceForm.__members__.values() if int(m) != current
        )
        target_label = surface_actions.form_label(target)
        target_action = next(a for a in form_actions if a.text() == target_label)
        target_action.trigger()
        assert int(project.system.surfaces[si].form) == target
        assert project.undo_label == "Set Form"
    finally:
        body.deleteLater()
