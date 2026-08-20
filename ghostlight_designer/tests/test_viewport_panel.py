from __future__ import annotations

import pytest

from ghostlight_designer.project import Project
from ghostlight_designer.viewport_panel import ViewportPanelBody
from ghostlight_designer.viewport_panel.menus import build_menus


def test_viewport_panel_receives_systemReplaced(qapp, sample_lens_path, monkeypatch):
    calls: list = []

    from ghostlight_viewport import LensViewport
    monkeypatch.setattr(
        LensViewport, "set_lens",
        lambda self, system, elements, fit_view=True, **kwargs:
            calls.append(("set_lens", system, fit_view)),
    )
    monkeypatch.setattr(LensViewport, "set_sensor", lambda self, sensor: None)

    project = Project()
    body = ViewportPanelBody(project)
    try:
        calls.clear()
        project.load(str(sample_lens_path))
        assert any(c[0] == "set_lens" and c[1] is project.system for c in calls)
    finally:
        body.deleteLater()


def test_two_viewport_panels_both_update_on_systemModified(qapp, sample_lens_path, monkeypatch):
    counters = {"a": 0, "b": 0}
    bodies_seen: list = []

    from ghostlight_viewport import LensViewport

    def fake_set_lens(self, system, elements, fit_view=True, **kwargs):
        bodies_seen.append(self)

    monkeypatch.setattr(LensViewport, "set_lens", fake_set_lens)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda self, sensor: None)

    project = Project()
    project.load(str(sample_lens_path))

    a = ViewportPanelBody(project)
    b = ViewportPanelBody(project)
    try:
        bodies_seen.clear()
        project.mark_modified()
        assert a.viewport in bodies_seen
        assert b.viewport in bodies_seen
    finally:
        a.deleteLater()
        b.deleteLater()


def test_viewport_panel_view_menu_present(qapp, monkeypatch):
    from ghostlight_viewport import LensViewport
    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    project = Project()
    body = ViewportPanelBody(project)
    try:
        menus = build_menus(body, project)
        assert len(menus) == 1
        assert menus[0].title() == "&View"
    finally:
        body.deleteLater()


def test_viewport_surface_signal_routes_to_project(qapp, sample_lens_path, monkeypatch):
    from ghostlight_viewport import LensViewport
    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)

    project = Project()
    project.load(str(sample_lens_path))
    # Pick the first element + its first surface so the project will accept.
    el = project.system.elements[0]
    si = el.resolve_surfaces(project.system)[0]
    project.set_selected_element(el)

    body = ViewportPanelBody(project)
    try:
        # Emit the viewport signal directly; the body routes it into the
        # project, which then echoes via surfaceSelectionChanged.
        body.viewport.surfaceSelected.emit(si)
        assert project.selected_surface_index == si
    finally:
        body.deleteLater()


def test_viewport_refits_on_load_but_not_on_undo(qapp, sample_lens_path, monkeypatch):
    """User-facing contract: opening a file refits the camera so the new
    lens is framed; undo / redo must NOT refit, because the whole point
    of pressing Ctrl+Z while inspecting a change is to compare before /
    after with the same framing."""
    from ghostlight_viewport import LensViewport

    fit_history: list[bool] = []
    monkeypatch.setattr(
        LensViewport, "set_lens",
        lambda self, system, elements, fit_view=True, **kwargs:
            fit_history.append(fit_view),
    )
    monkeypatch.setattr(LensViewport, "set_sensor", lambda self, sensor: None)

    project = Project()
    body = ViewportPanelBody(project)
    try:
        # User opens a file → camera refits.
        fit_history.clear()
        project.load(str(sample_lens_path))
        assert fit_history and all(fit_history), \
            f"load should refit: {fit_history}"

        # Make an edit that participates in undo, then undo it.
        from ghostlight_designer.optical_editor import element_actions
        element_actions.add_singlet(project)
        fit_history.clear()
        project.undo()
        # systemReplaced fired (the tree model rebuilds), so set_lens was
        # called — but every call must be no-fit.
        assert fit_history, "undo should still push set_lens for rebuild"
        assert not any(fit_history), \
            f"undo must NOT refit camera: {fit_history}"

        # And the same for redo.
        fit_history.clear()
        project.redo()
        assert fit_history
        assert not any(fit_history), \
            f"redo must NOT refit camera: {fit_history}"
    finally:
        body.deleteLater()


def test_project_last_replacement_kind_tracks_source(qapp, sample_lens_path):
    project = Project()
    assert project.last_replacement_kind == "load"
    project.load(str(sample_lens_path))
    assert project.last_replacement_kind == "load"

    from ghostlight_designer.optical_editor import element_actions
    element_actions.add_singlet(project)
    # Mutating edits don't fire systemReplaced, but the kind is still
    # whatever it was set to most recently.
    assert project.last_replacement_kind == "load"

    project.undo()
    assert project.last_replacement_kind == "restore"

    project.redo()
    assert project.last_replacement_kind == "restore"

    project.new()
    assert project.last_replacement_kind == "load"


def test_project_surface_change_pushes_to_viewport(qapp, sample_lens_path, monkeypatch):
    from ghostlight_viewport import LensViewport
    received: list = []
    monkeypatch.setattr(LensViewport, "set_lens", lambda *a, **k: None)
    monkeypatch.setattr(LensViewport, "set_sensor", lambda *a, **k: None)
    monkeypatch.setattr(
        LensViewport,
        "set_selected_surface",
        lambda self, si: received.append(si),
    )

    project = Project()
    project.load(str(sample_lens_path))
    el = project.system.elements[0]
    si = el.resolve_surfaces(project.system)[0]
    project.set_selected_element(el)

    body = ViewportPanelBody(project)
    try:
        received.clear()
        project.set_selected_surface_index(si)
        assert si in received
    finally:
        body.deleteLater()
