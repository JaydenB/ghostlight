"""The surface right-click 'Coating' submenu: present on optical surfaces
(the add path for bare surfaces), hidden on aperture stops, and its actions
apply / remove coatings through coating_actions."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMenu

import ghostlight

from ghostlight_designer.optical_editor import OpticalEditorBody
from ghostlight_designer.optical_editor.nodes import build_tree, SurfaceNode, CoatingFormNode
from ghostlight_designer.project import Project

_COATED, _UNCOATED, _STOP = 0, 1, 3


def _body(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    return project, OpticalEditorBody(project)


def test_coating_submenu_present_on_optical_surface(qapp, sample_lens_path):
    project, body = _body(qapp, sample_lens_path)
    try:
        menu = QMenu(body)
        sub = body._populate_coating_submenu(menu, _UNCOATED)
        assert sub is not None
        labels = [a.text() for a in sub.actions() if a.text()]
        # Every catalogue preset is offered; a bare surface gets no "Remove".
        assert "Vintage Amber (artist)" in labels
        assert "Remove Coating" not in labels
    finally:
        body.deleteLater()


def test_coating_submenu_offers_remove_when_coated(qapp, sample_lens_path):
    project, body = _body(qapp, sample_lens_path)
    try:
        menu = QMenu(body)
        sub = body._populate_coating_submenu(menu, _COATED)
        assert sub is not None
        assert "Remove Coating" in [a.text() for a in sub.actions()]
    finally:
        body.deleteLater()


def test_coating_submenu_absent_on_stop(qapp, sample_lens_path):
    project, body = _body(qapp, sample_lens_path)
    try:
        menu = QMenu(body)
        assert body._populate_coating_submenu(menu, _STOP) is None
    finally:
        body.deleteLater()


def test_submenu_apply_action_adds_coating_and_row(qapp, sample_lens_path):
    project, body = _body(qapp, sample_lens_path)
    try:
        menu = QMenu(body)
        sub = body._populate_coating_submenu(menu, _UNCOATED)
        amber = next(a for a in sub.actions() if a.text() == "Vintage Amber (artist)")
        amber.trigger()

        c = project.system.surfaces[_UNCOATED].coating
        assert int(c.model) == int(ghostlight.CoatingModel.ARTIST)
        # The tree now shows a coating row for the newly-coated surface.
        root = build_tree(project.system)
        found = False
        for el in root.children:
            for surf in el.children:
                if isinstance(surf, SurfaceNode) and surf.surface_index == _UNCOATED:
                    found = any(isinstance(ch, CoatingFormNode) for ch in surf.children)
        assert found
        assert project.can_undo
    finally:
        body.deleteLater()
