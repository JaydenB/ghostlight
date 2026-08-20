"""Tests for the reusable surface-level toolbar actions."""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.optical_editor import surface_actions
from ghostlight_designer.project import Project


def test_available_forms_covers_all_surface_form_members(qapp):
    forms = surface_actions.available_forms()
    expected_count = len(ghostlight.SurfaceForm.__members__)
    assert len(forms) == expected_count
    for form_int, label in forms:
        assert isinstance(form_int, int)
        assert isinstance(label, str) and label


def test_set_surface_form_changes_form_and_pushes_undo(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    si = 0
    original = int(project.system.surfaces[si].form)
    # Pick a target form that differs from the current one.
    target = next(
        int(m) for m in ghostlight.SurfaceForm.__members__.values() if int(m) != original
    )

    assert surface_actions.set_surface_form(project, si, target) is True

    assert int(project.system.surfaces[si].form) == target
    assert project.is_dirty is True
    assert project.can_undo is True
    assert project.undo_label == "Set Form"


def test_set_surface_form_noop_returns_false(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    si = 0
    current = int(project.system.surfaces[si].form)
    assert surface_actions.set_surface_form(project, si, current) is False
    assert project.can_undo is False


def test_set_surface_form_out_of_range_returns_false(qapp, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    n = project.system.num_surfaces()
    assert surface_actions.set_surface_form(project, n + 5, int(ghostlight.SurfaceForm.SPHERE)) is False
