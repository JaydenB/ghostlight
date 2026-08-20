"""The artist coating strength is an optimizable continuous variable via a
dotted attr path (``coating.tint_strength``)."""
from __future__ import annotations

import pytest

import ghostlight

from ghostlight_designer.project import Project
from ghostlight_designer.optimization_panel.variables import (
    VariableRef,
    collect_variables,
)


def _artist_singlet(qapp, sample_lens_path) -> Project:
    project = Project()
    project.load(str(sample_lens_path))
    c = project.system.surfaces[0].coating
    c.model = ghostlight.CoatingModel.ARTIST
    c.tint_r = c.tint_g = c.tint_b = 1.0
    c.tint_strength = 0.04
    return project


def test_variable_ref_reads_writes_nested_coating(qapp, sample_lens_path):
    project = _artist_singlet(qapp, sample_lens_path)
    ref = VariableRef(surface_index=0, attr="coating.tint_strength")
    assert ref.read(project.system) == pytest.approx(0.04)
    ref.write(project.system, 0.2)
    assert project.system.surfaces[0].coating.tint_strength == pytest.approx(0.2)
    assert ref.read(project.system) == pytest.approx(0.2)


def test_collect_variables_includes_flagged_coating_strength(qapp, sample_lens_path):
    project = _artist_singlet(qapp, sample_lens_path)
    uuid = list(project.system.surface_ids)[0]
    project.set_variable_flag(uuid, "coating.tint_strength")

    variables = collect_variables(project)
    coating_vars = [v for v in variables if v.attr == "coating.tint_strength"]
    assert len(coating_vars) == 1
    assert coating_vars[0].surface_index == 0
    # The collected ref round-trips through the live system.
    assert coating_vars[0].read(project.system) == pytest.approx(0.04)
