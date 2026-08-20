"""Tests for the panel-agnostic value scrubber.

Exercises the default ``is_scrubbable`` / ``compound_label`` callbacks and
verifies the popup also drives a non-optical-editor model — here, the
system_setup tree.
"""
from __future__ import annotations

from PySide6.QtCore import QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QTreeView

from ghostlight_designer.project import Project
from ghostlight_designer.system_setup.columns import Column as SsColumn
from ghostlight_designer.system_setup.model import SystemSetupTreeModel
from ghostlight_designer.value_scrubber import (
    ScrubPopup,
    attach_value_scrubber,
    default_compound_label,
    default_is_scrubbable,
)


def _find_index(model, path):
    cur = QModelIndex()
    for label in path:
        for r in range(model.rowCount(cur)):
            child = model.index(r, 0, cur)
            if model.data(child, Qt.DisplayRole) == label:
                cur = child
                break
        else:
            raise AssertionError(f"missing tree node: {label} under {path}")
    return cur


def _value_index(model, path):
    name_idx = _find_index(model, path)
    return model.index(name_idx.row(), int(SsColumn.VALUE), name_idx.parent())


# ---------------------------------------------------------------------------
# default_is_scrubbable
# ---------------------------------------------------------------------------


def test_default_is_scrubbable_true_for_editable_float(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    # Tilt Y under the first field is an editable float.
    tilt_y = _value_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Fields", "Axial", "Tilt Y (°)"],
    )
    assert default_is_scrubbable(tilt_y)


def test_default_is_scrubbable_false_for_string_value(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    # The Sequence name is editable but string-valued.
    seq_name = _value_index(model, ["Sequences", "Auto Sequence 1"])
    assert not default_is_scrubbable(seq_name)


def test_default_is_scrubbable_true_for_int_ray_count(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    # Int-valued editable cells are scrubbable too — the popup keeps a
    # private float accumulator and only pushes rounded ints to the model.
    rc = _value_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Distribution", "Ray Count"],
    )
    assert default_is_scrubbable(rc)


def test_default_is_scrubbable_false_for_readonly_category(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    seq_cat_name = model.index(0, int(SsColumn.NAME), QModelIndex())
    seq_cat_value = model.index(0, int(SsColumn.VALUE), QModelIndex())
    assert not default_is_scrubbable(seq_cat_name)
    assert not default_is_scrubbable(seq_cat_value)


# ---------------------------------------------------------------------------
# default_compound_label
# ---------------------------------------------------------------------------


def test_default_compound_label_uses_column_header(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    tilt_y = _value_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Fields", "Axial", "Tilt Y (°)"],
    )
    assert default_compound_label(tilt_y) == "Scrub Value"


def test_default_compound_label_for_invalid_index_is_plain(qapp):
    assert default_compound_label(QModelIndex()) == "Scrub"


# ---------------------------------------------------------------------------
# Cross-panel reuse: scrubber drives system_setup model values
# ---------------------------------------------------------------------------


def test_scrubpopup_writes_through_system_setup_model(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    tree = QTreeView()
    tree.setModel(model)

    tilt_y = _value_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Fields", "Axial", "Tilt Y (°)"],
    )
    original = project.system_setup.sequences[0].source.fields[0].tilt_y_deg

    popup = ScrubPopup(tree, tilt_y, QPoint(0, 0), project)
    try:
        # Simulate a write through the popup's model glue.
        popup._write_value(original + 1.5)
        assert (
            project.system_setup.sequences[0].source.fields[0].tilt_y_deg
            == original + 1.5
        )
    finally:
        popup.close()
        popup.deleteLater()


def test_attach_value_scrubber_returns_trigger_on_system_setup(qapp):
    project = Project()
    model = SystemSetupTreeModel(project)
    tree = QTreeView()
    tree.setModel(model)

    trigger = attach_value_scrubber(tree, project)
    assert trigger is not None
    assert trigger.parentWidget() is tree


def test_scrubpopup_int_mode_writes_rounded_ints(qapp):
    """An int-valued cell only sees int writes; the popup keeps a private
    float accumulator so sub-unit motion still resolves but the model
    only updates when the rounded int changes."""
    project = Project()
    model = SystemSetupTreeModel(project)
    tree = QTreeView()
    tree.setModel(model)

    rc = _value_index(
        model,
        ["Sequences", "Auto Sequence 1", "Source", "Distribution", "Ray Count"],
    )
    original = project.system_setup.sequences[0].source.distribution.ray_count
    assert isinstance(original, int)

    popup = ScrubPopup(tree, rc, QPoint(0, 0), project)
    try:
        assert popup._is_int is True

        # Sub-unit drift doesn't push a new value through.
        popup._write_value(original + 0.3)
        assert (
            project.system_setup.sequences[0].source.distribution.ray_count
            == original
        )

        # Crossing the rounding threshold writes the next int.
        popup._write_value(original + 0.7)
        assert (
            project.system_setup.sequences[0].source.distribution.ray_count
            == original + 1
        )
        # And the value pushed to the model is an int, not a float.
        assert isinstance(
            project.system_setup.sequences[0].source.distribution.ray_count, int
        )
    finally:
        popup.close()
        popup.deleteLater()


def test_attach_value_scrubber_accepts_custom_callbacks(qapp):
    """Consumer can override the defaults."""
    project = Project()
    model = SystemSetupTreeModel(project)
    tree = QTreeView()
    tree.setModel(model)

    calls = []

    def custom_label(idx):
        calls.append(idx.column())
        return "Custom"

    trigger = attach_value_scrubber(
        tree,
        project,
        is_scrubbable=lambda _idx: False,  # nothing is scrubbable
        compound_label=custom_label,
    )
    assert trigger is not None
