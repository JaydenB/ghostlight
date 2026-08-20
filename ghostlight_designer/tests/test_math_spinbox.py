"""Widget tests for MathSpinBox / MathDoubleSpinBox.

Values are driven through ``lineEdit().setText`` + ``interpretText()``
rather than synthesised key events — same approach as
test_value_scrubber_undo.py, and it exercises exactly the path Qt takes
on Enter, on focus-out, and from the item delegates' explicit
``interpretText()`` calls.
"""
from __future__ import annotations

import gc

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QDoubleSpinBox, QSpinBox

from ghostlight_designer.math_spinbox import MathDoubleSpinBox, MathSpinBox


def _drive(spin: QAbstractSpinBox, text: str):
    """Type ``text`` wholesale and interpret it, as Enter would."""
    spin.lineEdit().setText(text)
    spin.interpretText()
    return spin.value()


def _destroy(widget) -> None:
    """See test_sourceflare_panel._destroy — flush DeferredDelete inside
    the owning test so widget teardown never lands mid-way through a
    later test (0xc0000374 on Windows/PySide6)."""
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()


@pytest.fixture
def dbl(qapp):
    spin = MathDoubleSpinBox()
    spin.setDecimals(4)
    spin.setRange(-1.0e9, 1.0e9)
    spin.setValue(7.0)
    yield spin
    _destroy(spin)


@pytest.fixture
def integer(qapp):
    spin = MathSpinBox()
    spin.setRange(-1000, 1000)
    spin.setValue(5)
    yield spin
    _destroy(spin)


# ---------------------------------------------------------------------------
# Evaluating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [("12*2", 24.0), ("(2+3)*4", 20.0), ("100/8", 12.5), ("2.5*4", 10.0), ("10-3", 7.0)],
)
def test_double_evaluates_expressions(dbl, text, expected):
    assert _drive(dbl, text) == pytest.approx(expected)


def test_double_still_takes_plain_numbers(dbl):
    assert _drive(dbl, "3.5") == pytest.approx(3.5)


def test_negative_number_is_not_a_subtraction(dbl):
    """Typing -5 over -2 must land on -5, which is why there is no
    leading-operator 'relative' form."""
    dbl.setValue(-2.0)
    assert _drive(dbl, "-5") == pytest.approx(-5.0)


def test_result_is_rounded_to_the_field_decimals(dbl):
    dbl.setDecimals(2)
    assert _drive(dbl, "1/3") == pytest.approx(0.33)


def test_int_rounds_half_away_from_zero(integer):
    assert _drive(integer, "10/4") == 3          # 2.5 -> 3, not banker's 2
    assert _drive(integer, "3*4") == 12
    assert _drive(integer, "2.5*2") == 5


# ---------------------------------------------------------------------------
# Reverting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["12*", "*2", "abc", "1/0", "((1+2)", "2**8", ""])
def test_bad_input_keeps_the_previous_value(dbl, text):
    dbl.setValue(4.25)
    assert _drive(dbl, text) == pytest.approx(4.25)


def test_out_of_range_result_reverts_rather_than_clamping(qapp):
    """A blade-count style field: 8*4 is a valid number but not a valid
    value, so the field keeps what it had instead of pinning to the max."""
    spin = MathSpinBox()
    spin.setRange(3, 16)
    spin.setValue(6)
    try:
        assert _drive(spin, "8*4") == 6
        assert _drive(spin, "1-9") == 6
        assert _drive(spin, "2*4") == 8      # in range, so it takes
    finally:
        _destroy(spin)


def test_bad_input_leaves_the_double_field_out_of_range_alone(dbl):
    dbl.setRange(0.0, 10.0)
    dbl.setValue(4.0)
    assert _drive(dbl, "8*4") == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Prefix / suffix
# ---------------------------------------------------------------------------

def test_suffix_survives_evaluation(qapp):
    """System Setup's angle cell carries a '°' suffix; Qt hands it to
    validate/fixup still attached."""
    spin = MathDoubleSpinBox()
    spin.setDecimals(3)
    spin.setRange(-360.0, 360.0)
    spin.setSuffix("°")
    spin.setValue(1.0)
    try:
        assert _drive(spin, "2*3°") == pytest.approx(6.0)
        assert spin.lineEdit().text().endswith("°")
    finally:
        _destroy(spin)


def test_prefix_survives_evaluation(qapp):
    spin = MathDoubleSpinBox()
    spin.setDecimals(2)
    spin.setRange(0.0, 1000.0)
    spin.setPrefix("f/")
    spin.setValue(2.0)
    try:
        assert _drive(spin, "f/2*4") == pytest.approx(8.0)
    finally:
        _destroy(spin)


# ---------------------------------------------------------------------------
# Keystroke-level behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["12*", "12*2", "1/", "(2+3", "-1+"])
def test_operator_keystrokes_are_accepted_but_stay_intermediate(dbl, text):
    """Intermediate, not Acceptable — so the character reaches the line
    edit while valueChanged stays quiet until the expression is done."""
    assert dbl.validate(text, len(text))[0] == QValidator.Intermediate


@pytest.mark.parametrize("text", ["2.", "2.5", "2.5*2"], ids=["dot", "decimal", "expr"])
def test_int_field_accepts_decimal_keystrokes(integer, text):
    """An int spinbox rejects '.' outright, which would make `2.5*2`
    untypeable left to right."""
    assert integer.validate(text, len(text))[0] == QValidator.Intermediate


@pytest.mark.parametrize("text", ["12a", "hello", "1$2"])
def test_junk_keystrokes_are_still_rejected(dbl, text):
    assert dbl.validate(text, len(text))[0] == QValidator.Invalid


def test_plain_digits_stay_acceptable(dbl):
    assert dbl.validate("12", 2)[0] == QValidator.Acceptable


# ---------------------------------------------------------------------------
# The two real commit gestures: Enter, and focusing away
# ---------------------------------------------------------------------------

@pytest.fixture
def focus_pair(qapp):
    """Two spinboxes in a shown window, so focus can actually move."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    spin = MathDoubleSpinBox(host)
    spin.setDecimals(3)
    spin.setRange(-1.0e6, 1.0e6)
    spin.setValue(7.0)
    other = MathDoubleSpinBox(host)
    layout.addWidget(spin)
    layout.addWidget(other)
    host.show()
    QApplication.processEvents()
    yield spin, other
    _destroy(host)


def test_focusing_away_evaluates_the_expression(focus_pair):
    spin, other = focus_pair
    spin.setFocus()
    QApplication.processEvents()
    spin.lineEdit().setText("12*2")
    other.setFocus()
    QApplication.processEvents()
    assert spin.value() == pytest.approx(24.0)


def test_focusing_away_from_a_bad_expression_reverts(focus_pair):
    spin, other = focus_pair
    spin.setFocus()
    QApplication.processEvents()
    spin.lineEdit().setText("9*")
    other.setFocus()
    QApplication.processEvents()
    assert spin.value() == pytest.approx(7.0)


def test_enter_evaluates_the_expression(focus_pair):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    spin, _other = focus_pair
    spin.setFocus()
    QApplication.processEvents()
    spin.lineEdit().setText("5*5")
    QApplication.sendEvent(
        spin, QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
    )
    QApplication.processEvents()
    assert spin.value() == pytest.approx(25.0)


def test_no_value_change_is_emitted_mid_expression(dbl):
    seen = []
    dbl.valueChanged.connect(seen.append)
    dbl.lineEdit().setText("12*2")
    assert seen == [], "expression text must not emit until interpreted"
    dbl.interpretText()
    assert seen == [pytest.approx(24.0)]


# ---------------------------------------------------------------------------
# Contract with the rest of the app
# ---------------------------------------------------------------------------

def test_subclasses_stay_isinstance_of_the_qt_types(dbl, integer):
    """The value scrubber picks its int mode with isinstance(w, QSpinBox)
    (spinbox_scrub._SpinBoxAdapterModel), so this must keep holding."""
    assert isinstance(integer, QSpinBox)
    assert isinstance(dbl, QDoubleSpinBox)
    assert not isinstance(dbl, QSpinBox)


def test_correction_mode_is_pinned(dbl, integer):
    """Reverting on bad input depends on this mode."""
    for spin in (dbl, integer):
        assert spin.correctionMode() == QAbstractSpinBox.CorrectToPreviousValue


# ---------------------------------------------------------------------------
# Coverage — every numeric control in the app is math-capable
# ---------------------------------------------------------------------------

def _assert_all_math(widget) -> None:
    spins = widget.findChildren(QAbstractSpinBox)
    assert spins, f"{widget} should contain spinboxes"
    for spin in spins:
        assert isinstance(spin, (MathSpinBox, MathDoubleSpinBox)), (
            f"{spin.objectName() or spin} is a plain spinbox — use "
            f"MathSpinBox/MathDoubleSpinBox so it accepts typed calculations"
        )


def test_flare_render_settings_are_all_math(qapp):
    from ghostlight_designer.render_common import (
        MID_PRESET,
        RenderSettingsDialog,
    )

    dlg = RenderSettingsDialog(MID_PRESET)
    try:
        _assert_all_math(dlg)
    finally:
        _destroy(dlg)


def test_psf_render_settings_are_all_math(qapp):
    from ghostlight_designer.psf_panel.dialogs import (
        LOW_PRESET,
        PSFRenderSettingsDialog,
    )

    dlg = PSFRenderSettingsDialog(LOW_PRESET)
    try:
        _assert_all_math(dlg)
    finally:
        _destroy(dlg)


def test_sourceflare_body_spinboxes_are_all_math(qapp, isolated_settings):
    from ghostlight_designer.project import Project
    from ghostlight_designer.sourceflare_panel.body import SourceFlarePanelBody

    body = SourceFlarePanelBody(Project(), isolated_settings)
    try:
        _assert_all_math(body)
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# End to end through the optical-editor delegate
# ---------------------------------------------------------------------------

def _radius_cell(project, model):
    """(surface index, model index) of the first sphere-radius cell.

    Mirrors test_value_scrubber_undo._first_sphere_radius_index — the
    scrubbable cells and the math-capable cells are the same set.
    """
    import ghostlight
    from PySide6.QtCore import QModelIndex

    from ghostlight_designer.optical_editor.columns import Column

    for ei, el in enumerate(project.system.elements):
        for li, si in enumerate(el.resolve_surfaces(project.system)):
            if int(project.system.surfaces[si].form) != int(ghostlight.SurfaceForm.SPHERE):
                continue
            el_idx = model.index(ei, 0, QModelIndex())
            surface_row = len(el.material_glasses) + li
            return si, model.index(surface_row, int(Column.RADIUS), el_idx)
    pytest.skip("no sphere-radius cell in sample lens")


def _edit_cell(delegate, model, index, text):
    """Full delegate round trip: open the editor, type, commit."""
    from PySide6.QtWidgets import QStyleOptionViewItem, QTreeView

    parent = QTreeView()
    try:
        editor = delegate.createEditor(parent, QStyleOptionViewItem(), index)
        assert isinstance(editor, (MathSpinBox, MathDoubleSpinBox))
        delegate.setEditorData(editor, index)
        editor.lineEdit().setText(text)
        delegate.setModelData(editor, model, index)
    finally:
        _destroy(parent)


def test_ode_radius_cell_takes_a_typed_calculation(qapp, sample_lens_path):
    """The whole user story, through the real delegate and model:
    type an expression into a surface Radius cell, get the result."""
    from ghostlight_designer.optical_editor.delegates import SlotDelegate
    from ghostlight_designer.optical_editor.model import OpticalTreeModel
    from ghostlight_designer.project import Project

    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    delegate = SlotDelegate(project)
    si, index = _radius_cell(project, model)

    _edit_cell(delegate, model, index, "12*2")
    assert project.system.surfaces[si].radius == pytest.approx(24.0)

    # One edit, one undo entry — the expression commits like any typed value.
    undo_depth = len(project._undo)
    project.undo()
    assert len(project._undo) == undo_depth - 1
    assert project.system.surfaces[si].radius != pytest.approx(24.0)


def test_ode_radius_cell_keeps_its_value_on_a_bad_calculation(qapp, sample_lens_path):
    from ghostlight_designer.optical_editor.delegates import SlotDelegate
    from ghostlight_designer.optical_editor.model import OpticalTreeModel
    from ghostlight_designer.project import Project

    project = Project()
    project.load(str(sample_lens_path))
    model = OpticalTreeModel(project)
    delegate = SlotDelegate(project)
    si, index = _radius_cell(project, model)

    # Start from a value the 4-decimal editor round-trips exactly, so the
    # only thing that could move the radius is the expression itself.
    _edit_cell(delegate, model, index, "24")
    assert project.system.surfaces[si].radius == pytest.approx(24.0)
    undo_depth = len(project._undo)

    _edit_cell(delegate, model, index, "12*")
    assert project.system.surfaces[si].radius == pytest.approx(24.0)
    # Nothing changed, so nothing new lands on the undo stack either.
    assert len(project._undo) == undo_depth
