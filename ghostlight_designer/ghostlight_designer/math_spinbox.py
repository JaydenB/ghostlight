"""Spinboxes that accept a typed calculation instead of a bare number.

``MathSpinBox`` / ``MathDoubleSpinBox`` are drop-in replacements for
``QSpinBox`` / ``QDoubleSpinBox``. Type ``12*2`` and press Enter (or click
away) and the field becomes ``24``. If the expression doesn't compute, or
the result falls outside the field's range, the field goes back to the
value it had before.

How it works — no timers, no extra state, no signal wiring. Qt's own
``QAbstractSpinBoxPrivate::interpret()`` already runs this sequence on
Enter, on focus-out, and from the explicit ``editor.interpretText()``
calls the item delegates make:

    validate(text) -> if not Acceptable: fixup(text) -> validate again
                      -> if now Acceptable: valueFromText(text)
                      -> else: fall back to the *previous* value

So ``validate`` only has to keep operator keystrokes alive (by answering
``Intermediate`` instead of ``Invalid``), and ``fixup`` does the actual
arithmetic, rewriting the text as a plain number. Refusing to rewrite is
how we ask for the revert — which is why ``correctionMode`` is pinned to
``CorrectToPreviousValue`` below even though that is already Qt's default.

Two consequences worth knowing:

* Nothing is emitted mid-typing. The moment an operator appears the text
  is ``Intermediate``, so ``valueChanged`` stays quiet until the
  expression is interpreted. (Typing a plain number still emits per
  keystroke when ``keyboardTracking`` is on, exactly as before.)
* These are *subclasses*, so ``isinstance(w, QSpinBox)`` still holds and
  the value scrubber (which uses that test to pick its integer mode)
  needs no changes.

The evaluator itself lives in :mod:`ghostlight_designer.math_input`.
"""
from __future__ import annotations

import math

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox

from .math_input import MathInputError, evaluate, is_partial_expression, parse_number

__all__ = ["MathDoubleSpinBox", "MathSpinBox"]


class _MathSpinBoxMixin:
    """Shared validate/fixup halves; see the module docstring."""

    def _init_math_input(self) -> None:
        # The revert-on-bad-input contract depends on this mode, so pin it
        # rather than relying on it staying Qt's default.
        self.setCorrectionMode(QAbstractSpinBox.CorrectToPreviousValue)

    # -- text <-> expression plumbing ------------------------------------

    def _expression_body(self, text: str) -> str:
        """Strip prefix/suffix and normalise the decimal point.

        Qt's internal ``QSpinBoxValidator`` re-attaches the prefix and
        suffix *before* our ``validate`` is called, so the text we see for
        the System Setup angle field is e.g. ``"2*3°"``.
        """
        body = text
        prefix = self.prefix()
        if prefix and body.startswith(prefix):
            body = body[len(prefix):]
        suffix = self.suffix()
        if suffix and body.endswith(suffix):
            body = body[: len(body) - len(suffix)]
        point = self.locale().decimalPoint()
        if point and point != ".":
            body = body.replace(point, ".")
        return body.strip()

    def _decorate(self, text: str) -> str:
        return f"{self.prefix()}{text}{self.suffix()}"

    def _evaluated_text(self, text: str) -> str | None:
        """Expression result as display text, or ``None`` to leave it alone."""
        try:
            value = evaluate(self._expression_body(text))
        except MathInputError:
            return None
        return self._format_result(value)

    def _is_expression_candidate(self, body: str) -> bool:
        """Whether this half-typed text should be kept alive as arithmetic."""
        return is_partial_expression(body)

    def _format_result(self, value: float) -> str | None:  # pragma: no cover
        raise NotImplementedError

    # -- QAbstractSpinBox overrides --------------------------------------

    def validate(self, text: str, pos: int):  # type: ignore[override]
        state, fixed, new_pos = super().validate(text, pos)
        if state != QValidator.Invalid:
            # Plain numbers, blanks, negatives and specialValueText all
            # take the stock path untouched.
            return state, fixed, new_pos
        if self._is_expression_candidate(self._expression_body(text)):
            # Keep the keystroke, but stay Intermediate so no value is
            # emitted until the expression is interpreted.
            return QValidator.Intermediate, text, pos
        return state, fixed, new_pos

    def fixup(self, text: str) -> str:  # type: ignore[override]
        replacement = self._evaluated_text(text)
        if replacement is None:
            # Unchanged text tells Qt it could not be corrected, and
            # CorrectToPreviousValue restores the prior value.
            return super().fixup(text)
        return self._decorate(replacement)


class MathDoubleSpinBox(_MathSpinBoxMixin, QDoubleSpinBox):
    """``QDoubleSpinBox`` that evaluates typed arithmetic."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_math_input()

    def _format_result(self, value: float) -> str | None:
        if not (self.minimum() <= value <= self.maximum()):
            # Out of range reverts rather than clamping, so a fat-fingered
            # `8*40` never silently pins the field to its limit.
            return None
        # textFromValue, not locale().toString, so decimals and the
        # group-separator setting match what the spinbox displays.
        return self.textFromValue(float(value))


class MathSpinBox(_MathSpinBoxMixin, QSpinBox):
    """``QSpinBox`` that evaluates typed arithmetic, rounding the result.

    Fractional expressions are allowed on the way in — ``10/4`` lands on
    ``3`` — because rejecting them would be more surprising than rounding.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_math_input()

    def _is_expression_candidate(self, body: str) -> bool:
        if super()._is_expression_candidate(body):
            return True
        # An integer spinbox rejects '.' outright, which would make
        # `2.5*2` impossible to type left-to-right. Let decimal literals
        # through too; they get rounded on the way in.
        return "." in body and parse_number(body) is not None

    def _format_result(self, value: float) -> str | None:
        # Half away from zero, not Python's banker's rounding: a typed
        # `10/4` should read 3, not 2.
        rounded = int(math.copysign(math.floor(abs(value) + 0.5), value))
        if not (self.minimum() <= rounded <= self.maximum()):
            return None
        return self.textFromValue(rounded)
