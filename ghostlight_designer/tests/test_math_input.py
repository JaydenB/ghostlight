"""Unit tests for the arithmetic evaluator behind the math spinboxes.

Pure Python — no Qt. The widget half is covered by test_math_spinbox.py.
"""
from __future__ import annotations

import pytest

from ghostlight_designer.math_input import (
    MathInputError,
    evaluate,
    is_partial_expression,
    parse_number,
)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("12*2", 24.0),
        ("12 * 2", 24.0),
        ("100/4", 25.0),
        ("10-3", 7.0),
        ("10+3", 13.0),
        ("(2+3)*4", 20.0),
        ("2+3*4", 14.0),          # precedence, not left-to-right
        ("-2*-3", 6.0),
        ("1.5*2", 3.0),
        ("10/4", 2.5),
        ("7//2", 3.0),
        ("7%3", 1.0),
        ("1e-3*2", 0.002),
        ("-5", -5.0),             # a plain literal still evaluates
    ],
)
def test_evaluates_arithmetic(text, expected):
    assert evaluate(text) == pytest.approx(expected)


def test_one_third_is_a_float_not_integer_division():
    assert evaluate("1/3") == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "12*",           # incomplete
        "*2",            # no leading-operator relative form
        "((1+2)",        # unbalanced
        "1/0",           # ZeroDivisionError
        "7%0",
        "abc",
        "12a",
        "foo+1",         # name
        "__import__('os')",
        "len([1])",      # call
        "1 if 2 else 3",
        "[1,2]",
        "1,2",           # tuple
        "1<2",           # comparison
        "True*2",
        "nan+1",
        "inf",
    ],
)
def test_rejects_non_arithmetic(text):
    with pytest.raises(MathInputError):
        evaluate(text)


def test_power_is_rejected_so_the_ui_cannot_be_wedged():
    """``9**9**9`` would freeze the UI thread; the operator is not in the
    whitelist, and the rejection is a cheap parse-time decision."""
    with pytest.raises(MathInputError):
        evaluate("2**8")
    with pytest.raises(MathInputError):
        evaluate("9**9**9")


def test_rejects_overlong_input():
    with pytest.raises(MathInputError):
        evaluate("1+" * 200 + "1")


def test_rejects_non_finite_result():
    with pytest.raises(MathInputError):
        evaluate("1e308*10")


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [("5", 5.0), ("-5", -5.0), ("2.5", 2.5), ("  3 ", 3.0), ("1e-3", 0.001), ("2.", 2.0)],
)
def test_parse_number_accepts_plain_literals(text, expected):
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "12*2", "abc", "nan", "inf", "1,000"])
def test_parse_number_rejects_everything_else(text):
    assert parse_number(text) is None


# ---------------------------------------------------------------------------
# is_partial_expression — the per-keystroke gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["12*", "12*2", "1/", "(2+3", "1+2)", "2.5*2", "-1+"])
def test_partial_expression_keeps_half_typed_arithmetic_alive(text):
    assert is_partial_expression(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "5",
        "-5",       # a plain negative is a number, not an expression
        "1e-3",     # exponent literal, likewise
        "2.5",
        "12a",      # letters other than e/E never qualify
        "1+" * 40,  # too long
    ],
)
def test_partial_expression_rejects_non_expressions(text):
    assert not is_partial_expression(text)
