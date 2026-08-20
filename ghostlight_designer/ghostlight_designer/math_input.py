"""Tiny arithmetic-expression evaluator for numeric input fields.

Lets the user type a calculation instead of a number — ``12*2``, ``1/3``,
``(2+3)*4`` — into any designer spinbox. This module is deliberately
Qt-free so it can be unit-tested on its own; the widget half lives in
:mod:`ghostlight_designer.math_spinbox`.

Two questions the widgets ask, in the order they ask them:

``is_partial_expression(text)``
    Cheap character-level gate used *while the user is typing*: could this
    text still become an expression? ``"12*"`` says yes so the keystroke is
    accepted into the line edit; ``"12abc"`` says no so Qt rejects it.

``evaluate(text)``
    Strict parse, run once on Enter / focus-out. Raises
    :class:`MathInputError` for anything it won't compute, which is the
    caller's cue to leave the field on its previous value.

Only full expressions are supported. A leading operator is *not* a
relative edit — ``-5`` means negative five (so you can go from ``-2`` to
``-5`` by typing it), not "subtract five".

``**`` is deliberately not in the whitelist: ``9**9**9`` would wedge the
UI thread for minutes, and the feature only ever promised ``+ - * /``.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Optional

__all__ = [
    "MathInputError",
    "evaluate",
    "is_partial_expression",
    "parse_number",
]


class MathInputError(ValueError):
    """Raised when a string is not an arithmetic expression we will evaluate."""


# Guards a pathological paste from reaching the parser at all. Real
# expressions typed into a spinbox are a handful of characters.
MAX_EXPRESSION_LENGTH = 64

# Characters that may appear in a partially-typed expression. ``e``/``E``
# are here for exponent literals (``1e-3``); no other letter is allowed,
# which is what keeps names, calls and attributes out before we even parse.
_ALLOWED_CHARS = frozenset("0123456789.eE+-*/%() \t")

_OPERATOR_CHARS = frozenset("+-*/%()")

# The whole vocabulary. Anything not keyed here is a parse error, which
# is what keeps names, calls, attributes, comparisons and `**` out.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def parse_number(text: str) -> Optional[float]:
    """Return ``text`` as a float when it is a plain literal, else ``None``.

    Plain means anything Python's ``float()`` accepts on its own —
    ``"-5"``, ``"1e-3"``, ``"  2.5 "``. Such text is *not* an expression
    and must keep flowing through the spinbox's own validator untouched.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except (TypeError, ValueError):
        return None
    # "inf" / "nan" parse as floats but contain letters we never allow;
    # reject them here too so the two gates agree.
    if not math.isfinite(value):
        return None
    return value


def is_partial_expression(text: str) -> bool:
    """True when ``text`` could still grow into an expression we'd evaluate.

    Permissive on purpose — it runs per keystroke and only decides whether
    the character is allowed into the line edit. ``evaluate`` is the one
    that decides whether the finished text actually computes.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_EXPRESSION_LENGTH:
        return False
    if not _ALLOWED_CHARS.issuperset(stripped):
        return False
    if not _OPERATOR_CHARS.intersection(stripped):
        return False
    # A plain literal is not an expression even though "-5" and "1e-3"
    # contain operator characters.
    return parse_number(stripped) is None


def _eval_node(node: ast.AST) -> float:
    """Walk the tree, computing as we go and rejecting anything unlisted.

    Interpreting the AST directly (rather than ``compile`` + ``eval``)
    means no node outside the whitelist is ever executed, not merely
    inspected first.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.BinOp):
        func = _BINOPS.get(type(node.op))
        if func is None:
            raise MathInputError(f"unsupported operator {type(node.op).__name__}")
        return func(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        func = _UNARYOPS.get(type(node.op))
        if func is None:
            raise MathInputError(f"unsupported operator {type(node.op).__name__}")
        return func(_eval_node(node.operand))
    if isinstance(node, ast.Constant):
        # bool is an int subclass; `True*2` is not arithmetic anybody meant.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathInputError("only numeric literals are allowed")
        return float(node.value)
    raise MathInputError(f"unsupported syntax {type(node).__name__}")


def evaluate(text: str) -> float:
    """Evaluate ``text`` as arithmetic and return the result as a float.

    Raises :class:`MathInputError` for anything that does not parse, uses
    syntax outside the whitelist, divides by zero, or produces a
    non-finite result.
    """
    stripped = text.strip()
    if not stripped:
        raise MathInputError("empty expression")
    if len(stripped) > MAX_EXPRESSION_LENGTH:
        raise MathInputError("expression too long")
    if not _ALLOWED_CHARS.issuperset(stripped):
        raise MathInputError("expression contains disallowed characters")

    try:
        tree = ast.parse(stripped, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise MathInputError(f"could not parse {stripped!r}") from exc

    try:
        result = _eval_node(tree)
    except MathInputError:
        # Already specific (unsupported operator / syntax) — keep the reason.
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        # Chiefly ZeroDivisionError; float overflow yields `inf` instead
        # of raising and is caught by the finite check below.
        raise MathInputError(f"could not evaluate {stripped!r}") from exc

    if not math.isfinite(result):
        raise MathInputError("expression produced a non-finite result")
    return result
