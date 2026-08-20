"""Motion patterns for the source-flare animation exporter (Qt-free).

A *motion pattern* maps a normalised time ``t in [0, 1]`` to a flare
source position ``(sx, sy)`` in the panel's screen-fraction space
(``0.5, 0.5`` = frame centre; out-of-range is deliberately valid — the
source can sit off-frame, see ``PointFlareConfig.source_x``). The
exporter samples the pattern at ``n`` times and renders one frame per
sample, so the flare traces the path across the animation.

Adding a pattern is a three-line story: write a function that takes
``(t, ctx)`` and returns ``(sx, sy)``, decorate it with
``@pattern("Display Name")``, done — it appears in the export dialog's
pattern combo automatically (registration order = menu order). Mark it
``loop=True`` if the path is a closed loop, so the sampler omits the
duplicate endpoint and the last frame flows seamlessly into the first.

``ctx`` (:class:`MotionContext`) carries the panel's *current* source
position (the anchor for patterns that start "where the flare is now")
and the frame aspect ratio, so a pattern can trace a path that reads as
visually circular despite the non-square fraction space.

Nothing here imports Qt or ghostlight, so the registry is unit-testable
without a GPU or an event loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class MotionContext:
    """Immutable inputs a pattern function may consult.

    ``start_sx`` / ``start_sy`` are the panel's live source position at
    export time (screen fraction, ``0.5`` = centre), the anchor for
    patterns that begin from the current flare position. ``aspect`` is
    ``frame_width / frame_height`` in pixels: a pattern that wants a
    visually circular path scales its *x* excursion by ``1 / aspect``
    (an equal fraction-radius maps to a wider pixel-radius on a wide
    frame, so the x radius must shrink to match the y radius on screen).
    """

    start_sx: float = 0.5
    start_sy: float = 0.5
    aspect: float = 1.0


# fn(t, ctx) -> (sx, sy), with t in [0, 1].
MotionFn = Callable[[float, MotionContext], Tuple[float, float]]


@dataclass(frozen=True)
class MotionPattern:
    """A named motion path. ``loop`` marks a closed path (see
    :func:`sample_times`)."""

    name: str
    fn: MotionFn
    loop: bool = False


# Registration order == the order patterns appear in the UI. Populated by the
# @pattern decorator at import time.
PATTERNS: List[MotionPattern] = []


def pattern(name: str, *, loop: bool = False) -> Callable[[MotionFn], MotionFn]:
    """Decorator: register ``fn`` as a motion pattern. Returns ``fn`` unchanged
    so the decorated symbol stays directly callable in tests."""

    def _register(fn: MotionFn) -> MotionFn:
        PATTERNS.append(MotionPattern(name=name, fn=fn, loop=loop))
        return fn

    return _register


def get_pattern(name: str) -> Optional[MotionPattern]:
    """Return the registered pattern with ``name``, or ``None``."""
    for p in PATTERNS:
        if p.name == name:
            return p
    return None


def sample_times(n: int, loop: bool) -> List[float]:
    """Return ``n`` sample times in ``[0, 1]``.

    A looping path omits the duplicate endpoint (``t = i / n``) so the
    last frame flows seamlessly back into the first; an open path spans
    both endpoints (``t = i / (n - 1)``). A single frame samples ``t = 0``
    (no zero-division).
    """
    n = max(1, int(n))
    if n == 1:
        return [0.0]
    if loop:
        return [i / n for i in range(n)]
    return [i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Built-in patterns — each one is the whole "how to add a pattern" example.
# ---------------------------------------------------------------------------


@pattern("Sweep Left → Right")
def _sweep_left_right(t: float, ctx: MotionContext) -> Tuple[float, float]:
    """Straight horizontal pass at the source's current height, entering and
    leaving off-frame (exercises the out-of-frame source validity)."""
    return (-0.2 + 1.4 * t, ctx.start_sy)


@pattern("Sweep Diagonal")
def _sweep_diagonal(t: float, ctx: MotionContext) -> Tuple[float, float]:
    """Corner-to-corner diagonal pass, entering / leaving off-frame."""
    return (-0.2 + 1.4 * t, -0.2 + 1.4 * t)


@pattern("Orbit", loop=True)
def _orbit(t: float, ctx: MotionContext) -> Tuple[float, float]:
    """Closed circle around frame centre, anchored so frame 0 passes through
    the source's current position (radius / phase taken from where it sits).

    When the source starts within ~0.05 of centre there is no meaningful
    radius / phase to anchor to, so fall back to a quarter-frame circle
    starting at 3 o'clock.
    """
    dx = (ctx.start_sx - 0.5) * ctx.aspect
    dy = ctx.start_sy - 0.5
    r = math.hypot(dx, dy)
    if r < 0.05:
        r = 0.25
        phase = 0.0
    else:
        phase = math.atan2(dy, dx)
    ang = phase + 2.0 * math.pi * t
    return (0.5 + (r / ctx.aspect) * math.cos(ang), 0.5 + r * math.sin(ang))


@pattern("Figure Eight", loop=True)
def _figure_eight(t: float, ctx: MotionContext) -> Tuple[float, float]:
    """Closed Lissajous figure-eight centred on the frame (1:2 x:y frequency).
    The x amplitude is aspect-compensated so it doesn't over-stretch wide."""
    ax = 0.35 / ctx.aspect
    return (
        0.5 + ax * math.sin(2.0 * math.pi * t),
        0.5 + 0.25 * math.sin(4.0 * math.pi * t),
    )


@pattern("Sine Drift")
def _sine_drift(t: float, ctx: MotionContext) -> Tuple[float, float]:
    """Horizontal drift across the frame with a gentle vertical sine wobble
    around the source's current height."""
    return (
        -0.1 + 1.2 * t,
        ctx.start_sy + 0.15 * math.sin(4.0 * math.pi * t),
    )
