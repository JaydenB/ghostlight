"""Unit tests for the Qt-free motion-pattern registry."""
from __future__ import annotations

import math

import pytest

from ghostlight_designer.sourceflare_panel import motion_patterns as mp


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_non_empty_and_unique_names():
    assert len(mp.PATTERNS) >= 5
    names = [p.name for p in mp.PATTERNS]
    assert len(names) == len(set(names)), "pattern names must be unique"


def test_get_pattern_resolves_and_misses():
    first = mp.PATTERNS[0]
    assert mp.get_pattern(first.name) is first
    assert mp.get_pattern("no such pattern") is None


def test_decorator_registers_then_restore():
    before = list(mp.PATTERNS)
    try:
        @mp.pattern("TEST Only", loop=True)
        def _fn(t, ctx):
            return (t, 0.5)

        pat = mp.get_pattern("TEST Only")
        assert pat is not None and pat.loop is True
        assert pat.fn(0.25, mp.MotionContext()) == (0.25, 0.5)
    finally:
        # Don't pollute the module-global registry for other tests.
        mp.PATTERNS[:] = before
    assert mp.get_pattern("TEST Only") is None


# ---------------------------------------------------------------------------
# sample_times
# ---------------------------------------------------------------------------

def test_sample_times_open_spans_both_endpoints():
    ts = mp.sample_times(5, loop=False)
    assert ts[0] == 0.0
    assert ts[-1] == 1.0
    assert ts == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_sample_times_loop_omits_duplicate_endpoint():
    ts = mp.sample_times(4, loop=True)
    assert ts == [0.0, 0.25, 0.5, 0.75]
    assert 1.0 not in ts  # last frame flows back into the first


def test_sample_times_single_frame():
    assert mp.sample_times(1, loop=False) == [0.0]
    assert mp.sample_times(1, loop=True) == [0.0]


def test_sample_times_clamps_non_positive():
    assert mp.sample_times(0, loop=False) == [0.0]
    assert mp.sample_times(-3, loop=True) == [0.0]


# ---------------------------------------------------------------------------
# Built-in patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pat", mp.PATTERNS, ids=lambda p: p.name)
def test_builtin_patterns_finite_over_unit_interval(pat):
    ctx = mp.MotionContext(start_sx=0.7, start_sy=0.4, aspect=1.78)
    for i in range(21):
        t = i / 20.0
        sx, sy = pat.fn(t, ctx)
        assert math.isfinite(sx) and math.isfinite(sy)


def test_orbit_passes_through_start_at_t0():
    """Orbit is anchored: frame 0 sits on the current source position."""
    orbit = mp.get_pattern("Orbit")
    ctx = mp.MotionContext(start_sx=0.8, start_sy=0.3, aspect=1.5)
    sx, sy = orbit.fn(0.0, ctx)
    assert sx == pytest.approx(ctx.start_sx, abs=1e-9)
    assert sy == pytest.approx(ctx.start_sy, abs=1e-9)


def test_orbit_is_closed_loop():
    """t=0 and t=1 coincide (a closed circle)."""
    orbit = mp.get_pattern("Orbit")
    ctx = mp.MotionContext(start_sx=0.75, start_sy=0.5, aspect=1.0)
    assert orbit.fn(0.0, ctx) == pytest.approx(orbit.fn(1.0, ctx))


def test_orbit_central_start_uses_fallback_radius():
    """A source at frame centre has no anchor radius — fall back, don't NaN."""
    orbit = mp.get_pattern("Orbit")
    ctx = mp.MotionContext(start_sx=0.5, start_sy=0.5, aspect=1.0)
    sx, sy = orbit.fn(0.0, ctx)
    # Fallback: quarter-frame circle starting at 3 o'clock.
    assert sx == pytest.approx(0.75, abs=1e-9)
    assert sy == pytest.approx(0.5, abs=1e-9)


def test_sweep_left_right_enters_and_leaves_off_frame():
    sweep = mp.get_pattern("Sweep Left → Right")
    ctx = mp.MotionContext(start_sx=0.5, start_sy=0.42, aspect=1.0)
    x0, y0 = sweep.fn(0.0, ctx)
    x1, y1 = sweep.fn(1.0, ctx)
    assert x0 < 0.0 and x1 > 1.0          # off-frame both ends
    assert y0 == pytest.approx(0.42) and y1 == pytest.approx(0.42)


def test_figure_eight_is_closed_loop():
    fig = mp.get_pattern("Figure Eight")
    ctx = mp.MotionContext(aspect=1.0)
    assert fig.fn(0.0, ctx) == pytest.approx(fig.fn(1.0, ctx))
