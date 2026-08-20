"""Built-in merit function presets.

Each builder returns a fresh :class:`MeritFunction` with sensible default
goals. EFL targets are seeded from the live lens so "add Image Quality
MF" creates a MF whose EFL goal matches today's lens, i.e. "hold EFL
while reducing spot".
"""
from __future__ import annotations

from typing import Optional

import ghostlight

from .. import lens_metrics as lm
from .data import GoalEntry, GoalKind, MeritFunction
from .goals.base import default_params_for, default_target_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_efl_target(system: ghostlight.OpticalSystem) -> float:
    """Best-effort current EFL (mm) for seeding a target. Falls back to
    the evaluator default (50 mm) if the lens isn't ready to trace."""
    try:
        efl = lm._effective_focal_length_on_axis(system, "y")
    except Exception:
        efl = None
    if efl is None or efl <= 0.0:
        return default_target_for(GoalKind.EFL)
    return float(round(efl, 3))


def _seed_fnumber_target(system: ghostlight.OpticalSystem) -> float:
    """Best-effort current F-number for seeding a target."""
    try:
        from .goals.first_order import _entrance_pupil_radius  # local import to avoid cycle
        efl = lm._effective_focal_length_on_axis(system, "y")
        r = _entrance_pupil_radius(system)
    except Exception:
        return default_target_for(GoalKind.F_NUMBER)
    if efl is None or r is None or r <= 0.0:
        return default_target_for(GoalKind.F_NUMBER)
    return float(round(efl / (2.0 * r), 2))


def _new_goal(kind: GoalKind, **overrides) -> GoalEntry:
    """Build a goal with the evaluator's default params + supplied overrides."""
    entry = GoalEntry.make(
        kind=kind,
        target=default_target_for(kind),
        params=dict(default_params_for(kind)),
    )
    for k, v in overrides.items():
        if not hasattr(entry, k):
            raise TypeError(f"GoalEntry has no attribute {k!r}")
        setattr(entry, k, v)
    return entry


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_empty_merit_function() -> MeritFunction:
    return MeritFunction.make(name="New Merit Function")


def build_image_quality_mf(system: Optional[ghostlight.OpticalSystem] = None) -> MeritFunction:
    """Spot RMS (all fields, primary λ) + EFL hold + F# hold.

    Standard starting point for "tighten the image while keeping EFL
    and F# where they are".
    """
    efl_target = _seed_efl_target(system) if system is not None else 50.0
    return MeritFunction.make(
        name="Image Quality",
        goals=[
            _new_goal(GoalKind.SPOT_RMS, target=0.0, weight=1.0),
            _new_goal(GoalKind.EFL, target=efl_target, weight=10.0),
        ],
    )


def build_image_quality_with_fnumber_mf(
    system: Optional[ghostlight.OpticalSystem] = None,
) -> MeritFunction:
    efl_target = _seed_efl_target(system) if system is not None else 50.0
    fnum_target = _seed_fnumber_target(system) if system is not None else 2.8
    return MeritFunction.make(
        name="Image Quality + F#",
        goals=[
            _new_goal(GoalKind.SPOT_RMS, target=0.0, weight=1.0),
            _new_goal(GoalKind.EFL, target=efl_target, weight=10.0),
            _new_goal(GoalKind.F_NUMBER, target=fnum_target, weight=2.0),
        ],
    )


def build_anamorphic_mf(system: Optional[ghostlight.OpticalSystem] = None) -> MeritFunction:
    """Independent X / Y EFL targets — useful for tuning anamorphic squeeze."""
    if system is not None:
        try:
            efl_x = lm._effective_focal_length_on_axis(system, "x")
            efl_y = lm._effective_focal_length_on_axis(system, "y")
        except Exception:
            efl_x = efl_y = None
    else:
        efl_x = efl_y = None
    efl_x_target = float(round(efl_x, 3)) if efl_x and efl_x > 0.0 else 25.0
    efl_y_target = float(round(efl_y, 3)) if efl_y and efl_y > 0.0 else 50.0
    return MeritFunction.make(
        name="Anamorphic Squeeze",
        goals=[
            _new_goal(GoalKind.EFL_X, target=efl_x_target, weight=10.0),
            _new_goal(GoalKind.EFL_Y, target=efl_y_target, weight=10.0),
            _new_goal(GoalKind.SPOT_RMS, target=0.0, weight=1.0),
        ],
    )


def build_distortion_corrected_mf(
    system: Optional[ghostlight.OpticalSystem] = None,
) -> MeritFunction:
    efl_target = _seed_efl_target(system) if system is not None else 50.0
    return MeritFunction.make(
        name="Distortion Corrected",
        goals=[
            _new_goal(GoalKind.SPOT_RMS, target=0.0, weight=1.0),
            _new_goal(GoalKind.EFL, target=efl_target, weight=10.0),
            _new_goal(GoalKind.DISTORTION, target=0.0, weight=1.0),
            _new_goal(GoalKind.FIELD_CURVATURE, target=0.0, weight=1.0),
        ],
    )


# ---------------------------------------------------------------------------
# Registry — surfaced in the toolbar's "Add Merit Function" dropdown
# ---------------------------------------------------------------------------


# Order matters: the menu uses this list directly. Empty is first because
# it's the only preset that's never "wrong" for the current lens — every
# seeded preset bakes a target that only makes sense if the lens already
# looks like what the user is trying to optimise.
PRESETS: list[tuple[str, callable]] = [
    ("Empty", lambda _system: build_empty_merit_function()),
    ("Image Quality", build_image_quality_mf),
    ("Image Quality + F#", build_image_quality_with_fnumber_mf),
    ("Anamorphic Squeeze", build_anamorphic_mf),
    ("Distortion Corrected", build_distortion_corrected_mf),
]
