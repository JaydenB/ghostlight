"""Goal-evaluator registry.

Importing this package registers every built-in evaluator (EFL, F#, Spot
RMS, …) into :data:`base.GOAL_REGISTRY`. The optimizer and the tree-row
schema both look up evaluators here by ``GoalKind``.
"""
from __future__ import annotations

from . import first_order  # noqa: F401  — registers EFL / F# evaluators
from . import spot         # noqa: F401  — registers spot RMS evaluator
from . import field_distortion  # noqa: F401  — registers FC / distortion

from .base import GOAL_REGISTRY, evaluator_for, param_schema_for

__all__ = ("GOAL_REGISTRY", "evaluator_for", "param_schema_for")
