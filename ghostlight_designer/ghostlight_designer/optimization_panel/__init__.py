"""Optimization panel.

Surfaces a tree of merit functions + their goals, a "Run" action that
opens a modal :class:`OptimizationPreviewDialog` against a *cloned* lens
so the project lens is never touched mid-run, and Accept / Reject to
commit (or discard) the solution.

Public exports are the same shape as every other panel package:
``<TYPE_ID>`` and ``register_<panel>_panel_type``.
"""
from __future__ import annotations

from .type import OPTIMIZATION_TYPE_ID, register_optimization_panel_type

__all__ = (
    "OPTIMIZATION_TYPE_ID",
    "register_optimization_panel_type",
)
