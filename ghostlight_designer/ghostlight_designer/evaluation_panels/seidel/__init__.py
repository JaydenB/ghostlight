"""Seidel bar-chart evaluation panel.

Five third-order monochromatic Seidel sums (spherical, coma, astigmatism,
Petzval, distortion) plus the two third-order chromatic sums (axial +
lateral colour), all broken down per refracting surface and shown as a
stack of bar charts. The bar chart is the standard lens-design
diagnostic for "which surface is causing my aberration?" — it makes
cancellations and dominant contributors immediately visible.

Compute is a closed-form paraxial trace (no Monte-Carlo) and is cheap;
the panel debounces edits with the same 350 ms settle as the rest of
the evaluation family, but the actual work is sub-millisecond.
"""
from __future__ import annotations

from .body import SeidelBody
from .type import SEIDEL_TYPE_ID, register_seidel_panel_type

__all__ = [
    "SeidelBody",
    "SEIDEL_TYPE_ID",
    "register_seidel_panel_type",
]
