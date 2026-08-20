"""Matplotlib canvas painting a :class:`SeidelResult`.

One sub-plot per aberration type (spherical, coma, astigmatism,
Petzval, distortion, axial colour, lateral colour). Each sub-plot is a
horizontal grouping of bars — one bar per refracting surface plus a
distinct "Σ" bar showing the system total. Positive contributions in
warm colour, negative in cool, so the eye spots cancellations at a
glance.

Layout is a 4×2 grid (last cell empty) when all seven sub-plots are
on, or a tighter row when the user has hidden some.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .compute import SeidelResult


# Colour pairs (positive, negative). Slightly warm / cool tones —
# matches the field-diagrams palette so the user reads the same hue
# meaning across the evaluation family.
_BAR_POS = "#e6a23c"   # warm amber
_BAR_NEG = "#7ec8e3"   # cool blue
_BAR_SUM_POS = "#f56c6c"  # red — the totals stand out from per-surface
_BAR_SUM_NEG = "#67c23a"  # green — likewise

# Display catalogue: panel-spec-attr → (data array name, plot title, field-scaling tag).
#
# The field-scaling tag is rendered under each subplot title so users know
# *why* changing the Field control rescales but doesn't reshape these
# bars — third-order Seidel is fixed-power-of-field by construction; the
# per-surface signature is geometry, not field. (See ``compute.py`` for
# the formula derivations.)
_ENTRIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("show_spherical",     "spherical_per_surface",     "S_I  spherical",        "∝ field⁰"),
    ("show_coma",          "coma_per_surface",          "S_II  coma",            "∝ field¹"),
    ("show_astigmatism",   "astigmatism_per_surface",   "S_III  astigmatism",    "∝ field²"),
    ("show_petzval",       "petzval_per_surface",       "S_IV  Petzval",         "∝ field²"),
    ("show_distortion",    "distortion_per_surface",    "S_V  distortion",       "∝ field³"),
    ("show_axial_color",   "axial_color_per_surface",   "C_I  axial colour",     "∝ field⁰"),
    ("show_lateral_color", "lateral_color_per_surface", "C_II  lateral colour",  "∝ field¹"),
)


class SeidelCanvas(QWidget):
    """Matplotlib canvas painting the Seidel bar charts."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._figure = Figure(figsize=(9, 7), facecolor="#111")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)

        self._render_placeholder("Load a lens to compute Seidel coefficients")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result(self, result: Optional[SeidelResult]) -> None:
        if result is None:
            self._render_placeholder("No result")
            return
        self._render_result(result)

    def clear(self, placeholder: str = "") -> None:
        self._render_placeholder(placeholder)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_placeholder(self, text: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(1, 1, 1)
        ax.set_facecolor("#1a1a1a")
        ax.text(
            0.5, 0.5, text,
            ha="center", va="center", color="#999",
            fontsize=11, transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._canvas.draw()

    def _render_result(self, result: SeidelResult) -> None:
        self._figure.clear()
        spec = result.spec

        # Build the list of (data_array, title, scaling_tag) tuples for
        # sub-plots the user wants visible.
        active: List[Tuple[np.ndarray, str, str]] = []
        for show_attr, data_attr, title, scaling in _ENTRIES:
            if not getattr(spec, show_attr):
                continue
            active.append((getattr(result, data_attr), title, scaling))

        if not active:
            self._render_placeholder(
                "All sub-charts hidden — enable one from the View menu"
            )
            return

        # Layout: 4 cols × ceil(n/4) rows. With 7 active it's a tidy
        # 4×2 with one empty cell; with 5 it's 4×2 with three empty. The
        # bars + labels stay readable at this density.
        n = len(active)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)
        axes = self._figure.subplots(nrows, ncols, squeeze=False)

        for cell_idx in range(nrows * ncols):
            r, c = divmod(cell_idx, ncols)
            ax = axes[r][c]
            if cell_idx >= n:
                ax.set_visible(False)
                continue
            data, title, scaling = active[cell_idx]
            self._plot_one(ax, result, data, title, scaling)

        self._figure.suptitle(
            f"Seidel Sums  ·  λ = {result.primary_wavelength_nm:.1f} nm  ·  "
            f"field = {result.field_deg:.2f}°  ·  pupil R = {result.pupil_radius_mm:.2f} mm  ·  "
            f"H = {result.lagrange_invariant:+.3f}",
            color="#ddd", fontsize=10,
        )
        self._figure.subplots_adjust(
            left=0.06, right=0.98, top=0.92, bottom=0.08,
            wspace=0.30, hspace=0.45,
        )
        self._canvas.draw()

    def _plot_one(
        self,
        ax,
        result: SeidelResult,
        data: np.ndarray,
        title: str,
        scaling: str,
    ) -> None:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#777", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444")

        n = data.size
        if n == 0:
            ax.text(
                0.5, 0.5, "no surfaces",
                ha="center", va="center", color="#888",
                fontsize=8, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}   {scaling}", color="#ccc", fontsize=9)
            return

        total = float(np.nansum(data))
        # X positions: 0..n-1 for per-surface, plus n+0.5 for the gapped sum bar.
        xs_surf = np.arange(n)
        x_sum = n + 0.5

        # Colour every bar by sign so cancellations read at a glance.
        colours = [
            (_BAR_POS if v >= 0 else _BAR_NEG)
            for v in data
        ]
        ax.bar(xs_surf, data, width=0.78, color=colours, edgecolor="#222")
        ax.bar(
            [x_sum], [total], width=0.85,
            color=(_BAR_SUM_POS if total >= 0 else _BAR_SUM_NEG),
            edgecolor="#222",
        )

        # Tick labels: per-surface labels + a "Σ" for the sum bar.
        labels = list(result.surface_labels) + ["Σ"]
        ax.set_xticks(list(xs_surf) + [x_sum])
        ax.set_xticklabels(labels, color="#bbb", fontsize=7, rotation=0)

        # Zero line on top of the bars so the user reads sign easily.
        ax.axhline(0.0, color="#666", linewidth=0.8, zorder=2)

        # Scaling tag appended to the title so users can see at a glance
        # how each Seidel sum responds to the Field control. Dim grey
        # so it doesn't compete with the main title text.
        ax.set_title(title, color="#ccc", fontsize=9)
        ax.text(
            0.99, 1.02, scaling,
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888", fontsize=7, family="monospace",
        )

        # Y-limits: make sure both the per-surface bars AND the sum bar
        # all fit comfortably. A small floor prevents an all-zeros chart
        # from collapsing to a hairline.
        all_vals = np.concatenate([data, np.array([total])])
        peak = float(np.max(np.abs(all_vals))) if all_vals.size else 0.0
        if peak < 1e-12:
            ax.set_ylim(-1.0, 1.0)
        else:
            ax.set_ylim(-peak * 1.25, peak * 1.25)

        # Annotate sum value above (or below) the sum bar.
        offset = peak * 0.06 if peak > 0 else 0.05
        if total >= 0:
            ax.text(
                x_sum, total + offset, f"{total:+.3e}",
                ha="center", va="bottom",
                color="#ddd", fontsize=7, family="monospace",
            )
        else:
            ax.text(
                x_sum, total - offset, f"{total:+.3e}",
                ha="center", va="top",
                color="#ddd", fontsize=7, family="monospace",
            )
