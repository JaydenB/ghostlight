"""Matplotlib canvas that paints a :class:`SpotResult`.

Layout: rows = fields, columns = defocus offsets, with each cell holding
a scatter plot of ray landing positions colour-coded by wavelength.
Mirrors the textbook spot-diagram figure (Ch. 5.1.1 of Gerhard's
*Lens Design Basics*) — central spot is the Gaussian image plane, side
columns are ±N mm defocus slices.

**Scale handling.** Each field row uses ONE half-extent (mm) shared
across its defocus columns — that's what makes the textbook defocus
stack readable. The extent is resolved per row in priority order:

1. Explicit ``reference_extents`` passed by the panel body. The body
   captures the first render's auto-fit values and replays them on
   every subsequent render so a lens edit that shrinks (or grows) the
   bundle shows as a visibly smaller (or larger) blob against a
   *stable* reference — instead of silently rescaling to fit, which
   was the original bug.
2. ``spec.plot_half_extent_mm`` if positive — the user's manual lock.
3. Auto-fit from this render's data (per-field max spread). Used the
   first time a result is painted and on explicit "Auto-Fit Scale Now"
   menu invocations.

The canvas is dumb: no project knowledge, no thread state. It owns one
matplotlib :class:`~matplotlib.figure.Figure` and one ``FigureCanvasQTAgg``
and exposes a single :meth:`set_result` slot that the panel body calls
from the GUI thread.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Importing the canvas pulls matplotlib's Qt backend in — no pyplot
# import to avoid the global-state interactive figure manager.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .compute import SpotFieldResult, SpotResult


# Padding factor applied around the measured bundle radius when
# auto-scaling. 30% leaves a visible margin between the outermost ray
# and the subplot edge — tight enough to read the bundle's shape but
# generous enough that rays don't land on the frame.
_AUTO_EXTENT_PAD = 1.3

# Floor on the auto-scaled extent — 0.5 µm half-width. Below that the
# user is reading sub-wavelength residuals which aren't meaningful for
# the ray-traced geometry anyway, and a smaller floor risks vanishing
# tick labels.
_MIN_EXTENT_MM = 0.0005


# Wavelength → RGB rendering. Visible spectrum approximation: short
# wavelengths are blue-violet, mid are green/yellow, long are red.
# Out-of-visible falls back to neutral grey.
def _wavelength_to_rgb(nm: float) -> Tuple[float, float, float]:
    if nm < 380.0 or nm > 780.0:
        return (0.5, 0.5, 0.5)
    if nm < 440.0:
        r = (440.0 - nm) / 60.0
        g = 0.0
        b = 1.0
    elif nm < 490.0:
        r = 0.0
        g = (nm - 440.0) / 50.0
        b = 1.0
    elif nm < 510.0:
        r = 0.0
        g = 1.0
        b = (510.0 - nm) / 20.0
    elif nm < 580.0:
        r = (nm - 510.0) / 70.0
        g = 1.0
        b = 0.0
    elif nm < 645.0:
        r = 1.0
        g = (645.0 - nm) / 65.0
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    # Dim the edges of the visible band the way the eye does.
    if nm < 420.0:
        scale = 0.3 + 0.7 * (nm - 380.0) / 40.0
    elif nm > 700.0:
        scale = 0.3 + 0.7 * (780.0 - nm) / 80.0
    else:
        scale = 1.0
    return (r * scale, g * scale, b * scale)


def _field_centroid_for_slice(
    fr: SpotFieldResult, defocus_index: int
) -> Tuple[float, float]:
    """Mean (x, y) of valid rays in one defocus slice across all wavelengths.

    Used both to centre the subplot and (separately, per slice) to
    measure the bundle's spread. ``(nan, nan)`` when no valid rays.
    """
    mask = fr.valid_mask[defocus_index]
    if not mask.any():
        return (float("nan"), float("nan"))
    xs = fr.xs[defocus_index][mask]
    ys = fr.ys[defocus_index][mask]
    return (float(np.mean(xs)), float(np.mean(ys)))


def _resolve_row_extent(
    fr: SpotFieldResult,
    reference_extents: Optional[Dict[Tuple[float, float], float]],
    manual_extent_mm: Optional[float],
) -> float:
    """Pick the half-extent (mm) for one field row.

    Priority — manual (positive spec value) > reference map (set by the
    body to lock the first render's scale) > auto-fit from this render's
    data. Falls back gracefully when a reference is missing for this
    field tilt (e.g. the user added a new field since the reference was
    captured).
    """
    if manual_extent_mm is not None and manual_extent_mm > 0.0:
        return manual_extent_mm
    if reference_extents:
        key = (float(fr.tilt_x_deg), float(fr.tilt_y_deg))
        cached = reference_extents.get(key)
        if cached is not None and cached > 0.0:
            return float(cached)
    return _field_auto_extent(fr)


def compute_auto_extents(
    result: SpotResult,
) -> Dict[Tuple[float, float], float]:
    """Per-field auto-fit half-extent, keyed by ``(tilt_x_deg, tilt_y_deg)``.

    Public helper so the panel body can snapshot the first render's
    fit and replay it on subsequent renders — keeps lens-edit changes
    visible against a stable scale.
    """
    return {
        (float(fr.tilt_x_deg), float(fr.tilt_y_deg)): _field_auto_extent(fr)
        for fr in result.fields
    }


def _field_auto_extent(fr: SpotFieldResult) -> float:
    """Max distance from per-slice centroid across **all** defocus slices.

    Computed per slice (each slice is centred on its own centroid in
    the plot, so the relative *spread* is what we want — not the
    absolute deviation between slices, which is the centroid shift the
    plot already absorbs). Returns padded mm.
    """
    max_r = 0.0
    n_defocus = fr.xs.shape[0]
    for di in range(n_defocus):
        mask = fr.valid_mask[di]
        if not mask.any():
            continue
        xs = fr.xs[di][mask]
        ys = fr.ys[di][mask]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dx = xs - cx
        dy = ys - cy
        r = float(np.max(np.hypot(dx, dy)))
        if r > max_r:
            max_r = r
    return max(_MIN_EXTENT_MM, max_r * _AUTO_EXTENT_PAD)


def _format_extent_label(half_mm: float) -> Tuple[float, str, str]:
    """Pick a display unit (mm vs µm) and a tick-axis multiplier.

    Returns ``(multiplier_to_display, display_unit, scale_text)`` where
    ``multiplier_to_display`` is what to multiply mm-valued data by to
    get the value in display units. Sub-100µm spots read better as µm;
    larger ones stay in mm.
    """
    if half_mm < 0.1:
        return (1000.0, "µm", f"{2 * half_mm * 1000.0:.1f} µm wide")
    return (1.0, "mm", f"{2 * half_mm:.3f} mm wide")


class SpotDiagramCanvas(QWidget):
    """Matplotlib-backed canvas that lays out a grid of spot diagrams."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._figure = Figure(figsize=(8, 6), facecolor="#111")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)

        # Initial blank state — the first apply_result() call replaces this.
        self._render_placeholder("Load a lens and click Refresh")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result(
        self,
        result: Optional[SpotResult],
        reference_extents: Optional[Dict[Tuple[float, float], float]] = None,
    ) -> None:
        """Paint ``result`` into the figure.

        ``reference_extents`` maps ``(tilt_x_deg, tilt_y_deg)`` to a
        half-extent in mm — when present, those override per-row
        auto-fit so size changes between renders are visible. The body
        owns this map (captures it from the first auto-fit, replays it
        on subsequent renders, clears it on lens swap / explicit
        "Auto-Fit Scale Now").
        """
        if result is None:
            self._render_placeholder("No result")
            return
        self._render_result(result, reference_extents)

    def clear(self, placeholder: str = "") -> None:
        self._render_placeholder(placeholder)

    # ------------------------------------------------------------------
    # Internal — figure rendering
    # ------------------------------------------------------------------

    def _render_placeholder(self, text: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(1, 1, 1)
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color="#999", fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # draw() (synchronous) rather than draw_idle(): draw_idle merges
        # repeated requests into one paint on the next event-loop idle
        # cycle, which is fine for an interactive figure but invisible
        # to the user when a single update happens after a worker
        # finishes (no further events to coalesce). draw() guarantees
        # the new pixels hit the screen before set_result returns.
        self._canvas.draw()

    def _render_result(
        self,
        result: SpotResult,
        reference_extents: Optional[Dict[Tuple[float, float], float]] = None,
    ) -> None:
        self._figure.clear()
        n_fields = len(result.fields)
        n_defocus = len(result.defocus_offsets_mm)
        if n_fields == 0 or n_defocus == 0:
            self._render_placeholder("Empty spec — add at least one field")
            return

        # Each row gets its own per-field extent. ``sharex/sharey=False``
        # because the *whole point* is that field 0 (axial) and field N
        # (full field) can have wildly different scales — they should
        # not be coupled.
        axes_grid = self._figure.subplots(
            n_fields, n_defocus,
            squeeze=False,
            sharex=False, sharey=False,
        )
        wavelength_colours = [_wavelength_to_rgb(w) for w in result.wavelengths_nm]

        # If the user set a positive extent, lock every row to it.
        # Positive value beats both the auto-fit and the reference map
        # — it's an explicit manual setting.
        user_extent = float(result.spec.plot_half_extent_mm)
        manual_extent = user_extent if user_extent > 0.0 else None

        for row, field_result in enumerate(result.fields):
            row_extent_mm = _resolve_row_extent(
                field_result, reference_extents, manual_extent,
            )
            scale, unit, _scale_text = _format_extent_label(row_extent_mm)
            row_half = row_extent_mm * scale  # in display units

            for col, defocus in enumerate(result.defocus_offsets_mm):
                ax = axes_grid[row][col]
                ax.set_facecolor("#1a1a1a")

                # Each subplot is centred on its slice's centroid so the
                # bundle shape — not its absolute landing position — is
                # the thing the eye reads. The textbook spot diagram
                # does the same.
                cx, cy = _field_centroid_for_slice(field_result, col)

                if np.isnan(cx):
                    # No valid rays in this slice — skip drawing; the
                    # axis frame still renders so the layout stays
                    # symmetric across the row.
                    rms_mm: Optional[float] = None
                    n_valid = 0
                    n_total = int(field_result.valid_mask[col].size)
                else:
                    # Gather every valid ray across wavelengths for this
                    # slice so the RMS / valid-count annotation isn't
                    # biased by which colour happens to be drawn last.
                    flat_mask = field_result.valid_mask[col]
                    all_xs = field_result.xs[col][flat_mask]
                    all_ys = field_result.ys[col][flat_mask]
                    n_valid = int(flat_mask.sum())
                    n_total = int(flat_mask.size)
                    r2 = (all_xs - cx) ** 2 + (all_ys - cy) ** 2
                    rms_mm = float(np.sqrt(np.mean(r2))) if n_valid > 0 else None

                    for li, colour in enumerate(wavelength_colours):
                        xs = field_result.xs[col, li, :]
                        ys = field_result.ys[col, li, :]
                        mask = field_result.valid_mask[col, li, :]
                        if not mask.any():
                            continue
                        ax.scatter(
                            (xs[mask] - cx) * scale,
                            (ys[mask] - cy) * scale,
                            s=14, c=[colour], edgecolors="none", alpha=0.85,
                        )

                ax.set_xlim(-row_half, row_half)
                ax.set_ylim(-row_half, row_half)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(colors="#777", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#444")

                # In-subplot annotation. RMS spot radius is what the
                # user is actually trying to compare across defocus
                # slices — it shrinks at best focus, grows when
                # defocused — and is what changes when the lens edits
                # would otherwise leave the scaled picture looking the
                # same. Adding it here makes the numeric difference
                # visible regardless of the auto-fitted plot scale.
                if rms_mm is None:
                    annotation = f"no valid rays\n0 / {n_total}"
                else:
                    if rms_mm < 0.1:
                        rms_text = f"RMS {rms_mm * 1000.0:.1f} µm"
                    else:
                        rms_text = f"RMS {rms_mm:.3f} mm"
                    annotation = f"{rms_text}\n{n_valid} / {n_total} rays"
                ax.text(
                    0.03, 0.97, annotation,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    color="#cdcdcd", fontsize=7,
                    family="monospace",
                )

                # Column titles on the top row.
                if row == 0:
                    if defocus == 0.0:
                        title = "Gaussian"
                    else:
                        title = f"{defocus:+.3g} mm"
                    ax.set_title(title, color="#bbb", fontsize=9)

                # Row label (left column): field angle + the row's
                # display-unit scale, so the user knows whether they're
                # reading µm or mm without squinting at tick labels.
                if col == 0:
                    label = (
                        f"({field_result.tilt_x_deg:.1f}°, "
                        f"{field_result.tilt_y_deg:.1f}°)\n"
                        f"½-extent {row_extent_mm * scale:.2f} {unit}"
                    )
                    ax.set_ylabel(label, color="#bbb", fontsize=8)

        # Bottom-band legend: one swatch per wavelength. Use proxy
        # Line2D handles since the scatter calls produce one
        # PathCollection per (subplot, wavelength) and we want one
        # legend entry per wavelength across the whole figure.
        self._figure.legend(
            handles=[_proxy_artist(c) for c in wavelength_colours],
            labels=[f"{lam:.1f} nm" for lam in result.wavelengths_nm],
            loc="lower center", ncol=max(1, len(result.wavelengths_nm)),
            labelcolor="#ddd", facecolor="#111", edgecolor="#333",
            fontsize=8, frameon=True,
        )
        # Suptitle: just the pupil radius — sensor z is always 0 (the
        # convention's anchor) so showing it tells the user nothing and
        # clips the title on narrow panels.
        self._figure.suptitle(
            f"Spot Diagram  ·  pupil R = {result.pupil_radius_mm:.2f} mm",
            color="#ddd", fontsize=10,
        )

        # Tight layout with room for the bottom legend + suptitle.
        self._figure.subplots_adjust(
            left=0.10, right=0.98, top=0.92, bottom=0.12,
            wspace=0.30, hspace=0.30,
        )
        # See _render_placeholder for why this is draw() not draw_idle().
        self._canvas.draw()


def _proxy_artist(rgb: Tuple[float, float, float]):
    """Return a Line2D handle suitable for legend(handles=...)."""
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", color="none", markerfacecolor=rgb,
                  markeredgecolor="none", markersize=7, linestyle="None")
