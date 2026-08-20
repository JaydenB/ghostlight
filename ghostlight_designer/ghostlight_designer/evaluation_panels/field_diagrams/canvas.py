"""Matplotlib canvas painting a :class:`FieldDiagramResult`.

Layout follows the textbook figure 5.5: up to three side-by-side
sub-plots — astigmatism + Petzval, distortion, lateral chromatic —
each with **field angle on the Y axis** (origin at bottom, increasing
upward) and the diagram-specific aberration value on the X axis. The
shared Y axis means an artist can trace one field angle horizontally
across all three plots to read all the field-dependent aberrations
together.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .compute import FieldDiagramResult


# Same visible-spectrum approximation the spot diagram uses, so the
# colours line up between panels at a glance.
def _wavelength_to_rgb(nm: float) -> Tuple[float, float, float]:
    if nm < 380.0 or nm > 780.0:
        return (0.5, 0.5, 0.5)
    if nm < 440.0:
        r, g, b = (440.0 - nm) / 60.0, 0.0, 1.0
    elif nm < 490.0:
        r, g, b = 0.0, (nm - 440.0) / 50.0, 1.0
    elif nm < 510.0:
        r, g, b = 0.0, 1.0, (510.0 - nm) / 20.0
    elif nm < 580.0:
        r, g, b = (nm - 510.0) / 70.0, 1.0, 0.0
    elif nm < 645.0:
        r, g, b = 1.0, (645.0 - nm) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if nm < 420.0:
        scale = 0.3 + 0.7 * (nm - 380.0) / 40.0
    elif nm > 700.0:
        scale = 0.3 + 0.7 * (780.0 - nm) / 80.0
    else:
        scale = 1.0
    return (r * scale, g * scale, b * scale)


def _format_mm_or_um(value: float) -> str:
    """Pick mm vs µm for a tick / annotation."""
    if abs(value) < 0.1:
        return f"{value * 1000.0:.1f} µm"
    return f"{value:.3f} mm"


class FieldDiagramCanvas(QWidget):
    """Three-up matplotlib canvas for the field-diagrams panel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._figure = Figure(figsize=(9, 6), facecolor="#111")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas)

        self._render_placeholder("Load a lens to compute field diagrams")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result(self, result: Optional[FieldDiagramResult]) -> None:
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
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color="#999", fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._canvas.draw()

    def _render_result(self, result: FieldDiagramResult) -> None:
        self._figure.clear()
        spec = result.spec

        # Which subplots to draw. Disabled diagrams are skipped, the
        # remaining ones are arranged left-to-right so the user reads
        # them in the textbook order (astig → distortion → chromatic).
        active: List[str] = []
        if spec.show_astigmatism:
            active.append("astig")
        if spec.show_distortion:
            active.append("dist")
        if spec.show_lateral_chromatic:
            active.append("chrom")
        if not active:
            self._render_placeholder("All three sub-diagrams are hidden — enable one from the View menu")
            return

        # Field angle on the Y axis, shared across all subplots. ``squeeze=False``
        # so single-subplot results still get a 2-D axes_grid for uniform
        # indexing below.
        axes = self._figure.subplots(
            1, len(active),
            squeeze=False,
            sharey=True,
        )[0]
        max_field = float(spec.max_field_deg)

        for ax_idx, kind in enumerate(active):
            ax = axes[ax_idx]
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#777", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444")
            ax.set_ylim(0.0, max_field)
            ax.axvline(0.0, color="#444", linewidth=0.8, zorder=0)

            if ax_idx == 0:
                ax.set_ylabel("Field angle (°)", color="#ccc", fontsize=9)

            if kind == "astig":
                self._plot_astigmatism(ax, result)
            elif kind == "dist":
                self._plot_distortion(ax, result)
            elif kind == "chrom":
                self._plot_lateral_chromatic(ax, result)

        self._figure.suptitle(
            f"Field Diagrams  ·  λ_primary = {result.spec.primary_wavelength_nm:.1f} nm  ·  "
            f"pupil R = {result.pupil_radius_mm:.2f} mm",
            color="#ddd", fontsize=10,
        )
        self._figure.subplots_adjust(
            left=0.10, right=0.97, top=0.92, bottom=0.12,
            wspace=0.25,
        )
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Individual subplots
    # ------------------------------------------------------------------

    def _plot_astigmatism(self, ax, result: FieldDiagramResult) -> None:
        """Sagittal (dotted) and tangential (solid) defocus vs field.

        Both curves share the same units (mm relative to the sensor).
        Negative defocus means the field's focus sits in front of the
        sensor; positive, behind.
        """
        y = result.field_angles_deg
        x_sag = result.sagittal_defocus_mm
        x_tan = result.tangential_defocus_mm

        # Skip NaN samples per curve so a single bad field doesn't
        # break the line.
        valid_sag = ~np.isnan(x_sag)
        valid_tan = ~np.isnan(x_tan)
        if valid_tan.any():
            ax.plot(
                x_tan[valid_tan], y[valid_tan],
                color="#e6a23c", linewidth=1.6,
                label="tangential (M)",
            )
        if valid_sag.any():
            ax.plot(
                x_sag[valid_sag], y[valid_sag],
                color="#7ec8e3", linewidth=1.6, linestyle=(0, (4, 2)),
                label="sagittal (S)",
            )

        ax.set_title("Astigmatism", color="#ccc", fontsize=9)
        ax.set_xlabel("defocus (mm)", color="#bbb", fontsize=8)
        ax.legend(loc="lower right", facecolor="#111", edgecolor="#333",
                  labelcolor="#ddd", fontsize=7, frameon=True)

        # Symmetric autoscale around 0 so the eye reads positive vs
        # negative defocus relative to the sensor — never let the axis
        # collapse to a hairline when defocus is below ~1 µm.
        all_vals = np.concatenate([x_tan[valid_tan], x_sag[valid_sag]])
        _set_symmetric_xlim(ax, all_vals, floor=0.001)

    def _plot_distortion(self, ax, result: FieldDiagramResult) -> None:
        """Distortion percent vs field angle. Negative = barrel,
        positive = pincushion (Ch. 4.1.6 convention)."""
        y = result.field_angles_deg
        x = result.distortion_pct
        valid = ~np.isnan(x)
        if valid.any():
            ax.plot(x[valid], y[valid], color="#c47b9a", linewidth=1.8)

        ax.set_title("Distortion", color="#ccc", fontsize=9)
        ax.set_xlabel("distortion (%)", color="#bbb", fontsize=8)

        _set_symmetric_xlim(ax, x[valid], floor=0.1)

        # Distortion at the largest field is the canonical D_max readout
        # in the textbook — surface it as a small annotation since the
        # plot itself can be hard to read at small percentages.
        if valid.any():
            d_max = float(x[valid][np.argmax(np.abs(x[valid]))])
            ax.text(
                0.04, 0.96,
                f"D_max = {d_max:+.3f}%",
                transform=ax.transAxes, ha="left", va="top",
                color="#cdcdcd", fontsize=7, family="monospace",
            )

    def _plot_lateral_chromatic(self, ax, result: FieldDiagramResult) -> None:
        """Per-wavelength chief-ray landing deviation from the primary
        wavelength's landing, vs field angle.

        One curve per non-primary wavelength. The primary wavelength is
        the zero line by construction (shown as the axvline already
        drawn for every subplot).
        """
        y = result.field_angles_deg
        chief_y = result.chief_y_per_wavelength_mm
        primary_col = chief_y[:, result.primary_wavelength_index]

        any_drawn = False
        for li, lam in enumerate(result.wavelengths_nm):
            if li == result.primary_wavelength_index:
                continue
            dy = chief_y[:, li] - primary_col
            valid = ~np.isnan(dy)
            if not valid.any():
                continue
            ax.plot(
                dy[valid], y[valid],
                color=_wavelength_to_rgb(lam),
                linewidth=1.4,
                label=f"{lam:.1f} nm",
            )
            any_drawn = True

        ax.set_title("Lateral chromatic", color="#ccc", fontsize=9)
        ax.set_xlabel("Δy (mm)", color="#bbb", fontsize=8)
        if any_drawn:
            ax.legend(loc="lower right", facecolor="#111", edgecolor="#333",
                      labelcolor="#ddd", fontsize=7, frameon=True)
        else:
            ax.text(
                0.5, 0.5,
                "Only one wavelength configured\n— enable more for lateral chromatic",
                transform=ax.transAxes, ha="center", va="center",
                color="#888", fontsize=7,
            )

        # Symmetric scale around 0 so primary stays in the middle.
        # Gather the deviations we actually drew so the autoscale
        # isn't dominated by NaN columns.
        deltas = []
        for li in range(chief_y.shape[1]):
            if li == result.primary_wavelength_index:
                continue
            col = chief_y[:, li] - primary_col
            deltas.append(col[~np.isnan(col)])
        if deltas:
            _set_symmetric_xlim(ax, np.concatenate(deltas), floor=0.001)


def _set_symmetric_xlim(ax, values: np.ndarray, *, floor: float) -> None:
    """Set the X-axis to ``[-half, +half]`` where ``half`` is the
    largest absolute value in ``values`` (with a small floor so a
    perfectly aberration-free lens doesn't collapse the axis)."""
    if values.size == 0:
        ax.set_xlim(-floor, floor)
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_xlim(-floor, floor)
        return
    half = max(float(floor), float(np.max(np.abs(finite))) * 1.1)
    ax.set_xlim(-half, half)
