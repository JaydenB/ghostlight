"""Spec dataclass for the Seidel bar-chart panel.

One spec drives the paraxial Seidel computation and selects which of
the seven aberration bar charts get rendered. Per-panel custom fields
(pupil radius, field angle, wavelengths) match the rest of the
evaluation panel family — sync from System Setup is opt-in via the
View menu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Default wavelengths: F, d, C — gives a meaningful axial / lateral
# chromatic sum out of the box. The d-line drives the monochromatic
# (S_I..S_V) trace; F and C bracket it for the C_I / C_II differences.
DEFAULT_WAVELENGTHS_NM: Tuple[float, ...] = (486.13, 587.56, 656.27)
DEFAULT_PRIMARY_WAVELENGTH_INDEX = 1


@dataclass
class SeidelSpec:
    """Inputs to one Seidel-bar-chart render.

    The five monochromatic Seidel sums are computed once at the primary
    wavelength using a paraxial marginal ray (object at infinity,
    pupil_radius_mm at the entrance pupil) and a paraxial chief ray
    tilted by ``field_deg`` whose launch height is set so it passes
    through the aperture stop centre.

    Chromatic sums use the bracketing pair (F-line, C-line by
    convention) — pick those as wavelengths[0] and wavelengths[2] with
    the primary at index 1 to get sensible defaults.
    """

    # Field angle for the chief-ray trace. Distortion + coma scale with
    # field; pick something representative of the lens' working FOV.
    field_deg: float = 5.0

    # Wavelengths participating. Index ``primary_wavelength_index`` is
    # the d-line / monochromatic trace; chromatic sums bracket using
    # the lowest and highest in the tuple.
    wavelengths_nm: Tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_WAVELENGTHS_NM
    )
    primary_wavelength_index: int = DEFAULT_PRIMARY_WAVELENGTH_INDEX

    # Pupil radius for the marginal-ray height. ``0`` = auto from the
    # aperture stop's semi-aperture (matches the rest of the evaluation
    # panels).
    pupil_radius_mm: float = 0.0

    # Per-chart visibility. The monochromatic five are always available;
    # chromatic two need ≥ 2 distinct wavelengths to compute.
    show_spherical: bool = True
    show_coma: bool = True
    show_astigmatism: bool = True
    show_petzval: bool = True
    show_distortion: bool = True
    show_axial_color: bool = True
    show_lateral_color: bool = True

    def clamp(self) -> "SeidelSpec":
        """Return a copy with values pinned to safe ranges."""
        wavelengths = tuple(
            float(w) for w in self.wavelengths_nm if float(w) > 0.0
        ) or DEFAULT_WAVELENGTHS_NM
        primary_idx = max(0, min(int(self.primary_wavelength_index), len(wavelengths) - 1))
        return SeidelSpec(
            field_deg=max(0.0, float(self.field_deg)),
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary_idx,
            pupil_radius_mm=max(0.0, float(self.pupil_radius_mm)),
            show_spherical=bool(self.show_spherical),
            show_coma=bool(self.show_coma),
            show_astigmatism=bool(self.show_astigmatism),
            show_petzval=bool(self.show_petzval),
            show_distortion=bool(self.show_distortion),
            show_axial_color=bool(self.show_axial_color),
            show_lateral_color=bool(self.show_lateral_color),
        )

    @property
    def primary_wavelength_nm(self) -> float:
        wls = self.wavelengths_nm or DEFAULT_WAVELENGTHS_NM
        idx = max(0, min(self.primary_wavelength_index, len(wls) - 1))
        return float(wls[idx])
