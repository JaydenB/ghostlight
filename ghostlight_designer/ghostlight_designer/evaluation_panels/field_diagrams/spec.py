"""Spec dataclass for the field diagrams panel.

One spec drives three side-by-side aberration plots (astigmatism +
Petzval, distortion, lateral chromatic) that all share a field-angle
axis. The user controls the field-angle sample range and density,
which wavelengths participate, and which sub-diagrams are visible.

Per-panel custom fields rather than reading from System Setup — same
rationale as the spot diagram. A "Sync from System Setup" menu action
copies setup values across when the user wants them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Default wavelengths — F, d, C lines. Same trio as the spot diagram
# and the project's System Setup fall-back.
DEFAULT_WAVELENGTHS_NM: Tuple[float, ...] = (486.13, 587.56, 656.27)

# Default primary wavelength index points at d-line (587.56 nm). The
# astigmatism and distortion plots trace at this wavelength only;
# lateral chromatic uses every wavelength in the list.
DEFAULT_PRIMARY_WAVELENGTH_INDEX = 1


@dataclass
class FieldDiagramSpec:
    """Inputs to one field-diagrams render.

    Field axis runs from 0 to ``max_field_deg`` in ``field_samples``
    evenly-spaced points (inclusive of both endpoints). For the
    default ``14.0 / 11`` that's 0°, 1.4°, 2.8°, …, 14.0°.
    """

    # Field-angle axis.
    max_field_deg: float = 14.0
    field_samples: int = 11

    # Wavelengths participating in the analysis. Astigmatism + distortion
    # trace at ``wavelengths_nm[primary_wavelength_index]`` only;
    # lateral chromatic traces at every wavelength and plots each
    # non-primary relative to the primary.
    wavelengths_nm: Tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_WAVELENGTHS_NM
    )
    primary_wavelength_index: int = DEFAULT_PRIMARY_WAVELENGTH_INDEX

    # Sagittal / tangential focus is found by tracing a small fan of
    # rays in X (sagittal) or Y (tangential) at each field, then fitting
    # the z at which the bundle's spread along that axis is minimised.
    # ``0`` = auto pupil radius (from front surface semi_aperture).
    pupil_radius_mm: float = 0.0
    rays_per_fan: int = 5

    # Per-diagram visibility. All on by default; the user toggles them
    # from the View menu.
    show_astigmatism: bool = True
    show_distortion: bool = True
    show_lateral_chromatic: bool = True

    def clamp(self) -> "FieldDiagramSpec":
        """Return a copy with values pinned to safe ranges.

        Field samples must be ≥ 2 to draw a line; wavelength tuple
        must be non-empty; primary index must point inside the tuple.
        """
        wavelengths = tuple(
            float(w) for w in self.wavelengths_nm if float(w) > 0.0
        ) or DEFAULT_WAVELENGTHS_NM
        primary_idx = max(0, min(int(self.primary_wavelength_index), len(wavelengths) - 1))
        return FieldDiagramSpec(
            max_field_deg=max(0.1, float(self.max_field_deg)),
            field_samples=max(2, int(self.field_samples)),
            wavelengths_nm=wavelengths,
            primary_wavelength_index=primary_idx,
            pupil_radius_mm=max(0.0, float(self.pupil_radius_mm)),
            rays_per_fan=max(3, int(self.rays_per_fan)),
            show_astigmatism=bool(self.show_astigmatism),
            show_distortion=bool(self.show_distortion),
            show_lateral_chromatic=bool(self.show_lateral_chromatic),
        )

    @property
    def primary_wavelength_nm(self) -> float:
        wls = self.wavelengths_nm or DEFAULT_WAVELENGTHS_NM
        idx = max(0, min(self.primary_wavelength_index, len(wls) - 1))
        return float(wls[idx])
