"""Spec dataclass for the spot diagram panel.

A spec captures everything the panel needs to recompute its image: which
wavelengths, which field tilts, how many pupil rings × fans, how much
defocus to scan, and (display-only) the plot extent in mm.

Spec values are owned by the panel — they're independent of the project's
System Setup unless the user explicitly clicks "Sync from System Setup",
which copies the current setup values into the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# Wavelength rendering colours follow the Ghostlight convention: short
# = blue, mid = green, long = red. Spot points get coloured by their
# wavelength's position within the panel's spectral range.

# Default wavelengths: F-line, d-line, C-line. Same trio the project's
# System Setup falls back to.
DEFAULT_WAVELENGTHS_NM: Tuple[float, ...] = (486.13, 587.56, 656.27)

# Default field tilts in degrees (x, y). One axial + two off-axis is enough
# to give the spot diagram visible coma/astigmatism behaviour out of the box.
DEFAULT_FIELDS_DEG: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 7.0),
    (0.0, 14.0),
)

# Defocus offsets in mm. Three slices around the Gaussian image plane
# is the smallest set that makes a focus/defocus comparison obvious at
# a glance — exactly what users open the panel for. The textbook
# layout uses 5 slices (±2, ±1, 0); 3 is the same idea, less subplot
# clutter. Single-element ``(0.0,)`` is supported but doesn't tell the
# user whether their lens is focused at the sensor.
DEFAULT_DEFOCUS_OFFSETS_MM: Tuple[float, ...] = (-0.5, 0.0, 0.5)


@dataclass
class SpotDiagramSpec:
    """Inputs to one spot-diagram render.

    All collections are stored as tuples so equality / hashing is well-
    behaved and copies are cheap to compare for "did this change?".
    """

    wavelengths_nm: Tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_WAVELENGTHS_NM
    )
    fields_deg: Tuple[Tuple[float, float], ...] = field(
        default_factory=lambda: DEFAULT_FIELDS_DEG
    )
    rings: int = 4
    fans: int = 8
    # Entrance-pupil radius in mm. ``0.0`` asks the body to auto-pick
    # from the system's front surface semi_aperture at compute time, so
    # a brand-new spec works on any reasonable lens without manual
    # tweaking.
    pupil_radius_mm: float = 0.0  # 0.0 → auto from front surface
    defocus_offsets_mm: Tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_DEFOCUS_OFFSETS_MM
    )
    # Half-width of each subplot in mm. ``0.0`` = auto: the canvas
    # measures the spread of the actual ray landings and sizes each
    # field's subplots to fit. Set a positive value to force a fixed
    # scale (useful when you want to compare against a separate panel
    # at the same magnification).
    #
    # A fixed default would be wrong here: well-focused lenses produce
    # spots in the tens of microns, defocused ones span millimetres,
    # and there's no single value that shows both as a recognisable
    # pattern — the user would otherwise see a single dot in the
    # well-focused case and a clipped blob in the defocused one.
    plot_half_extent_mm: float = 0.0

    def clamp(self) -> "SpotDiagramSpec":
        """Return a copy with values pinned to safe ranges.

        The dialog/widget editors don't enforce limits themselves, so a
        user typo (zero rings, negative pupil) can't propagate into the
        compute and crash the worker.
        """
        return SpotDiagramSpec(
            wavelengths_nm=tuple(
                float(w) for w in self.wavelengths_nm if float(w) > 0.0
            ) or DEFAULT_WAVELENGTHS_NM,
            fields_deg=tuple((float(fx), float(fy)) for fx, fy in self.fields_deg)
                or DEFAULT_FIELDS_DEG,
            rings=max(0, int(self.rings)),
            fans=max(1, int(self.fans)),
            pupil_radius_mm=max(0.0, float(self.pupil_radius_mm)),
            defocus_offsets_mm=tuple(float(d) for d in self.defocus_offsets_mm)
                or (0.0,),
            plot_half_extent_mm=max(0.0, float(self.plot_half_extent_mm)),
        )
