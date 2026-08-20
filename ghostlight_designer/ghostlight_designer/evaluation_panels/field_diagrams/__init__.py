"""Field diagrams evaluation panel.

Three field-angle-dependent aberration plots in one panel, mirroring the
textbook figure 5.5 (Gerhard, *Lens Design Basics*):

* **Astigmatism + Petzval field curvature** — tangential and sagittal
  focus position as a function of field angle.
* **Distortion** — percent deviation of actual chief-ray image height
  from the paraxial expected height, vs field angle.
* **Lateral chromatic aberration** — chief-ray landing deviation per
  wavelength, vs field angle.

All three share a single field-angle axis (Y), so they read as a
side-by-side stack. Field angle along Y is the textbook convention.
"""
from __future__ import annotations

from .body import FieldDiagramBody
from .type import FIELD_DIAGRAMS_TYPE_ID, register_field_diagrams_panel_type

__all__ = [
    "FieldDiagramBody",
    "FIELD_DIAGRAMS_TYPE_ID",
    "register_field_diagrams_panel_type",
]
