"""Colour helpers: wavelength → RGB and element tint palette."""

from __future__ import annotations


# Soft VFX-ish palette tuned for the lens visualisation.
PALETTE = {
    "background_top":    (0.20, 0.22, 0.26),
    "background_bottom": (0.04, 0.05, 0.06),
    "glass_base":        (0.40, 0.575, 0.875),
    "glass_warm":        (0.80, 0.75, 0.80),
    "lens_wall":         (0.20, 0.22, 0.25),
    "stop":              (0.18, 0.18, 0.18),
    "sensor_border":     (0.95, 0.85, 0.50),
    "sensor_fill":       (0.08, 0.10, 0.12),
    "calibrated_sensor_border": (0.0, 0.0, 0.0),
    "selection_outline": (1.00, 0.55, 0.10),
    "hover_outline":     (0.95, 0.90, 0.35),
    "axis_x":            (0.85, 0.30, 0.30),
    "axis_y":            (0.30, 0.85, 0.40),
    "axis_z":            (0.30, 0.55, 0.95),
    "ray_default":       (0.95, 0.95, 0.95),
    # Element centre-of-rotation marker. Deliberately NOT one of the axis_*
    # colours — it's drawn as a small 3-axis cross, so reusing red/green/blue
    # would read as a second world gizmo floating in the lens. Magenta is
    # unused elsewhere in the scene and stays legible against glass.
    "pivot_marker":      (0.95, 0.35, 0.85),
}


def wavelength_to_rgb(lam_nm: float) -> tuple[float, float, float]:
    """Approximate visible-spectrum colour for a single wavelength in nm.

    Piecewise linear across CIE-ish bands with intensity falloff at the
    visible-spectrum edges.  Returns (r, g, b) in [0, 1].
    """
    lam = float(lam_nm)
    if lam < 380.0:
        lam = 380.0
    elif lam > 700.0:
        lam = 700.0

    if 380.0 <= lam < 440.0:
        r = -(lam - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440.0 <= lam < 490.0:
        r = 0.0
        g = (lam - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490.0 <= lam < 510.0:
        r = 0.0
        g = 1.0
        b = -(lam - 510.0) / (510.0 - 490.0)
    elif 510.0 <= lam < 580.0:
        r = (lam - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580.0 <= lam < 645.0:
        r = 1.0
        g = -(lam - 645.0) / (645.0 - 580.0)
        b = 0.0
    else:  # 645..700
        r = 1.0
        g = 0.0
        b = 0.0

    # Edge falloff so deep violet / deep red don't render as full intensity.
    if lam < 420.0:
        factor = 0.3 + 0.7 * (lam - 380.0) / (420.0 - 380.0)
    elif lam > 645.0:
        factor = 0.3 + 0.7 * (700.0 - lam) / (700.0 - 645.0)
    else:
        factor = 1.0

    gamma = 0.8
    return (
        (r * factor) ** gamma if r > 0.0 else 0.0,
        (g * factor) ** gamma if g > 0.0 else 0.0,
        (b * factor) ** gamma if b > 0.0 else 0.0,
    )


def element_tint(mean_ior: float) -> tuple[float, float, float]:
    """Tint for a glass element from its mean refractive index."""
    t = max(0.0, min(1.0, (mean_ior - 1.45) / (1.85 - 1.45)))
    smooth = t * t * (3.0 - 2.0 * t)
    base = PALETTE["glass_base"]
    warm = PALETTE["glass_warm"]
    return (
        base[0] + (warm[0] - base[0]) * smooth,
        base[1] + (warm[1] - base[1]) * smooth,
        base[2] + (warm[2] - base[2]) * smooth,
    )
