"""Optional helpers for reshaping Ghostlight planar outputs into HWC arrays."""

from __future__ import annotations

import numpy as np


def ghost_to_hwc(result: dict) -> np.ndarray:
    """Stack ghost_r/g/b channels into a (H, W, 3) float32 array."""
    return np.stack([result["ghost_r"], result["ghost_g"], result["ghost_b"]], axis=-1)


def starburst_to_hwc(result: dict) -> np.ndarray:
    """Stack starburst_r/g/b channels into a (H, W, 3) float32 array.

    Returns None when the starburst layer is absent (DiffractionConfig.starburst
    off) — the additive-composite caller then simply skips the term.
    """
    if "starburst_r" not in result:
        return None
    return np.stack(
        [result["starburst_r"], result["starburst_g"], result["starburst_b"]], axis=-1
    )


def veil_to_hwc(result: dict) -> np.ndarray:
    """Stack veil_r/g/b channels into a (H, W, 3) float32 array.

    Returns None when the veil layer is absent (DiffractionConfig.veil off).
    """
    if "veil_r" not in result:
        return None
    return np.stack([result["veil_r"], result["veil_g"], result["veil_b"]], axis=-1)


def gate_to_hwc(result: dict) -> np.ndarray:
    """Stack gate_r/g/b channels into a (H, W, 3) float32 array.

    Returns None when the gate layer is absent (GateConfig.enabled off) — the
    additive-composite caller then simply skips the term. The gate layer is the
    aperture plate's cut edge scattering light back into frame from just outside
    it; like the veil it joins the metered flare layer (ghost + veil + gate).
    """
    if "gate_r" not in result:
        return None
    return np.stack([result["gate_r"], result["gate_g"], result["gate_b"]], axis=-1)


def planar_from_hwc(hwc: np.ndarray):
    """Split a (H, W, 3) float32 array into three planar (H, W) arrays."""
    hwc = np.ascontiguousarray(hwc, dtype=np.float32)
    return hwc[..., 0], hwc[..., 1], hwc[..., 2]
