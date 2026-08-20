"""Machinery shared by the flare-rendering panels.

The source-flare renderer, the ghost explorer and the PSF panel all need
the same canvas, the same per-panel render settings + quality presets, the
same exposure dialog and the same spinbox value-scrubber. It lives here
rather than in any one panel, so no renderer owns it.
"""
from __future__ import annotations

from .canvas import FlareCanvas
from .constants import (
    FLARE_GAIN,
    MIN_SURFACES,
    POLL_INTERVAL_MS,
    PUPIL_JITTER,
    SLIDER_MAX,
    SLIDER_MIN,
    SLIDER_SCALE,
    SOURCE_RGB,
)
from .dialogs import (
    DRAFT_PRESET,
    EXPOSURE_STOPS_MAX,
    EXPOSURE_STOPS_MIN,
    EXPOSURE_STOPS_STEP,
    HIGH_PLUS_PRESET,
    HIGH_PRESET,
    MID_PRESET,
    ExposureDialog,
    RenderSettings,
    RenderSettingsDialog,
)
from .spinbox_scrub import attach_spinbox_scrubber

__all__ = [
    "FlareCanvas",
    "FLARE_GAIN",
    "MIN_SURFACES",
    "POLL_INTERVAL_MS",
    "PUPIL_JITTER",
    "SLIDER_MAX",
    "SLIDER_MIN",
    "SLIDER_SCALE",
    "SOURCE_RGB",
    "DRAFT_PRESET",
    "MID_PRESET",
    "HIGH_PRESET",
    "HIGH_PLUS_PRESET",
    "EXPOSURE_STOPS_MIN",
    "EXPOSURE_STOPS_MAX",
    "EXPOSURE_STOPS_STEP",
    "ExposureDialog",
    "RenderSettings",
    "RenderSettingsDialog",
    "attach_spinbox_scrubber",
]
