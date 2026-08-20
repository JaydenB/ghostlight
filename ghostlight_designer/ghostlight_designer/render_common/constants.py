"""Constants shared by every flare-rendering panel.

Resolution / ray-grid / spectral count are per-panel and live on each
panel's :class:`~ghostlight_designer.render_common.dialogs.RenderSettings`;
what lives here is the handful of values that are the same everywhere so
the source-flare and ghost-explorer panels agree on framing, source
brightness and threading cadence.
"""
from __future__ import annotations

# Static render constants.
PUPIL_JITTER = 2
SOURCE_RGB = (10.0, 10.0, 10.0)
FLARE_GAIN = 5000.0

# Source-position sliders cover [-2, 2] in fractional-sensor units (0.5
# = on-axis); QSlider is integer-only, so map through SLIDER_SCALE.
SLIDER_MIN = -2.0
SLIDER_MAX = 2.0
SLIDER_SCALE = 1000

# A lens needs at least two surfaces for the calibration / ghost-pair
# enumeration to produce anything; below that panels show a placeholder.
MIN_SURFACES = 2

POLL_INTERVAL_MS = 50
