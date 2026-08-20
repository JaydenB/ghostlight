"""Shared test fixtures for the per-issue validation suite.

Centralises the offscreen Qt setup, a programmatic doublet lens for headless
rendering, and a reference Python implementation of ghostlight's asphere sag
(so the sag-parity test in :mod:`test_sag_matches_ghostlight` doesn't need a
rebuilt C++ binding to be runnable).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from typing import Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Path bootstrap so tests run from a clean checkout
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent                  # repository root


# ---------------------------------------------------------------------------
# Offscreen Qt + GL availability
# ---------------------------------------------------------------------------

def require_qt():
    """Return (QApplication, LensViewport) or skip the test."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
        from ghostlight_viewport import LensViewport
        from ghostlight_viewport.widget import set_default_surface_format
    except Exception as exc:
        pytest.skip(f"PySide6/ghostlight_viewport unavailable: {exc}")
    set_default_surface_format()
    app = QApplication.instance() or QApplication([])
    return app, LensViewport


def require_ghostlight():
    try:
        import ghostlight  # noqa: F401
    except Exception as exc:
        pytest.skip(f"ghostlight unavailable: {exc}")
    return ghostlight


# ---------------------------------------------------------------------------
# Reference doublet for headless tests
# ---------------------------------------------------------------------------

def example_doublet_path() -> pathlib.Path:
    return (_ROOT / "ghostlight" / "bindings" / "python" / "tests"
            / "fixtures" / "example_doublet.lens")


def load_example_lens():
    ghostlight = require_ghostlight()
    return ghostlight.OpticalSystem.load(str(example_doublet_path()))


def load_example_elements():
    ghostlight = require_ghostlight()
    return ghostlight.Element.from_lens_file(str(example_doublet_path()))


# ---------------------------------------------------------------------------
# Reference sag matching ghostlight's `asphere_sag` C++ inline (lens_calibration
# notes upstream).  Tests can compare geometry.sag against this for parity
# without needing to call into the binding.
# ---------------------------------------------------------------------------

def reference_asphere_sag(r: float, radius: float, conic_k: float,
                           terms: list[float]) -> float:
    if abs(radius) < 1e-12:
        # Flat — first term collapses to 0; only asphere series contribute.
        z = 0.0
    else:
        inv_R = 1.0 / radius
        r2 = r * r
        D = 1.0 - (1.0 + conic_k) * inv_R * inv_R * r2
        if D <= 0.0:
            return 1e30
        z = inv_R * r2 / (1.0 + math.sqrt(D))
    r2 = r * r
    rp = r2 * r2          # r^4
    for a in terms:
        z += a * rp
        rp *= r2
    return z


# ---------------------------------------------------------------------------
# Frame-rendering helper for any test that needs paintGL to have run.
# ---------------------------------------------------------------------------

def render_one_frame(widget, width: int = 480, height: int = 320,
                     processes: int = 4) -> None:
    """Resize, show off-screen, and trigger a frame to be drawn."""
    from PySide6.QtCore import QCoreApplication
    widget.resize(width, height)
    widget.show()
    for _ in range(processes):
        QCoreApplication.processEvents()
    widget.repaint()
    for _ in range(processes):
        QCoreApplication.processEvents()
