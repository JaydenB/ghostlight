"""Qt-free image / video sequence writers for the designer's animation export.

Kept free of Qt (and of any panel imports) so the frame writers can be reused
by other render panels later and unit-tested without an event loop. The Qt
orchestration (worker thread, progress dialog) lives in the panel packages.
"""
from __future__ import annotations

from .writers import (
    ExportDependencyError,
    ExportError,
    FrameWriter,
    WRITER_SPECS,
    WriterSpec,
    check_writer_available,
    frame_path,
    make_writer,
    quantize_display,
    sanitize_hdr,
)

__all__ = [
    "ExportDependencyError",
    "ExportError",
    "FrameWriter",
    "WRITER_SPECS",
    "WriterSpec",
    "check_writer_available",
    "frame_path",
    "make_writer",
    "quantize_display",
    "sanitize_hdr",
]
