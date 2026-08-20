"""Pack a display-referred float image into an owned RGB888 QImage.

The single home for float -> uint8 -> QImage conversion, shared by every
render panel so the packing cannot drift between them.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def to_qimage(disp: np.ndarray) -> QImage:
    """``(H, W, 3)`` display-referred float ``[0, 1]`` -> owned RGB888 QImage.

    Rounds to nearest 8-bit code (``+ 0.5``) rather than truncating, matching a
    compositor's display quantization.
    """
    rgb8 = np.ascontiguousarray(
        (np.asarray(disp, dtype=np.float32) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    )
    h, w, _ = rgb8.shape
    img = QImage(rgb8.data, w, h, w * 3, QImage.Format_RGB888)
    # .copy() detaches the buffer from the soon-to-be-freed numpy array.
    return img.copy()
