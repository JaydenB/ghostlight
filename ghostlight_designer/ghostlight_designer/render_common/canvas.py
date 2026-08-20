"""Image canvas shared by the flare-rendering panels.

The source-flare panel and the ghost explorer both subclass
:class:`FlareCanvas`, so the marker/de-squeeze/vignette painting contract
lives here rather than in any one panel.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class FlareCanvas(QWidget):
    """Image canvas with a draggable red marker.

    Coordinates emitted by :attr:`sourceDragged` are in fractional-sensor
    units where (0.5, 0.5) is on-axis and the image extent is [0, 1] on
    each axis (matches ``demo_interactive``).
    """

    sourceDragged = Signal(float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._sx: float = 0.5
        self._sy: float = 0.5
        self._dragging: bool = False
        self._placeholder: str = "Load a lens to render"
        # Half-red vignette overlay (regions no primary ray can reach). A small
        # RGBA mask image, smooth-scaled to the frame; ``None`` = nothing to
        # draw. ``_vignette_visible`` is the panel's enable toggle.
        self._vignette_img: Optional[QImage] = None
        self._vignette_visible: bool = False
        # Horizontal stretch applied when painting the rendered image.
        # 1.0 = no stretch; 2.0 displays a 2× anamorphic frame de-squeezed.
        # Source-marker coords and mouse picking stay in image-fractional
        # units, so the rect change keeps both consistent.
        self._squeeze: float = 1.0
        self.setMinimumSize(QSize(256, 256))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, img: Optional[QImage]) -> None:
        self._image = img
        self.update()

    def clear_image(self, placeholder: str) -> None:
        self._image = None
        self._placeholder = placeholder
        self.update()

    def set_source(self, sx: float, sy: float) -> None:
        if sx == self._sx and sy == self._sy:
            return
        self._sx = sx
        self._sy = sy
        self.update()

    def set_vignette_image(self, img: Optional[QImage]) -> None:
        """Set (or clear) the vignette mask. Repaints only when the overlay
        is currently shown — a background recompute while hidden is silent."""
        self._vignette_img = img
        if self._vignette_visible:
            self.update()

    def set_vignette_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._vignette_visible:
            return
        self._vignette_visible = visible
        self.update()

    def set_squeeze(self, squeeze: float) -> None:
        """Set the horizontal stretch factor for the displayed image.

        ``1.0`` is no stretch; values > 1 stretch horizontally (anamorphic
        de-squeeze). Non-finite or non-positive values fall back to 1.0.
        """
        s = float(squeeze)
        if not (s > 0.0) or s != s:
            s = 1.0
        if s == self._squeeze:
            return
        self._squeeze = s
        self.update()

    def _image_rect(self) -> QRectF:
        """Aspect-preserving rect for the current image inside this widget.

        The image's effective aspect is ``(iw * squeeze) : ih`` — when a
        de-squeeze factor is set the rect is computed against the
        stretched-width version of the image.
        """
        if self._image is None:
            return QRectF(self.rect())
        wid = float(self.width())
        hgt = float(self.height())
        iw = float(self._image.width()) * self._squeeze
        ih = float(self._image.height())
        if iw <= 0.0 or ih <= 0.0:
            return QRectF(self.rect())
        scale = min(wid / iw, hgt / ih)
        out_w = iw * scale
        out_h = ih * scale
        x = (wid - out_w) / 2.0
        y = (hgt - out_h) / 2.0
        return QRectF(x, y, out_w, out_h)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(13, 13, 13))
        if self._image is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return
        r = self._image_rect()
        p.drawImage(r, self._image)
        # Vignette overlay — drawn over the frame but under the source marker.
        # Smooth transform so the coarse mask reads as a soft-edged red region.
        if self._vignette_visible and self._vignette_img is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(r, self._vignette_img)
        dot_x = r.left() + self._sx * r.width()
        dot_y = r.top() + self._sy * r.height()
        # Only draw when the marker falls inside the widget — slider
        # values outside [0, 1] put the source off-frame and the marker
        # has nowhere meaningful to sit.
        if (0.0 <= dot_x <= self.width()) and (0.0 <= dot_y <= self.height()):
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(QColor(255, 255, 255), 1.0))
            p.setBrush(QColor(255, 64, 64))
            p.drawEllipse(QPointF(dot_x, dot_y), 3.5, 3.5)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None or event.button() != Qt.LeftButton:
            return
        r = self._image_rect()
        if not r.contains(event.position()):
            return
        self._dragging = True
        self._emit_from_event(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._emit_from_event(event)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._dragging = False

    def _emit_from_event(self, event: QMouseEvent) -> None:
        r = self._image_rect()
        if r.width() <= 0.0 or r.height() <= 0.0:
            return
        sx = (event.position().x() - r.left()) / r.width()
        sy = (event.position().y() - r.top()) / r.height()
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        self.sourceDragged.emit(sx, sy)
