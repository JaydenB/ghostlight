"""Floating bottom-of-viewport info text for :class:`LensViewport`.

A non-interactive overlay widget that paints a single line of small
sans-serif text with a hard (unblurred) 1-pixel black drop shadow.  The
host sets the text via :meth:`InfoBar.set_text`; passing ``None`` or an
empty string hides the widget entirely so an incomplete lens doesn't
leave stale numbers on screen.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QWidget


# Visual constants — small sans-serif text with a one-pixel hard shadow.
_FONT_POINT_SIZE = 8
_SHADOW_OFFSET_PX = 1
# Padding around the text inside the widget; the shadow needs room past
# the text's natural advance so it isn't clipped at the right/bottom.
_PADDING_X = 4
_PADDING_Y = 2


class InfoBar(QWidget):
    """Bottom-overlay text row.

    Hidden by default and whenever the active text is empty.  Sized to
    fit its current text exactly — the host positions it.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # The overlay shouldn't steal clicks or hover events from the GL
        # surface beneath it; it's pure read-only annotation.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.NoFocus)

        font = QFont()
        font.setStyleHint(QFont.SansSerif)
        font.setPointSize(_FONT_POINT_SIZE)
        self.setFont(font)

        self._text: str = ""
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text(self) -> str:
        return self._text

    def set_text(self, text: Optional[str]) -> None:
        """Replace the displayed text.  Empty / ``None`` hides the bar.

        Calls :meth:`adjustSize` so the host's positioning code can read
        the new width/height immediately.
        """
        new_text = "" if text is None else str(text)
        if new_text == self._text:
            return
        self._text = new_text
        if not new_text:
            self.hide()
            self.update()
            return
        self.adjustSize()
        self.show()
        self.update()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def sizeHint(self):  # type: ignore[override]
        fm = QFontMetrics(self.font())
        w = fm.horizontalAdvance(self._text) + _SHADOW_OFFSET_PX + _PADDING_X * 2
        h = fm.height() + _SHADOW_OFFSET_PX + _PADDING_Y * 2
        from PySide6.QtCore import QSize
        return QSize(w, h)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _ev) -> None:  # type: ignore[override]
        if not self._text:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(self.font())
            fm = painter.fontMetrics()
            baseline_y = _PADDING_Y + fm.ascent()
            x = _PADDING_X
            # Hard black shadow, offset by one pixel down-right.
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(
                x + _SHADOW_OFFSET_PX,
                baseline_y + _SHADOW_OFFSET_PX,
                self._text,
            )
            # White foreground on top.
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, baseline_y, self._text)
        finally:
            painter.end()
