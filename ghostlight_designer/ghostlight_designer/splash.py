"""Startup splash screen.

Text-only today. To swap in artwork later: drop
``ghostlight_designer/resources/splash.png`` into the resources subpackage and
replace the ``QPixmap(...)`` block below with::

    from importlib.resources import files
    path = files('ghostlight_designer.resources').joinpath('splash.png')
    pixmap = QPixmap(str(path))
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

SPLASH_W, SPLASH_H = 480, 280


def build_splash() -> QSplashScreen:
    pixmap = QPixmap(SPLASH_W, SPLASH_H)
    pixmap.fill(QColor(30, 30, 36))

    painter = QPainter(pixmap)
    painter.setPen(QColor(220, 220, 230))
    font = painter.font()
    font.setPointSize(20)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "Ghostlight Designer")
    painter.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setAttribute(Qt.WA_DeleteOnClose)
    return splash
