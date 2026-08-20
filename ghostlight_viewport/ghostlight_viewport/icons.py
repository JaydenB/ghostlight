"""Painted-glyph icons shared between the viewport overlay and host apps.

Icons are drawn at runtime from :data:`colors.PALETTE` so they pick up the
viewport's look without bundling asset files. Both the floating
:class:`ViewportToolbar` and the designer's optical-editor tree call
:func:`make_icon` with a kind string (e.g. ``"elem"``, ``"surf"``,
``"material"``).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication

from .colors import PALETTE


_DEFAULT_ICON_PX = 20
_ICON_CACHE: dict[tuple[str, int], QIcon] = {}


def _palette_color(key: str, alpha: int = 255) -> QColor:
    r, g, b = PALETTE[key]
    return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)


_SUPERSAMPLE = 3


def _new_pixmap(size: int) -> QPixmap:
    """Logical-``size`` pixmap with a fixed supersample factor baked in.

    Painting onto the returned pixmap uses logical coordinates (0..``size``)
    but rasterises at ``size * _SUPERSAMPLE`` device pixels, so thin pens and
    curves stay crisp when Qt scales the icon for HiDPI screens. The
    ``devicePixelRatio`` tells :class:`QIcon` to display it at logical size.
    """
    physical = size * _SUPERSAMPLE
    pm = QPixmap(physical, physical)
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(float(_SUPERSAMPLE))
    return pm


def _draw_axis_plane(p: QPainter, size: int, axis: str) -> None:
    """Plane-outline icon for X / Y / XY cutaway.

    The kept half is shown as a filled translucent quad; the discard half is
    just an outlined edge.  Axis letter is overlaid bottom-right.
    """
    margin = max(2, size // 8)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default")
    accent = _palette_color({
        "x": "axis_x",
        "y": "axis_y",
        "xy": "axis_z",
    }[axis])

    p.setPen(QPen(fg, 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRect(rect)

    fill = QColor(accent)
    fill.setAlpha(110)
    if axis == "x":
        kept = QRectF(rect.left(), rect.top(), rect.width() / 2.0, rect.height())
    elif axis == "y":
        kept = QRectF(rect.left(), rect.top() + rect.height() / 2.0,
                      rect.width(), rect.height() / 2.0)
    else:  # "xy"
        kept = QRectF(rect.left(), rect.top() + rect.height() / 2.0,
                      rect.width() / 2.0, rect.height() / 2.0)
    p.setPen(Qt.NoPen)
    p.setBrush(fill)
    p.drawRect(kept)

    font = QFont(p.font())
    font.setPixelSize(max(7, size // 2 - 2))
    font.setBold(True)
    p.setFont(font)
    p.setPen(QPen(fg))
    label = {"x": "X", "y": "Y", "xy": "XY"}[axis]
    p.drawText(rect.adjusted(0, 0, -1, -1), Qt.AlignBottom | Qt.AlignRight, label)


def _draw_none(p: QPainter, size: int) -> None:
    margin = max(2, size // 8)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default", alpha=170)
    p.setPen(QPen(fg, 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRect(rect)
    strike = _palette_color("selection_outline")
    p.setPen(QPen(strike, 1.8))
    p.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.top()))


def _draw_elem(p: QPainter, size: int) -> None:
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default")
    fill = _palette_color("glass_base", alpha=160)
    p.setPen(QPen(fg, 1.0))
    p.setBrush(fill)
    p.drawEllipse(rect)


def _draw_stop(p: QPainter, size: int) -> None:
    """Aperture-stop glyph — dark-grey octagonal iris with a polygon opening.

    Drawn as an octagon-shaped ring evoking an iris diaphragm, so Stop
    elements read as apertures rather than refracting glass elements.
    """
    margin = max(2, size // 6)
    cx = size / 2.0
    cy = size / 2.0
    outer_r = (size - 2 * margin) / 2.0
    inner_r = outer_r * 0.42

    fg = _palette_color("ray_default")
    fill = _palette_color("stop", alpha=235)

    sides = 8
    rotation = math.pi / sides

    def polygon(radius: float) -> QPainterPath:
        path = QPainterPath()
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides + rotation
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        return path

    ring = QPainterPath()
    ring.setFillRule(Qt.OddEvenFill)
    ring.addPath(polygon(outer_r))
    ring.addPath(polygon(inner_r))

    p.setPen(QPen(fg, 1.0))
    p.setBrush(fill)
    p.drawPath(ring)


def _draw_surf(p: QPainter, size: int) -> None:
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default")
    p.setPen(QPen(fg, 1.6))
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(rect.left() + rect.width() * 0.25, rect.top())
    path.cubicTo(
        QPointF(rect.left(), rect.top() + rect.height() * 0.35),
        QPointF(rect.left(), rect.bottom() - rect.height() * 0.35),
        QPointF(rect.left() + rect.width() * 0.25, rect.bottom()),
    )
    p.drawPath(path)


def _draw_material(p: QPainter, size: int) -> None:
    """Glass-slab glyph with a refracted ray passing through it.

    A thin translucent vertical block represents the optical medium; the
    incoming line bends as it enters and again as it exits, hinting at
    refraction without spelling out Snell's law.
    """
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default")
    fill = _palette_color("glass_base", alpha=140)

    slab_w = rect.width() * 0.34
    slab = QRectF(
        rect.center().x() - slab_w / 2.0,
        rect.top(),
        slab_w,
        rect.height(),
    )
    p.setPen(QPen(fg, 1.0))
    p.setBrush(fill)
    p.drawRect(slab)

    # Ray: enters top-left, refracts through the slab, exits bottom-right.
    enter = QPointF(rect.left(), rect.top() + rect.height() * 0.25)
    inside_top = QPointF(slab.left(), rect.top() + rect.height() * 0.40)
    inside_bot = QPointF(slab.right(), rect.top() + rect.height() * 0.60)
    exit_pt = QPointF(rect.right(), rect.top() + rect.height() * 0.75)
    p.setPen(QPen(fg, 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawLine(enter, inside_top)
    p.drawLine(inside_top, inside_bot)
    p.drawLine(inside_bot, exit_pt)


def _draw_add(p: QPainter, size: int) -> None:
    """Plus glyph — horizontal + vertical stroke through the middle."""
    margin = max(2, size // 5)
    fg = _palette_color("ray_default")
    pen = QPen(fg, max(1.6, size / 10.0))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    cx = size / 2.0
    cy = size / 2.0
    p.drawLine(QPointF(margin, cy), QPointF(size - margin, cy))
    p.drawLine(QPointF(cx, margin), QPointF(cx, size - margin))


def _draw_remove(p: QPainter, size: int) -> None:
    """Minus glyph — single horizontal stroke through the middle."""
    margin = max(2, size // 5)
    fg = _palette_color("ray_default")
    pen = QPen(fg, max(1.6, size / 10.0))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    cy = size / 2.0
    p.drawLine(QPointF(margin, cy), QPointF(size - margin, cy))


def _draw_chevron_pair(p: QPainter, size: int, direction: str) -> None:
    """Two stacked chevrons — ``"down"`` for expand-all, ``"up"`` for
    collapse-all. Reads as ``⌄⌄`` / ``⌃⌃``; the doubling implies "all",
    matching the accepted convention in tree-view toolbars."""
    margin = max(2, size // 5)
    fg = _palette_color("ray_default")
    pen = QPen(fg, max(1.4, size / 11.0))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx = size / 2.0
    left = margin
    right = size - margin
    inset = (right - left) * 0.10
    if direction == "down":
        upper_y, upper_tip = size * 0.28, size * 0.46
        lower_y, lower_tip = size * 0.54, size * 0.72
    else:  # "up"
        upper_tip, upper_y = size * 0.28, size * 0.46
        lower_tip, lower_y = size * 0.54, size * 0.72
    path = QPainterPath()
    path.moveTo(left + inset, upper_y)
    path.lineTo(cx, upper_tip)
    path.lineTo(right - inset, upper_y)
    path.moveTo(left + inset, lower_y)
    path.lineTo(cx, lower_tip)
    path.lineTo(right - inset, lower_y)
    p.drawPath(path)


def _draw_surf_solo(p: QPainter, size: int) -> None:
    """Solo'd-surface glyph — the standard surf curve plus a small accent
    dot that reads as "ghosts from this surface are picked out".

    Uses ``selection_outline`` for the dot so it matches the accent the
    rest of the UI uses for "you've targeted this". The curve stays in
    ``ray_default`` so the underlying glyph still reads as a surface.
    """
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default")
    accent = _palette_color("selection_outline")
    p.setPen(QPen(fg, 1.6))
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(rect.left() + rect.width() * 0.25, rect.top())
    path.cubicTo(
        QPointF(rect.left(), rect.top() + rect.height() * 0.35),
        QPointF(rect.left(), rect.bottom() - rect.height() * 0.35),
        QPointF(rect.left() + rect.width() * 0.25, rect.bottom()),
    )
    p.drawPath(path)
    # Accent dot on the curve mid-point so the badge is visually attached
    # to the surface, not floating in the corner.
    dot_r = max(1.5, size / 7.0)
    cx = rect.left() + rect.width() * 0.15
    cy = rect.center().y()
    p.setPen(Qt.NoPen)
    p.setBrush(accent)
    p.drawEllipse(QPointF(cx, cy), dot_r, dot_r)


def _draw_elem_muted(p: QPainter, size: int) -> None:
    """Muted-element glyph — desaturated element circle with a diagonal
    strike.

    Reuses the ``elem`` ellipse so the silhouette is still recognisably
    an element, but drains the fill alpha and overlays a strike from
    ``selection_outline`` (the same accent the "none" glyph uses) so a
    glance reads "this element is off". The strike runs top-right to
    bottom-left, mirroring familiar mute / no-entry iconography.
    """
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default", alpha=160)
    fill = _palette_color("glass_base", alpha=70)
    p.setPen(QPen(fg, 1.0))
    p.setBrush(fill)
    p.drawEllipse(rect)
    strike = _palette_color("selection_outline")
    p.setPen(QPen(strike, max(1.6, size / 9.0)))
    p.drawLine(QPointF(rect.right(), rect.top()),
               QPointF(rect.left(), rect.bottom()))


def _draw_unsolo_all(p: QPainter, size: int) -> None:
    """Un-Solo All glyph — the ``surf-solo`` accent dot with a diagonal
    strike through it, reading as "clear the solo highlight from every
    surface". Uses the same accent as ``surf-solo`` so the visual link
    between "solo" and "un-solo" is obvious."""
    fg = _palette_color("ray_default")
    accent = _palette_color("selection_outline")
    cx = size / 2.0
    cy = size / 2.0
    dot_r = max(2.0, size * 0.22)
    p.setPen(Qt.NoPen)
    p.setBrush(accent)
    p.drawEllipse(QPointF(cx, cy), dot_r, dot_r)
    margin = max(2, size // 5)
    pen = QPen(fg, max(1.6, size / 9.0))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(margin, size - margin),
               QPointF(size - margin, margin))


def _draw_unvariable_all(p: QPainter, size: int) -> None:
    """Un-Flag All Variables glyph — a short amber vertical bar (mirrors
    the delegate's left-edge stripe on flagged cells) with a diagonal
    strike, reading as "clear the variable-flag stripe from every cell".
    Same strike pattern as :func:`_draw_unsolo_all` so the family of
    "clear all X" toolbar buttons stays visually consistent.
    """
    # Amber matches ``SlotDelegate._VAR_STRIPE_COLOR`` — palette doesn't
    # expose a matching key today (the stripe is drawn as a literal
    # QColor in the delegate), so we duplicate the value here.
    amber = QColor(0xF5, 0xA6, 0x23)
    fg = _palette_color("ray_default")
    # Stripe: centre a short vertical bar in the icon rect.
    bar_w = max(2.0, size * 0.16)
    bar_h = size * 0.6
    bar_x = size / 2.0 - bar_w / 2.0
    bar_y = (size - bar_h) / 2.0
    p.setPen(Qt.NoPen)
    p.setBrush(amber)
    p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 1.0, 1.0)
    # Strike diagonally across it — matches the unsolo-all family.
    margin = max(2, size // 5)
    pen = QPen(fg, max(1.6, size / 9.0))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(margin, size - margin),
               QPointF(size - margin, margin))


def _draw_unmute_all(p: QPainter, size: int) -> None:
    """Un-Mute All glyph — the ``elem-muted`` circle-with-strike, with a
    short perpendicular segment cancelling the mute-strike near its
    centre. Reads as "the muted state is cleared from every element".
    Keeps the muted glyph visible so the visual link between "mute" and
    "un-mute" is obvious."""
    margin = max(2, size // 6)
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    fg = _palette_color("ray_default", alpha=160)
    fill = _palette_color("glass_base", alpha=70)
    p.setPen(QPen(fg, 1.0))
    p.setBrush(fill)
    p.drawEllipse(rect)
    strike = _palette_color("selection_outline")
    strike_pen = QPen(strike, max(1.6, size / 9.0))
    strike_pen.setCapStyle(Qt.RoundCap)
    p.setPen(strike_pen)
    p.drawLine(QPointF(rect.right(), rect.top()),
               QPointF(rect.left(), rect.bottom()))
    # Cancel segment perpendicular to the strike, crossing its centre.
    fg_full = _palette_color("ray_default")
    cancel_pen = QPen(fg_full, max(1.6, size / 9.0))
    cancel_pen.setCapStyle(Qt.RoundCap)
    p.setPen(cancel_pen)
    cx = rect.center().x()
    cy = rect.center().y()
    # Half-length along the perpendicular direction (also diagonal, but
    # rotated 90° from the mute-strike): top-left → bottom-right axis,
    # centred on the ellipse midpoint.
    span = rect.width() * 0.28
    p.drawLine(QPointF(cx - span, cy - span),
               QPointF(cx + span, cy + span))


def make_icon(kind: str, size: int = _DEFAULT_ICON_PX) -> QIcon:
    """Return a cached painted-glyph icon for ``kind``.

    Returns an empty :class:`QIcon` if no :class:`QApplication` exists yet
    (icon construction requires one); callers in tests that don't run a Qt
    event loop can still safely use the menu-action machinery.
    """
    if QApplication.instance() is None:
        return QIcon()
    cached = _ICON_CACHE.get((kind, size))
    if cached is not None:
        return cached
    pm = _new_pixmap(size)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)
    if kind in ("x", "y", "xy"):
        _draw_axis_plane(p, size, kind)
    elif kind == "none":
        _draw_none(p, size)
    elif kind == "elem":
        _draw_elem(p, size)
    elif kind == "elem-muted":
        _draw_elem_muted(p, size)
    elif kind == "stop":
        _draw_stop(p, size)
    elif kind == "surf":
        _draw_surf(p, size)
    elif kind == "surf-solo":
        _draw_surf_solo(p, size)
    elif kind == "material":
        _draw_material(p, size)
    elif kind == "sel_none":
        _draw_none(p, size)
    elif kind == "add":
        _draw_add(p, size)
    elif kind == "remove":
        _draw_remove(p, size)
    elif kind == "expand-all":
        _draw_chevron_pair(p, size, "down")
    elif kind == "collapse-all":
        _draw_chevron_pair(p, size, "up")
    elif kind == "unsolo-all":
        _draw_unsolo_all(p, size)
    elif kind == "unmute-all":
        _draw_unmute_all(p, size)
    elif kind == "unvariable-all":
        _draw_unvariable_all(p, size)
    else:
        pass
    p.end()
    icon = QIcon(pm)
    _ICON_CACHE[(kind, size)] = icon
    return icon
