"""Right-click context popup for the lens viewport.

A frameless popup anchored at the cursor with a column of icon action
rows on top and drag-scrub value rows below, populated from the picked
element (element mode) or surface (surface mode). All optical mutations
route through :mod:`optical_editor.element_actions` /
:mod:`optical_editor.surface_actions`, so undo / redo and live viewport
repaint come for free.

Because a ``Qt.Popup`` funnels every mouse event to the top-level popup
(not its children), the popup hit-tests which row is under the cursor and
drives it — the same approach the tree value-scrubber uses.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import ghostlight

from ..project import Project
from ..settings import AppSettings
from ..optical_editor import element_actions as ea
from ..optical_editor import surface_actions as sa
from .scrub_row import (
    BG,
    BG_ARMED,
    BG_HOVER,
    BORDER,
    FG,
    FG_DISABLED,
    FG_MUTED,
    ScrubRow,
    ScrubRowSpec,
    Separator,
)

_log = logging.getLogger("ghostlight_designer.viewport_panel.context_popup")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


class _ActionRow(QFrame):
    """A clickable icon + label row. Enabled/checked/hover states are
    painted; the owning popup handles the actual click (Qt.Popup grab)."""

    def __init__(
        self,
        glyph: str,
        label_fn: Callable[[], str],
        enabled: bool,
        on_click: Callable[[], None],
        closes: bool,
        checked_fn: Optional[Callable[[], bool]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._label_fn = label_fn
        self.row_enabled = enabled
        self.on_click = on_click
        self.closes = closes
        self._checked_fn = checked_fn
        self._hover = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 5, 14, 5)
        layout.setSpacing(10)
        self._glyph = QLabel(glyph, self)
        self._glyph.setFixedWidth(16)
        self._text = QLabel(label_fn(), self)
        layout.addWidget(self._glyph)
        layout.addWidget(self._text)
        layout.addStretch(1)
        self._apply_fg()

    def _apply_fg(self) -> None:
        fg = FG if self.row_enabled else FG_DISABLED
        self._glyph.setStyleSheet(f"color: {fg.name()};")
        self._text.setStyleSheet(f"color: {fg.name()};")

    def refresh_label(self) -> None:
        self._text.setText(self._label_fn())
        self._apply_fg()
        self.update()

    def set_hover(self, hover: bool) -> None:
        if hover == self._hover:
            return
        self._hover = hover
        self.update()

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        if self._checked_fn is not None and self._checked_fn():
            painter.fillRect(self.rect(), BG_ARMED)
        elif self._hover and self.row_enabled:
            painter.fillRect(self.rect(), BG_HOVER)
        else:
            painter.fillRect(self.rect(), BG)


class _UnitToggle(QFrame):
    """Small clickable pill at the right of the Focus row (mm ⇄ dpt)."""

    def __init__(
        self, text_fn: Callable[[], str], on_click: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._text_fn = text_fn
        self.on_click = on_click
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 6, 1)
        layout.setSpacing(0)
        self._label = QLabel(text_fn(), self)
        self._label.setStyleSheet(f"color: {FG_MUTED.name()};")
        f = QFont(self._label.font())
        f.setPointSizeF(max(1.0, f.pointSizeF() - 1.0))
        self._label.setFont(f)
        layout.addWidget(self._label)

    def refresh(self) -> None:
        self._label.setText(self._text_fn())

    def paintEvent(self, _ev) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_HOVER)


# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------


class ViewportContextPopup(QFrame):
    """Context menu shown on a viewport right-click."""

    def __init__(
        self,
        project: Project,
        settings: Optional[AppSettings],
        viewport: QWidget,
        pick_info: dict,
    ) -> None:
        super().__init__(
            viewport,
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)

        self._project = project
        self._settings = settings
        self._viewport = viewport
        self._mode = pick_info.get("mode")
        self._element: Optional[ghostlight.Element] = pick_info.get("element")
        self._surface_index: Optional[int] = pick_info.get("surface_index")
        self._focus_unit = (
            settings.viewport_focus_unit() if settings is not None else "mm"
        )

        self._scrub_rows: List[ScrubRow] = []
        self._action_rows: List[_ActionRow] = []
        self._unit_toggle: Optional[_UnitToggle] = None
        self._focus_row: Optional[ScrubRow] = None
        self._active_scrub: Optional[ScrubRow] = None
        self._cursor_override = False
        self._hovered: Optional[QWidget] = None
        self._signals_connected = False

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(1, 1, 1, 1)
        self._outer.setSpacing(0)

        if self._mode == "surface":
            self._build_surface()
        else:
            self._build_element()

        # Live refresh + auto-close on structural replacement.
        project.systemModified.connect(self._on_system_changed)
        project.ghostSoloChanged.connect(self._on_system_changed)
        project.systemReplaced.connect(self._on_system_replaced)
        self._signals_connected = True

        self.adjustSize()
        global_pos = pick_info.get("global_pos") or QCursor.pos()
        self.move(self._clamp_to_screen(global_pos))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _add_action(
        self,
        glyph: str,
        label_fn,
        enabled: bool,
        on_click,
        closes: bool,
        checked_fn=None,
    ) -> None:
        if isinstance(label_fn, str):
            text = label_fn
            label_fn = lambda t=text: t  # noqa: E731
        row = _ActionRow(
            glyph, label_fn, enabled, on_click, closes, checked_fn, self
        )
        self._outer.addWidget(row)
        self._action_rows.append(row)

    def _add_scrub(self, spec: ScrubRowSpec) -> ScrubRow:
        row = ScrubRow(spec, self._project, self)
        self._outer.addWidget(row)
        self._scrub_rows.append(row)
        return row

    def _add_separator(self) -> None:
        self._outer.addWidget(Separator(self))

    def _build_element(self) -> None:
        el = self._element
        system = self._project.system
        is_singlet = (
            el is not None
            and el.kind == ghostlight.ElementKind.GLASS
            and len(el.surface_ids) == 2
        )
        is_stop = el is not None and el.kind == ghostlight.ElementKind.STOP
        n_surfaces = len(el.surface_ids) if el is not None else 0

        # --- structural actions -------------------------------------
        self._add_action(
            "⇄", "Flip Element", n_surfaces >= 2,
            lambda: ea.flip_element(self._project, el), closes=True,
        )
        self._add_action(
            "◫", "To Doublet", is_singlet,
            lambda: ea.convert_to_doublet(self._project, el), closes=True,
        )
        self._add_action(
            "«", "Merge Prev.",
            ea.can_merge_with_previous(self._project, el) if el else False,
            lambda: ea.merge_with_previous(self._project, el), closes=True,
        )
        self._add_action(
            "»", "Merge Next",
            ea.can_merge_with_next(self._project, el) if el else False,
            lambda: ea.merge_with_next(self._project, el), closes=True,
        )
        self._add_action(
            "∅",
            lambda: ("Un-Mute" if (el is not None and el.is_muted(system))
                     else "Mute"),
            not is_stop,
            self._toggle_mute, closes=False,
            checked_fn=lambda: el is not None and el.is_muted(system),
        )
        self._add_action(
            "◎",
            lambda: ("Un-Solo Ghosts"
                     if (el is not None
                         and ea.is_element_ghost_solo(self._project, el))
                     else "Solo Ghosts"),
            not is_stop,
            self._toggle_ghost_solo, closes=False,
            checked_fn=lambda: (
                el is not None
                and ea.is_element_ghost_solo(self._project, el)
            ),
        )
        self._add_action(
            "✕", "Delete", True,
            lambda: ea.remove_element(self._project, el), closes=True,
        )

        # --- scrub rows ---------------------------------------------
        self._add_separator()

        if is_singlet:
            self._unit_toggle = _UnitToggle(
                lambda: self._focus_unit, self._toggle_focus_unit,
            )
            focus_spec = self._focus_spec()
            focus_spec.trailing = self._unit_toggle
            self._focus_row = self._add_scrub(focus_spec)
            # Bend scrubs the front curvature (back auto-solved to hold
            # power) so a flat-flat singlet can still be bent — shape factor
            # is undefined there. Display the resulting front radius.
            self._add_scrub(ScrubRowSpec(
                label="Bend",
                get=lambda: ea.element_front_curvature(self._project.system, el),
                set=lambda c: ea.set_element_bend(self._project, el, c),
                fmt=self._bend_fmt,
                sensitivity=0.0005,   # front curvature (1/mm) per pixel
                undo_label="Scrub Bend",
            ))

        self._add_scrub(ScrubRowSpec(
            label="Move Z",
            get=self._element_z_get,
            set=lambda z: ea.move_element_z(self._project, el, z),
            fmt=lambda v: f"{v:.2f}",
            sensitivity=0.2,
            undo_label="Move Element Z",
        ))
        if not is_stop:
            self._add_scrub(ScrubRowSpec(
                label="Aperture",
                get=self._element_aperture_get,
                set=lambda a: ea.offset_element_aperture(self._project, el, a),
                fmt=lambda v: f"{v:.2f}",
                sensitivity=0.1,
                undo_label="Scrub Aperture",
            ))

    def _build_surface(self) -> None:
        # Ghost-solo toggle — mirrors the ODE surface menu's "Solo Ghost
        # Reflections"; hidden on stop surfaces (soloing the pupil surface
        # doesn't fit the "which ghosts does this surface bounce" model).
        is_stop = (
            self._surface_valid()
            and bool(self._project.system.surfaces[self._surface_index].is_stop)
        )
        if self._surface_valid() and not is_stop:
            self._add_action(
                "◎",
                lambda: ("Un-Solo Ghosts" if self._surface_ghost_solo()
                         else "Solo Ghosts"),
                True,
                self._toggle_surface_ghost_solo, closes=False,
                checked_fn=self._surface_ghost_solo,
            )
            self._add_separator()

        self._add_scrub(ScrubRowSpec(
            label="Pos Z",
            get=self._surface_z_get,
            set=self._surface_z_set,
            fmt=lambda v: f"{v:.2f}",
            sensitivity=0.2,
            undo_label="Scrub Pos Z",
        ))
        self._add_scrub(ScrubRowSpec(
            label="Radius",
            get=self._surface_radius_get,
            set=self._surface_radius_set,
            fmt=lambda v: f"{v:.3f}",
            sensitivity=0.5,
            undo_label="Scrub Radius",
        ))

    # ------------------------------------------------------------------
    # Focus row (unit-aware)
    # ------------------------------------------------------------------

    def _focus_spec(self) -> ScrubRowSpec:
        return ScrubRowSpec(
            label="Focus",
            get=self._focus_get,
            set=self._focus_set,
            fmt=self._focus_fmt,
            sensitivity=0.05,   # per-pixel; interpreted in the active unit
            undo_label="Scrub Focus",
        )

    def _focus_get(self) -> Optional[float]:
        phi = ea.element_power(self._project.system, self._element)
        if phi is None:
            return None
        if self._focus_unit == "dpt":
            return phi * 1000.0
        # EFL mm — undefined at (near-)zero power; disable so the value
        # doesn't blow up. The dpt unit still works through zero.
        if abs(phi) < 1e-9:
            return None
        return 1.0 / phi

    def _focus_set(self, shown_value: float) -> None:
        if self._focus_unit == "dpt":
            phi = shown_value / 1000.0
        else:
            f = shown_value
            if abs(f) < 0.1:
                f = 0.1 if f >= 0.0 else -0.1
            phi = 1.0 / f
        ea.set_element_power(self._project, self._element, phi)

    def _focus_fmt(self, v: float) -> str:
        if self._focus_unit == "dpt":
            return f"{v:+.3f} dpt"
        return f"{v:.2f} mm"

    def _toggle_focus_unit(self) -> None:
        self._focus_unit = "dpt" if self._focus_unit == "mm" else "mm"
        if self._settings is not None:
            self._settings.set_viewport_focus_unit(self._focus_unit)
        if self._unit_toggle is not None:
            self._unit_toggle.refresh()
        if self._focus_row is not None:
            self._focus_row.refresh()

    # ------------------------------------------------------------------
    # Scrub get/set closures
    # ------------------------------------------------------------------

    def _bend_fmt(self, c1: float) -> str:
        # The scrubbed quantity is front curvature; show the front radius,
        # which is the number a lens designer reads (flat at zero curvature).
        if abs(c1) < 1e-6:
            return "flat"
        return f"{1.0 / c1:.1f} mm"

    def _element_z_get(self) -> Optional[float]:
        el = self._element
        system = self._project.system
        try:
            idx = el.resolve_surfaces(system)
        except (KeyError, AttributeError):
            return None
        if not idx:
            return None
        return float(system.surfaces[idx[0]].z)

    def _element_aperture_get(self) -> Optional[float]:
        el = self._element
        system = self._project.system
        try:
            idx = el.resolve_surfaces(system)
        except (KeyError, AttributeError):
            return None
        if not idx:
            return None
        return float(system.surfaces[idx[0]].semi_aperture)

    def _surface_valid(self) -> bool:
        si = self._surface_index
        return (
            si is not None
            and 0 <= si < len(self._project.system.surfaces)
        )

    def _surface_z_get(self) -> Optional[float]:
        if not self._surface_valid():
            return None
        return float(self._project.system.surfaces[self._surface_index].z)

    def _surface_z_set(self, new_z: float) -> None:
        if not self._surface_valid():
            return
        system = self._project.system
        si = self._surface_index
        cur_z = float(system.surfaces[si].z)
        cur_t = float(system.surfaces[si].thickness)
        # Keep the surface at new_z with downstream fixed: thickness is the
        # gap to the next surface, and finalize() lays z back from the
        # sensor, so new_t = cur_t - (new_z - cur_z) lands surf.z on new_z.
        sa.set_surface_thickness(self._project, si, cur_t - (new_z - cur_z))

    def _surface_radius_get(self) -> Optional[float]:
        if not self._surface_valid():
            return None
        surf = self._project.system.surfaces[self._surface_index]
        if bool(surf.is_stop):
            return None
        if int(surf.form) != int(ghostlight.SurfaceForm.SPHERE):
            return None
        return float(surf.radius)

    def _surface_radius_set(self, r: float) -> None:
        if not self._surface_valid():
            return
        sa.set_surface_radius(self._project, self._surface_index, r)

    def _surface_ghost_solo(self) -> bool:
        if not self._surface_valid():
            return False
        uuid = str(self._project.system.surface_ids[self._surface_index])
        return self._project.is_surface_ghost_solo(uuid)

    def _toggle_surface_ghost_solo(self) -> None:
        if not self._surface_valid():
            return
        sa.set_surface_ghost_solo(
            self._project, self._surface_index, not self._surface_ghost_solo()
        )
        self._refresh_actions()

    # ------------------------------------------------------------------
    # Toggle actions (popup stays open)
    # ------------------------------------------------------------------

    def _toggle_mute(self) -> None:
        el = self._element
        system = self._project.system
        if el is None:
            return
        ea.set_element_muted(self._project, el, not el.is_muted(system))
        self._refresh_actions()

    def _toggle_ghost_solo(self) -> None:
        el = self._element
        if el is None:
            return
        ea.set_element_ghost_solo(
            self._project, el, not ea.is_element_ghost_solo(self._project, el)
        )
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        for row in self._action_rows:
            row.refresh_label()

    # ------------------------------------------------------------------
    # Live refresh / structural close
    # ------------------------------------------------------------------

    def _on_system_changed(self) -> None:
        for row in self._scrub_rows:
            row.refresh()
        self._refresh_actions()

    def _on_system_replaced(self, _system=None) -> None:
        # Undo/redo/flip/merge rebuild the system — our element wrapper is
        # orphaned, so dismiss rather than act on a stale handle.
        self.close()

    # ------------------------------------------------------------------
    # Mouse handling (popup-level hit-testing)
    # ------------------------------------------------------------------

    def _interactive_under(self, local: QPoint) -> Optional[QWidget]:
        widget = self.childAt(local)
        while widget is not None and widget is not self:
            if isinstance(widget, (_UnitToggle, _ActionRow, ScrubRow)):
                return widget
            widget = widget.parentWidget()
        return None

    def _invoke_action(self, row: _ActionRow) -> None:
        if not row.row_enabled:
            return
        if row.closes:
            # Dismiss first: close() disconnects our signals and schedules
            # deleteLater (self stays valid this frame), so the action's
            # systemReplaced/systemModified won't re-enter a live popup.
            self.close()
            try:
                row.on_click()
            except Exception:
                _log.exception("Viewport context action failed")
        else:
            try:
                row.on_click()
            except Exception:
                _log.exception("Viewport context toggle failed")

    def mousePressEvent(self, ev) -> None:
        local = ev.position().toPoint()
        if not self.rect().contains(local):
            self.close()
            ev.accept()
            return
        target = self._interactive_under(local)
        if isinstance(target, _UnitToggle):
            target.on_click()
            ev.accept()
            return
        if isinstance(target, _ActionRow):
            self._invoke_action(target)
            ev.accept()
            return
        if isinstance(target, ScrubRow):
            if ev.button() == Qt.LeftButton and target.is_scrubbable():
                if target.begin_scrub(ev.globalPosition().toPoint()):
                    self._active_scrub = target
                    QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
                    self._cursor_override = True
            ev.accept()
            return
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._active_scrub is not None:
            self._active_scrub.update_scrub(
                ev.globalPosition().toPoint(), ev.modifiers()
            )
            ev.accept()
            return
        self._update_hover(ev.position().toPoint())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if self._active_scrub is not None and ev.button() == Qt.LeftButton:
            self._active_scrub.end_scrub()
            self._active_scrub = None
            self._restore_cursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(ev)

    def _update_hover(self, local: QPoint) -> None:
        target = self._interactive_under(local)
        if target is self._hovered:
            return
        if isinstance(self._hovered, (_ActionRow, ScrubRow)):
            self._hovered.set_hover(False)
        if isinstance(target, (_ActionRow, ScrubRow)):
            target.set_hover(True)
        self._hovered = target

    # ------------------------------------------------------------------
    # Layout / cleanup
    # ------------------------------------------------------------------

    def _clamp_to_screen(self, pos: QPoint) -> QPoint:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return pos
        avail = screen.availableGeometry()
        size = self.sizeHint()
        x = max(avail.left(), min(pos.x(), avail.right() - size.width()))
        y = max(avail.top(), min(pos.y(), avail.bottom() - size.height()))
        return QPoint(x, y)

    def _restore_cursor(self) -> None:
        if self._cursor_override:
            QApplication.restoreOverrideCursor()
            self._cursor_override = False

    def _disconnect_signals(self) -> None:
        if not self._signals_connected:
            return
        for sig, slot in (
            (self._project.systemModified, self._on_system_changed),
            (self._project.ghostSoloChanged, self._on_system_changed),
            (self._project.systemReplaced, self._on_system_replaced),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._signals_connected = False

    def paintEvent(self, _ev) -> None:
        from PySide6.QtGui import QPen
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG)
        painter.setPen(QPen(BORDER, 2))
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

    def closeEvent(self, ev) -> None:
        # Defensive: a popup dismissed mid-drag must still close the compound
        # and restore the cursor.
        if self._active_scrub is not None:
            self._active_scrub.end_scrub()
            self._active_scrub = None
        self._restore_cursor()
        self._disconnect_signals()
        super().closeEvent(ev)
