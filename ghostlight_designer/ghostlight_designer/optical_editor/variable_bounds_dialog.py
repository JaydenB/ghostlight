"""Modal dialog for editing a variable-flagged cell's bounds.

Opened from the tree's right-click menu on any cell whose slot carries
``options["variable_attr"]`` (currently the SURFACE Radius, SURFACE Pos Z,
and the ASPHERE / CYLINDRICAL form Radius slots).

Layout::

    ┌──────────────────────────────────────┐
    │ Variable Bounds                       │
    │ ─────────────────────────────────────│
    │ Surface 3 — Radius                    │
    │                                       │
    │  ☑ Enable as optimization variable    │
    │                                       │
    │  Min: ┌─────────────┐  (blank = -∞)  │
    │       └─────────────┘                 │
    │  Max: ┌─────────────┐  (blank = +∞)  │
    │       └─────────────┘                 │
    │                                       │
    │              [Cancel] [OK]            │
    └──────────────────────────────────────┘

Empty text fields = unbounded on that side. Toggling "Enable as
optimization variable" off is the same as picking "Toggle Variable" from
the menu — it clears the flag, so the dialog doubles as an
un-flag affordance.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..project import VariableBounds


def _parse_optional_float(text: str) -> Optional[float]:
    """Blank / whitespace → ``None`` (unbounded). Otherwise parse a float.

    Callers get ``None`` on any parse failure — the dialog's validators
    already restricted input to well-formed floats, so the fallback is
    just a paranoid guard.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _format_optional_float(value: Optional[float]) -> str:
    """Empty string for ``None`` (unbounded), fixed-precision otherwise."""
    if value is None:
        return ""
    return f"{float(value):g}"


class VariableBoundsDialog(QDialog):
    """Small modal for editing (or removing) a variable flag's bounds.

    Constructed with the current state; ``exec()`` returns ``QDialog.Accepted``
    on OK. Read the outcome via :meth:`enabled` and :meth:`bounds`.
    """

    def __init__(
        self,
        *,
        title_summary: str,
        current_enabled: bool,
        current_bounds: VariableBounds,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Variable Bounds")
        self.setModal(True)

        header = QLabel(title_summary, self)
        header.setStyleSheet("font-weight: bold;")

        self._enable = QCheckBox("Enable as optimization variable", self)
        self._enable.setChecked(bool(current_enabled))
        self._enable.toggled.connect(self._on_enable_toggled)

        # Fixed-decimals validators would over-restrict the empty-string
        # case (validators reject ""). Use QDoubleValidator with plain
        # notation; blank inputs are handled by _parse_optional_float.
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)

        self._min_edit = QLineEdit(self)
        self._min_edit.setPlaceholderText("blank = -∞")
        self._min_edit.setValidator(validator)
        self._min_edit.setText(_format_optional_float(current_bounds.lo))

        self._max_edit = QLineEdit(self)
        self._max_edit.setPlaceholderText("blank = +∞")
        self._max_edit.setValidator(validator)
        self._max_edit.setText(_format_optional_float(current_bounds.hi))

        form = QFormLayout()
        form.addRow("Min:", self._min_edit)
        form.addRow("Max:", self._max_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)
        outer.addWidget(header)
        outer.addWidget(self._enable)
        outer.addLayout(form)
        outer.addWidget(buttons)

        self._on_enable_toggled(self._enable.isChecked())

    def _on_enable_toggled(self, enabled: bool) -> None:
        # Disable the bounds edits when the user's about to clear the
        # flag — no point letting them type bounds that will be dropped.
        self._min_edit.setEnabled(enabled)
        self._max_edit.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Outcome accessors
    # ------------------------------------------------------------------

    def enabled(self) -> bool:
        """``True`` if the user wants this attribute flagged variable."""
        return bool(self._enable.isChecked())

    def bounds(self) -> VariableBounds:
        """Bounds the user typed. Empty min / max → unbounded on that side.

        If :meth:`enabled` returns ``False`` the caller should ignore the
        bounds — they're only meaningful when the flag stays set.
        """
        lo = _parse_optional_float(self._min_edit.text())
        hi = _parse_optional_float(self._max_edit.text())
        # If the user typed both and put them in the wrong order, swap so
        # the flag stays usable. A friendlier alternative would be a
        # validation error, but silent swap matches how ``pack_bounds``
        # already tolerates lo > hi.
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return VariableBounds(lo=lo, hi=hi)


def edit_variable_bounds(
    parent: QWidget,
    *,
    title_summary: str,
    current_enabled: bool,
    current_bounds: VariableBounds,
) -> Optional[Tuple[bool, VariableBounds]]:
    """Convenience runner — pop the dialog and return ``(enabled, bounds)``.

    Returns ``None`` on Cancel. Callers use the tuple to decide whether
    to call ``project.set_variable_flag`` or ``project.clear_variable_flag``.
    """
    dlg = VariableBoundsDialog(
        title_summary=title_summary,
        current_enabled=current_enabled,
        current_bounds=current_bounds,
        parent=parent,
    )
    if dlg.exec() != QDialog.Accepted:
        return None
    return (dlg.enabled(), dlg.bounds())
