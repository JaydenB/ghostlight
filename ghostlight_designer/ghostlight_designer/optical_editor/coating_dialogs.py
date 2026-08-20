"""Modal editors + cell pickers for the coating row.

The Model combo on a coating row picks a coating tier; the richer models
(spectral / angular / spectral-angular tables, TMM layer stacks, Gaussian
attenuators) are edited through the modal dialogs here, launched from the
row's "Data" picker cell. The "Tint" cell uses ``QColorDialog`` and the
"Preset" cell browses the coating catalogue.

Each picker returns a **string** (the delegate coerces picker returns via
``str()``): the tint picker returns ``"#RRGGBB"``; the data + preset pickers
return a compact JSON payload that the matching ``row_schemas`` writer
decodes and applies via the ``OpticalSystem.set_coating_*`` accessors —
inside the model's ``project.edit(...)`` transaction, so the edit is
undoable and round-trips through the writer.
"""
from __future__ import annotations

import csv
import json
from typing import Any, Optional

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..coating_catalogue import get_coating_catalogue
from ..math_spinbox import MathDoubleSpinBox


# Thin-film materials offered by the layer-stack editor, with dispersion-free
# n,k across the visible (λ in μm). Enough to author plausible AR stacks
# without pulling a full optical-constants database into the designer.
_THIN_FILM_MATERIALS: dict[str, list[list[float]]] = {
    "MgF2":  [[0.40, 1.384, 0.0], [0.55, 1.380, 0.0], [0.70, 1.377, 0.0]],
    "SiO2":  [[0.40, 1.470, 0.0], [0.55, 1.460, 0.0], [0.70, 1.455, 0.0]],
    "Al2O3": [[0.40, 1.680, 0.0], [0.55, 1.660, 0.0], [0.70, 1.650, 0.0]],
    "ZrO2":  [[0.40, 2.150, 0.0], [0.55, 2.100, 0.0], [0.70, 2.060, 0.0]],
    "Ta2O5": [[0.40, 2.160, 0.0], [0.55, 2.100, 0.0], [0.70, 2.070, 0.0]],
    "TiO2":  [[0.40, 2.550, 0.0], [0.55, 2.400, 0.0], [0.70, 2.320, 0.0]],
}


# ---------------------------------------------------------------------------
# index → (system, surface_index) helper
# ---------------------------------------------------------------------------

def _resolve(index: QModelIndex):
    """Return (system, surface_index) for a coating-row cell, or (None, -1)."""
    model = index.model()
    if model is None or not hasattr(model, "project"):
        return None, -1
    node = index.internalPointer()
    si = getattr(node, "surface_index", -1)
    system = model.project.system
    if system is None or not (0 <= si < len(system.surfaces)):
        return None, -1
    return system, si


# ---------------------------------------------------------------------------
# 1-D table dialog (spectral / angular)
# ---------------------------------------------------------------------------

class SpectralTableDialog(QDialog):
    """Editor for a 1-D reflectance table (spectral or angular).

    ``kind`` is ``"spectral"`` (key = wavelength nm) or ``"angular"``
    (key = incidence angle deg, plus an angle-reference-IOR field).
    """

    def __init__(self, kind: str, rows: list[tuple[float, float]],
                 out_of_range_discard: bool, angle_ref_ior: float,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._kind = kind
        key_label = "λ (nm)" if kind == "spectral" else "Angle (°)"
        self.setWindowTitle(
            "Spectral coating" if kind == "spectral" else "Angular coating")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Reflectance table (linear interpolation between rows):"))

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels([key_label, "R (0–1)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self._table)
        for k, r in rows:
            self._append_row(k, r)
        if not rows:
            self._append_row(400.0 if kind == "spectral" else 0.0, 0.02)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add row")
        add_btn.clicked.connect(lambda: self._append_row(0.0, 0.0))
        del_btn = QPushButton("Remove row")
        del_btn.clicked.connect(self._remove_selected)
        csv_btn = QPushButton("Import CSV…")
        csv_btn.clicked.connect(self._import_csv)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        btn_row.addWidget(csv_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        form = QFormLayout()
        self._discard = QCheckBox("Discard rays outside the table range")
        self._discard.setChecked(bool(out_of_range_discard))
        form.addRow(self._discard)
        self._angle_ref: Optional[QDoubleSpinBox] = None
        if kind == "angular":
            self._angle_ref = MathDoubleSpinBox()
            self._angle_ref.setRange(0.1, 5.0)
            self._angle_ref.setDecimals(4)
            self._angle_ref.setSingleStep(0.01)
            self._angle_ref.setValue(float(angle_ref_ior))
            form.addRow("Angle reference IOR", self._angle_ref)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(360, 380)

    def _append_row(self, key: float, r: float) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(f"{float(key):g}"))
        self._table.setItem(row, 1, QTableWidgetItem(f"{float(r):g}"))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV (key, r per row)", "", "CSV files (*.csv);;All (*)")
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                rows = [
                    (float(r[0]), float(r[1]))
                    for r in csv.reader(fh)
                    if len(r) >= 2 and _is_number(r[0]) and _is_number(r[1])
                ]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "CSV import failed", str(exc))
            return
        if not rows:
            QMessageBox.warning(self, "CSV import", "No numeric rows found.")
            return
        self._table.setRowCount(0)
        for k, r in rows:
            self._append_row(k, r)

    def result_payload(self) -> Optional[dict]:
        rows: list[list[float]] = []
        for i in range(self._table.rowCount()):
            k_item = self._table.item(i, 0)
            r_item = self._table.item(i, 1)
            if k_item is None or r_item is None:
                continue
            try:
                k = float(k_item.text())
                r = max(0.0, min(1.0, float(r_item.text())))
            except ValueError:
                continue
            rows.append([k, r])
        if not rows:
            return None
        rows.sort(key=lambda kr: kr[0])
        payload: dict[str, Any] = {
            "kind": self._kind,
            "data": rows,
            "out_of_range_discard": bool(self._discard.isChecked()),
        }
        if self._angle_ref is not None:
            payload["angle_ref_ior"] = float(self._angle_ref.value())
        return payload


# ---------------------------------------------------------------------------
# 2-D table dialog (spectral × angular)
# ---------------------------------------------------------------------------

class SATableDialog(QDialog):
    """Editor for a spectral×angular reflectance grid.

    Rows are wavelengths, columns are angles. The first column holds the
    wavelength values; the header holds the angle values (editable).
    """

    def __init__(self, wavelengths: list[float], angles: list[float],
                 r: list[list[float]], angle_ref_ior: float,
                 out_of_range_discard: bool,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectral×Angular coating")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Rows = wavelength (nm), columns = angle (°). "
            "Cell = reflectance (0–1)."))

        if not wavelengths:
            wavelengths = [400.0, 550.0, 700.0]
        if not angles:
            angles = [0.0, 30.0, 60.0]
        if not r or len(r) != len(wavelengths):
            r = [[0.02 for _ in angles] for _ in wavelengths]

        self._table = QTableWidget(len(wavelengths), len(angles) + 1, self)
        self._sync_headers(angles)
        for i, wl in enumerate(wavelengths):
            self._table.setItem(i, 0, QTableWidgetItem(f"{wl:g}"))
            for j in range(len(angles)):
                val = r[i][j] if j < len(r[i]) else 0.02
                self._table.setItem(i, j + 1, QTableWidgetItem(f"{val:g}"))
        layout.addWidget(self._table)

        btns = QHBoxLayout()
        for text, fn in (
            ("Import CSV…", self._import_csv),
        ):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        layout.addLayout(btns)

        form = QFormLayout()
        self._angle_ref = MathDoubleSpinBox()
        self._angle_ref.setRange(0.1, 5.0)
        self._angle_ref.setDecimals(4)
        self._angle_ref.setValue(float(angle_ref_ior))
        form.addRow("Angle reference IOR", self._angle_ref)
        self._discard = QCheckBox("Discard rays outside the table range")
        self._discard.setChecked(bool(out_of_range_discard))
        form.addRow(self._discard)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(480, 360)

    def _sync_headers(self, angles: list[float]) -> None:
        headers = ["λ (nm)"] + [f"{a:g}°" for a in angles]
        self._table.setHorizontalHeaderLabels(headers)

    def _import_csv(self) -> None:
        """CSV: first row = angles (blank first cell), each subsequent row =
        wavelength then reflectances."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import spectral×angular CSV", "", "CSV files (*.csv);;All (*)")
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                grid = [row for row in csv.reader(fh) if row]
            angles = [float(x) for x in grid[0][1:] if _is_number(x)]
            wl, r = [], []
            for row in grid[1:]:
                if not _is_number(row[0]):
                    continue
                wl.append(float(row[0]))
                r.append([float(x) for x in row[1:1 + len(angles)]])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "CSV import failed", str(exc))
            return
        if not wl or not angles:
            QMessageBox.warning(self, "CSV import", "Could not parse a grid.")
            return
        self._table.setRowCount(len(wl))
        self._table.setColumnCount(len(angles) + 1)
        self._sync_headers(angles)
        for i, w in enumerate(wl):
            self._table.setItem(i, 0, QTableWidgetItem(f"{w:g}"))
            for j in range(len(angles)):
                val = r[i][j] if j < len(r[i]) else 0.0
                self._table.setItem(i, j + 1, QTableWidgetItem(f"{val:g}"))

    def result_payload(self) -> Optional[dict]:
        n_ang = self._table.columnCount() - 1
        angles: list[float] = []
        for j in range(n_ang):
            header = self._table.horizontalHeaderItem(j + 1)
            txt = header.text().rstrip("°") if header else ""
            try:
                angles.append(float(txt))
            except ValueError:
                angles.append(float(j))
        wl: list[float] = []
        r: list[list[float]] = []
        for i in range(self._table.rowCount()):
            k_item = self._table.item(i, 0)
            if k_item is None or not _is_number(k_item.text()):
                continue
            wl.append(float(k_item.text()))
            row_r = []
            for j in range(n_ang):
                cell = self._table.item(i, j + 1)
                try:
                    row_r.append(max(0.0, min(1.0, float(cell.text()))))
                except (ValueError, AttributeError):
                    row_r.append(0.0)
            r.append(row_r)
        if not wl or not angles:
            return None
        # Sort both axes ascending (the accessor requires it).
        ang_order = sorted(range(len(angles)), key=lambda j: angles[j])
        wl_order = sorted(range(len(wl)), key=lambda i: wl[i])
        angles = [angles[j] for j in ang_order]
        wl_sorted = [wl[i] for i in wl_order]
        r_sorted = [[r[i][j] for j in ang_order] for i in wl_order]
        return {
            "kind": "spectral_angular",
            "wavelengths": wl_sorted,
            "angles": angles,
            "r": r_sorted,
            "angle_ref_ior": float(self._angle_ref.value()),
            "out_of_range_discard": bool(self._discard.isChecked()),
        }


# ---------------------------------------------------------------------------
# Layer-stack dialog (TMM)
# ---------------------------------------------------------------------------

class LayerStackDialog(QDialog):
    """Ordered thin-film layer stack editor (outer medium → substrate)."""

    def __init__(self, layers: list[dict], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Layer stack (TMM)")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Layers, outer medium first (substrate is the lens glass):"))

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Material", "Thickness (nm)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self._table)
        # Remember the (possibly custom) nk table per row by material key.
        self._nk_by_material = dict(_THIN_FILM_MATERIALS)
        for ly in layers:
            mat = str(ly.get("material", "MgF2"))
            nk = ly.get("nk_table")
            if nk is not None:
                self._nk_by_material[mat] = [list(map(float, e)) for e in nk]
            self._append_layer(mat, float(ly.get("thickness_nm", 100.0)))
        if not layers:
            self._append_layer("MgF2", 99.6)

        btns = QHBoxLayout()
        add = QPushButton("Add layer")
        add.clicked.connect(lambda: self._append_layer("MgF2", 100.0))
        rem = QPushButton("Remove layer")
        rem.clicked.connect(self._remove_selected)
        btns.addWidget(add)
        btns.addWidget(rem)
        btns.addStretch(1)
        layout.addLayout(btns)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(360, 320)

    def _append_layer(self, material: str, thickness_nm: float) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        combo = QComboBox()
        materials = list(self._nk_by_material.keys())
        combo.addItems(materials)
        if material in materials:
            combo.setCurrentText(material)
        elif material:
            combo.addItem(material)
            combo.setCurrentText(material)
        self._table.setCellWidget(row, 0, combo)
        self._table.setItem(row, 1, QTableWidgetItem(f"{thickness_nm:g}"))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def result_payload(self) -> Optional[dict]:
        layers: list[dict] = []
        for i in range(self._table.rowCount()):
            combo = self._table.cellWidget(i, 0)
            t_item = self._table.item(i, 1)
            if combo is None or t_item is None:
                continue
            try:
                thickness = float(t_item.text())
            except ValueError:
                continue
            if thickness <= 0.0:
                continue
            mat = combo.currentText()
            nk = self._nk_by_material.get(mat, _THIN_FILM_MATERIALS["MgF2"])
            layers.append({
                "material": mat,
                "thickness_nm": thickness,
                "nk_table": [list(map(float, e)) for e in nk],
            })
        if not layers:
            return None
        return {"kind": "layers", "layers": layers}


# ---------------------------------------------------------------------------
# Attenuator dialog
# ---------------------------------------------------------------------------

class AttenuatorDialog(QDialog):
    """Editor for a Gaussian positional attenuator coating."""

    def __init__(self, sigma: float, background: float, peak: float,
                 decenter_x: float, decenter_y: float,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gaussian attenuator")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        def _spin(val, lo, hi, step=0.1, dec=3):
            s = MathDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(float(val))
            return s

        self._sigma = _spin(sigma, 0.001, 1e4, 1.0)
        self._bg = _spin(background, 0.0, 1.0)
        self._peak = _spin(peak, 0.0, 1.0)
        self._dx = _spin(decenter_x, -1e4, 1e4, 1.0)
        self._dy = _spin(decenter_y, -1e4, 1e4, 1.0)
        form.addRow("Sigma (mm)", self._sigma)
        form.addRow("Background transmission", self._bg)
        form.addRow("Peak transmission (added)", self._peak)
        form.addRow("Decenter X (mm)", self._dx)
        form.addRow("Decenter Y (mm)", self._dy)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_payload(self) -> dict:
        return {
            "kind": "attenuator",
            "sigma": float(self._sigma.value()),
            "background": float(self._bg.value()),
            "peak": float(self._peak.value()),
            "decenter_x": float(self._dx.value()),
            "decenter_y": float(self._dy.value()),
        }


# ---------------------------------------------------------------------------
# Pickers registered on the delegate
# ---------------------------------------------------------------------------

def open_tint_picker(parent, index: QModelIndex) -> Optional[str]:
    """QColorDialog → ``"#RRGGBB"`` (or None on cancel)."""
    system, si = _resolve(index)
    initial = QColor(255, 255, 255)
    if system is not None:
        c = system.surfaces[si].coating
        initial = QColor(
            max(0, min(255, int(round(float(c.tint_r) * 255)))),
            max(0, min(255, int(round(float(c.tint_g) * 255)))),
            max(0, min(255, int(round(float(c.tint_b) * 255)))),
        )
    chosen = QColorDialog.getColor(initial, parent, "Ghost tint")
    if not chosen.isValid():
        return None
    return f"#{chosen.red():02X}{chosen.green():02X}{chosen.blue():02X}"


def open_coating_data_dialog(parent, index: QModelIndex) -> Optional[str]:
    """Dispatch to the right modal editor by the surface's current model,
    prefilled with its data. Returns the edited payload as a JSON string."""
    import ghostlight

    system, si = _resolve(index)
    if system is None:
        return None
    surf = system.surfaces[si]
    model = int(surf.coating.model)

    dlg: Optional[QDialog] = None
    if model == int(ghostlight.CoatingModel.SPECTRAL):
        table = system.get_coating_table(si)
        rows = [(float(k), float(r)) for k, r in table]
        dlg = SpectralTableDialog("spectral", rows,
                                  bool(surf.coating.out_of_range_discard),
                                  1.0, parent)
    elif model == int(ghostlight.CoatingModel.ANGULAR):
        table = system.get_coating_table(si)
        rows = [(float(k), float(r)) for k, r in table]
        dlg = SpectralTableDialog("angular", rows,
                                  bool(surf.coating.out_of_range_discard),
                                  float(surf.coating.angle_ref_ior), parent)
    elif model == int(ghostlight.CoatingModel.SPECTRAL_ANGULAR):
        layers = system.get_coating_layers(si)
        if layers:
            dlg = LayerStackDialog(
                [{"material": ly["material"],
                  "thickness_nm": ly["thickness_nm"],
                  "nk_table": ly["nk_table"].tolist()}
                 for ly in layers], parent)
        else:
            wl, ang, r = system.get_coating_sa_table(si)
            dlg = SATableDialog(
                [float(x) for x in wl], [float(x) for x in ang],
                [[float(v) for v in row] for row in r],
                float(surf.coating.angle_ref_ior),
                bool(surf.coating.out_of_range_discard), parent)
    elif model == int(ghostlight.CoatingModel.ATTENUATOR_GAUSS):
        c = surf.coating
        dlg = AttenuatorDialog(
            float(c.gauss_sigma), float(c.gauss_background),
            float(c.gauss_peak), float(c.gauss_decenter_x),
            float(c.gauss_decenter_y), parent)

    if dlg is None:
        return None
    if dlg.exec() != QDialog.Accepted:
        return None
    payload = dlg.result_payload()
    if not payload:
        return None
    return json.dumps(payload)


def open_coating_preset_picker(parent, index: QModelIndex) -> Optional[str]:
    """Browse the coating catalogue; return the chosen preset's payload JSON."""
    cat = get_coating_catalogue()
    presets = cat.all()
    if not presets:
        QMessageBox.information(parent, "No coating presets",
                               "No coating catalogue is bundled.")
        return None
    labels = [f"{p.display_name}  —  {p.description}" for p in presets]
    chosen, ok = QInputDialog.getItem(
        parent, "Apply coating preset", "Preset:", labels, 0, editable=False)
    if not ok:
        return None
    idx = labels.index(chosen)
    return json.dumps(presets[idx].payload)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False
