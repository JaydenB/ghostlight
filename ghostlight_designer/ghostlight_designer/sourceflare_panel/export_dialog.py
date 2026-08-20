"""Modal dialog that collects animation-export options for the source panel.

Unlike the panel's other (modeless, apply-live) dialogs this one is modal: it
gathers a motion pattern, frame count, fps, output format, optional EXR AOV
layers, width, and a destination path, then the body runs the export off the
GUI thread. **No value scrubber** is attached here (the scrubber's hidden-tree
teardown corrupts the heap when it rides on transient dialog widgets; see
feedback-scrubber-teardown). The math spinboxes are fine — they are plain
subclasses that add no child widgets and so no teardown surface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..export import writers
from ..math_spinbox import MathDoubleSpinBox, MathSpinBox
from ..settings import AppSettings
from . import motion_patterns
from .export_worker import desqueezed_width

# Dialog defaults, in one place so ExportOptions and the widgets agree. The
# width is the exporter's own default — deliberately independent of the panel's
# live render width, which is tuned for interactive speed.
DEFAULT_N_FRAMES = 24
DEFAULT_FPS = 8.0
DEFAULT_WIDTH_PX = 256


@dataclass(frozen=True)
class ExportOptions:
    """Result of the dialog — everything the body needs to build a job."""

    pattern_name: str = ""
    n_frames: int = DEFAULT_N_FRAMES
    fps: float = DEFAULT_FPS
    writer_key: str = "gif"
    exr_layers: bool = False
    width_px: int = DEFAULT_WIDTH_PX
    out_path: str = ""
    # Bake the lens's anamorphic de-squeeze into the exported frames (on by
    # default, exactly like the panel's own view toggle). A no-op for a
    # spherical lens, whose squeeze factor is 1.0.
    desqueeze: bool = True


class ExportAnimationDialog(QDialog):
    """Collects :class:`ExportOptions`. Call :meth:`exec`; on ``Accepted`` read
    :meth:`result_options`."""

    def __init__(
        self,
        app_settings: AppSettings,
        default_width: int = DEFAULT_WIDTH_PX,
        preselect_pattern: Optional[str] = None,
        squeeze_factor: float = 1.0,
        default_desqueeze: bool = True,
        dims_for_width: Optional[Callable[[int], Tuple[int, int]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_settings = app_settings
        self._out_path: str = ""
        # Lens squeeze factor (1.0 = spherical) + the body's dims resolver, used
        # to spell out the frame size the chosen width actually produces.
        sq = float(squeeze_factor)
        self._squeeze_factor: float = sq if (sq > 0.0 and sq == sq) else 1.0
        self._dims_for_width = dims_for_width
        self.setWindowTitle("Render Animation")
        self.setModal(True)

        # --- motion pattern -------------------------------------------------
        self._pattern = QComboBox(self)
        for pat in motion_patterns.PATTERNS:
            label = pat.name + ("  (loop)" if pat.loop else "")
            self._pattern.addItem(label, pat.name)
        if preselect_pattern:
            idx = self._pattern.findData(preselect_pattern)
            if idx >= 0:
                self._pattern.setCurrentIndex(idx)
        self._pattern.setToolTip(
            "Path the flare source follows across the animation. Add your own "
            "in motion_patterns.py — one decorated function per pattern."
        )

        # --- frames / fps ---------------------------------------------------
        self._frames = MathSpinBox(self)
        self._frames.setRange(1, 2000)
        self._frames.setValue(DEFAULT_N_FRAMES)
        self._frames.setToolTip("Number of frames rendered along the path.")

        self._fps = MathDoubleSpinBox(self)
        self._fps.setRange(1.0, 120.0)
        self._fps.setDecimals(2)
        self._fps.setValue(DEFAULT_FPS)
        self._fps.setSuffix(" fps")
        self._fps.setToolTip("Playback rate baked into the GIF / MOV timing.")

        # --- output format --------------------------------------------------
        self._format = QComboBox(self)
        first_available = -1
        for i, spec in enumerate(writers.WRITER_SPECS.values()):
            reason = writers.check_writer_available(spec.key)
            label = spec.label if reason is None else f"{spec.label} — unavailable"
            self._format.addItem(label, spec.key)
            if reason is not None:
                item = self._format.model().item(i)
                if item is not None:
                    item.setEnabled(False)
                    item.setToolTip(reason)
            elif first_available < 0:
                first_available = i
        if first_available >= 0:
            self._format.setCurrentIndex(first_available)
        self._format.setToolTip(
            "GIF / MOV / JPEG are display-referred (the ACES view transform + "
            "the panel's exposure are baked in). EXR is scene-linear ACEScg HDR."
        )

        self._exr_layers = QCheckBox("Include AOV layers (ghost / starburst / veil)", self)
        self._exr_layers.setToolTip(
            "EXR only: write the ghost, starburst and veil passes as separate "
            "named layers alongside the combined image."
        )

        # --- width ----------------------------------------------------------
        self._width = MathSpinBox(self)
        self._width.setRange(32, 1024)
        self._width.setSingleStep(16)
        self._width.setValue(max(32, min(1024, int(default_width))))
        self._width.setSuffix(" px")
        self._width.setToolTip(
            "Rendered frame width. Height tracks the sensor aspect ratio. "
            "Independent of the panel's own resolution."
        )

        # --- desqueeze ------------------------------------------------------
        # An anamorphic render is only the right shape once unsqueezed, so this
        # is on by default — matching the panel's own View ▸ Desqueeze toggle.
        anamorphic = self._squeeze_factor > 1.0
        self._desqueeze = QCheckBox(
            f"Desqueeze — stretch the frame ×{self._squeeze_factor:.2f} horizontally"
            if anamorphic
            else "Desqueeze (this lens is spherical — 1.00×, no effect)",
            self,
        )
        self._desqueeze.setChecked(bool(default_desqueeze))
        self._desqueeze.setEnabled(anamorphic)
        self._desqueeze.setToolTip(
            "Bake the lens's anamorphic de-squeeze into the exported frames, "
            "so the file has the same shape as the panel's de-squeezed view. "
            "Off writes the squeezed (as-shot) frame."
        )

        self._dims_label = QLabel("", self)
        self._dims_label.setStyleSheet("color: #888;")

        # --- output path ----------------------------------------------------
        self._path_edit = QLineEdit(self)
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("Choose an output file…")
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._on_browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse)

        self._caption = QLabel("", self)
        self._caption.setStyleSheet("color: #888;")
        self._caption.setWordWrap(True)

        note = QLabel(
            "Frames use the panel's current render settings and exposure. For a "
            "moving source the MDFT starburst engine (High preset) stays crisp; "
            "the Sprite engine can shimmer. The de-squeeze is applied in "
            "scene-linear light, so the EXR layers carry it too.",
            self,
        )
        note.setStyleSheet("color: #888;")
        note.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Motion pattern:", self._pattern)
        form.addRow("Frames:", self._frames)
        form.addRow("Frame rate:", self._fps)
        form.addRow("Format:", self._format)
        form.addRow("", self._exr_layers)
        form.addRow("Width:", self._width)
        form.addRow("", self._desqueeze)
        form.addRow("", self._dims_label)
        form.addRow("Output:", path_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(self._caption)
        outer.addWidget(note)
        outer.addWidget(self._buttons)

        self._format.currentIndexChanged.connect(self._on_format_changed)
        self._width.valueChanged.connect(self._update_dims_label)
        self._desqueeze.toggled.connect(self._update_dims_label)
        self._on_format_changed()
        self._update_dims_label()
        self._update_ok_enabled()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_spec(self) -> writers.WriterSpec:
        return writers.WRITER_SPECS[str(self._format.currentData())]

    def _on_format_changed(self, *_a) -> None:
        spec = self._current_spec()
        self._exr_layers.setEnabled(spec.key == "exr_seq")
        # Keep a chosen path's extension in step with the format.
        if self._out_path:
            p = Path(self._out_path)
            self._set_out_path(str(p.with_suffix(spec.suffix)))
        else:
            self._update_caption()
        self._update_ok_enabled()

    def _effective_squeeze(self) -> float:
        """The stretch this export will actually apply (1.0 when off)."""
        return self._squeeze_factor if self._desqueeze.isChecked() else 1.0

    def _update_dims_label(self, *_a) -> None:
        """Spell out the written frame size — with a de-squeeze the file is
        wider than the "Width" the user typed, which is worth saying out loud."""
        if self._dims_for_width is None:
            self._dims_label.setText("")
            return
        try:
            w, h = self._dims_for_width(int(self._width.value()))
        except Exception:
            self._dims_label.setText("")
            return
        out_w = desqueezed_width(int(w), self._effective_squeeze())
        if out_w != int(w):
            self._dims_label.setText(
                f"Output: {out_w} × {h} px  (rendered {w} × {h}, de-squeezed)"
            )
        else:
            self._dims_label.setText(f"Output: {out_w} × {h} px")

    def _on_browse(self) -> None:
        spec = self._current_spec()
        start_dir = self._app_settings.last_export_dir() or os.path.expanduser("~")
        seed_name = f"flare_{self._pattern.currentData() or 'anim'}"
        seed_name = seed_name.lower().replace(" ", "_").replace("→", "to")
        seed = os.path.join(start_dir, seed_name + spec.suffix)
        # We run our own overwrite check in _on_accept (it must cover the
        # ####-expanded sequence names too), so suppress QFileDialog's.
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export animation", seed, spec.file_filter,
            options=QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return
        # Force the format's extension (the picker may not add one).
        path = str(Path(path).with_suffix(spec.suffix))
        self._app_settings.set_last_export_dir(str(Path(path).parent))
        self._set_out_path(path)

    def _set_out_path(self, path: str) -> None:
        self._out_path = path
        self._path_edit.setText(path)
        self._update_caption()
        self._update_ok_enabled()

    def _update_caption(self) -> None:
        spec = self._current_spec()
        if not self._out_path:
            self._caption.setText("")
            return
        if spec.is_sequence:
            example = writers.frame_path(self._out_path, 1, spec.suffix).name
            base = Path(self._out_path).stem
            self._caption.setText(
                f"Writes a numbered sequence: {base}.####{spec.suffix}  "
                f"(first file {example})."
            )
        else:
            self._caption.setText(f"Writes a single file: {Path(self._out_path).name}.")

    def _update_ok_enabled(self) -> None:
        spec = writers.WRITER_SPECS.get(str(self._format.currentData()))
        ok = bool(self._out_path) and spec is not None and (
            writers.check_writer_available(spec.key) is None
        )
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def _on_accept(self) -> None:
        if not self._out_path:
            return
        spec = self._current_spec()
        if spec.is_sequence:
            target = writers.frame_path(self._out_path, 1, spec.suffix)
            exists_msg = (
                f"A sequence starting at {target.name} already exists in this "
                "folder. Overwrite these frames?"
            )
        else:
            target = Path(self._out_path)
            exists_msg = f"{target.name} already exists. Overwrite it?"
        if target.exists():
            reply = QMessageBox.question(
                self, "Overwrite?", exists_msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def result_options(self) -> ExportOptions:
        spec = self._current_spec()
        return ExportOptions(
            pattern_name=str(self._pattern.currentData() or ""),
            n_frames=int(self._frames.value()),
            fps=float(self._fps.value()),
            writer_key=spec.key,
            exr_layers=bool(self._exr_layers.isChecked() and spec.key == "exr_seq"),
            width_px=int(self._width.value()),
            out_path=self._out_path,
            desqueeze=bool(self._desqueeze.isChecked()),
        )
