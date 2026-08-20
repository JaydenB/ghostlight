"""Off-GUI-thread animation renderer for the source-flare exporter.

:class:`AnimationWorker` is a ``QObject`` moved onto a ``QThread`` (the
optimization-panel ``_Worker`` idiom). For each frame it moves the source
along the chosen motion pattern, renders every sample chunk through the
panel's own ``_do_render_chunk`` (injected — live and batch run identical
bytecode), accumulates the weighted mean, and hands the frame to a
:class:`~ghostlight_designer.export.writers.FrameWriter`.

Display-referred formats (GIF / MOV / JPEG) receive the ACES-view-transformed
frame; the scene-linear EXR format receives the raw ACEScg HDR (plus the
ghost / starburst / veil AOVs when layer export is on). A ``frameReady`` QImage
is emitted per frame for a live canvas preview whenever the view transform is
available.

An anamorphic de-squeeze (``job.squeeze``) is applied in scene-linear light,
before the view transform, so every format — including the EXR AOV layers —
comes out unsqueezed and the preview matches the panel's display-only stretch.

The worker never touches the GUI: it emits cross-thread signals the panel body
translates into progress-dialog / canvas / status updates. Cancellation is a
``threading.Event`` checked between chunks and between frames; on cancel the
writer's ``abort()`` deletes every partial file and the worker emits
``failed(CANCELLED)``, which the body maps to a quiet "cancelled" status rather
than an error box.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from .. import viewtransform as vt
from ..export import writers

_log = logging.getLogger("ghostlight_designer.sourceflare_panel.export_worker")

# Sentinel emitted through ``failed`` on a user cancel, distinguishing it from a
# real error so the body can show a quiet status instead of a warning box.
CANCELLED = "__export_cancelled__"


@dataclass(frozen=True)
class AnimationJob:
    """Immutable snapshot of everything one export needs, assembled on the GUI
    thread so the worker never reads live panel / project / settings state.

    Mirrors the live render-dispatch snapshot (lens + frozen calibration, ghost
    filter, pre-chunked offsets, dims, render settings, matte box, exposure
    stops, view-transform spec) plus the animation parameters (pattern, motion
    context, frame count, fps) and the output target.
    """

    lens: object
    calib: object
    ghost_filter: object
    chunks: Tuple[np.ndarray, ...]
    total_weight: float
    width: int
    height: int
    half_w: float
    half_h: float
    settings: object
    matte: object
    stops: float
    view_spec: Optional["vt.ViewTransformSpec"]
    pattern: object            # motion_patterns.MotionPattern
    ctx: object                # motion_patterns.MotionContext
    n_frames: int
    fps: float
    writer_key: str
    exr_layers: bool
    out_path: str
    # Anamorphic de-squeeze: horizontal stretch applied to every finished frame
    # (1.0 = off / spherical lens). Resolved on the GUI thread from the export
    # option + the lens's squeeze factor, so the worker never asks the lens.
    squeeze: float = 1.0


def desqueezed_width(width: int, squeeze: float) -> int:
    """Output pixel width for ``width`` stretched by ``squeeze``.

    The single definition of the exported frame width — the dialog's "Output"
    label, the writer's dimensions and the resample target all go through it,
    so they cannot drift apart. Non-finite / non-positive factors give
    ``width`` unchanged.
    """
    s = float(squeeze)
    if not (s > 0.0) or s != s:
        s = 1.0
    return max(1, int(round(int(width) * s)))


def desqueeze_frame(hwc: np.ndarray, out_w: int) -> np.ndarray:
    """Stretch an ``(H, W, …)`` frame horizontally to ``out_w`` columns.

    Linear interpolation between source pixel *centres* — the array analogue of
    the panel canvas's smooth-scaled ``drawImage``. Values (not energy) are
    interpolated, so a de-squeezed frame keeps the same peak brightness and the
    same exposure as the panel preview. A no-op when the widths already match.
    """
    arr = np.asarray(hwc)
    src_w = int(arr.shape[1])
    out_w = int(out_w)
    if out_w == src_w or src_w <= 0 or out_w <= 0:
        return arr
    if src_w == 1:
        return np.repeat(arr, out_w, axis=1)
    # Output centre -> source centre, then blend the two straddling columns.
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    x = np.clip(x, 0.0, src_w - 1.0)
    i0 = np.floor(x).astype(np.intp)
    i1 = np.minimum(i0 + 1, src_w - 1)
    frac = (x - i0).astype(np.float32)
    # Broadcast the per-column weights over whatever trailing axes exist.
    shape = [1] * arr.ndim
    shape[1] = out_w
    w1 = frac.reshape(shape)
    return (arr[:, i0, ...] * (1.0 - w1) + arr[:, i1, ...] * w1).astype(
        arr.dtype, copy=False
    )


# render_chunk(lens, calib, offsets, sx, sy, w, h, half_w, half_h, settings,
#              matte, *, ghost_filter=None, with_layers=False)
RenderChunk = Callable[..., Tuple]


class AnimationWorker(QObject):
    """Renders an :class:`AnimationJob` frame-by-frame off the GUI thread."""

    # done_units, total_units, label — units are rendered chunks (fine-grained
    # so the progress bar advances within a frame, not just between frames).
    progress = Signal(int, int, str)
    frameReady = Signal(object)      # QImage | None, for the live canvas preview
    finished = Signal(str)           # human-readable summary
    failed = Signal(str)             # error message, or CANCELLED

    def __init__(
        self,
        job: AnimationJob,
        render_chunk: RenderChunk,
        cancel: Event,
        make_writer_fn: Callable[..., "writers.FrameWriter"] = writers.make_writer,
    ) -> None:
        super().__init__()
        self._job = job
        self._render_chunk = render_chunk
        self._cancel = cancel
        self._make_writer = make_writer_fn

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Thread entry point. Emits exactly one of finished / failed."""
        job = self._job
        from .motion_patterns import sample_times

        spec = writers.WRITER_SPECS.get(job.writer_key)
        if spec is None:
            self.failed.emit(f"Unknown export format {job.writer_key!r}.")
            return
        if spec.needs_display and job.view_spec is None:
            self.failed.emit(
                "The display (view) transform is unavailable, so a display-"
                "referred format cannot be written. Export EXR instead."
            )
            return

        times = sample_times(int(job.n_frames), bool(job.pattern.loop))
        n_chunks = max(1, len(job.chunks))
        total_units = len(times) * n_chunks
        done_units = 0

        # Frames leave the renderer at job.width and are written de-squeezed,
        # so the writer must be told the stretched width (it is what ffmpeg's
        # rawvideo input is sized by).
        out_w = desqueezed_width(int(job.width), float(job.squeeze))

        writer: Optional["writers.FrameWriter"] = None
        try:
            writer = self._make_writer(
                job.writer_key, job.out_path, float(job.fps),
                out_w, int(job.height), exr_layers=bool(job.exr_layers),
            )
        except Exception as exc:
            _log.exception("Export: writer construction failed")
            self.failed.emit(str(exc))
            return

        try:
            for frame_idx, t in enumerate(times):
                if self._cancel.is_set():
                    return self._do_cancel(writer)
                sx, sy = job.pattern.fn(float(t), job.ctx)

                acc: Optional[np.ndarray] = None
                acc_layers: Dict[str, np.ndarray] = {}
                for chunk in job.chunks:
                    if self._cancel.is_set():
                        return self._do_cancel(writer)
                    hwc, layers = self._render_one(sx, sy, chunk, spec)
                    acc = hwc if acc is None else acc + hwc
                    if layers:
                        for name, arr in layers.items():
                            acc_layers[name] = (
                                arr if name not in acc_layers
                                else acc_layers[name] + arr
                            )
                    done_units += 1
                    self.progress.emit(
                        done_units, total_units,
                        f"Rendering frame {frame_idx + 1}/{len(times)}…",
                    )

                weight = float(job.total_weight) if job.total_weight > 0.0 else 1.0
                norm = writers.sanitize_hdr(acc / weight)
                norm_layers = {k: v / weight for k, v in acc_layers.items()}

                # De-squeeze in scene-linear light, before the view transform,
                # so the EXR (and its AOV layers) carry the same geometry as
                # the display-referred formats and the preview.
                if out_w != int(job.width):
                    norm = desqueeze_frame(norm, out_w)
                    norm_layers = {
                        k: desqueeze_frame(v, out_w) for k, v in norm_layers.items()
                    }

                self._write_frame(writer, spec, norm, norm_layers)

            writer.finish()
        except Exception as exc:
            _log.exception("Export: render / write failed")
            try:
                writer.abort()
            except Exception:
                _log.exception("Export: abort after failure also failed")
            self.failed.emit(str(exc))
            return

        self.finished.emit(
            f"Exported {len(times)} frames → {job.out_path}"
        )

    # ------------------------------------------------------------------

    def _render_one(
        self, sx: float, sy: float, chunk: np.ndarray, spec,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Render one sample chunk, returning ``(hwc, layers|None)``.

        ``layers`` (raw ghost / starburst / veil components) is only requested
        for the EXR layer-export path — everything else takes the cheaper
        two-value ``_do_render_chunk`` return."""
        job = self._job
        want_layers = (not spec.needs_display) and bool(job.exr_layers)
        if want_layers:
            hwc, _flare, layers = self._render_chunk(
                job.lens, job.calib, chunk, sx, sy, job.width, job.height,
                job.half_w, job.half_h, job.settings, job.matte,
                ghost_filter=job.ghost_filter, with_layers=True,
            )
            return hwc, layers
        hwc, _flare = self._render_chunk(
            job.lens, job.calib, chunk, sx, sy, job.width, job.height,
            job.half_w, job.half_h, job.settings, job.matte,
            ghost_filter=job.ghost_filter,
        )
        return hwc, None

    def _write_frame(self, writer, spec, norm, norm_layers) -> None:
        """Hand one accumulated frame to the writer and emit a preview.

        Display formats write the view-transformed frame; EXR writes the raw
        linear frame (+ layers). A preview QImage is emitted whenever the view
        transform is available, independent of the output format."""
        job = self._job
        disp = None
        if job.view_spec is not None:
            try:
                disp = vt.apply_view(norm, float(job.stops), job.view_spec)
            except Exception:
                _log.exception("Export: view transform failed on a frame")
                disp = None

        if spec.needs_display:
            if disp is None:
                raise writers.ExportError(
                    "The view transform failed, so this frame cannot be "
                    "written to a display-referred format."
                )
            writer.add_frame(disp)
        else:
            writer.add_frame(norm, layers=norm_layers)

        if disp is not None:
            try:
                self.frameReady.emit(vt.to_qimage(disp))
            except Exception:
                _log.exception("Export: preview QImage build failed")

    def _do_cancel(self, writer) -> None:
        try:
            writer.abort()
        except Exception:
            _log.exception("Export: abort on cancel failed")
        self.failed.emit(CANCELLED)
