"""Frame writers for animation export — GIF, MOV (ProRes / H.264), JPEG and
EXR sequences. Qt-free and free of any panel imports.

Two families of format:

* **Display-referred** (GIF / MOV / JPEG) consume already-view-transformed
  float frames in ``[0, 1]`` (``needs_display`` is True) and quantize them to
  8-bit. The caller applies the designer's ACES view transform + viewer
  exposure before handing frames over, so what's written matches the panel.
* **Scene-linear** (EXR) consumes the raw ACEScg HDR float (``needs_display``
  is False) and stores it as ``float16`` — the same buffer a Nuke session
  would receive. Optionally splits the ghost / starburst / veil / gate AOVs into
  named layers.

Heavy / optional dependencies (Pillow, OpenEXR, the ffmpeg binary) are
imported lazily inside each writer so importing this module never fails;
:func:`check_writer_available` reports a human-readable reason when a format
can't be used, and the dialog disables it rather than erroring at write time.

Every writer supports :meth:`FrameWriter.abort`, which closes handles and
deletes every file it created — so a cancelled export leaves no partial
output (including a half-muxed .mov).
"""
from __future__ import annotations

import collections
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Half-float (IEEE binary16) maximum finite magnitude. HDR peaks are clamped to
# this so an over-bright pixel survives the float16 cast as a bright finite
# value instead of overflowing to +inf.
_HALF_MAX = 65504.0


class ExportError(RuntimeError):
    """A frame could not be written (bad data, encoder failure, I/O error)."""


class ExportDependencyError(ExportError):
    """A required optional dependency (Pillow / OpenEXR / ffmpeg) is missing."""


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def sanitize_hdr(hwc: np.ndarray) -> np.ndarray:
    """Return a finite float32 copy of an HDR frame.

    ``NaN`` -> 0, ``+inf`` -> the float16 max (so it survives an EXR cast),
    ``-inf`` -> 0. Applied before the view transform (display formats) and
    before the float16 cast (EXR) so no non-finite value reaches an encoder.
    """
    arr = np.asarray(hwc, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=_HALF_MAX, neginf=0.0)


def quantize_display(disp: np.ndarray) -> np.ndarray:
    """Display-referred float ``(H, W, 3)`` in ``[0, 1]`` -> C-contiguous uint8.

    Rounds to the nearest 8-bit code (``+ 0.5``) rather than truncating,
    matching :func:`ghostlight_designer.viewtransform.qimage.to_qimage` and a
    compositor's display quantization.
    """
    arr = np.asarray(disp, dtype=np.float32)
    return np.ascontiguousarray((arr * 255.0 + 0.5).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterSpec:
    """Static description of an output format, for the dialog + factory.

    ``needs_display`` — frames must be view-transformed (display-referred)
    before writing. ``is_sequence`` — one file per frame (``name.####.ext``)
    rather than a single container. ``suffix`` includes the dot.
    """

    key: str
    label: str
    needs_display: bool
    is_sequence: bool
    suffix: str
    file_filter: str


# Insertion order == dialog order. The first available entry is the default.
WRITER_SPECS: Dict[str, WriterSpec] = {
    spec.key: spec
    for spec in (
        WriterSpec("gif", "Animated GIF (.gif)", True, False, ".gif",
                   "Animated GIF (*.gif)"),
        WriterSpec("mov_prores", "QuickTime — ProRes 4444 (.mov)", True, False,
                   ".mov", "QuickTime movie (*.mov)"),
        WriterSpec("mov_h264", "QuickTime — H.264 (.mov)", True, False, ".mov",
                   "QuickTime movie (*.mov)"),
        WriterSpec("jpeg_seq", "JPEG sequence (.jpg)", True, True, ".jpg",
                   "JPEG image (*.jpg)"),
        WriterSpec("exr_seq", "OpenEXR sequence (.exr, scene-linear)", False, True,
                   ".exr", "OpenEXR image (*.exr)"),
    )
}


def check_writer_available(key: str) -> Optional[str]:
    """Return ``None`` if ``key`` can be written now, else a human-readable
    reason (missing Pillow / OpenEXR / ffmpeg). Never raises."""
    spec = WRITER_SPECS.get(key)
    if spec is None:
        return f"Unknown export format {key!r}."
    if key in ("gif", "jpeg_seq"):
        try:
            import PIL  # noqa: F401
        except Exception:
            return ("Pillow is required for GIF / JPEG export "
                    "(pip install Pillow).")
        return None
    if key == "exr_seq":
        try:
            import OpenEXR  # noqa: F401
        except Exception:
            return ("The OpenEXR Python module is required for EXR export "
                    "(pip install OpenEXR).")
        return None
    if key in ("mov_prores", "mov_h264"):
        if shutil.which("ffmpeg") is None:
            return ("ffmpeg was not found on PATH — required for .mov export. "
                    "Install ffmpeg or export a GIF / JPEG / EXR sequence.")
        return None
    return f"Unknown export format {key!r}."


def frame_path(out_path: str, index: int, suffix: str) -> Path:
    """Per-frame path for a sequence: ``<dir>/<stem>.<index:04d><suffix>``.

    ``index`` is 1-based. The stem is taken from ``out_path`` (its own
    extension, if any, is dropped) so ``anim.exr`` frame 1 -> ``anim.0001.exr``.
    """
    p = Path(out_path)
    return p.parent / f"{p.stem}.{int(index):04d}{suffix}"


# ---------------------------------------------------------------------------
# Writer base + implementations
# ---------------------------------------------------------------------------


class FrameWriter:
    """Base class. Subclasses append frames, then :meth:`finish` (success) or
    :meth:`abort` (cancel — deletes all output)."""

    def add_frame(self, frame: np.ndarray,
                  layers: Optional[Dict[str, np.ndarray]] = None) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        """Flush / close. Safe to call once after the last frame."""

    def abort(self) -> None:
        """Close handles and delete every file this writer created."""


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


class GifWriter(FrameWriter):
    """Animated GIF. Buffers frames (PIL needs them all at ``save`` time) and
    writes the whole file on :meth:`finish`."""

    def __init__(self, out_path: str, fps: float) -> None:
        self._out = Path(out_path)
        self._fps = max(1.0, float(fps))
        self._frames: List = []

    def add_frame(self, frame, layers=None) -> None:
        from PIL import Image

        rgb8 = quantize_display(frame)
        self._frames.append(Image.fromarray(rgb8, mode="RGB"))

    def finish(self) -> None:
        if not self._frames:
            raise ExportError("GIF export produced no frames.")
        duration_ms = int(round(1000.0 / self._fps))
        first, rest = self._frames[0], self._frames[1:]
        first.save(
            str(self._out),
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,            # infinite
            disposal=2,        # restore to background between frames
        )

    def abort(self) -> None:
        self._frames = []
        if self._out.exists():
            _unlink_quietly(self._out)


class JpegSequenceWriter(FrameWriter):
    """One JPEG per frame, written immediately (``name.0001.jpg`` …)."""

    def __init__(self, out_path: str, suffix: str) -> None:
        self._out = out_path
        self._suffix = suffix
        self._written: List[Path] = []

    def add_frame(self, frame, layers=None) -> None:
        from PIL import Image

        rgb8 = quantize_display(frame)
        path = frame_path(self._out, len(self._written) + 1, self._suffix)
        Image.fromarray(rgb8, mode="RGB").save(
            str(path), quality=95, subsampling=0,
        )
        self._written.append(path)

    def abort(self) -> None:
        for path in self._written:
            _unlink_quietly(path)
        self._written = []


class ExrSequenceWriter(FrameWriter):
    """One scene-linear (ACEScg) ``float16`` EXR per frame.

    Always writes the combined image as the default ``R`` / ``G`` / ``B``
    channels. With ``exr_layers`` on, each supplied AOV (``ghost`` /
    ``starburst`` / ``veil`` / ``gate``) is written as its own named layer;
    absent AOVs are simply omitted.
    """

    def __init__(self, out_path: str, suffix: str, exr_layers: bool) -> None:
        self._out = out_path
        self._suffix = suffix
        self._layers = bool(exr_layers)
        self._written: List[Path] = []

    @staticmethod
    def _to_half(hwc: np.ndarray) -> np.ndarray:
        # Sanitize (NaN/inf -> finite) then clamp magnitude to the half-float
        # range so a stray HDR spike can't become +inf. This is the single
        # choke point for every EXR channel, so the AOV layers are covered too,
        # not just the combined frame. Negatives (out-of-gamut ACEScg) survive.
        arr = sanitize_hdr(hwc)
        return np.ascontiguousarray(
            np.clip(arr, -_HALF_MAX, _HALF_MAX).astype(np.float16)
        )

    def add_frame(self, frame, layers=None) -> None:
        import OpenEXR

        channels = {"RGB": OpenEXR.Channel("RGB", self._to_half(frame))}
        if self._layers and layers:
            for name in ("ghost", "starburst", "veil", "gate"):
                arr = layers.get(name)
                if arr is not None:
                    channels[name] = OpenEXR.Channel(name, self._to_half(arr))
        header = {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "type": OpenEXR.scanlineimage,
        }
        # Best-effort provenance note; never let an unsupported attribute abort
        # the write (the pixels are what matter).
        try:
            header["comments"] = "scene-linear ACEScg (AP1) — Ghostlight export"
        except Exception:
            pass
        path = frame_path(self._out, len(self._written) + 1, self._suffix)
        try:
            OpenEXR.File(header, channels).write(str(path))
        except Exception as exc:
            raise ExportError(f"EXR write failed: {exc}") from exc
        self._written.append(path)

    def abort(self) -> None:
        for path in self._written:
            _unlink_quietly(path)
        self._written = []


# ffmpeg codec argument groups, keyed by writer.
_MOV_CODECS = {
    "mov_prores": [
        "-c:v", "prores_ks", "-profile:v", "3",
        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
    ],
    "mov_h264": [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ],
}


class FfmpegMovWriter(FrameWriter):
    """Streams rgb24 frames into ffmpeg's stdin and muxes a .mov.

    The output dimensions are padded up to even (``pad=ceil(iw/2)*2:…``) because
    ProRes / yuv420p require even sides and the sensor-derived height is often
    odd. stderr is drained on a daemon thread into a bounded ring buffer so the
    pipe can't deadlock; a nonzero exit raises :class:`ExportError` with the
    stderr tail.
    """

    def __init__(self, key: str, out_path: str, fps: float, width: int,
                 height: int) -> None:
        self._out = Path(out_path)
        self._codec = _MOV_CODECS[key]
        self._fps = max(1.0, float(fps))
        self._w = int(width)
        self._h = int(height)
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_tail: "collections.deque" = collections.deque(maxlen=200)
        self._stderr_thread: Optional[threading.Thread] = None
        self._start()

    def _start(self) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self._w}x{self._h}", "-r", f"{self._fps}",
            "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            *self._codec,
            str(self._out),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise ExportDependencyError(
                "ffmpeg was not found on PATH — required for .mov export."
            ) from exc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, b""):
            self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())

    def add_frame(self, frame, layers=None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ExportError("ffmpeg process is not running.")
        rgb8 = quantize_display(frame)
        try:
            self._proc.stdin.write(rgb8.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise ExportError(
                "ffmpeg closed the pipe early:\n" + self._stderr_text()
            ) from exc

    def finish(self) -> None:
        proc = self._proc
        if proc is None:
            raise ExportError("ffmpeg process is not running.")
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
        self._proc = None
        if rc != 0:
            raise ExportError(
                f"ffmpeg exited with code {rc}:\n" + self._stderr_text()
            )

    def abort(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._out.exists():
            _unlink_quietly(self._out)

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail)


def make_writer(key: str, out_path: str, fps: float, width: int, height: int,
                *, exr_layers: bool = False) -> FrameWriter:
    """Construct the :class:`FrameWriter` for ``key``.

    Raises :class:`ExportDependencyError` if the format's dependency is missing
    (mirrors :func:`check_writer_available`, which the dialog uses to disable
    the option up front).
    """
    spec = WRITER_SPECS.get(key)
    if spec is None:
        raise ExportError(f"Unknown export format {key!r}.")
    reason = check_writer_available(key)
    if reason is not None:
        raise ExportDependencyError(reason)
    if key == "gif":
        return GifWriter(out_path, fps)
    if key == "jpeg_seq":
        return JpegSequenceWriter(out_path, spec.suffix)
    if key == "exr_seq":
        return ExrSequenceWriter(out_path, spec.suffix, exr_layers)
    if key in ("mov_prores", "mov_h264"):
        return FfmpegMovWriter(key, out_path, fps, width, height)
    raise ExportError(f"Unknown export format {key!r}.")
