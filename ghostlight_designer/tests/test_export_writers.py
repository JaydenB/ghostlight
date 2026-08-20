"""Unit tests for the Qt-free frame writers (GIF / JPEG / EXR / MOV)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from ghostlight_designer.export import writers

H, W, N = 6, 8, 4


def _disp_frame(i: int) -> np.ndarray:
    base = np.linspace(0.0, 1.0, H * W, dtype=np.float32).reshape(H, W)
    return np.stack([base, base * (i / N), base * 0.5], axis=-1)


def _lin_frame(i: int) -> np.ndarray:
    return _disp_frame(i) * 12.0


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def test_sanitize_hdr_replaces_non_finite():
    f = np.zeros((2, 2, 3), np.float32)
    f[0, 0, 0] = np.nan
    f[0, 1, 0] = np.inf
    f[1, 0, 0] = -np.inf
    out = writers.sanitize_hdr(f)
    assert np.isfinite(out).all()
    assert out[0, 0, 0] == 0.0
    assert out[0, 1, 0] == pytest.approx(65504.0)
    assert out[1, 0, 0] == 0.0


def test_quantize_display_rounds_and_clips():
    disp = np.array([[[0.0, 1.0, 2.0]]], np.float32)  # 2.0 clips to 255
    q = writers.quantize_display(disp)
    assert q.dtype == np.uint8
    assert list(q[0, 0]) == [0, 255, 255]


def test_frame_path_pads_index():
    p = writers.frame_path(r"C:\x\anim.exr", 7, ".exr")
    assert p.name == "anim.0007.exr"


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

def test_gif_writes_animated_file(tmp_path):
    from PIL import Image

    out = tmp_path / "anim.gif"
    w = writers.make_writer("gif", str(out), 12.0, W, H)
    for i in range(N):
        w.add_frame(writers.sanitize_hdr(_disp_frame(i)))
    w.finish()
    assert out.exists()
    img = Image.open(str(out))
    assert getattr(img, "is_animated", False) is True
    assert img.n_frames == N


# ---------------------------------------------------------------------------
# JPEG sequence
# ---------------------------------------------------------------------------

def test_jpeg_sequence_names_and_count(tmp_path):
    out = tmp_path / "seq.jpg"
    w = writers.make_writer("jpeg_seq", str(out), 12.0, W, H)
    for i in range(N):
        w.add_frame(writers.sanitize_hdr(_disp_frame(i)))
    w.finish()
    files = sorted(tmp_path.glob("seq.*.jpg"))
    assert [p.name for p in files] == [
        "seq.0001.jpg", "seq.0002.jpg", "seq.0003.jpg", "seq.0004.jpg",
    ]
    assert all(p.stat().st_size > 0 for p in files)


def test_jpeg_abort_leaves_no_files(tmp_path):
    out = tmp_path / "ab.jpg"
    w = writers.make_writer("jpeg_seq", str(out), 12.0, W, H)
    w.add_frame(writers.sanitize_hdr(_disp_frame(0)))
    w.add_frame(writers.sanitize_hdr(_disp_frame(1)))
    w.abort()
    assert list(tmp_path.glob("ab.*.jpg")) == []


# ---------------------------------------------------------------------------
# EXR sequence
# ---------------------------------------------------------------------------

def test_exr_combined_only_channels(tmp_path):
    import OpenEXR

    out = tmp_path / "hdr.exr"
    w = writers.make_writer("exr_seq", str(out), 12.0, W, H, exr_layers=False)
    w.add_frame(writers.sanitize_hdr(_lin_frame(0)))
    w.finish()
    files = sorted(tmp_path.glob("hdr.*.exr"))
    assert len(files) == 1
    part = OpenEXR.File(str(files[0])).parts[0]
    assert sorted(part.channels.keys()) == ["RGB"]


def test_exr_layers_written_and_absent_omitted(tmp_path):
    import OpenEXR

    out = tmp_path / "hdr.exr"
    w = writers.make_writer("exr_seq", str(out), 12.0, W, H, exr_layers=True)
    frame = writers.sanitize_hdr(_lin_frame(0))
    # veil deliberately absent -> its channel must be omitted.
    w.add_frame(frame, layers={"ghost": frame * 0.4, "starburst": frame * 0.2})
    w.finish()
    part = OpenEXR.File(str(sorted(tmp_path.glob("hdr.*.exr"))[0])).parts[0]
    keys = sorted(part.channels.keys())
    assert keys == ["RGB", "ghost", "starburst"]
    assert "veil" not in keys


def test_exr_roundtrip_within_half_precision(tmp_path):
    """EXR is float16, so compare ppm-of-peak, never byte-exact."""
    import OpenEXR

    out = tmp_path / "hdr.exr"
    frame = writers.sanitize_hdr(_lin_frame(2))
    w = writers.make_writer("exr_seq", str(out), 12.0, W, H)
    w.add_frame(frame)
    w.finish()
    part = OpenEXR.File(str(sorted(tmp_path.glob("hdr.*.exr"))[0])).parts[0]
    back = np.asarray(part.channels["RGB"].pixels, dtype=np.float32)
    peak = float(frame.max())
    assert np.abs(back - frame).max() <= peak * 1e-3


def test_exr_sanitizes_non_finite(tmp_path):
    import OpenEXR

    out = tmp_path / "hdr.exr"
    frame = _lin_frame(1)
    frame[0, 0, 0] = np.nan
    frame[0, 1, 1] = np.inf
    w = writers.make_writer("exr_seq", str(out), 12.0, W, H)
    w.add_frame(writers.sanitize_hdr(frame))
    w.finish()
    part = OpenEXR.File(str(sorted(tmp_path.glob("hdr.*.exr"))[0])).parts[0]
    assert np.isfinite(np.asarray(part.channels["RGB"].pixels)).all()


def test_exr_layers_are_sanitized(tmp_path):
    """A NaN/inf in an AOV layer (not just the combined frame) must not leak
    into the EXR — the writer sanitizes every channel."""
    import OpenEXR

    out = tmp_path / "hdr.exr"
    frame = writers.sanitize_hdr(_lin_frame(0))
    bad = frame.copy()
    bad[0, 0, 0] = np.nan
    bad[1, 1, 2] = np.inf
    w = writers.make_writer("exr_seq", str(out), 12.0, W, H, exr_layers=True)
    w.add_frame(frame, layers={"ghost": bad})   # combined clean, layer dirty
    w.finish()
    part = OpenEXR.File(str(sorted(tmp_path.glob("hdr.*.exr"))[0])).parts[0]
    assert np.isfinite(np.asarray(part.channels["ghost"].pixels)).all()


# ---------------------------------------------------------------------------
# MOV (needs ffmpeg)
# ---------------------------------------------------------------------------

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("key", ["mov_prores", "mov_h264"])
def test_mov_writes_quicktime_container(tmp_path, key):
    out = tmp_path / "movie.mov"
    w = writers.make_writer(key, str(out), 12.0, W, H)
    for i in range(N):
        w.add_frame(writers.sanitize_hdr(_disp_frame(i)))
    w.finish()
    assert out.exists() and out.stat().st_size > 0
    head = out.read_bytes()[:64]
    # QuickTime / ISO-BMFF files carry an 'ftyp' box near the start.
    assert b"ftyp" in head


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_mov_abort_leaves_no_file(tmp_path):
    out = tmp_path / "abmov.mov"
    w = writers.make_writer("mov_h264", str(out), 12.0, W, H)
    w.add_frame(writers.sanitize_hdr(_disp_frame(0)))
    w.abort()
    assert not out.exists()


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_mov_handles_odd_dimensions(tmp_path):
    """Sensor-derived height is often odd; the pad filter must even it out."""
    out = tmp_path / "odd.mov"
    w = writers.make_writer("mov_h264", str(out), 12.0, 7, 5)  # both odd
    for i in range(N):
        frame = np.full((5, 7, 3), 0.3, np.float32)
        w.add_frame(frame)
    w.finish()
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_check_writer_available_unknown_key():
    assert "Unknown" in writers.check_writer_available("nope")


def test_check_writer_available_mov_needs_ffmpeg(monkeypatch):
    monkeypatch.setattr(writers.shutil, "which", lambda _name: None)
    msg = writers.check_writer_available("mov_prores")
    assert msg is not None and "ffmpeg" in msg


def test_make_writer_raises_on_missing_dependency(monkeypatch):
    monkeypatch.setattr(writers.shutil, "which", lambda _name: None)
    with pytest.raises(writers.ExportDependencyError):
        writers.make_writer("mov_h264", "x.mov", 12.0, W, H)
