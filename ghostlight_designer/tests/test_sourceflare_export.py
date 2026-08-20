"""Tests for the source-flare animation export — the worker's frame loop
(driven synchronously with a fake renderer + stub writer, no GPU), the panel's
export guards, and the modal options dialog.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from ghostlight_designer.project import Project
from ghostlight_designer.sourceflare_panel import export_worker, motion_patterns
from ghostlight_designer.sourceflare_panel.body import SourceFlarePanelBody
from ghostlight_designer.sourceflare_panel.export_dialog import ExportAnimationDialog

from _corpus import EXAMPLE_DOUBLET


# ---------------------------------------------------------------------------
# Harness (mirrors test_sourceflare_panel.py — scrubber teardown must flush
# DeferredDelete + gc so bodies don't corrupt the heap in a later test)
# ---------------------------------------------------------------------------

def _make_body(qapp, isolated_settings):
    project = Project()
    body = SourceFlarePanelBody(project, isolated_settings)
    return project, body


def _destroy(widget) -> None:
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()


def _example_lens_path():
    return EXAMPLE_DOUBLET


class _FakeWriter:
    """Records add_frame / finish / abort without touching disk."""

    def __init__(self):
        self.frames = []
        self.finished_called = False
        self.aborted = False

    def add_frame(self, frame, layers=None):
        self.frames.append((np.asarray(frame).copy(), layers))

    def finish(self):
        self.finished_called = True

    def abort(self):
        self.aborted = True


def _make_job(pattern_name, n_frames, *, writer_key="exr_seq", exr_layers=False,
              n_chunks=2, squeeze=1.0):
    offs = np.array([[0.0, 0.0, 0.5]] * n_chunks, np.float32)
    chunks = tuple(offs[i:i + 1] for i in range(n_chunks))
    return export_worker.AnimationJob(
        lens=None, calib=None, ghost_filter=None, chunks=chunks,
        total_weight=float(offs[:, 2].sum()), width=8, height=6,
        half_w=12.0, half_h=9.0, settings=None, matte=None, stops=0.0,
        view_spec=None, pattern=motion_patterns.get_pattern(pattern_name),
        ctx=motion_patterns.MotionContext(start_sx=0.5, start_sy=0.5, aspect=1.0),
        n_frames=n_frames, fps=24.0, writer_key=writer_key,
        exr_layers=exr_layers, out_path="unused.exr", squeeze=squeeze,
    )


# ---------------------------------------------------------------------------
# Worker frame loop (synchronous — direct signal connections, no event loop)
# ---------------------------------------------------------------------------

def test_worker_renders_one_frame_per_sample_time(qapp):
    positions = []

    def fake_rc(lens, calib, offsets, sx, sy, w, h, hw, hh, settings, matte,
                *, ghost_filter=None, with_layers=False):
        positions.append((sx, sy))
        hwc = np.full((h, w, 3), 0.1, np.float32)
        return (hwc, hwc, {"ghost": hwc}) if with_layers else (hwc, hwc)

    fw = _FakeWriter()
    job = _make_job("Sweep Left → Right", 4)
    worker = export_worker.AnimationWorker(
        job, fake_rc, threading.Event(), make_writer_fn=lambda *a, **k: fw,
    )
    finished, failed, progress = [], [], []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.progress.connect(lambda d, t, l: progress.append((d, t)))
    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert fw.finished_called and not fw.aborted
    assert len(fw.frames) == 4

    # Every chunk of every frame renders at that frame's pattern position.
    times = motion_patterns.sample_times(4, loop=False)
    expected = [job.pattern.fn(t, job.ctx) for t in times]
    assert positions == [p for p in expected for _ in range(len(job.chunks))]

    # One progress tick per chunk; final tick is (total, total).
    assert len(progress) == 4 * len(job.chunks)
    assert progress[-1] == (8, 8)


def test_worker_loop_pattern_omits_duplicate_endpoint(qapp):
    positions = []

    def fake_rc(lens, calib, offsets, sx, sy, *a, ghost_filter=None,
                with_layers=False):
        positions.append((sx, sy))
        hwc = np.full((6, 8, 3), 0.1, np.float32)
        return (hwc, hwc)

    job = _make_job("Orbit", 4, n_chunks=1)
    worker = export_worker.AnimationWorker(
        job, fake_rc, threading.Event(), make_writer_fn=lambda *a, **k: _FakeWriter(),
    )
    worker.run()
    times = motion_patterns.sample_times(4, loop=True)
    assert positions == [job.pattern.fn(t, job.ctx) for t in times]


def test_worker_cancel_after_first_frame_aborts(qapp):
    cancel = threading.Event()
    calls = {"n": 0}

    def fake_rc(lens, calib, offsets, sx, sy, w, h, *a, ghost_filter=None,
                with_layers=False):
        calls["n"] += 1
        if calls["n"] == 2:            # 2 chunks/frame -> end of frame 0
            cancel.set()
        hwc = np.full((h, w, 3), 0.1, np.float32)
        return (hwc, hwc)

    fw = _FakeWriter()
    job = _make_job("Sweep Diagonal", 4, n_chunks=2)
    worker = export_worker.AnimationWorker(
        job, fake_rc, cancel, make_writer_fn=lambda *a, **k: fw,
    )
    finished, failed = [], []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()

    assert finished == []
    assert failed == [export_worker.CANCELLED]
    assert fw.aborted and not fw.finished_called
    assert len(fw.frames) == 1        # only frame 0 was written


def test_worker_requests_layers_only_for_exr_aov(qapp):
    seen = {"with_layers": []}

    def fake_rc(lens, calib, offsets, sx, sy, w, h, hw, hh, settings, matte,
                *, ghost_filter=None, with_layers=False):
        seen["with_layers"].append(with_layers)
        hwc = np.full((h, w, 3), 0.1, np.float32)
        return (hwc, hwc, {"ghost": hwc}) if with_layers else (hwc, hwc)

    job = _make_job("Sine Drift", 2, writer_key="exr_seq", exr_layers=True,
                    n_chunks=1)
    fw = _FakeWriter()
    worker = export_worker.AnimationWorker(
        job, fake_rc, threading.Event(), make_writer_fn=lambda *a, **k: fw,
    )
    worker.run()
    assert all(seen["with_layers"])                  # every chunk asked for layers
    assert all(layers is not None for _f, layers in fw.frames)


# ---------------------------------------------------------------------------
# Anamorphic de-squeeze
# ---------------------------------------------------------------------------

def test_desqueezed_width_rounds_and_guards():
    assert export_worker.desqueezed_width(192, 2.0) == 384
    assert export_worker.desqueezed_width(192, 1.0) == 192
    assert export_worker.desqueezed_width(101, 1.33) == 134     # rounds
    # Nonsense factors fall back to "no stretch" rather than a zero-width frame.
    assert export_worker.desqueezed_width(192, 0.0) == 192
    assert export_worker.desqueezed_width(192, float("nan")) == 192


def test_desqueeze_frame_geometry_and_brightness():
    src = np.zeros((2, 4, 3), np.float32)
    src[:, :, 0] = np.array([0.0, 1.0, 2.0, 3.0], np.float32)

    same = export_worker.desqueeze_frame(src, 4)
    assert same.shape == src.shape and np.array_equal(same, src)

    out = export_worker.desqueeze_frame(src, 8)
    assert out.shape == (2, 8, 3) and out.dtype == np.float32
    # Edges keep the source's end values (clamped), the ramp stays monotonic,
    # and nothing exceeds the source range — no interpolation overshoot.
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[0, -1, 0] == pytest.approx(3.0)
    assert np.all(np.diff(out[0, :, 0]) > 0)
    assert out.min() >= 0.0 and out.max() <= 3.0

    # A flat field is unchanged in value: the stretch must not rescale energy,
    # or the export would land at a different exposure than the panel.
    flat = np.full((3, 5, 3), 0.37, np.float32)
    assert np.allclose(export_worker.desqueeze_frame(flat, 11), 0.37)


def test_worker_desqueezes_frames_and_layers(qapp):
    """squeeze=2 -> frames (and EXR AOV layers) are written twice as wide, and
    the writer is constructed with the stretched width."""
    def fake_rc(lens, calib, offsets, sx, sy, w, h, hw, hh, settings, matte,
                *, ghost_filter=None, with_layers=False):
        hwc = np.full((h, w, 3), 0.25, np.float32)
        return (hwc, hwc, {"ghost": hwc}) if with_layers else (hwc, hwc)

    fw = _FakeWriter()
    made = {}

    def make_writer(key, path, fps, width, height, *, exr_layers=False):
        made.update(width=width, height=height)
        return fw

    job = _make_job("Orbit", 2, writer_key="exr_seq", exr_layers=True,
                    n_chunks=1, squeeze=2.0)
    worker = export_worker.AnimationWorker(
        job, fake_rc, threading.Event(), make_writer_fn=make_writer,
    )
    failed = []
    worker.failed.connect(failed.append)
    worker.run()

    assert failed == []
    assert made == {"width": 16, "height": 6}       # rendered 8 wide
    assert fw.frames and all(f.shape == (6, 16, 3) for f, _l in fw.frames)
    assert all(l["ghost"].shape == (6, 16, 3) for _f, l in fw.frames)
    # Value-preserving: a flat frame keeps the level it had before the stretch
    # (0.25 rendered / 0.5 sample weight), so exposure is unchanged.
    assert np.allclose(fw.frames[0][0], 0.5)


def test_worker_leaves_frames_untouched_without_desqueeze(qapp):
    def fake_rc(lens, calib, offsets, sx, sy, w, h, *a, ghost_filter=None,
                with_layers=False):
        hwc = np.full((h, w, 3), 0.25, np.float32)
        return (hwc, hwc)

    fw = _FakeWriter()
    made = {}

    def make_writer(key, path, fps, width, height, *, exr_layers=False):
        made.update(width=width, height=height)
        return fw

    job = _make_job("Orbit", 2, n_chunks=1, squeeze=1.0)
    worker = export_worker.AnimationWorker(
        job, fake_rc, threading.Event(), make_writer_fn=make_writer,
    )
    worker.run()
    assert made == {"width": 8, "height": 6}
    assert all(f.shape == (6, 8, 3) for f, _l in fw.frames)


def test_worker_reports_unknown_format(qapp):
    job = _make_job("Orbit", 2)
    job = export_worker.AnimationJob(**{**job.__dict__, "writer_key": "bogus"})
    failed = []
    worker = export_worker.AnimationWorker(
        job, lambda *a, **k: None, threading.Event(),
        make_writer_fn=lambda *a, **k: _FakeWriter(),
    )
    worker.failed.connect(failed.append)
    worker.run()
    assert failed and "bogus" in failed[0]


# ---------------------------------------------------------------------------
# Panel guards
# ---------------------------------------------------------------------------

def test_maybe_launch_noops_while_exporting(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    try:
        lens_path = _example_lens_path()
        if not lens_path.exists():
            pytest.skip("sample lens not present")
        project.load(str(lens_path))
        body._is_active = True
        body._exporting = True
        with body._lock:
            body._pending = True
            body._busy = False
        body._maybe_launch()
        # Guard fired: no live worker launched (would have set _busy True).
        assert body._busy is False
        assert body._pending is True
    finally:
        _destroy(body)


def test_request_render_defers_while_exporting(qapp, isolated_settings):
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._exporting = True
        body._dirty_pending = False
        body.request_render()
        assert body._dirty_pending is True
    finally:
        _destroy(body)


def test_export_animation_ignored_when_ineligible(qapp, isolated_settings):
    """No lens -> refuse quietly (no dialog, no crash)."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body.export_animation("Orbit")
        assert body._exporting is False
    finally:
        _destroy(body)


def test_export_animation_noop_when_already_exporting(qapp, isolated_settings,
                                                      monkeypatch):
    import ghostlight_designer.sourceflare_panel.body as bmod

    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._exporting = True

        def _boom(*_a, **_k):
            raise AssertionError("dialog must not open while exporting")

        monkeypatch.setattr(bmod, "ExportAnimationDialog", _boom)
        body.export_animation("Orbit")  # returns immediately, no dialog
    finally:
        _destroy(body)


def _load_body_with_lens(qapp, isolated_settings):
    project, body = _make_body(qapp, isolated_settings)
    lens_path = _example_lens_path()
    if not lens_path.exists():
        _destroy(body)
        pytest.skip("sample lens not present")
    project.load(str(lens_path))
    return project, body


def test_begin_export_puts_squeeze_on_the_job(qapp, isolated_settings, monkeypatch,
                                              tmp_path):
    """The option + the lens's squeeze factor resolve to job.squeeze on the GUI
    thread; the pattern context keeps the RENDERED aspect."""
    from ghostlight_designer.sourceflare_panel.export_dialog import ExportOptions

    _project, body = _load_body_with_lens(qapp, isolated_settings)
    try:
        # Stand in for an anamorphic lens: pin the factor and stop the
        # recompute from clearing it.
        monkeypatch.setattr(body, "_refresh_squeeze_factor", lambda: None)
        body._squeeze_factor = 2.0

        jobs = []
        monkeypatch.setattr(body, "_begin_export_when_idle", jobs.append)

        body._export_options = ExportOptions(
            pattern_name="Orbit", n_frames=2, fps=24.0, writer_key="exr_seq",
            width_px=64, out_path=str(tmp_path / "a.exr"), desqueeze=True,
        )
        body._begin_export()
        assert len(jobs) == 1
        assert jobs[0].squeeze == pytest.approx(2.0)
        assert jobs[0].width == 64
        assert jobs[0].ctx.aspect == pytest.approx(64.0 / jobs[0].height)

        body._exporting = False
        body._export_options = replace(body._export_options, desqueeze=False)
        body._begin_export()
        assert jobs[1].squeeze == pytest.approx(1.0)
    finally:
        body._exporting = False
        _destroy(body)


def test_export_preview_squeeze_avoids_double_stretch(qapp, isolated_settings):
    """Baked-in de-squeeze -> the canvas must not stretch the preview again;
    finishing restores the panel's own setting."""
    _project, body = _make_body(qapp, isolated_settings)
    try:
        body._desqueeze = True
        body._squeeze_factor = 2.0
        body._push_squeeze_to_canvas()
        assert body._canvas._squeeze == pytest.approx(2.0)

        body._apply_export_preview_squeeze(_make_job("Orbit", 1, squeeze=2.0))
        assert body._canvas._squeeze == pytest.approx(1.0)

        body._finish_export("done", ok=True)
        assert body._canvas._squeeze == pytest.approx(2.0)

        # A squeezed (option off) export leaves the panel's stretch alone.
        body._apply_export_preview_squeeze(_make_job("Orbit", 1, squeeze=1.0))
        assert body._canvas._squeeze == pytest.approx(2.0)
    finally:
        _destroy(body)


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

def test_dialog_prefills_width(qapp, isolated_settings):
    dlg = ExportAnimationDialog(isolated_settings, default_width=320)
    try:
        assert dlg._width.value() == 320
    finally:
        _destroy(dlg)


def test_dialog_opens_on_the_export_defaults(qapp, isolated_settings):
    """24 frames @ 8 fps, 256 px — the exporter's own defaults, independent of
    the panel's live render width."""
    dlg = ExportAnimationDialog(isolated_settings)
    try:
        assert dlg._frames.value() == 24
        assert dlg._fps.value() == pytest.approx(8.0)
        assert dlg._width.value() == 256
        opts = dlg.result_options()
        assert (opts.n_frames, opts.fps, opts.width_px) == (24, 8.0, 256)
    finally:
        _destroy(dlg)


def test_dialog_disables_unavailable_formats(qapp, isolated_settings, monkeypatch):
    from ghostlight_designer.export import writers as w

    def fake_avail(key):
        return "ffmpeg missing" if key.startswith("mov") else None

    monkeypatch.setattr(w, "check_writer_available", fake_avail)
    dlg = ExportAnimationDialog(isolated_settings, default_width=192)
    try:
        for i in range(dlg._format.count()):
            key = str(dlg._format.itemData(i))
            enabled = dlg._format.model().item(i).isEnabled()
            assert enabled is (not key.startswith("mov")), key
        # The initial selection lands on an available (non-mov) format.
        assert not str(dlg._format.currentData()).startswith("mov")
    finally:
        _destroy(dlg)


def test_dialog_result_options_roundtrip(qapp, isolated_settings, tmp_path):
    dlg = ExportAnimationDialog(
        isolated_settings, default_width=192, preselect_pattern="Orbit",
    )
    try:
        dlg._frames.setValue(30)
        dlg._fps.setValue(12.0)
        dlg._format.setCurrentIndex(dlg._format.findData("exr_seq"))
        dlg._exr_layers.setChecked(True)
        dlg._width.setValue(256)
        dlg._set_out_path(str(tmp_path / "shot.exr"))

        opts = dlg.result_options()
        assert opts.pattern_name == "Orbit"
        assert opts.n_frames == 30
        assert opts.fps == pytest.approx(12.0)
        assert opts.writer_key == "exr_seq"
        assert opts.exr_layers is True
        assert opts.width_px == 256
        assert opts.out_path.endswith("shot.exr")
        assert opts.desqueeze is True          # default-on
    finally:
        _destroy(dlg)


def test_dialog_desqueeze_defaults_on_and_reports_output_size(qapp,
                                                              isolated_settings):
    dlg = ExportAnimationDialog(
        isolated_settings, default_width=192, squeeze_factor=2.0,
        dims_for_width=lambda w: (w, w // 2),
    )
    try:
        assert dlg._desqueeze.isChecked() is True
        assert dlg._desqueeze.isEnabled() is True
        assert "2.00" in dlg._desqueeze.text()
        assert dlg.result_options().desqueeze is True
        assert dlg._dims_label.text().startswith("Output: 384 × 96 px")

        dlg._desqueeze.setChecked(False)
        assert dlg.result_options().desqueeze is False
        assert dlg._dims_label.text() == "Output: 192 × 96 px"
    finally:
        _destroy(dlg)


def test_dialog_desqueeze_disabled_for_spherical_lens(qapp, isolated_settings):
    dlg = ExportAnimationDialog(
        isolated_settings, default_width=192, squeeze_factor=1.0,
        dims_for_width=lambda w: (w, w // 2),
    )
    try:
        assert dlg._desqueeze.isEnabled() is False
        # Checked but inert — the width is unchanged either way.
        assert dlg._dims_label.text() == "Output: 192 × 96 px"
    finally:
        _destroy(dlg)


def test_dialog_honours_panel_desqueeze_off(qapp, isolated_settings):
    dlg = ExportAnimationDialog(
        isolated_settings, default_width=192, squeeze_factor=2.0,
        default_desqueeze=False,
    )
    try:
        assert dlg._desqueeze.isChecked() is False
        assert dlg.result_options().desqueeze is False
    finally:
        _destroy(dlg)


def test_dialog_aov_forced_off_for_non_exr(qapp, isolated_settings, tmp_path):
    dlg = ExportAnimationDialog(isolated_settings, default_width=192)
    try:
        # Select EXR, tick AOV, then switch to GIF: AOV is disabled + excluded.
        dlg._format.setCurrentIndex(dlg._format.findData("exr_seq"))
        dlg._exr_layers.setChecked(True)
        dlg._format.setCurrentIndex(dlg._format.findData("gif"))
        dlg._set_out_path(str(tmp_path / "a.gif"))
        assert dlg._exr_layers.isEnabled() is False
        assert dlg.result_options().exr_layers is False
    finally:
        _destroy(dlg)


def test_dialog_extension_follows_format(qapp, isolated_settings, tmp_path):
    dlg = ExportAnimationDialog(isolated_settings, default_width=192)
    try:
        dlg._format.setCurrentIndex(dlg._format.findData("gif"))
        dlg._set_out_path(str(tmp_path / "clip.gif"))
        dlg._format.setCurrentIndex(dlg._format.findData("jpeg_seq"))
        assert dlg._out_path.endswith(".jpg")
    finally:
        _destroy(dlg)
