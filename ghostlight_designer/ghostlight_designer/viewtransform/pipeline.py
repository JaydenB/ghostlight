"""OCIO ACES 2.0 view-transform pipeline (Qt-free core).

Everything here is pure numpy + OpenColorIO, so it can be reused verbatim by
any host's colour-management code. The Qt QImage packing lives in
:mod:`.qimage`.

``opencolorio`` is a hard dependency of ``ghostlight_designer`` (declared in
``pyproject.toml``): ACES 2.0's single-stage display-rendering transform, with
its per-pixel chroma compression and highlight desaturation, cannot be
faithfully reproduced offline, so there is no numpy fallback.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import PyOpenColorIO as ocio

_log = logging.getLogger("ghostlight_designer.viewtransform")


class ViewTransformError(RuntimeError):
    """Raised when a config / display / view cannot be resolved."""


# The renderer output plane is ACEScg. Try the canonical name, the ACES
# reference config's prefixed name, the common CG-config raw name, then the
# scene_linear role — so both the builtin ACES configs and a studio's custom
# ``$OCIO`` config resolve.
_INPUT_SPACE_CANDIDATES = ("ACEScg", "ACES - ACEScg", "lin_ap1")

# Config-key sentinels stored in AppSettings under ``view/ocio_config``:
#   ""      -> the bundled builtin ACES 2.0 studio config
#   "$OCIO" -> resolve the ``$OCIO`` environment config (match a studio's Nuke)
#   <path>  -> load an .ocio config file
_ENV_CONFIG_KEY = "$OCIO"

_lock = threading.Lock()
_config_cache: dict = {}
_processor_cache: dict = {}
_default_name_cache: Optional[str] = None


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewTransformSpec:
    """Immutable snapshot of the designer-wide display transform selection.

    Snapshotted on the GUI thread at render dispatch and handed to the worker
    thread, so a render worker never reads ``AppSettings`` / ``QSettings``
    off-thread (mirrors the existing ``scale_hint`` snapshot pattern).
    """

    config_key: str = ""
    display: str = ""
    view: str = ""


# ---------------------------------------------------------------------------
# Config resolution (cached)
# ---------------------------------------------------------------------------


def _config_version_key(name: str) -> Tuple[int, ...]:
    """Sort key for a builtin config name, newest-highest.

    e.g. ``studio-config-v4.0.0_aces-v2.0_ocio-v2.5`` -> (4, 0, 0, 2, 5).
    """
    m = re.search(
        r"-v(\d+)\.(\d+)\.(\d+)_aces-v[\d.]+_ocio-v(\d+)\.(\d+)", name
    )
    if not m:
        return (0,)
    return tuple(int(g) for g in m.groups())


def default_config_name() -> str:
    """Name of the bundled builtin config: the newest ACES 2.0 *studio* config.

    Pins the target to ACES 2.0 while still
    tracking OCIO point releases within it; falls back to the ``-latest`` alias
    if the registry can't be enumerated.
    """
    global _default_name_cache
    if _default_name_cache is not None:
        return _default_name_cache
    name = "studio-config-latest"
    try:
        names = [str(n) for n in ocio.BuiltinConfigRegistry()]
        aces2 = [
            n
            for n in names
            if n.startswith("studio-config") and "aces-v2.0" in n
        ]
        if aces2:
            name = max(aces2, key=_config_version_key)
    except Exception:  # pragma: no cover - registry always present in practice
        _log.warning("Could not enumerate builtin OCIO configs; using latest alias")
    _default_name_cache = name
    return name


def _create_config(config_key: str) -> ocio.Config:
    if config_key == _ENV_CONFIG_KEY:
        return ocio.Config.CreateFromEnv()
    if config_key:
        return ocio.Config.CreateFromFile(config_key)
    return ocio.Config.CreateFromBuiltinConfig(default_config_name())


def _get_config(config_key: str) -> ocio.Config:
    with _lock:
        cfg = _config_cache.get(config_key)
    if cfg is not None:
        return cfg
    try:
        cfg = _create_config(config_key)
    except Exception as exc:  # bad path / missing $OCIO / unreadable file
        raise ViewTransformError(
            f"Could not load OCIO config ({config_key or 'builtin'}): {exc}"
        ) from exc
    with _lock:
        _config_cache[config_key] = cfg
    return cfg


def _resolve_input_space(cfg: ocio.Config) -> str:
    for cand in _INPUT_SPACE_CANDIDATES:
        cs = cfg.getColorSpace(cand)
        if cs is not None:
            return cs.getName()
    cs = cfg.getColorSpace(ocio.ROLE_SCENE_LINEAR)
    if cs is not None:
        return cs.getName()
    raise ViewTransformError(
        "OCIO config has no ACEScg / scene_linear colour space to view from"
    )


def _validate_display_view(
    cfg: ocio.Config, display: str, view: str
) -> Tuple[str, str]:
    displays = list(cfg.getDisplays())
    if not displays:
        raise ViewTransformError("OCIO config defines no displays")
    if display not in displays:
        display = cfg.getDefaultDisplay()
    views = list(cfg.getViews(display))
    if view not in views:
        view = cfg.getDefaultView(display)
    return display, view


# ---------------------------------------------------------------------------
# Processor (cached, thread-safe)
# ---------------------------------------------------------------------------


def get_processor(spec: ViewTransformSpec) -> "ocio.CPUProcessor":
    """Return a cached OCIO CPU processor for ``spec``.

    ``applyRGB`` on the returned processor is const/thread-safe, so all panels
    and their render workers share one instance. Construction is done outside
    the lock (a rare duplicate build on first use is harmless).
    """
    key = (spec.config_key, spec.display, spec.view)
    with _lock:
        proc = _processor_cache.get(key)
    if proc is not None:
        return proc

    cfg = _get_config(spec.config_key)
    display, view = _validate_display_view(cfg, spec.display, spec.view)
    src = _resolve_input_space(cfg)
    dvt = ocio.DisplayViewTransform(src=src, display=display, view=view)
    try:
        processor = cfg.getProcessor(dvt)
        cpu = processor.getOptimizedCPUProcessor(
            ocio.BIT_DEPTH_F32, ocio.BIT_DEPTH_F32, ocio.OPTIMIZATION_DEFAULT
        )
    except Exception as exc:
        raise ViewTransformError(
            f"Could not build view transform {display!r}/{view!r}: {exc}"
        ) from exc
    with _lock:
        _processor_cache[key] = cpu
    return cpu


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_view(
    hwc: np.ndarray, stops: float, spec: ViewTransformSpec
) -> np.ndarray:
    """Scene-linear ACEScg ``(H, W, 3)`` -> display-referred ``(H, W, 3)``.

    ``stops`` is the viewer exposure (``* 2**stops`` in linear, before the view
    transform — same as Nuke's Viewer gain). Returns a display-encoded float32
    array nominally in ``[0, 1]`` (may slightly exceed on the brightest
    highlights); pack it with :func:`.qimage.to_qimage`. Negatives and HDR
    peaks are passed through the transform unclamped.
    """
    scale = np.float32(2.0 ** float(stops))
    lin = np.ascontiguousarray(np.asarray(hwc, dtype=np.float32) * scale)
    cpu = get_processor(spec)
    # applyRGB mutates in place; lin is our own copy, and reshape(-1, 3) is a
    # view into it (lin is contiguous), so the result lands back in lin.
    cpu.applyRGB(lin.reshape(-1, 3))
    return lin


# ---------------------------------------------------------------------------
# Auto-exposure meter (single canonical copy; drives the Auto-Expose action)
# ---------------------------------------------------------------------------


def compute_exposure_scale(hwc: np.ndarray) -> float:
    """Auto-exposure *linear gain* that maps an HDR render to display range.

    Meters off the 99th percentile so a few blown highlights don't drag the
    whole frame dark. That works for a normally-filled frame, but collapses on
    a *mostly-dark* one — e.g. a bare aperture starburst, whose diffraction
    floor fills the buffer with ~1e-8-of-peak values. There the 99th percentile
    lands in that floor, so ``0.9 / p99`` explodes and blows the whole feature
    into a white block. Detect that collapse and meter off the pixels carrying
    real signal instead; a normally-filled frame keeps the original metering
    exactly.
    """
    p99 = float(np.percentile(hwc, 99))
    peak = float(hwc.max())
    if peak < 1e-8:
        return 0.0
    if p99 >= peak * 1e-3:
        return (0.9 / p99) if p99 >= 1e-8 else 0.0

    lum = hwc.max(axis=-1) if hwc.ndim == 3 else hwc
    signal = lum[lum > peak * 1e-4]
    ref = float(np.percentile(signal, 90)) if signal.size else peak
    return (0.9 / ref) if ref >= 1e-8 else 0.0


def meter_auto_stops(hwc: np.ndarray) -> float:
    """Auto-exposure expressed in stops (``log2`` of the linear meter gain).

    Returns ``0.0`` for an empty / black frame (the Nuke-native default).
    """
    scale = compute_exposure_scale(hwc)
    if scale <= 0.0:
        return 0.0
    return float(np.log2(scale))


# ---------------------------------------------------------------------------
# Menu / settings helpers
# ---------------------------------------------------------------------------


def resolve_default_display_view(config_key: str = "") -> Tuple[str, str]:
    """The config's default (display, view) — used to seed the initial setting."""
    cfg = _get_config(config_key)
    display = cfg.getDefaultDisplay()
    return display, cfg.getDefaultView(display)


def available_views(config_key: str = "") -> List[Tuple[str, List[str]]]:
    """``[(display, [view, ...]), ...]`` for the active config, for the menu."""
    cfg = _get_config(config_key)
    return [(d, list(cfg.getViews(d))) for d in cfg.getDisplays()]


def spec_from_settings(settings) -> ViewTransformSpec:
    """Build a :class:`ViewTransformSpec` from AppSettings.

    Empty display/view (first run) are filled from the config's defaults. Runs
    on the GUI thread only — workers receive the returned snapshot instead.
    """
    config_key = settings.view_ocio_config() or ""
    display, view = settings.view_display_view()
    if not display or not view:
        display, view = resolve_default_display_view(config_key)
    return ViewTransformSpec(config_key=config_key, display=display, view=view)
