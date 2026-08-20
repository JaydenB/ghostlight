"""Ghostlight Python bindings.

The compiled extension module ``_ghostlight`` exposes the C++ types directly.
This package re-exports them and adds the artist-facing layer:

- :class:`OpticalSystem` — Python subclass of the bound C++ ``_OpticalSystem``
  with cached calibration / ghost data, element + pivot accessors, and a
  ``save()`` round-trip path.
- :class:`Element` — one entry from the ``optical_system`` array (N surfaces
  + N-1 materials, or a 1-surface stop / mirror / interface), reconstructed
  from the source JSON.
- :class:`Pivot` — one entry from the top-level ``pivots`` rig, with
  translation + rotation offsets and a list of exposed artist controls.

Quick start::

    import ghostlight

    system = ghostlight.OpticalSystem.load("my_lens.lens")

    # CPU ray-trace diagnostics
    ray  = ghostlight.Ray(ghostlight.Vec3f(0, 0, -100), ghostlight.Vec3f(0, 0, 1), wavelength=587.56)
    path = ghostlight.trace_primary_ray_diagnostic(ray, system)
    for ev in path.events:
        print(ev.surface_index, ev.hit_point, ev.fresnel_weight)

    # GPU render
    cfg = ghostlight.PointFlareConfig()
    cfg.source_x = 0.7
    cfg.spectral_samples = 16
    out = system.render_point_flare(1920, 1080, cfg)
    # out["ghost_r"] is a (1080, 1920) float32 numpy array
"""

def _ensure_cuda_dll_path() -> None:
    """Make the CUDA runtime libraries (notably cuFFT) loadable.

    The extension links cuFFT dynamically (``cufft64_*.dll`` on Windows).  On
    CUDA 12/13 those DLLs live in ``<toolkit>/bin/x64``, which is not on the
    default loader path, so importing the extension fails with "DLL load
    failed".  Add the toolkit library directories to the search path before the
    extension loads.  A no-op on non-Windows (loader uses RPATH / LD_LIBRARY_PATH).
    """
    import os
    import sys

    if not sys.platform.startswith("win") or not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    for var in ("CUDA_PATH",):
        root = os.environ.get(var)
        if root:
            candidates += [os.path.join(root, "bin", "x64"), os.path.join(root, "bin")]
    # Fall back to scanning the standard Windows toolkit install location,
    # newest version first, so a machine without CUDA_PATH set still resolves.
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        for ver in sorted(os.listdir(base), reverse=True):
            candidates += [os.path.join(base, ver, "bin", "x64"),
                           os.path.join(base, ver, "bin")]

    seen = set()
    for d in candidates:
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        try:
            os.add_dll_directory(d)
        except OSError:
            pass


_ensure_cuda_dll_path()

from ._ghostlight import (
    # Format constants (defined in optical_system.h — the single source of
    # truth for the lens-file version and the asphere cap)
    LENS_FORMAT_MAJOR,
    LENS_FORMAT_MINOR,
    MAX_ASPHERE_TERMS,
    # Geometry
    Vec3f,
    Ray,
    dot,
    cross,
    # Enums
    CoatingModel,
    SurfaceForm,
    DispersionModel,
    CylinderAxis,
    ApertureShape,
    GhostAovMode,
    SensorModel,
    InputColorSpace,
    OutputColorSpace,
    StarburstEngine,
    BaffleShape,
    HurbKickDistribution,
    GateLobe,
    TraceStatus,
    PSFCenterMode,
    PSFCellStatus,
    # Structures
    Coating,
    Surface,
    SurfaceList,
    StringList,
    ApertureImage,
    ApertureImageList,
    LensCalibration,
    TraceResult,
    TraceEvent,
    RayPath,
    GhostPair,
    GhostFilter,
    Baffle,
    # Config
    GateConfig,
    RenderConfig,
    FlareConfig,
    PointFlareConfig,
    PSFConfig,
    # Free functions
    calibrate_lens,
    build_spectral_lambdas,
    enumerate_ghost_pairs,
    filter_ghost_pairs,
    trace_ghost_ray,
    trace_ghost_ray_diagnostic,
    trace_primary_ray,
    trace_primary_ray_diagnostic,
    render_point_flare,
    render_source_flare,
    render_psf,
    _cuda_available,
)

from .optical_system import OpticalSystem
from .element import Element, ElementKind, Pivot, ExposedAttr, load_elements_and_pivots
from .writer import (build_optical_system_doc, lens_format_version,
                     write_optical_system)
from .pose import bake_system_poses, element_world_pivot, make_rotation
from . import _arrays
from . import source_sampling

__all__ = [
    # Geometry
    "Vec3f", "Ray", "dot", "cross",
    # Enums
    "LENS_FORMAT_MAJOR", "LENS_FORMAT_MINOR", "MAX_ASPHERE_TERMS",
    "CoatingModel", "SurfaceForm", "DispersionModel", "CylinderAxis", "ApertureShape",
    "GhostAovMode", "SensorModel", "InputColorSpace", "OutputColorSpace", "StarburstEngine",
    "BaffleShape", "HurbKickDistribution", "GateLobe", "TraceStatus", "PSFCenterMode", "PSFCellStatus",
    # Structures
    "Coating", "Surface", "SurfaceList", "StringList",
    "ApertureImage", "ApertureImageList",
    "LensCalibration",
    "TraceResult", "TraceEvent", "RayPath",
    "GhostPair", "GhostFilter", "Baffle", "GateConfig",
    # Config
    "RenderConfig", "FlareConfig",
    "PointFlareConfig",
    "PSFConfig",
    # Free functions
    "calibrate_lens", "build_spectral_lambdas",
    "enumerate_ghost_pairs", "filter_ghost_pairs",
    "trace_ghost_ray", "trace_ghost_ray_diagnostic",
    "trace_primary_ray", "trace_primary_ray_diagnostic",
    "render_point_flare", "render_source_flare", "render_psf",
    "_cuda_available",
    # Artist-facing types
    "OpticalSystem",
    "Element", "ElementKind",
    "Pivot", "ExposedAttr", "load_elements_and_pivots",
    "build_optical_system_doc", "lens_format_version",
    "write_optical_system",
    "bake_system_poses", "element_world_pivot", "make_rotation",
    # Array helpers
    "_arrays",
    # Extended-source samplers
    "source_sampling",
]
