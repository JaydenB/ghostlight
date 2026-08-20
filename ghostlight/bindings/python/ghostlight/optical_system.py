"""High-level OpticalSystem subclass with cached LensCalibration + ghost pairs.

The compiled extension module ``_ghostlight`` exposes the bound C++ class as
``_OpticalSystem``.  :class:`OpticalSystem` here subclasses it and folds in
the caching layer, the convenient render-method wrappers, and the
element/pivot dataclasses that get reconstructed from the JSON file at load
time.

Usage::

    import ghostlight

    system = ghostlight.OpticalSystem.load("my_lens.lens")
    cfg    = ghostlight.PointFlareConfig()
    out    = system.render_point_flare(1920, 1080, cfg)

    for el in system.elements:
        print(el.name, el.position)
    for piv in system.pivots:
        print(piv.name, piv.offset_position)
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import TYPE_CHECKING, Optional

from ._ghostlight import (
    ApertureShape,
    _OpticalSystem,
    LensCalibration,
    FlareConfig,
    PointFlareConfig,
    PSFConfig,
    calibrate_lens,
    enumerate_ghost_pairs,
    filter_ghost_pairs,
    render_point_flare as _render_point_flare,
    render_source_flare as _render_source_flare,
    render_psf as _render_psf,
)

if TYPE_CHECKING:
    from .element import Element, Pivot


class OpticalSystem(_OpticalSystem):
    """Bound C++ optical system + Python-side caching, element grouping, and pivots.

    The C++ side holds the flat ``surfaces`` array (and parallel
    ``surface_ids`` / ``aperture_images``).  The Python side adds:

    - cached :class:`LensCalibration` (rebuilt on geometry mutation)
    - cached ghost pair list + filtered ghost results
    - reconstructed element groupings (``elements``) and pivot rig (``pivots``),
      re-parsed from the original JSON file at load time
    - render convenience methods (``render_point_flare``, etc.)

    The cache invalidates on any mutation to ``surfaces[i]``'s geometry —
    radius, thickness, ior, abbe_v, semi_aperture, is_stop, coating layers,
    surface form, dispersion, aperture shape, **and** the pivot-baked
    ``z``/``decenter_x``/``decenter_y``/``rot`` (those become user-editable
    once pivots are baked, so they must contribute to the cache key).
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache_key: Optional[bytes] = None
        self._calib: Optional[LensCalibration] = None
        self._calib_args: Optional[tuple] = None
        self._ghost_pairs: Optional[list] = None
        self._filtered: dict = {}
        # Element / pivot grouping reconstructed from the JSON file. None
        # until load() populates them; an empty list means "loaded but no
        # entries" (e.g. programmatic construction with no file).
        self._elements: Optional[list] = None
        self._pivots: Optional[list] = None
        # JSON blocks retained outside the flattened C++ representation.
        self._source_path: Optional[str] = None
        self._raw_metadata: dict = {}
        self._raw_glass_catalogue: dict = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "OpticalSystem":
        """Load a ghostlight-optical JSON lens file.

        The C++ loader fully populates ``surfaces`` (and bakes any pivots in
        the file).  We then re-parse the JSON on the Python side to recover
        the element groupings and pivot definitions for editing / display.
        """
        sys = cls()
        super(OpticalSystem, sys).load(path)
        sys._source_path = os.fspath(path)
        sys._reparse_json_layer(path)
        return sys

    def reload(self, path: Optional[str] = None) -> None:
        """Reload from disk.  Used by editors after writing pivot edits."""
        target = path if path is not None else self._source_path
        if target is None:
            raise ValueError("OpticalSystem.reload(): no source path known")
        super().load(target)
        self._source_path = os.fspath(target)
        self._reparse_json_layer(target)
        self._invalidate()

    def _reparse_json_layer(self, path: str) -> None:
        """Re-read the JSON to populate elements/pivots + stash raw blocks."""
        from .element import load_elements_and_pivots, _parse_lens_json

        raw = _parse_lens_json(path)
        self._raw_metadata        = raw.get("metadata") or {}
        self._raw_glass_catalogue = raw.get("glass_catalogue") or {}
        elements, pivots = load_elements_and_pivots(path)
        self._elements = elements
        self._pivots   = pivots

    def finalize(self) -> None:
        """Recompute Surface.z so the chain ends at z=0 (sensor).

        Call after programmatically building or mutating the surfaces list.

        Element positions are synchronized because the loader derives
        inter-element air gaps from them.
        """
        super().finalize()
        self._sync_element_positions()
        self._invalidate()

    def _sync_element_positions(self) -> None:
        """Re-derive each element's ``position.z`` from the live surface chain.

        Walks elements in order, keeping ``elements[0].position.z`` as the
        user-authored anchor (the loader rebases the chain to sensor=0
        anyway, so the absolute value is informational) and setting each
        subsequent element's ``position.z`` to
        ``previous.position.z + sum(previous.surface_thicknesses)``.

        That formula is the inverse of the loader's inter-element gap
        patch (``next.position.z - this_last_surf_nominal``): the patch
        reproduces exactly the in-memory thicknesses, so per-surface
        thickness edits survive save / reload.

        ``x`` and ``y`` are preserved — those carry authored decenter
        that isn't derivable from on-axis surface vertices.
        """
        if self._elements is None:
            return
        ids = list(self.surface_ids)
        lookup = {uuid: i for i, uuid in enumerate(ids)}
        prev_pos_z: Optional[float] = None
        prev_thickness_sum: float = 0.0
        for el in self._elements:
            if not el.surface_ids:
                continue
            indices = [lookup.get(u) for u in el.surface_ids]
            if any(idx is None or not (0 <= idx < len(self.surfaces)) for idx in indices):
                # Stale UUID — skip without disturbing the running anchor;
                # ``prev_pos_z`` stays pointed at the last known-good element.
                continue
            if prev_pos_z is None:
                new_z = float(el.position[2])
            else:
                new_z = prev_pos_z + prev_thickness_sum
            x, y, _z = el.position
            el.position = (x, y, new_z)
            prev_pos_z = new_z
            prev_thickness_sum = float(
                sum(self.surfaces[idx].thickness for idx in indices)
            )

    def load_aperture_images(self, *, root_dir: Optional[str] = None) -> None:
        """Decode bitmap data for any surface with ``aperture_shape == IMAGE``.

        For each surface whose ``aperture_shape`` is ``IMAGE``, read the
        ``source_path`` recorded on ``self.aperture_images[i]`` (resolved
        against ``root_dir`` when relative), decode the image to
        single-channel float32 in [0, 1], and fill ``.pixels`` / ``.width``
        / ``.height``.

        Idempotent: surfaces whose ``pixels`` are already populated are
        skipped.  Cache state is invalidated so the next renderer call
        reflects the new mask data.

        Uses PIL (Pillow) to decode; install it explicitly if your
        environment doesn't already pull it in.
        """
        from PIL import Image
        import numpy as np

        n = self.num_surfaces()
        images = self.aperture_images
        if len(images) < n:
            # Defensive: parser normally sizes this to n_surfaces; pad if a
            # programmatic builder didn't.
            while len(images) < n:
                images.append(type(images[0])() if len(images) else None)

        loaded_any = False
        for i in range(n):
            s = self.surfaces[i]
            if s.aperture_shape != ApertureShape.IMAGE:
                continue
            img = images[i]
            if img.pixels.size > 0:
                continue
            if not img.source_path:
                continue
            path = img.source_path
            if root_dir is not None and not os.path.isabs(path):
                path = os.path.join(root_dir, path)
            with Image.open(path) as pil_img:
                gray = pil_img.convert("F")
                arr = np.asarray(gray, dtype=np.float32)
            peak = float(arr.max()) if arr.size else 0.0
            if peak > 1.5:
                arr = arr / 255.0
            img.pixels = arr
            loaded_any = True

        if loaded_any:
            self._invalidate()

    # ------------------------------------------------------------------
    # Element / pivot accessors
    # ------------------------------------------------------------------

    @property
    def elements(self) -> list:
        """List of :class:`Element` reconstructed from the source JSON.

        Returns an empty list for systems built programmatically without
        going through :meth:`load`.
        """
        return list(self._elements) if self._elements is not None else []

    @property
    def pivots(self) -> list:
        """List of :class:`Pivot` reconstructed from the source JSON."""
        return list(self._pivots) if self._pivots is not None else []

    # ------------------------------------------------------------------
    # Save / round-trip
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Emit this system as a v1.0 ``ghostlight-optical`` JSON file.

        Uses the in-memory :attr:`elements` / :attr:`pivots` (the editable
        copies) plus the stashed glass catalogue and metadata to produce a
        file that re-loads to the same bind-pose + rig state.
        """
        from .writer import write_optical_system

        write_optical_system(
            path,
            system=self,
            metadata=self._raw_metadata,
            glass_catalogue=self._raw_glass_catalogue,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _current_key(self) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(struct.pack("f", self.focal_length))
        for s in self.surfaces:
            h.update(struct.pack(
                "fffffiii",
                s.radius, s.thickness, s.ior, s.abbe_v, s.semi_aperture,
                int(s.is_stop), s.coating.ar_layers,
                int(s.is_active),
            ))
            h.update(struct.pack("iif", s.form, s.disp_model, s.conic_k))
            h.update(struct.pack(
                "iifff",
                s.aperture_shape, s.aperture_blades,
                s.aperture_rotation_rad, s.aperture_aspect,
                s.aperture_semi_diameter,
            ))
            # Blade geometry affects the calibrated pupil and ghost pairs.
            h.update(struct.pack(
                "ffff",
                s.aperture_curvature, s.aperture_twist,
                s.aperture_notch_rad, s.aperture_notch_angle_rad,
            ))
            # Baked pose affects calibration and ghost enumeration.
            h.update(struct.pack("fff", s.z, s.decenter_x, s.decenter_y))
            for i in range(9):
                h.update(struct.pack("f", float(s.rot[i])))
            # Surface form and dispersion affect calibration and ghost pairs.
            h.update(struct.pack("ii", int(s.n_asphere_terms), int(s.cyl_axis)))
            h.update(s.asphere_terms.tobytes())
            h.update(s.sellmeier_B.tobytes())
            h.update(s.sellmeier_C.tobytes())
            c = s.coating
            h.update(struct.pack("ifffff",
                                 int(c.model), c.gauss_sigma, c.gauss_background,
                                 c.gauss_peak, c.gauss_decenter_x, c.gauss_decenter_y))
            h.update(struct.pack("fffff",
                                 c.tint_r, c.tint_g, c.tint_b, c.tint_strength,
                                 c.angle_ref_ior))
        h.update(struct.pack("<Q", self.aperture_image_state_hash()))
        # Table-backed coating contents (spectral/angular/SA tables + layer
        # stacks) are hashed in C++; one uint64 covers every table byte.
        h.update(struct.pack("<Q", self.coating_state_hash()))
        return h.digest()

    def _check_invalidate(self) -> None:
        k = self._current_key()
        if k != self._cache_key:
            self._invalidate()
            self._cache_key = k

    def _invalidate(self) -> None:
        self._calib = None
        self._calib_args = None
        self._ghost_pairs = None
        self._filtered.clear()
        self._cache_key = None

    # ------------------------------------------------------------------
    # Cached accessors
    # ------------------------------------------------------------------

    def calibration(
        self,
        *,
        d_line_nm: float = 587.56,
    ) -> LensCalibration:
        """Return (cached) LensCalibration.  Recomputes only when the
        system geometry or the calibration parameters change."""
        self._check_invalidate()
        args = (d_line_nm,)
        if self._calib is None or self._calib_args != args:
            self._calib = calibrate_lens(self, d_line_nm)
            self._calib_args = args
        return self._calib

    def ghost_pairs(self) -> list:
        """Return (cached) list of all GhostPair objects for this system."""
        self._check_invalidate()
        if self._ghost_pairs is None:
            self._ghost_pairs = enumerate_ghost_pairs(self)
        return self._ghost_pairs

    def filter_ghosts(
        self,
        config: FlareConfig,
        *,
        calib: Optional[LensCalibration] = None,
    ) -> tuple:
        """Return (pairs, area_boosts) after filtering ghost pairs for the
        given FlareConfig.  Results are cached per unique config settings."""
        self._check_invalidate()
        cal = calib if calib is not None else self.calibration()
        cfg_key = (
            config.min_ghost_intensity,
            config.ghost_normalize,
            config.max_area_boost,
            cal.sensor_half_w,
            cal.sensor_half_h,
        )
        if cfg_key not in self._filtered:
            self._filtered[cfg_key] = filter_ghost_pairs(
                self, cal.sensor_half_w, cal.sensor_half_h, config
            )
        return self._filtered[cfg_key]

    # ------------------------------------------------------------------
    # Renderer convenience wrappers
    # ------------------------------------------------------------------

    def render_point_flare(
        self,
        width: int,
        height: int,
        config: PointFlareConfig,
        *,
        calib: Optional[LensCalibration] = None,
    ) -> dict:
        """Render ghost flares from a single point source.

        Returns a dict of float32 numpy arrays: ghost_r/g/b (and starburst_r/g/b
        when DiffractionConfig.starburst is set).
        """
        cal = calib if calib is not None else self.calibration()
        return _render_point_flare(width, height, self, cal, config)

    def render_source_flare(
        self,
        offsets,
        width: int,
        height: int,
        config: PointFlareConfig,
        *,
        calib: Optional[LensCalibration] = None,
    ) -> dict:
        """Render ghost flares from an extended source sampled as angular offsets.

        offsets: (N, 3) float32 array of [d_angle_x, d_angle_y, weight] rows —
                 angular offsets in radians around the screen-space center
                 config.source_x/source_y, each scaled by its quadrature
                 weight.  Weights summing to 1 average an area source; pass a
                 subset of those weights to accumulate a progressive chunk
                 (results are linear, so chunk sums equal a single full call).
                 See ghostlight.source_sampling for shape samplers.

        Returns a dict of float32 numpy arrays: ghost_r/g/b (and starburst_r/g/b
        when DiffractionConfig.starburst is set).
        """
        import numpy as np

        arr = np.ascontiguousarray(offsets, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            raise ValueError(
                "render_source_flare: offsets must be a (N, 3) float32 array "
                "of [d_angle_x, d_angle_y, weight] rows with N >= 1"
            )
        cal = calib if calib is not None else self.calibration()
        return _render_source_flare(arr, width, height, self, cal, config)

    def render_psf(
        self,
        sources,
        config: "PSFConfig",
        *,
        weights=None,
        targets_mm=None,
    ) -> dict:
        """Render a grid of geometric point-spread functions.

        sources: (N, 2) float32 array of [angle_x, angle_y] in radians, or any
                 (N, 5) array of [angle_x, angle_y, r, g, b] rows.  In
                 FIXED_TARGET mode angle_* is the aim seed direction.
        weights: optional (N, 3) float32 of [r, g, b] per-source weights.  Only
                 used when sources is (N, 2); ignored otherwise.  Defaults to
                 white (1, 1, 1) for every source.
        targets_mm: optional (N, 2) float32 of [target_x_mm, target_y_mm] cell
                 centres for FIXED_TARGET mode (config.center_mode must be
                 FIXED_TARGET).  When given, the chief ray of each cell is aimed
                 at its target and the tile is anchored there.

        N must be <= config.grid_nx * config.grid_ny.  Source i maps to tile
        (i % grid_nx, i // grid_nx) of the returned composite buffer.

        Returns a dict with r/g/b composite arrays (composite_h, composite_w),
        per-source chief_x_mm / chief_y_mm, status (uint8 PSFCellStatus),
        pupil_fraction, aim_residual_mm arrays, and tile metadata.
        """
        import numpy as np
        arr = np.ascontiguousarray(sources, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] not in (2, 5):
            raise ValueError(
                "render_psf: sources must be a (N, 2) or (N, 5) float32 array"
            )
        n = arr.shape[0]
        if arr.shape[1] == 2:
            if weights is None:
                w = np.ones((n, 3), dtype=np.float32)
            else:
                w = np.ascontiguousarray(weights, dtype=np.float32)
                if w.shape != (n, 3):
                    raise ValueError("render_psf: weights must have shape (N, 3)")
            arr = np.concatenate([arr, w], axis=1)

        if targets_mm is not None:
            t = np.ascontiguousarray(targets_mm, dtype=np.float32)
            if t.shape != (n, 2):
                raise ValueError("render_psf: targets_mm must have shape (N, 2)")
            arr = np.concatenate([arr, t], axis=1)   # -> (N, 7)

        return _render_psf(arr, self, config)

    def __repr__(self) -> str:
        return (
            f"OpticalSystem(name={self.name!r}, "
            f"surfaces={self.num_surfaces()}, "
            f"elements={len(self._elements) if self._elements else 0}, "
            f"pivots={len(self._pivots) if self._pivots else 0})"
        )
