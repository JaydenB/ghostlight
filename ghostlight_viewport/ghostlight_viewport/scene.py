"""Scene graph + dirty flags for the viewport.

This module is GL-agnostic: it stores per-element mesh data (vertex / normal
arrays) and bookkeeping (tints, picking IDs, bbox), but it never calls GL.
The widget reads from a Scene to upload VBOs and issue draw calls.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import geometry
from .colors import PALETTE, element_tint

if TYPE_CHECKING:
    from ghostlight import Element, OpticalSystem


def _accent_blend(
    rgb: tuple[float, float, float],
    accent: tuple[float, float, float],
    strength: float,
) -> tuple[float, float, float]:
    """Linear blend ``rgb`` toward ``accent`` by ``strength`` (0..1).

    Used by ghost-solo highlighting: pulls a glass element's tint toward
    the UI accent so the viewport reads "this element is participating
    in the ghosts you've isolated" without losing its hue identity.
    """
    s = max(0.0, min(1.0, float(strength)))
    return (
        rgb[0] + (float(accent[0]) - rgb[0]) * s,
        rgb[1] + (float(accent[1]) - rgb[1]) * s,
        rgb[2] + (float(accent[2]) - rgb[2]) * s,
    )


def _desaturate(rgb: tuple[float, float, float], strength: float) -> tuple[float, float, float]:
    """Pull an RGB tint toward its luminance by ``strength`` (0..1).

    Used by muted-element styling: keeps the original hue identifiable
    while draining the saturation, so the user reads "still this element,
    just off" rather than "different element". ``strength=0`` is a no-op;
    ``strength=1`` collapses fully to grey.
    """
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    # ITU-R BT.709 luminance.
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    s = max(0.0, min(1.0, float(strength)))
    return (
        r + (y - r) * s,
        g + (y - g) * s,
        b + (y - b) * s,
    )


def _region_to_mesh(region: geometry.SubmeshRegion) -> geometry.Mesh:
    return geometry.Mesh(
        vertices=region.vertices,
        normals=region.normals,
        indices=region.indices,
        kinds=region.kinds,
    )


def _combine_subsolids(subsolids: list[geometry.SubSolid]) -> geometry.Mesh:
    """Merge every region of every sub-solid into one mesh for bbox/framing."""
    meshes = [
        _region_to_mesh(r)
        for ss in subsolids
        for r in ss.regions
        if r.vertex_count > 0
    ]
    if not meshes:
        return geometry.Mesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
        )
    if len(meshes) == 1:
        return meshes[0]
    return geometry.merge_meshes(meshes)


@dataclass
class SceneElement:
    """Per-element scene record.

    ``subsolids`` holds one :class:`geometry.SubSolid` per closed sub-solid
    (singlet → 1, doublet → 2, triplet → 3, stop → 1).  Each sub-solid is
    further decomposed into ``regions`` (front cap, back cap, wall halves, or
    iris) carrying a surface index — the renderer issues one draw call per
    region and depth-sorts across sub-solids so cemented n-lets blend
    correctly from either side of the optical axis.  ``mesh`` is the merged
    convenience view used for bbox computation and "frame to selection" —
    drawing it directly would defeat the sub-solid sort.
    """
    index: int                          # stable element index (used as pick ID)
    element: "Element"
    subsolids: list[geometry.SubSolid]
    mesh: geometry.Mesh
    tint: tuple[float, float, float]
    alpha: float
    is_stop: bool
    # True when the underlying Element is ghost-muted. Widget.render
    # skips the fill / cap / depth-prepass passes for muted elements so
    # only their outlines draw — a stronger visual cue than translucent
    # geometry that OIT tends to swallow.
    muted: bool = False
    content_hash: bytes = b""


class Scene:
    """Owns the list of :class:`SceneElement` plus the global bbox.

    The widget queries:
      * :attr:`elements`     to draw the scene
      * :attr:`bbox_min` / :attr:`bbox_max` for camera framing and cap quads
    """

    def __init__(self) -> None:
        self.elements: list[SceneElement] = []
        self.bbox_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.bbox_max = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)
        # World positions of every element's centre of rotation, for elements
        # that declare a non-zero one. Only these are worth marking: a zero
        # pivot sits on the element's front vertex, so drawing it would put a
        # marker on every element and say nothing.
        self.pivot_points: list[np.ndarray] = []
        self._subsolid_cache: dict[bytes, list[geometry.SubSolid]] = {}
        # Global indices of surfaces the user has marked "ghost-solo" in
        # the designer. Set by the widget via rebuild(); per-element the
        # tint/alpha picks up a highlight when any of its surfaces is in
        # this set, so the user can spot which elements participate in
        # the ghosts they're isolating.
        self._ghost_solo_surface_indices: set[int] = set()

    def clear(self) -> None:
        self.elements.clear()
        self.pivot_points.clear()
        self.bbox_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.bbox_max = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)

    def rebuild(
        self,
        system: "OpticalSystem",
        elements: list,
        *,
        ghost_solo_surface_indices: set | frozenset | None = None,
    ) -> None:
        """Tessellate every element + recompute bbox.

        Mesh cache is keyed on a blake2b of the element's geometry-affecting
        attributes so re-pushing an unchanged lens reuses previous meshes.

        ``ghost_solo_surface_indices`` is the set of global surface
        indices the user has marked ghost-solo; elements whose surface
        set intersects it get a brighter tint + nearly-opaque alpha so
        the viewport hints at "ghosts I'm filtering involve these
        elements". None or empty is the unfiltered default.
        """
        self._ghost_solo_surface_indices = (
            set(ghost_solo_surface_indices) if ghost_solo_surface_indices else set()
        )
        self.elements.clear()
        for i, el in enumerate(elements):
            key = self._element_hash(el, system)
            subsolids = self._subsolid_cache.get(key)
            if subsolids is None:
                subsolids = geometry.build_element_subsolids(el, system)
                self._subsolid_cache[key] = subsolids
            kind = getattr(el.kind, "name", "GLASS")
            is_stop = kind == "STOP"
            if is_stop:
                tint = PALETTE["stop"]
                alpha = 0.9
            else:
                indices = el.resolve_surfaces(system)
                iors = [
                    float(getattr(system.surfaces[idx], "ior", 1.5))
                    for idx in indices
                ]
                mean_ior = float(np.mean(iors)) if iors else 1.5
                tint = element_tint(mean_ior)
                alpha = 0.75
            # Muted elements are rendered outline-only by the widget —
            # the tint/alpha stashed here are only consulted if that
            # path is ever bypassed, so a desaturated tint is still a
            # helpful fallback. The try/except covers Element-like test
            # stubs without ``is_muted``.
            muted = False
            try:
                muted = bool(el.is_muted(system))
            except AttributeError:
                pass
            if muted:
                tint = _desaturate(tint, 0.35)
                alpha = 0.2
            # Ghost-solo'd elements (any of their surfaces in the solo set)
            # get a brighter tint + near-opaque alpha so the viewport
            # echoes the tree's surf-solo accent. Muting wins when both
            # apply — the user explicitly asked for it to be off.
            elif self._ghost_solo_surface_indices:
                solo_hit = any(
                    int(idx) in self._ghost_solo_surface_indices
                    for idx in el.resolve_surfaces(system)
                )
                if solo_hit:
                    tint = _accent_blend(tint, PALETTE["selection_outline"], 0.45)
                    alpha = 0.95
            self.elements.append(SceneElement(
                index=i,
                element=el,
                subsolids=subsolids,
                mesh=_combine_subsolids(subsolids),
                tint=tuple(tint),
                alpha=alpha,
                is_stop=is_stop,
                muted=muted,
                content_hash=key,
            ))
        self._rebuild_pivot_points(system, elements)
        meshes = [se.mesh for se in self.elements]
        mn, mx = geometry.mesh_bbox(meshes)
        self.bbox_min, self.bbox_max = mn, mx
        # A pivot can legitimately sit outside the glass (that's how you tilt
        # an element about a point in front of it), so fold the markers into
        # the bbox or "frame all" would crop them.
        for p in self.pivot_points:
            self.bbox_min = np.minimum(self.bbox_min, p)
            self.bbox_max = np.maximum(self.bbox_max, p)

    def _rebuild_pivot_points(self, system: "OpticalSystem", elements: list) -> None:
        """Resolve each element's centre of rotation into world space.

        Delegates to ``ghostlight.element_world_pivot``, which derives the point
        from the element's *baked* first-surface pose rather than from
        ``Element.position``. That matters: the loader rebases the whole chain
        so the last surface lands on the sensor at z = 0, which puts authored
        element z and baked surface z in different frames.
        """
        self.pivot_points.clear()
        try:
            from ghostlight import element_world_pivot
        except ImportError:  # pragma: no cover - core package without the helper
            return
        for el in elements:
            try:
                point = element_world_pivot(system, el)
            except Exception:  # pragma: no cover - test stubs / mid-edit state
                continue
            if point is not None:
                self.pivot_points.append(np.asarray(point, dtype=np.float32))

    def expand_bbox_with_sensor(self, sensor) -> None:
        if sensor is None:
            return
        smn, smx = sensor.bbox()
        smn = np.asarray(smn, dtype=np.float32)
        smx = np.asarray(smx, dtype=np.float32)
        self.bbox_min = np.minimum(self.bbox_min, smn)
        self.bbox_max = np.maximum(self.bbox_max, smx)

    def expand_bbox_with_calibrated_sensor(self, calibrated) -> None:
        if calibrated is None:
            return
        smn, smx = calibrated.bbox()
        smn = np.asarray(smn, dtype=np.float32)
        smx = np.asarray(smx, dtype=np.float32)
        self.bbox_min = np.minimum(self.bbox_min, smn)
        self.bbox_max = np.maximum(self.bbox_max, smx)

    def update_element_at(self, system: "OpticalSystem", element_index: int, element) -> None:
        if not (0 <= element_index < len(self.elements)):
            return
        key = self._element_hash(element, system)
        subsolids = self._subsolid_cache.get(key)
        if subsolids is None:
            subsolids = geometry.build_element_subsolids(element, system)
            self._subsolid_cache[key] = subsolids
        se = self.elements[element_index]
        se.element = element
        se.subsolids = subsolids
        se.mesh = _combine_subsolids(subsolids)
        se.content_hash = key

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _element_hash(el, system: "OpticalSystem") -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(el.name.encode("utf-8"))
        h.update(el.element_id.encode("utf-8"))
        h.update(struct.pack("iii",
                              int(getattr(el.kind, "value", el.kind)),
                              len(el.surface_ids),
                              len(el.material_glasses)))
        h.update(struct.pack("fff", *el.position))
        h.update(struct.pack("fff", *el.rotation_euler_deg))
        try:
            indices = el.resolve_surfaces(system)
        except KeyError:
            return h.digest()
        # Mix the resolved global indices into the hash. SubmeshRegion bakes
        # the global surface_index into each region at build time, and the
        # renderer + picking compare those values directly. When a new
        # element is inserted in front, this element's surfaces shift to
        # higher indices in system.surfaces but none of its own properties
        # (position, geometry) change — without this, the cache would hand
        # back stale subsolids whose regions still carry the OLD indices,
        # so a surface selection in any one element would also highlight
        # the same local-position surface in every other element.
        h.update(struct.pack(f"{len(indices)}i", *(int(i) for i in indices)))
        for idx in indices:
            s = system.surfaces[idx]
            h.update(struct.pack(
                "fffffi",
                float(getattr(s, "radius", 0.0)),
                float(getattr(s, "thickness", 0.0)),
                float(getattr(s, "semi_aperture", 1.0)),
                float(getattr(s, "conic_k", 0.0)),
                float(getattr(s, "z", 0.0)),
                int(bool(getattr(s, "is_stop", False))),
            ))
            form = getattr(s, "form", None)
            form_v = int(getattr(form, "value", form)) if form is not None else 0
            shape = getattr(s, "aperture_shape", None)
            shape_v = int(getattr(shape, "value", shape)) if shape is not None else 0
            cyl_axis = getattr(s, "cyl_axis", None)
            cyl_axis_v = int(getattr(cyl_axis, "value", cyl_axis)) if cyl_axis is not None else 0
            h.update(struct.pack(
                "iiiifff",
                form_v,
                shape_v,
                cyl_axis_v,
                int(getattr(s, "aperture_blades", 0)),
                float(getattr(s, "aperture_rotation_rad", 0.0)),
                float(getattr(s, "aperture_aspect", 1.0)),
                float(getattr(s, "aperture_semi_diameter", 0.0)),
            ))
            n_terms = int(getattr(s, "n_asphere_terms", 0))
            for i in range(n_terms):
                try:
                    h.update(struct.pack("f", float(s.asphere_terms[i])))
                except (IndexError, TypeError):
                    break
            h.update(struct.pack(
                "ff", float(getattr(s, "decenter_x", 0.0)),
                       float(getattr(s, "decenter_y", 0.0))
            ))
            # ``rot`` matters as much as decenter — geometry.py rotates every
            # vertex by it. Leaving it out meant a pure tilt edit produced an
            # unchanged hash, so the cached mesh was reused and the element
            # never visibly turned.
            rot = getattr(s, "rot", None)
            if rot is not None:
                try:
                    h.update(struct.pack("9f", *(float(v) for v in rot)))
                except (TypeError, ValueError, struct.error):
                    pass
        # The element's own pivot doesn't reach the surfaces (it's already
        # baked into their poses) but it does drive the viewport's pivot
        # marker, so a pivot-only edit has to invalidate too.
        pivot = getattr(el, "pivot", None)
        if pivot is not None:
            try:
                h.update(struct.pack("3f", *(float(v) for v in pivot)))
            except (TypeError, ValueError, struct.error):
                pass
        return h.digest()
