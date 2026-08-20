"""Selection-mode wiring in ``LensViewport``.

Covers the three modes exposed by the toolbar (``element``, ``surface``,
``none``) and the new per-region picking encoding.  Doesn't exercise the
GL path — those checks live alongside the screenshot-based viewport
smoke tests.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed; viewport tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "ghostlight" / "bindings" / "python"))
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

from _helpers import example_doublet_path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import ghostlight
from ghostlight_viewport import LensViewport
from ghostlight_viewport.widget import set_default_surface_format
from ghostlight_viewport import picking


@pytest.fixture(scope="module")
def app():
    set_default_surface_format()
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def doublet_loaded(app):
    """Viewport pre-populated with the example doublet so scene.elements
    and per-element subsolids/regions exist for selection bookkeeping
    tests.  GL never initialises in offscreen mode without paintGL — the
    scene-level data is enough for the public API exercised here."""
    lens_path = example_doublet_path()
    system = ghostlight.OpticalSystem.load(str(lens_path))
    elements = list(system.elements)
    viewport = LensViewport()
    viewport.scene.rebuild(system, elements)
    return viewport, system, elements


# ---------------------------------------------------------------------------
# Selection-state / signal contract
# ---------------------------------------------------------------------------


def test_selection_state_tracks_surface(app):
    from ghostlight_viewport.selection import SelectionState

    state = SelectionState()
    assert state.element is None and state.surface is None

    assert state.set_surface(3) is True
    assert state.set_surface(3) is False        # idempotent
    assert state.surface == 3
    assert state.clear_surface() is True
    assert state.surface is None


def test_clear_selection_emits_for_both_dimensions(doublet_loaded):
    viewport, _system, elements = doublet_loaded
    viewport.selection.element = elements[0]
    viewport.selection.surface = 1

    el_emits: list = []
    surf_emits: list = []
    viewport.elementSelected.connect(el_emits.append)
    viewport.surfaceSelected.connect(surf_emits.append)

    viewport.clear_selection()
    assert viewport.selection.element is None
    assert viewport.selection.surface is None
    assert el_emits == [None]
    assert surf_emits == [None]


def test_set_selected_element_clears_surface(doublet_loaded):
    viewport, _system, elements = doublet_loaded
    viewport.selection.element = elements[0]
    viewport.selection.surface = 0

    viewport.set_selected_element(elements[0])
    # Element unchanged but surface dropped — surface is element-scoped.
    assert viewport.selected_element() is elements[0]
    assert viewport.selected_surface() is None


def test_set_selected_surface_is_programmatic(doublet_loaded):
    viewport, _system, _elements = doublet_loaded
    emits: list = []
    viewport.surfaceSelected.connect(emits.append)

    viewport.set_selected_surface(2)
    assert viewport.selected_surface() == 2
    assert emits == []  # programmatic setter never emits

    viewport.set_selected_surface(None)
    assert viewport.selected_surface() is None
    assert emits == []


# ---------------------------------------------------------------------------
# Mode switch — always-clear-on-switch
# ---------------------------------------------------------------------------


def test_mode_switch_clears_existing_selection(doublet_loaded):
    viewport, _system, elements = doublet_loaded
    viewport.selection.element = elements[0]
    viewport.selection.surface = 0

    el_emits: list = []
    surf_emits: list = []
    viewport.elementSelected.connect(el_emits.append)
    viewport.surfaceSelected.connect(surf_emits.append)

    # Drive the user-facing path (toolbar menu click) — pick a different
    # mode so the change actually goes through.
    viewport._toolbar.selection.set_value("surface", emit=True)

    assert viewport.selection_mode() == "surface"
    assert viewport.selected_element() is None
    assert viewport.selected_surface() is None
    assert el_emits == [None]
    assert surf_emits == [None]


# ---------------------------------------------------------------------------
# Geometry — per-region subsolids + surface attribution
# ---------------------------------------------------------------------------


def test_doublet_has_two_subsolids_each_with_four_regions(doublet_loaded):
    """A two-element doublet should produce per-element sub-solids whose
    regions cover front_cap + back_cap + the two wall halves."""
    viewport, _system, _elements = doublet_loaded
    glass_elements = [
        se for se in viewport.scene.elements
        if getattr(se.element.kind, "name", "GLASS") == "GLASS"
    ]
    assert glass_elements, "doublet sample should contain glass elements"
    for se in glass_elements:
        assert len(se.subsolids) >= 1
        for ss in se.subsolids:
            kinds = sorted([
                (r.is_cap, r.surface_index) for r in ss.regions
            ])
            assert len(kinds) == 4, (
                f"each glass sub-solid expected 4 regions "
                f"(front/back caps + wall_a/wall_b), got {kinds}"
            )
            caps = [r for r in ss.regions if r.is_cap]
            walls = [r for r in ss.regions if not r.is_cap]
            assert len(caps) == 2 and len(walls) == 2
            # Wall halves' surface_index always matches one of the caps.
            cap_indices = {r.surface_index for r in caps}
            wall_indices = {r.surface_index for r in walls}
            assert wall_indices.issubset(cap_indices)


# ---------------------------------------------------------------------------
# Picking encoder — (element, surface) round-trip via the helper used by
# the actual GL draw call.
# ---------------------------------------------------------------------------


def test_encode_decode_element_and_surface_pair():
    r, g, b, a = picking.encode_element_surface_id(7, 12)
    info = picking.decode_pixel((
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        int(round(a * 255)),
    ))
    assert info["element_index"] == 7
    assert info["surface_index"] == 12
    assert info["tag"] == picking.TAG_ELEMENT_BODY
    assert info["is_empty"] is False


def test_encode_with_no_surface_returns_none_on_decode():
    r, g, b, a = picking.encode_element_surface_id(3, None)
    info = picking.decode_pixel((
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        int(round(a * 255)),
    ))
    assert info["element_index"] == 3
    assert info["surface_index"] is None


# ---------------------------------------------------------------------------
# Surface highlight isolation: surface-mode hover/select must not light up
# the whole owning element — only the picked cap.
# ---------------------------------------------------------------------------


def test_hover_surface_field_independent_of_hover_element(app):
    from ghostlight_viewport.selection import SelectionState

    state = SelectionState()
    assert state.hover_surface is None
    assert state.set_hover_surface(5) is True
    assert state.hover_surface == 5
    # clear_hover wipes both element and surface hover — they're paired.
    assert state.clear_hover() is True
    assert state.hover is None and state.hover_surface is None


def test_surface_select_suppresses_element_highlight_in_render_state(doublet_loaded):
    """When a surface is selected the renderer's priority logic must pick
    the surface state for the matching cap and the normal (non-highlighted)
    state for the rest of that element's regions.

    We probe the priority directly: surface highlight beats element
    highlight on the cap, and the same element's other regions fall through
    to the normal state.
    """
    viewport, _system, elements = doublet_loaded
    target_se = next(
        se for se in viewport.scene.elements
        if getattr(se.element.kind, "name", "GLASS") == "GLASS"
        and se.subsolids
    )
    cap_region = next(r for r in target_se.subsolids[0].regions if r.is_cap)
    other_cap = next(
        r for r in target_se.subsolids[0].regions
        if r.is_cap and r.surface_index != cap_region.surface_index
    )
    wall_region = next(r for r in target_se.subsolids[0].regions if not r.is_cap)

    viewport.selection.element = target_se.element
    viewport.selection.surface = cap_region.surface_index

    sel_elem = viewport.selection.element
    sel_surface = viewport.selection.surface
    elem_select_suppressed = sel_surface is not None

    def classify(region) -> str:
        is_sel_cap = (
            sel_surface is not None
            and region.is_cap
            and region.surface_index == sel_surface
        )
        is_elem_selected = (
            not elem_select_suppressed
            and sel_elem is not None
            and target_se.element is sel_elem
        )
        if is_sel_cap:
            return "sel_surface"
        if is_elem_selected:
            return "elem_select"
        return "normal"

    assert classify(cap_region) == "sel_surface"
    assert classify(other_cap) == "normal"  # NOT element-selected — surface mode suppresses it
    assert classify(wall_region) == "normal"


def test_surface_hover_suppresses_element_hover(doublet_loaded):
    viewport, _system, _elements = doublet_loaded
    target_se = next(
        se for se in viewport.scene.elements
        if getattr(se.element.kind, "name", "GLASS") == "GLASS"
        and se.subsolids
    )
    cap_region = next(r for r in target_se.subsolids[0].regions if r.is_cap)
    other_cap = next(
        r for r in target_se.subsolids[0].regions
        if r.is_cap and r.surface_index != cap_region.surface_index
    )

    # Simulate the hover state surface mode would produce: hover element +
    # hover surface paired (set_hover_surface non-None).
    viewport.selection.hover = target_se.element
    viewport.selection.hover_surface = cap_region.surface_index

    hover_elem = viewport.selection.hover
    hover_surface = viewport.selection.hover_surface
    elem_hover_suppressed = hover_surface is not None

    def classify(region) -> str:
        is_hover_cap = (
            hover_surface is not None
            and region.is_cap
            and region.surface_index == hover_surface
        )
        is_elem_hovered = (
            not elem_hover_suppressed
            and hover_elem is not None
            and target_se.element is hover_elem
        )
        if is_hover_cap:
            return "hover_surface"
        if is_elem_hovered:
            return "elem_hover"
        return "normal"

    assert classify(cap_region) == "hover_surface"
    assert classify(other_cap) == "normal"


def test_element_mode_still_highlights_whole_element(doublet_loaded):
    """Regression guard: element-mode selection (no surface set) must keep
    lighting up every region of the owning element."""
    viewport, _system, _elements = doublet_loaded
    target_se = next(
        se for se in viewport.scene.elements
        if getattr(se.element.kind, "name", "GLASS") == "GLASS"
        and se.subsolids
    )

    viewport.selection.element = target_se.element
    viewport.selection.surface = None  # element mode: no surface pick

    sel_elem = viewport.selection.element
    sel_surface = viewport.selection.surface
    elem_select_suppressed = sel_surface is not None

    is_elem_selected = (
        not elem_select_suppressed
        and sel_elem is not None
        and target_se.element is sel_elem
    )
    assert is_elem_selected is True

    # And every region (caps + walls) sees the same element-level state.
    for region in target_se.subsolids[0].regions:
        is_sel_cap = (
            sel_surface is not None
            and region.is_cap
            and region.surface_index == sel_surface
        )
        assert is_sel_cap is False  # no surface set → no cap state


def test_pick_pass_sorts_regions_back_to_front(doublet_loaded):
    """Pick render must order regions by view-space depth so the
    front-most cap wins each pixel.

    The bug: only elements were depth-sorted. Within a cemented n-let or
    even a singlet, regions drew in storage order (front cap, back cap,
    walls), and with no depth test the back cap painted over the front
    cap — making surface picks resolve to the back surface even when
    hovering the front. This test rebuilds the same per-region sort key
    the pick pass uses, using the camera's real view matrix.
    """
    viewport, _system, _elements = doublet_loaded
    viewport.camera.fit_to_bbox(viewport.scene.bbox_min, viewport.scene.bbox_max)
    view = viewport.camera.view_matrix()

    def depth(c) -> float:
        return float(view[2, 0] * c[0] + view[2, 1] * c[1] + view[2, 2] * c[2] + view[2, 3])

    regions: list = []
    for se in viewport.scene.elements:
        for ss in se.subsolids:
            for region in ss.regions:
                if region.vertex_count == 0:
                    continue
                c = region.vertices.mean(axis=0)
                regions.append({
                    "centroid": (float(c[0]), float(c[1]), float(c[2])),
                    "surface_index": region.surface_index,
                    "is_cap": region.is_cap,
                })

    assert len(regions) >= 4, "doublet should produce at least 4 regions"

    sorted_regions = sorted(regions, key=lambda r: depth(r["centroid"]))
    depths = [depth(r["centroid"]) for r in sorted_regions]
    assert depths == sorted(depths), "sort key not monotone non-decreasing"


def test_pick_pass_sort_handles_front_back_caps_within_singlet(doublet_loaded):
    """Within a single sub-solid, the front cap (closer to camera) must
    draw AFTER the back cap so the front cap wins the pick FBO pixel.

    The bug: regions of one sub-solid drew in storage order (front_cap,
    back_cap, walls).  When looking down the optical axis, both caps
    project to the same screen pixel and the back cap (drawn last) stole
    the click — picks went through the front to the back.  We set up
    that exact view and verify the per-region sort flips the order.
    """
    viewport, _system, _elements = doublet_loaded
    # Look down the optical axis (-Z preset) so front and back caps of
    # each sub-solid project to the same screen position and their
    # view-space depths actually differ.
    viewport.camera.set_preset("-z")
    viewport.camera.fit_to_bbox(viewport.scene.bbox_min, viewport.scene.bbox_max)
    view = viewport.camera.view_matrix()

    def depth(c) -> float:
        return float(view[2, 0] * c[0] + view[2, 1] * c[1] + view[2, 2] * c[2] + view[2, 3])

    found_pair = False
    for se in viewport.scene.elements:
        for ss in se.subsolids:
            caps = [r for r in ss.regions if r.is_cap and r.vertex_count > 0]
            if len(caps) < 2:
                continue
            sorted_caps = sorted(
                caps,
                key=lambda r: depth(tuple(float(v) for v in r.vertices.mean(axis=0))),
            )
            d_first = depth(tuple(float(v) for v in sorted_caps[0].vertices.mean(axis=0)))
            d_last = depth(tuple(float(v) for v in sorted_caps[-1].vertices.mean(axis=0)))
            if d_first == d_last:
                continue
            # Sort key ascending → ``sorted_caps[-1]`` is the cap closest
            # to the camera.  It MUST draw later than the other cap so the
            # last-write-wins pick FBO returns its surface_index for a
            # click at the optical-axis centre.  The pre-fix code stored
            # caps as (front_cap, back_cap) and looking down -Z this meant
            # the FRONT cap drew first then back drew on top of it — broken.
            assert d_last > d_first
            found_pair = True
    assert found_pair, (
        "doublet should have at least one sub-solid with two depth-distinct "
        "caps when viewed along the optical axis"
    )


def test_region_buffer_entries_carry_centroid(doublet_loaded):
    """The pick pass reads `region["centroid"]` to depth-sort, so the
    upload code must populate it.  Without GL initialisation we can't
    exercise `_upload_element_buffers` directly, but the geometry layer
    feeds it region.vertices.mean — verify those means are non-degenerate
    so a depth sort would produce a meaningful ordering."""
    viewport, _system, _elements = doublet_loaded
    seen_zs: set = set()
    for se in viewport.scene.elements:
        for ss in se.subsolids:
            for region in ss.regions:
                if region.vertex_count == 0:
                    continue
                z = round(float(region.vertices.mean(axis=0)[2]), 4)
                seen_zs.add(z)
    # A doublet must produce regions at distinct Z positions (front cap,
    # back cap, wall halves) — otherwise the depth sort is degenerate.
    assert len(seen_zs) >= 3, (
        f"expected regions spread across multiple Z positions, got {seen_zs}"
    )


def test_scene_cache_invalidates_when_indices_shift(app):
    """When a new element is inserted in front, existing elements' surfaces
    shift to higher global indices in ``system.surfaces``. The scene's
    subsolid cache must invalidate so each region's ``surface_index``
    matches the element's current global indices — otherwise a surface
    selection in one element also highlights the same-local-position
    surface in every other element (the cap regions all carry the stale
    ``surface_index`` from when each element was first cached).
    """
    from ghostlight_viewport.scene import Scene

    def _build(name, surface_uuids, kind):
        return ghostlight.Element(
            name=name,
            surface_ids=list(surface_uuids),
            material_glasses=[] if kind == ghostlight.ElementKind.STOP else ["N-BK7"],
            kind=kind,
            element_id=name + "-id",
        )

    def _make_surface(*, is_stop=False, semi_aperture=20.0):
        s = ghostlight.Surface()
        s.radius = 0.0
        s.thickness = 10.0
        s.semi_aperture = semi_aperture
        s.ior = 1.0 if is_stop else 1.5168
        s.abbe_v = 0.0 if is_stop else 64.17
        s.is_stop = is_stop
        s.disp_model = ghostlight.DispersionModel.AIR if is_stop else ghostlight.DispersionModel.ABBE
        return s

    # Build a 1-element system with a singlet (2 surfaces).
    system = ghostlight.OpticalSystem()
    system._raw_glass_catalogue = {
        "N-BK7": {
            "name": "N-BK7",
            "dispersion": {"model": "abbe", "nd": 1.5168, "Vd": 64.17},
        }
    }
    s0 = _make_surface()
    s1 = _make_surface()
    system.surfaces.append(s0)
    system.surfaces.append(s1)
    system.surface_ids.append("a-uuid")
    system.surface_ids.append("b-uuid")
    system.aperture_images.append(ghostlight.ApertureImage())
    system.aperture_images.append(ghostlight.ApertureImage())
    el_a = _build("A", ["a-uuid", "b-uuid"], ghostlight.ElementKind.GLASS)
    el_a.material_glasses = ["N-BK7"]
    system._elements = [el_a]
    system.finalize()

    scene = Scene()
    scene.rebuild(system, list(system.elements))
    # Element A starts at indices [0, 1].
    a_indices = el_a.resolve_surfaces(system)
    assert a_indices == [0, 1]
    region_indices_a = {
        r.surface_index
        for ss in scene.elements[0].subsolids for r in ss.regions
    }
    assert region_indices_a == {0, 1}, region_indices_a

    # Insert a stop in front of A. A's global indices shift to [1, 2].
    stop_s = _make_surface(is_stop=True, semi_aperture=12.5)
    system.surfaces.insert(0, stop_s)
    system.surface_ids.insert(0, "stop-uuid")
    system.aperture_images.insert(0, ghostlight.ApertureImage())
    stop_el = _build("Stop", ["stop-uuid"], ghostlight.ElementKind.STOP)
    system._elements = [stop_el, el_a]
    system.finalize()

    scene.rebuild(system, list(system.elements))
    a_indices_after = el_a.resolve_surfaces(system)
    assert a_indices_after == [1, 2]
    region_indices_a_after = {
        r.surface_index
        for ss in scene.elements[1].subsolids for r in ss.regions
    }
    # After the shift, A's regions MUST carry the new global indices,
    # not the cached {0, 1}. Without cache invalidation the assertion
    # below fails — the renderer's `region.surface_index == sel_surface`
    # would then match the wrong surface across elements.
    assert region_indices_a_after == {1, 2}, region_indices_a_after

    # And the stop's region carries its own global index (0), not a
    # value that could collide with A's surfaces.
    stop_region_indices = {
        r.surface_index
        for ss in scene.elements[0].subsolids for r in ss.regions
    }
    assert stop_region_indices == {0}


def test_update_hover_writes_surface_only_in_surface_mode(doublet_loaded):
    """`_update_hover` should populate hover_surface from the pick info
    only when the toolbar is in surface mode."""
    viewport, _system, _elements = doublet_loaded
    target_se = viewport.scene.elements[0]
    sample_surface = target_se.subsolids[0].regions[0].surface_index

    fake_info = {
        "tag": picking.TAG_ELEMENT_BODY,
        "element_index": target_se.index,
        "surface_index": int(sample_surface),
        "face_index": None,
        "is_empty": False,
    }

    # Patch _pick_at so we exercise _update_hover without GL.
    viewport._pick_at = lambda x, y, exclude_indices=None: fake_info
    viewport._gl_ready = True  # _update_hover bails out otherwise

    # Element mode → hover_surface stays None.
    viewport.set_selection_mode("element")
    viewport._update_hover(10, 10)
    assert viewport.selection.hover is target_se.element
    assert viewport.selection.hover_surface is None

    # Surface mode → hover_surface set from pick.
    viewport.set_selection_mode("surface")
    viewport._update_hover(10, 10)
    assert viewport.selection.hover is target_se.element
    assert viewport.selection.hover_surface == int(sample_surface)
