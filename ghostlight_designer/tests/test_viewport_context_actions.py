"""Tests for the viewport right-click context popup: new element/surface
actions, the thick-singlet focus/bend math, the scrub-row undo coalescing,
and popup population / gating."""
from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QWidget

import ghostlight

from ghostlight_designer.project import Project
from ghostlight_designer.optical_editor import element_actions as ea
from ghostlight_designer.optical_editor import surface_actions as sa
from ghostlight_designer.viewport_panel.context_popup import (
    ViewportContextPopup,
    _ActionRow,
)
from ghostlight_designer.viewport_panel.scrub_row import ScrubRow, ScrubRowSpec


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _curved_singlet(project: Project, r1=50.0, r2=-50.0, t=5.0, after=None):
    el = ea.add_singlet(project, after=after)
    idx = el.resolve_surfaces(project.system)
    project.system.surfaces[idx[0]].radius = r1
    project.system.surfaces[idx[1]].radius = r2
    project.system.surfaces[idx[0]].thickness = t
    project.system.finalize()
    return el


# ---------------------------------------------------------------------------
# Pure focus / bend math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,t,r1,r2", [
    (1.5168, 5.0, 50.0, -50.0),
    (1.6727, 8.0, 40.0, -120.0),
    (1.5168, 3.0, 80.0, 200.0),   # meniscus
])
def test_singlet_power_shape_solve_roundtrip(n, t, r1, r2):
    phi = ea.singlet_power(n, t, r1, r2)
    x = ea.singlet_shape(r1, r2)
    assert x is not None
    solved = ea.solve_singlet_radii(n, t, phi, x, current_d=(1.0 / r1 - 1.0 / r2))
    assert solved is not None
    sr1, sr2 = solved
    assert ea.singlet_power(n, t, sr1, sr2) == pytest.approx(phi, rel=1e-6, abs=1e-9)
    assert ea.singlet_shape(sr1, sr2) == pytest.approx(x, rel=1e-6, abs=1e-9)


def test_shape_plus_one_gives_flat_rear():
    solved = ea.solve_singlet_radii(1.5168, 5.0, 1.0 / 80.0, 1.0, current_d=0.02)
    assert solved is not None
    r1, r2 = solved
    assert r2 == 0.0
    assert r1 != 0.0


def test_shape_minus_one_gives_flat_front():
    solved = ea.solve_singlet_radii(1.5168, 5.0, 1.0 / 80.0, -1.0, current_d=-0.02)
    assert solved is not None
    r1, r2 = solved
    assert r1 == 0.0
    assert r2 != 0.0


def test_unreachable_power_returns_none():
    # An absurd power at a high-curvature shape can be past the discriminant.
    assert ea.solve_singlet_radii(1.5, 5.0, 1e9, 0.0, 0.0) is None


def test_flat_flat_shape_is_none():
    assert ea.singlet_shape(0.0, 0.0) is None
    assert ea.singlet_power(1.5, 5.0, 0.0, 0.0) == pytest.approx(0.0)


def test_root_continuity_picks_near_branch():
    # Two solutions exist for a thick lens; perturbing power slightly should
    # keep the solved d near the previous one, not jump to the far branch.
    n, t, x = 1.5168, 8.0, 0.0
    r1, r2 = ea.solve_singlet_radii(n, t, 1.0 / 60.0, x, 0.0)
    d0 = (0.0 if r1 == 0 else 1.0 / r1) - (0.0 if r2 == 0 else 1.0 / r2)
    r1b, r2b = ea.solve_singlet_radii(n, t, 1.0 / 60.5, x, d0)
    d1 = (0.0 if r1b == 0 else 1.0 / r1b) - (0.0 if r2b == 0 else 1.0 / r2b)
    assert abs(d1 - d0) < 0.01


# ---------------------------------------------------------------------------
# set_element_power / set_element_shape + EFL cross-check
# ---------------------------------------------------------------------------


def test_set_element_power_holds_shape(qapp):
    p = Project()
    el = _curved_singlet(p)
    x_before = ea.element_shape_factor(p.system, el)
    assert ea.set_element_power(p, el, 1.0 / 80.0) is True
    assert 1.0 / ea.element_power(p.system, el) == pytest.approx(80.0, rel=1e-6)
    assert ea.element_shape_factor(p.system, el) == pytest.approx(x_before, abs=1e-6)


def test_set_element_shape_holds_power(qapp):
    p = Project()
    el = _curved_singlet(p)
    phi_before = ea.element_power(p.system, el)
    assert ea.set_element_shape(p, el, 1.0) is True
    assert ea.element_power(p.system, el) == pytest.approx(phi_before, abs=1e-9)
    idx = el.resolve_surfaces(p.system)
    assert p.system.surfaces[idx[1]].radius == 0.0  # X=+1 → flat rear


def test_set_element_bend_scrubs_front_only(qapp):
    # Bend sets the front curvature and leaves the back fixed; focal length
    # is free to change.
    p = Project()
    el = _curved_singlet(p)
    idx = el.resolve_surfaces(p.system)
    r2_before = p.system.surfaces[idx[1]].radius
    for c1 in (0.04, 0.0, -0.01):
        assert ea.set_element_bend(p, el, c1) is True
        assert ea.element_front_curvature(p.system, el) == pytest.approx(c1, abs=1e-9)
        assert p.system.surfaces[idx[1]].radius == r2_before   # back untouched


def test_set_element_bend_works_on_flat_flat(qapp):
    # The whole point: a flat-flat singlet has no shape factor, but bending
    # by front curvature still works — it becomes plano-convex with power.
    p = Project()
    el = ea.add_singlet(p)   # radii [0, 0]
    assert ea.element_shape_factor(p.system, el) is None      # undefined
    assert ea.element_front_curvature(p.system, el) == 0.0    # defined
    assert ea.set_element_bend(p, el, 0.02) is True
    idx = el.resolve_surfaces(p.system)
    r1, r2 = p.system.surfaces[idx[0]].radius, p.system.surfaces[idx[1]].radius
    assert r1 == pytest.approx(50.0)               # 1/0.02
    assert r2 == 0.0                               # back stays flat (plano-convex)
    assert ea.element_power(p.system, el) > 0.0    # gained power


def test_set_element_bend_back_to_flat_is_clean(qapp):
    p = Project()
    el = ea.add_singlet(p)
    ea.set_element_bend(p, el, 0.03)
    ea.set_element_bend(p, el, 0.0)
    idx = el.resolve_surfaces(p.system)
    assert p.system.surfaces[idx[0]].radius == 0.0   # snapped, not 3e7


def test_set_element_bend_rejects_non_singlet(qapp):
    p = Project()
    doub = ea.add_doublet(p)
    assert ea.set_element_bend(p, doub, 0.02) is False
    assert ea.element_front_curvature(p.system, doub) is None


def test_element_power_matches_paraxial_tracer(qapp):
    from ghostlight_designer import lens_metrics as lm
    p = Project()
    el = _curved_singlet(p, r1=50.0, r2=-50.0, t=5.0)
    my_efl = 1.0 / ea.element_power(p.system, el)
    efl_axis = lm._effective_focal_length_on_axis(p.system, "y")
    assert my_efl == pytest.approx(efl_axis, rel=1e-3)


def test_power_shape_setters_reject_non_singlet(qapp):
    p = Project()
    doub = ea.add_doublet(p)
    assert ea.set_element_power(p, doub, 1.0 / 80.0) is False
    assert ea.set_element_shape(p, doub, 0.0) is False
    assert ea.element_power(p.system, doub) is None
    assert ea.element_shape_factor(p.system, doub) is None


# ---------------------------------------------------------------------------
# convert_to_doublet
# ---------------------------------------------------------------------------


def test_convert_to_doublet(qapp):
    p = Project()
    el = _curved_singlet(p, r1=40.0, r2=-60.0, t=8.0)
    idx = el.resolve_surfaces(p.system)
    front_r, back_r = p.system.surfaces[idx[0]].radius, p.system.surfaces[idx[1]].radius
    back_z = p.system.surfaces[idx[1]].z

    assert ea.convert_to_doublet(p, el) is True
    assert len(el.surface_ids) == 3
    assert el.material_glasses == [ea.DEFAULT_GLASS_KEY, ea.DEFAULT_FLINT_KEY]
    idx2 = el.resolve_surfaces(p.system)
    assert p.system.surfaces[idx2[0]].radius == front_r
    assert p.system.surfaces[idx2[-1]].radius == back_r
    assert p.system.surfaces[idx2[1]].radius == 0.0            # flat interface
    assert p.system.surfaces[idx2[-1]].z == pytest.approx(back_z)
    # Parallel vectors stay aligned.
    assert len(p.system.surfaces) == len(p.system.surface_ids) == len(p.system.aperture_images)


def test_convert_to_doublet_rejects_doublet_and_stop(qapp):
    p = Project()
    doub = ea.add_doublet(p)
    stop = ea.add_aperture_stop(p)
    assert ea.convert_to_doublet(p, doub) is False
    assert ea.convert_to_doublet(p, stop) is False


def test_convert_to_doublet_undo(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    el = _curved_singlet(p, after=p.system.elements[0])
    n_surf = len(p.system.surfaces)
    ea.convert_to_doublet(p, el)
    assert len(p.system.surfaces) == n_surf + 1
    p.undo()
    assert len(p.system.surfaces) == n_surf


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_with_next_cements_two_singlets(qapp):
    p = Project()
    a = _curved_singlet(p, r1=30.0, r2=-30.0)
    _curved_singlet(p, r1=25.0, r2=-25.0, after=a)
    front = p.system.elements[0]
    n_elem, n_surf = len(p.system.elements), len(p.system.surfaces)
    downstream_z = p.system.surfaces[-1].z

    assert ea.merge_with_next(p, front) is True
    assert len(p.system.elements) == n_elem - 1
    assert len(p.system.surfaces) == n_surf - 1
    merged = p.system.elements[0]
    assert len(merged.surface_ids) == 3
    assert len(merged.material_glasses) == 2
    assert p.system.surfaces[-1].z == pytest.approx(downstream_z)
    assert len(p.system.surface_ids) == len(p.system.surfaces)


def test_merge_single_undo_entry(qapp):
    p = Project()
    a = _curved_singlet(p)
    _curved_singlet(p, after=a)
    n_undo = len(p._undo)
    front = p.system.elements[0]
    ea.merge_with_next(p, front)
    assert len(p._undo) == n_undo + 1
    p.undo()
    assert len(p.system.elements) == 2


def test_merge_rejects_stop_neighbour_and_edges(qapp):
    p = Project()
    a = _curved_singlet(p)
    stop = ea.add_aperture_stop(p, after=a)   # stop behind the singlet
    # a is front element; next is the stop → refused.
    assert ea.can_merge_with_next(p, p.system.elements[0]) is False
    assert ea.merge_with_next(p, p.system.elements[0]) is False
    # first element has no previous.
    assert ea.can_merge_with_previous(p, p.system.elements[0]) is False
    assert ea.merge_with_previous(p, p.system.elements[0]) is False


# ---------------------------------------------------------------------------
# move_element_z / offset_element_aperture
# ---------------------------------------------------------------------------


def test_move_element_z_shifts_only_that_element(qapp):
    p = Project()
    a = _curved_singlet(p)
    mid = _curved_singlet(p, after=a)   # element between a and (nothing behind)
    idx = mid.resolve_surfaces(p.system)
    z_before = [float(s.z) for s in p.system.surfaces]
    ea.move_element_z(p, mid, float(p.system.surfaces[idx[0]].z) + 3.0)
    z_after = [float(s.z) for s in p.system.surfaces]
    for k in range(len(z_before)):
        expected = z_before[k] + (3.0 if k in idx else 0.0)
        assert z_after[k] == pytest.approx(expected, abs=1e-6)


def test_offset_element_aperture_is_additive(qapp):
    p = Project()
    el = _curved_singlet(p)
    idx = el.resolve_surfaces(p.system)
    p.system.surfaces[idx[0]].semi_aperture = 20.0
    p.system.surfaces[idx[1]].semi_aperture = 18.0
    ea.offset_element_aperture(p, el, 22.0)     # +2 to the first → +2 to all
    assert p.system.surfaces[idx[0]].semi_aperture == pytest.approx(22.0)
    assert p.system.surfaces[idx[1]].semi_aperture == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# surface actions
# ---------------------------------------------------------------------------


def test_set_surface_radius_and_thickness(qapp):
    p = Project()
    el = _curved_singlet(p)
    si = el.resolve_surfaces(p.system)[0]
    assert sa.set_surface_radius(p, si, 15.0) is True
    assert p.system.surfaces[si].radius == 15.0
    assert sa.set_surface_radius(p, si, 15.0) is False   # no-op
    t0 = float(p.system.surfaces[si].thickness)
    assert sa.set_surface_thickness(p, si, t0 + 1.5) is True
    assert p.system.surfaces[si].thickness == pytest.approx(t0 + 1.5)


def test_set_surface_radius_rejects_stop(qapp):
    p = Project()
    stop = ea.add_aperture_stop(p)
    si = stop.resolve_surfaces(p.system)[0]
    assert sa.set_surface_radius(p, si, 5.0) is False


# ---------------------------------------------------------------------------
# ScrubRow drag → single undo entry
# ---------------------------------------------------------------------------


def test_scrub_row_coalesces_one_undo_entry(qapp):
    p = Project()
    el = _curved_singlet(p)
    idx = el.resolve_surfaces(p.system)
    spec = ScrubRowSpec(
        label="Move Z",
        get=lambda: float(p.system.surfaces[idx[0]].z),
        set=lambda z: ea.move_element_z(p, el, z),
        undo_label="Move Element Z",
    )
    row = ScrubRow(spec, p)
    try:
        n_undo = len(p._undo)
        assert row.begin_scrub(QPoint(0, 0)) is True
        for _ in range(3):
            row.update_scrub(QPoint(10, 0), Qt.NoModifier)
        row.end_scrub()
        assert len(p._undo) == n_undo + 1
    finally:
        row.deleteLater()


def test_scrub_row_press_without_move_no_undo(qapp):
    p = Project()
    el = _curved_singlet(p)
    idx = el.resolve_surfaces(p.system)
    spec = ScrubRowSpec(
        label="Move Z",
        get=lambda: float(p.system.surfaces[idx[0]].z),
        set=lambda z: ea.move_element_z(p, el, z),
    )
    row = ScrubRow(spec, p)
    try:
        n_undo = len(p._undo)
        row.begin_scrub(QPoint(0, 0))
        row.end_scrub()   # never moved
        assert len(p._undo) == n_undo
    finally:
        row.deleteLater()


def test_scrub_row_disabled_when_get_none(qapp):
    p = Project()
    spec = ScrubRowSpec(label="X", get=lambda: None, set=lambda v: None)
    row = ScrubRow(spec, p)
    try:
        assert row.is_scrubbable() is False
        assert row.begin_scrub(QPoint(0, 0)) is False
    finally:
        row.deleteLater()


# ---------------------------------------------------------------------------
# Popup population / gating
# ---------------------------------------------------------------------------


def _labels(popup):
    return {r._text.text(): r.row_enabled for r in popup._action_rows}


def _make_popup(qapp, project, mode, element, surface_index=None):
    parent = QWidget()
    idx = list(project.system.elements).index(element)
    info = {
        "mode": mode,
        "element_index": idx,
        "element": element,
        "surface_index": surface_index,
        "global_pos": QPoint(100, 100),
    }
    return ViewportContextPopup(project, None, parent, info), parent


def test_popup_element_singlet_enables_to_doublet(qapp):
    p = Project()
    el = _curved_singlet(p)
    popup, parent = _make_popup(qapp, p, "element", el)
    try:
        labels = _labels(popup)
        assert labels.get("To Doublet") is True
        assert labels.get("Flip Element") is True
        # Singlet-only scrub rows present: Focus + Bend + Move Z + Aperture.
        scrub_labels = {r._spec.label for r in popup._scrub_rows}
        assert {"Focus", "Bend", "Move Z", "Aperture"} <= scrub_labels
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_bend_row_enabled_on_flat_flat_singlet(qapp):
    p = Project()
    el = ea.add_singlet(p)   # flat-flat: shape factor undefined
    popup, parent = _make_popup(qapp, p, "element", el)
    try:
        bend = next(r for r in popup._scrub_rows if r._spec.label == "Bend")
        # Previously disabled (shape factor None); now scrubbable via curvature.
        assert bend.is_scrubbable() is True
        assert bend.begin_scrub(QPoint(0, 0)) is True
        bend.end_scrub()
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_doublet_disables_to_doublet(qapp):
    p = Project()
    doub = ea.add_doublet(p)
    idx = doub.resolve_surfaces(p.system)
    for k, si in enumerate(idx):
        p.system.surfaces[si].radius = 40.0 if k == 0 else -40.0
    p.system.finalize()
    popup, parent = _make_popup(qapp, p, "element", doub)
    try:
        labels = _labels(popup)
        assert labels.get("To Doublet") is False
        # Doublet has no Focus/Bend rows (singlet-only).
        scrub_labels = {r._spec.label for r in popup._scrub_rows}
        assert "Focus" not in scrub_labels
        assert "Bend" not in scrub_labels
        assert "Move Z" in scrub_labels
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_merge_prev_disabled_at_first_element(qapp):
    p = Project()
    a = _curved_singlet(p)
    _curved_singlet(p, after=a)
    first = p.system.elements[0]
    popup, parent = _make_popup(qapp, p, "element", first)
    try:
        labels = _labels(popup)
        assert labels.get("Merge Prev.") is False
        assert labels.get("Merge Next") is True
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_surface_mode_rows(qapp):
    p = Project()
    el = _curved_singlet(p)
    si = el.resolve_surfaces(p.system)[0]
    popup, parent = _make_popup(qapp, p, "surface", el, surface_index=si)
    try:
        # Ghost-solo toggle + the two scrub rows.
        assert _labels(popup) == {"Solo Ghosts": True}
        scrub_labels = {r._spec.label for r in popup._scrub_rows}
        assert scrub_labels == {"Pos Z", "Radius"}
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_surface_ghost_solo_toggles(qapp):
    p = Project()
    el = _curved_singlet(p)
    si = el.resolve_surfaces(p.system)[0]
    uuid = str(p.system.surface_ids[si])
    popup, parent = _make_popup(qapp, p, "surface", el, surface_index=si)
    try:
        solo_row = popup._action_rows[0]
        assert p.is_surface_ghost_solo(uuid) is False
        solo_row.on_click()                       # toggle on
        assert p.is_surface_ghost_solo(uuid) is True
        assert solo_row._text.text() == "Un-Solo Ghosts"
        solo_row.on_click()                       # toggle off
        assert p.is_surface_ghost_solo(uuid) is False
        assert solo_row._text.text() == "Solo Ghosts"
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_surface_stop_hides_ghost_solo(qapp):
    p = Project()
    stop = ea.add_aperture_stop(p)
    si = stop.resolve_surfaces(p.system)[0]
    popup, parent = _make_popup(qapp, p, "surface", stop, surface_index=si)
    try:
        # Stop surfaces don't offer ghost-solo (matches the ODE menu).
        assert len(popup._action_rows) == 0
    finally:
        popup.close()
        parent.deleteLater()


def test_popup_closes_on_system_replaced(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    el = p.system.elements[0]
    popup, parent = _make_popup(qapp, p, "element", el)
    popup.show()
    assert popup.isVisible()
    # A reload (undo/redo/new/load) swaps the system → popup must dismiss.
    p.load(str(sample_lens_path))
    assert not popup.isVisible()
    parent.deleteLater()


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


def test_focus_unit_setting_roundtrip(isolated_settings):
    assert isolated_settings.viewport_focus_unit() == "mm"      # default
    isolated_settings.set_viewport_focus_unit("dpt")
    assert isolated_settings.viewport_focus_unit() == "dpt"
    isolated_settings.set_viewport_focus_unit("bogus")          # ignored
    assert isolated_settings.viewport_focus_unit() == "dpt"
