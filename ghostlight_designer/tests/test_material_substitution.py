"""Tests for the catalogue-hammer material-substitution feature (Slice 1).

Covers:

* Project storage — SubstitutionSpec + set / clear / toggle / getters,
  signal fan-out, cleared on new / load, pruned when an element or
  material index disappears.
* material_substitution helpers — vendor enumeration, hull, candidate
  filtering by spec, nearest-glass lookup.
* Optimization panel tree — MaterialSubstitutionEntryNode appears under
  the Variables header, renders element / vendor / current glass, is
  read-only, and Delete + right-click Unflag drop the flag.
* ODE Material row right-click submenu — vendor items with radio-check
  reflecting the current spec + Off entry to clear.
"""
from __future__ import annotations

import pathlib

import pytest

from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt
from PySide6.QtWidgets import QMenu

import ghostlight

from ghostlight_designer.material_catalogue import (
    CatalogueMaterial,
    MaterialCatalogue,
)
from ghostlight_designer.material_substitution import (
    VendorHull,
    candidates_for_vendor,
    candidates_within_spec,
    nearest_glass,
    vendor_hull,
    vendors_in_catalogue,
)
from ghostlight_designer.optical_editor.body import OpticalEditorBody
from ghostlight_designer.optical_editor.nodes import MaterialNode
from ghostlight_designer.optimization_panel.body import OptimizationPanelBody
from ghostlight_designer.optimization_panel.columns import Column
from ghostlight_designer.optimization_panel.nodes import (
    MaterialSubstitutionEntryNode,
    VariablesHeaderNode,
)
from ghostlight_designer.project import Project, SubstitutionSpec


# ---------------------------------------------------------------------------
# SubstitutionSpec
# ---------------------------------------------------------------------------


def test_substitution_spec_defaults():
    s = SubstitutionSpec()
    assert s.vendor == ""
    assert s.nd_lo is None and s.nd_hi is None
    assert s.vd_lo is None and s.vd_hi is None


def test_substitution_spec_equality():
    assert SubstitutionSpec(vendor="Schott") == SubstitutionSpec(vendor="Schott")
    assert SubstitutionSpec(vendor="Schott") != SubstitutionSpec(vendor="Ohara")


# ---------------------------------------------------------------------------
# Project storage API
# ---------------------------------------------------------------------------


def test_project_starts_with_no_material_flags(qapp):
    project = Project()
    assert project.all_material_flags() == {}
    assert project.get_material_flags("anything") == {}
    assert project.is_material_flagged("anything", 0) is False
    assert project.material_flag_spec("anything", 0) is None


def test_set_get_clear_material_flag(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id

    spec = SubstitutionSpec(vendor="Schott")
    assert project.set_material_flag(eid, 0, spec) is True  # changed
    assert project.set_material_flag(eid, 0, spec) is False  # no-op
    assert project.is_material_flagged(eid, 0)
    assert project.material_flag_spec(eid, 0) == spec

    # Overwrite with different vendor.
    other = SubstitutionSpec(vendor="Ohara")
    assert project.set_material_flag(eid, 0, other) is True
    assert project.material_flag_spec(eid, 0) == other

    assert project.clear_material_flag(eid, 0) is True
    assert not project.is_material_flagged(eid, 0)
    assert project.clear_material_flag(eid, 0) is False


def test_toggle_material_flag(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id

    assert project.toggle_material_flag(
        eid, 0, default_spec=SubstitutionSpec(vendor="Schott"),
    ) is True
    assert project.is_material_flagged(eid, 0)
    assert project.toggle_material_flag(eid, 0) is False
    assert not project.is_material_flagged(eid, 0)


def test_material_flag_signals_fire(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    changes: list = []
    project.materialFlagChanged.connect(
        lambda e, i: changes.append(("changed", e, i))
    )
    project.materialFlagsReplaced.connect(
        lambda: changes.append(("replaced",))
    )
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    project.clear_material_flag(eid, 0)
    assert changes == [
        ("changed", eid, 0),
        ("changed", eid, 0),
    ]


def test_new_clears_material_flags(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    replaced: list = []
    project.materialFlagsReplaced.connect(lambda: replaced.append(True))
    project.new()
    assert project.all_material_flags() == {}
    assert replaced  # signal fired


def test_clear_all_material_flags(qapp, sample_lens_path: pathlib.Path):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    project.set_material_flag(eid, 1, SubstitutionSpec(vendor="Ohara"))
    replaced: list = []
    project.materialFlagsReplaced.connect(lambda: replaced.append(True))
    assert project.clear_all_material_flags() is True
    assert project.all_material_flags() == {}
    assert replaced
    replaced.clear()
    assert project.clear_all_material_flags() is False
    assert not replaced


def test_material_flags_pruned_when_element_disappears(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    elements = list(project.system.elements)
    if not elements:
        pytest.skip("Sample lens has no elements to remove")
    eid = elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    # Remove the flagged element — pruning must drop the entry.
    from ghostlight_designer.optical_editor import element_actions
    element_actions.remove_element(project, elements[0])
    live_ids = {getattr(e, "element_id", None) for e in project.system.elements}
    for e_id in project.all_material_flags():
        assert e_id in live_ids, f"stale material flag on removed element {e_id}"


def test_material_flag_ignored_for_bad_inputs(qapp):
    project = Project()
    assert project.set_material_flag("", 0, SubstitutionSpec()) is False
    assert project.set_material_flag("some-id", -1, SubstitutionSpec()) is False
    assert project.clear_material_flag("", 0) is False


# ---------------------------------------------------------------------------
# material_substitution helpers
# ---------------------------------------------------------------------------


def _fake_material(
    key: str, vendor: str, nd: float, vd: float,
) -> CatalogueMaterial:
    """Minimal CatalogueMaterial suitable for hull / distance tests."""
    return CatalogueMaterial(
        key=key,
        display_name=key,
        catalogue_ref=f"{vendor}:{key}",
        source_vendor=vendor,
        tags=(),
        dispersion={"model": "abbe", "nd": nd, "Vd": vd},
        nd=nd,
        vd=vd,
        wavelength_range_nm=None,
        glass_code="",
        glass_status="",
        density_g_cm3=None,
        description="",
        comments="",
        references="",
        source_file="",
        raw_source=None,
    )


def _mini_catalogue() -> MaterialCatalogue:
    return MaterialCatalogue([
        _fake_material("N-BK7", "Schott", 1.5168, 64.17),
        _fake_material("SF5",   "Schott", 1.6727, 32.25),
        _fake_material("N-SF6", "Schott", 1.8052, 25.36),
        _fake_material("S-BSL7", "Ohara", 1.5163, 64.14),
        _fake_material("S-TIH53", "Ohara", 1.8467, 23.79),
    ])


def test_vendors_in_catalogue_returns_sorted_unique():
    cat = _mini_catalogue()
    assert vendors_in_catalogue(cat) == ["Ohara", "Schott"]


def test_candidates_for_vendor_filters_by_source_vendor():
    cat = _mini_catalogue()
    schott = candidates_for_vendor(cat, "Schott")
    assert {m.key for m in schott} == {"N-BK7", "SF5", "N-SF6"}
    assert candidates_for_vendor(cat, "Unknown") == []
    assert candidates_for_vendor(cat, "") == []


def test_vendor_hull_axis_aligned_bounding_box():
    cat = _mini_catalogue()
    hull = vendor_hull(cat, "Schott")
    assert isinstance(hull, VendorHull)
    assert hull.count == 3
    assert hull.nd_min == pytest.approx(1.5168)
    assert hull.nd_max == pytest.approx(1.8052)
    assert hull.vd_min == pytest.approx(25.36)
    assert hull.vd_max == pytest.approx(64.17)


def test_vendor_hull_empty_vendor_flags_unusable():
    cat = _mini_catalogue()
    hull = vendor_hull(cat, "NoSuchVendor")
    assert hull.is_empty()
    assert hull.nd_min is None and hull.nd_max is None


def test_nearest_glass_finds_closest_by_nd_vd():
    cat = _mini_catalogue()
    # Ask for something very close to N-BK7 (1.5168, 64.17).
    m = nearest_glass(cat, 1.517, 64.0)
    assert m is not None
    assert m.key in ("N-BK7", "S-BSL7")  # essentially the same glass


def test_nearest_glass_scoped_to_vendor():
    cat = _mini_catalogue()
    m = nearest_glass(cat, 1.517, 64.0, vendor="Ohara")
    assert m is not None
    assert m.source_vendor == "Ohara"
    assert m.key == "S-BSL7"


def test_candidates_within_spec_uses_hull_when_bounds_unset():
    cat = _mini_catalogue()
    spec = SubstitutionSpec(vendor="Schott")
    matches = candidates_within_spec(cat, spec)
    assert {m.key for m in matches} == {"N-BK7", "SF5", "N-SF6"}


def test_candidates_within_spec_respects_nd_bounds():
    cat = _mini_catalogue()
    spec = SubstitutionSpec(vendor="Schott", nd_lo=1.6, nd_hi=1.75)
    matches = candidates_within_spec(cat, spec)
    # Only SF5 (nd=1.6727) fits between 1.6 and 1.75.
    assert {m.key for m in matches} == {"SF5"}


# ---------------------------------------------------------------------------
# Optimization panel tree — material entries under Variables header
# ---------------------------------------------------------------------------


def test_optimization_panel_material_entry_appears_on_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OptimizationPanelBody(project)
    try:
        eid = project.system.elements[0].element_id
        project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))

        header_idx = body.model.index(0, 0, QModelIndex())
        assert isinstance(header_idx.internalPointer(), VariablesHeaderNode)
        assert "Variables" in str(body.model.data(header_idx))
        assert "(1)" in str(body.model.data(header_idx))

        entry_idx = body.model.index(0, 0, header_idx)
        node = entry_idx.internalPointer()
        assert isinstance(node, MaterialSubstitutionEntryNode)
        assert node.element_id == eid
        assert node.material_index == 0

        # NAME cell renders "Element N · material M".
        name = str(body.model.data(entry_idx))
        assert "Element 0" in name
        assert "material 0" in name

        # TYPE column names the target vendor.
        type_idx = body.model.index(0, int(Column.TYPE), header_idx)
        assert body.model.data(type_idx) == "Substitute · Schott"

        # VALUE column shows the current glass on the lens.
        value_idx = body.model.index(0, int(Column.VALUE), header_idx)
        assert body.model.data(value_idx) == "N-BK7"
    finally:
        body.deleteLater()


def test_optimization_panel_material_entry_is_read_only(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    body = OptimizationPanelBody(project)
    try:
        header_idx = body.model.index(0, 0, QModelIndex())
        for col in range(body.model.columnCount()):
            idx = body.model.index(0, col, header_idx)
            flags = body.model.flags(idx)
            assert not (flags & Qt.ItemIsEditable), (
                f"material row col {col!r} should not be editable"
            )
    finally:
        body.deleteLater()


def test_optimization_panel_remove_shortcut_unflags_material(
    qapp, sample_lens_path: pathlib.Path,
):
    """Delete key / _on_remove drops the material flag."""
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    body = OptimizationPanelBody(project)
    try:
        header_idx = body.model.index(0, 0, QModelIndex())
        entry_idx = body.model.index(0, 0, header_idx)
        body.tree.selectionModel().setCurrentIndex(
            entry_idx,
            QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
        )
        body._on_remove()
        assert not project.is_material_flagged(eid, 0)
        # Whole section disappears when the last flag is cleared.
        assert body.model.rowCount(QModelIndex()) == 0
    finally:
        body.deleteLater()


def test_optimization_panel_mixed_surface_and_material_flags(
    qapp, sample_lens_path: pathlib.Path,
):
    """Both flag kinds coexist under one header, count reflects both."""
    project = Project()
    project.load(str(sample_lens_path))
    project.set_variable_flag(str(project.system.surface_ids[0]), "radius")
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 1, SubstitutionSpec(vendor="Ohara"))
    body = OptimizationPanelBody(project)
    try:
        header_idx = body.model.index(0, 0, QModelIndex())
        assert body.model.rowCount(header_idx) == 2
        # Surface variables come first, material entries after.
        first = body.model.index(0, 0, header_idx).internalPointer()
        second = body.model.index(1, 0, header_idx).internalPointer()
        from ghostlight_designer.optimization_panel.nodes import VariableEntryNode
        assert isinstance(first, VariableEntryNode)
        assert isinstance(second, MaterialSubstitutionEntryNode)
        # Header count sums both.
        assert "(2)" in str(body.model.data(header_idx))
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# ODE Material row right-click submenu
# ---------------------------------------------------------------------------


def _find_material_node(body: OpticalEditorBody, element_index: int) -> MaterialNode:
    root = body.model._root
    element_node = root.children[element_index]
    for child in element_node.children:
        if isinstance(child, MaterialNode):
            return child
    raise AssertionError("no MaterialNode found on element")


def _submenu(menu: QMenu, title: str) -> QMenu:
    """Look up a submenu by its title.

    ``QAction.menu()`` returns a Python wrapper that shiboken can GC out
    from under us — see feedback-qt-cell-editor-quirks. ``findChildren``
    returns wrappers that hold a strong C++ ref for the caller.
    """
    for sub in menu.findChildren(QMenu):
        if sub.title() == title:
            return sub
    raise AssertionError(f"submenu {title!r} not found in menu")


def test_material_menu_has_substitution_submenu(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    body = OpticalEditorBody(project)
    try:
        node = _find_material_node(body, element_index=0)
        menu = QMenu(body)
        body._populate_material_menu(menu, node)
        sub = _submenu(menu, "Substitute in Optimization")
        actions = sub.actions()
        # First action is "Off"; disabled when nothing's flagged.
        assert actions[0].text() == "Off"
        assert not actions[0].isEnabled()
        # Subsequent items are vendors, all checkable, none checked yet.
        vendor_items = [a for a in actions if a.text() and a.text() != "Off"]
        assert vendor_items, "expected vendor items"
        for act in vendor_items:
            assert act.isCheckable()
            assert not act.isChecked()
        menu.deleteLater()
    finally:
        body.deleteLater()


def test_material_menu_checks_currently_flagged_vendor(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    body = OpticalEditorBody(project)
    try:
        node = _find_material_node(body, element_index=0)
        menu = QMenu(body)
        body._populate_material_menu(menu, node)
        sub = _submenu(menu, "Substitute in Optimization")
        actions = {a.text(): a for a in sub.actions()}
        # Off is now enabled because a spec exists.
        assert actions["Off"].isEnabled()
        # Schott is the checked radio.
        schott = actions.get("Schott")
        assert schott is not None
        assert schott.isChecked()
        menu.deleteLater()
    finally:
        body.deleteLater()


def test_material_menu_off_action_clears_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Schott"))
    body = OpticalEditorBody(project)
    try:
        node = _find_material_node(body, element_index=0)
        menu = QMenu(body)
        body._populate_material_menu(menu, node)
        sub = _submenu(menu, "Substitute in Optimization")
        off = next(a for a in sub.actions() if a.text() == "Off")
        off.trigger()
        assert not project.is_material_flagged(eid, 0)
        menu.deleteLater()
    finally:
        body.deleteLater()


def test_material_menu_vendor_action_sets_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    body = OpticalEditorBody(project)
    try:
        node = _find_material_node(body, element_index=0)
        menu = QMenu(body)
        body._populate_material_menu(menu, node)
        sub = _submenu(menu, "Substitute in Optimization")
        # Grab an arbitrary vendor action and trigger it.
        vendor_act = next(
            a for a in sub.actions() if a.text() and a.text() != "Off"
        )
        vendor_act.trigger()
        spec = project.material_flag_spec(eid, 0)
        assert spec is not None
        assert spec.vendor == vendor_act.text()
        menu.deleteLater()
    finally:
        body.deleteLater()


# ---------------------------------------------------------------------------
# collect_material_flags + install_candidate_glass (hammer variable layer)
# ---------------------------------------------------------------------------


def test_collect_material_flags_returns_refs_in_order(
    qapp, sample_lens_path: pathlib.Path,
):
    from ghostlight_designer.optimization_panel.variables import (
        collect_material_flags,
    )
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(eid, 1, SubstitutionSpec(vendor="Schott"))
    project.set_material_flag(eid, 0, SubstitutionSpec(vendor="Ohara"))
    refs = collect_material_flags(project)
    # Sorted by material_index within an element.
    assert [r.material_index for r in refs] == [0, 1]
    assert refs[0].current_key == "N-BK7"
    assert refs[1].current_key == "SF5"


def test_collect_material_flags_skips_empty_vendor(
    qapp, sample_lens_path: pathlib.Path,
):
    from ghostlight_designer.optimization_panel.variables import (
        collect_material_flags,
    )
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    # Flag with an empty vendor — user opened submenu but never picked.
    project.set_material_flag(eid, 0, SubstitutionSpec())
    assert collect_material_flags(project) == []


def test_install_candidate_glass_updates_element_and_surface(
    qapp, sample_lens_path: pathlib.Path,
):
    from ghostlight_designer.material_catalogue import get_catalogue
    from ghostlight_designer.material_substitution import candidates_for_vendor
    from ghostlight_designer.optimization_panel.variables import (
        MaterialFlagRef,
        install_candidate_glass,
    )
    project = Project()
    project.load(str(sample_lens_path))
    system = project.system
    element = system.elements[0]
    eid = element.element_id
    catalogue = get_catalogue()
    # Pick a specific catalogue entry to install as the swap target.
    candidates = candidates_for_vendor(catalogue, "Schott")
    assert candidates
    glass = candidates[0]
    ref = MaterialFlagRef(
        element_id=eid,
        material_index=0,
        spec=SubstitutionSpec(vendor="Schott"),
        current_key=element.material_glasses[0],
    )
    install_candidate_glass(system, ref, glass)
    # Element's material_glasses updated.
    assert element.material_glasses[0] == glass.key
    # Surface's ior/abbe_v updated to match.
    surface_uuid = element.surface_ids[0]
    idx = list(system.surface_ids).index(surface_uuid)
    surf = system.surfaces[idx]
    assert surf.ior == pytest.approx(float(glass.nd), rel=1e-6)
    assert surf.abbe_v == pytest.approx(float(glass.vd), rel=1e-6)
    # Glass now in the project catalogue for round-trip.
    assert glass.key in system._raw_glass_catalogue


# ---------------------------------------------------------------------------
# Hammer optimizer end-to-end
# ---------------------------------------------------------------------------


def _wait_for_run(qapp, run, timeout_s: float = 45.0):
    """Spin the event loop until ``run.runFinished`` fires."""
    import time
    captured: dict = {}
    run.runFinished.connect(lambda r: captured.setdefault("result", r))
    if run.is_finished and run.result is not None:
        return run.result
    deadline = time.monotonic() + timeout_s
    while "result" not in captured and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    return captured.get("result")


def test_hammer_material_only_run_reports_ok(
    qapp, sample_lens_path: pathlib.Path,
):
    """Material flags with no geometric variables — runs the hammer,
    picks a best glass from the vendor's candidate list, finishes ok.

    Restrict the candidate pool to just a handful via tight nd bounds
    so the run stays under a second on CI.
    """
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer.optimization_panel.data import (
        GoalEntry,
        GoalKind,
        MeritFunction,
    )
    from ghostlight_designer.optimization_panel.optimizer import OptimizationRun

    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    # Very tight nd window: 1.51..1.53 → a handful of Schott crowns.
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=1.51, nd_hi=1.53),
    )

    from ghostlight_designer import lens_metrics as lm
    efl0 = lm._effective_focal_length_on_axis(project.system, "y")
    assert efl0 is not None and efl0 > 0
    mf = MeritFunction.make(
        name="Hammer EFL",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=efl0, weight=1.0)],
        max_hammer_rounds=1,
        hammer_sub_max_nfev=3,
    )
    run = OptimizationRun(project, mf, project.system_setup)
    # The run should use the hammer path.
    assert run.material_flags
    assert len(run.material_flags) == 1

    ticks: list = []
    run.hammerProgress.connect(lambda p: ticks.append(p))
    run.start()
    result = _wait_for_run(qapp, run)
    assert result is not None, "Hammer run did not finish"
    assert result.status == "ok", f"status={result.status} msg={result.message!r}"
    # Progress signals should have fired at least once (candidate ≥ 1).
    assert ticks, "hammerProgress never fired"
    # The virtual system's flagged material should end pointing at some
    # Schott key (may or may not equal N-BK7 — the search may pick the
    # baseline back).
    new_key = run.virtual_system.elements[0].material_glasses[0]
    assert isinstance(new_key, str) and new_key
    # Source system still untouched.
    assert project.system.elements[0].material_glasses[0] == "N-BK7"


def test_hammer_no_vendor_candidates_finishes_cleanly(
    qapp, sample_lens_path: pathlib.Path,
):
    """A spec whose bounds admit no glass reports 0 candidates and
    finishes ok — the flag is a no-op, not a crash."""
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer.optimization_panel.data import (
        GoalEntry,
        GoalKind,
        MeritFunction,
    )
    from ghostlight_designer.optimization_panel.optimizer import OptimizationRun

    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    # Impossible bounds: nd must be > 3.0.
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=3.0, nd_hi=4.0),
    )
    mf = MeritFunction.make(
        name="No-candidate Hammer",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=50.0)],
        max_hammer_rounds=1,
        hammer_sub_max_nfev=3,
    )
    run = OptimizationRun(project, mf, project.system_setup)
    ticks: list = []
    run.hammerProgress.connect(lambda p: ticks.append(p))
    run.start()
    result = _wait_for_run(qapp, run)
    assert result is not None
    assert result.status == "ok"
    # The zero-candidate tick should have emitted for the flag.
    assert any(p.total_candidates == 0 for p in ticks)


def test_hammer_cancel_reports_cancelled_not_failed(
    qapp, sample_lens_path: pathlib.Path,
):
    """Cancelling mid-hammer must resolve as ``cancelled`` — regression
    for the bug where ``_Cancelled`` inside the sub-run propagated to
    ``_Worker.run``'s generic ``BaseException`` catcher and got reported
    as ``failed``.
    """
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer.optimization_panel.data import (
        GoalEntry,
        GoalKind,
        MeritFunction,
    )
    from ghostlight_designer.optimization_panel.optimizer import OptimizationRun

    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    # A handful of Schott crowns → enough candidates for the run to be
    # active when cancel arrives, but small enough to keep the test fast.
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=1.51, nd_hi=1.53),
    )
    mf = MeritFunction.make(
        name="Cancel me",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=50.0)],
        max_hammer_rounds=3,
        hammer_sub_max_nfev=3,
    )
    run = OptimizationRun(project, mf, project.system_setup)
    # Set cancel BEFORE start() so the worker sees it immediately and
    # the test doesn't depend on cross-thread timing.
    run.request_cancel()
    run.start()
    result = _wait_for_run(qapp, run)
    assert result is not None
    assert result.status == "cancelled", (
        f"expected cancelled, got status={result.status} "
        f"msg={result.message!r}"
    )


def test_hammer_updates_preview_during_sweep(
    qapp, sample_lens_path: pathlib.Path,
):
    """`previewUpdated` must fire multiple times during a hammer sweep,
    not just at flag boundaries — otherwise the dialog viewport sits
    stale for the whole sweep (regression for the "viewport not updating
    during hammer" bug).
    """
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer.optimization_panel.data import (
        GoalEntry,
        GoalKind,
        MeritFunction,
    )
    from ghostlight_designer.optimization_panel.optimizer import OptimizationRun

    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=1.51, nd_hi=1.53),
    )
    mf = MeritFunction.make(
        name="Preview sweep",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=50.0)],
        max_hammer_rounds=1,
        hammer_sub_max_nfev=3,
    )
    run = OptimizationRun(project, mf, project.system_setup)
    preview_count: list = []
    run.previewUpdated.connect(lambda: preview_count.append(True))
    run.start()
    result = _wait_for_run(qapp, run)
    assert result is not None
    assert result.status == "ok"
    # Should fire at least once per candidate (a handful for Schott
    # nd ∈ [1.51, 1.53]) plus the per-flag flush + the final commit.
    assert len(preview_count) >= 3, (
        f"previewUpdated fired {len(preview_count)}× — expected several "
        "per-candidate + per-flag + final flushes"
    )


def test_material_flag_alone_triggers_hammer_not_no_variables(
    qapp, sample_lens_path: pathlib.Path,
):
    """Prior to Slice 2 a run with no geometric variables would emit
    ``no_variables``. With material flags now counting as opt-in the
    hammer path takes over instead."""
    pytest.importorskip("scipy.optimize", exc_type=ImportError)
    from ghostlight_designer.optimization_panel.data import (
        GoalEntry,
        GoalKind,
        MeritFunction,
    )
    from ghostlight_designer.optimization_panel.optimizer import OptimizationRun

    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=1.51, nd_hi=1.53),
    )
    mf = MeritFunction.make(
        name="Alone",
        goals=[GoalEntry.make(kind=GoalKind.EFL, target=50.0)],
        max_hammer_rounds=1,
        hammer_sub_max_nfev=3,
    )
    run = OptimizationRun(project, mf, project.system_setup)
    run.start()
    result = _wait_for_run(qapp, run)
    assert result is not None
    assert result.status != "no_variables"


# ---------------------------------------------------------------------------
# Variables header runtime-estimate display
# ---------------------------------------------------------------------------


def test_variables_header_shows_candidate_count_with_material_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    """Header TYPE column augments with candidate count when at least
    one material flag is set — user sees the "this run will search N
    catalogue glasses" cue right where they're looking."""
    project = Project()
    project.load(str(sample_lens_path))
    eid = project.system.elements[0].element_id
    project.set_material_flag(
        eid, 0,
        SubstitutionSpec(vendor="Schott", nd_lo=1.51, nd_hi=1.53),
    )
    body = OptimizationPanelBody(project)
    try:
        # The Variables header sits at root row 0. Its TYPE column is
        # what carries the candidate-count summary.
        header_type_idx = body.model.index(0, int(Column.TYPE), QModelIndex())
        display = str(body.model.data(header_type_idx))
        assert "candidates" in display
        # Tooltip carries the full explanation.
        tooltip = str(body.model.data(header_type_idx, Qt.ToolTipRole))
        assert "Material substitution" in tooltip
        assert "candidates" in tooltip
    finally:
        body.deleteLater()


def test_variables_header_no_candidate_count_without_material_flag(
    qapp, sample_lens_path: pathlib.Path,
):
    """No material flag → header TYPE stays empty (standard path)."""
    project = Project()
    project.load(str(sample_lens_path))
    project.set_variable_flag(str(project.system.surface_ids[0]), "radius")
    body = OptimizationPanelBody(project)
    try:
        header_type_idx = body.model.index(0, int(Column.TYPE), QModelIndex())
        display = str(body.model.data(header_type_idx))
        assert "candidates" not in display
    finally:
        body.deleteLater()
