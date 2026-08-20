"""Unit tests for the ghost-explorer panel.

Covers the Qt-free survey helpers (labelling, culling, peak normalisation)
plus the panel parts that don't need a GPU: the pinned source, the
layers-always-off settings contract, and the scrubber's selection
bookkeeping across a cull toggle.
"""
from __future__ import annotations

import gc

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

import ghostlight

from ghostlight_designer.ghost_explorer_panel import GhostExplorerPanelBody
from ghostlight_designer.ghost_explorer_panel import body as body_mod
from ghostlight_designer.ghost_explorer_panel import ghost_survey as gs
from ghostlight_designer.ghost_explorer_panel.body import SOURCE_X, SOURCE_Y
from ghostlight_designer.ghost_explorer_panel.ghost_survey import GhostEntry
from ghostlight_designer.ghost_explorer_panel.menus import build_menus
from ghostlight_designer.ghost_explorer_panel.type import (
    GHOST_EXPLORER_TYPE_ID,
    register_ghost_explorer_panel_type,
)
from ghostlight_designer.panel_system import registry
from ghostlight_designer.project import Project


def _destroy(body) -> None:
    """Destroy a body deterministically inside the owning test.

    The body's spinboxes carry scrubber plumbing (hidden QTreeView + adapter
    model); letting bodies pile up in the deleteLater queue and die mid-way
    through a *later* test's event processing corrupts the heap on
    Windows/PySide6. See feedback_scrubber_teardown.
    """
    body.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    gc.collect()


def _entry(number: int, rel: float, *, metered: bool = True) -> GhostEntry:
    return GhostEntry(
        number=number, surf_a=number, surf_b=number + 1,
        label_a="a", label_b="b",
        peak=rel, rel=rel, metered=metered,
    )


@pytest.fixture
def loaded_body(qapp, isolated_settings, sample_lens_path):
    project = Project()
    project.load(str(sample_lens_path))
    body = GhostExplorerPanelBody(project, isolated_settings)
    yield project, body
    _destroy(body)


# ---------------------------------------------------------------------------
# Culling
# ---------------------------------------------------------------------------

def test_cull_off_returns_everything():
    entries = [_entry(i, i / 10.0) for i in range(1, 11)]
    assert gs.visible_ghosts(entries, cull=False, rel_threshold=0.5) == entries


def test_cull_drops_entries_below_threshold():
    entries = [_entry(1, 1.0), _entry(2, 0.2), _entry(3, 0.001)]
    kept = gs.visible_ghosts(entries, cull=True, rel_threshold=0.1)
    assert [e.number for e in kept] == [1, 2]


def test_cull_never_hides_unmetered_entries():
    """No measurement means no evidence to hide on."""
    entries = [_entry(1, 1.0), _entry(2, 0.0, metered=False)]
    kept = gs.visible_ghosts(entries, cull=True, rel_threshold=0.5)
    assert [e.number for e in kept] == [1, 2]


def test_cull_keeps_the_brightest_when_everything_would_go():
    """The slider must never end up with nothing to render."""
    entries = [_entry(1, 0.0), _entry(2, 0.0)]
    kept = gs.visible_ghosts(entries, cull=True, rel_threshold=0.5)
    assert len(kept) == 1


def test_cull_threshold_is_clamped():
    entries = [_entry(1, 1.0), _entry(2, 1e-9)]
    # A threshold below CULL_REL_MIN clamps up, so the 1e-9 entry still goes.
    kept = gs.visible_ghosts(entries, cull=True, rel_threshold=0.0)
    assert [e.number for e in kept] == [1]


def test_cull_empty_list():
    assert gs.visible_ghosts([], cull=True) == []


def test_sort_by_brightness_orders_brightest_first():
    entries = [_entry(1, 0.2), _entry(2, 1.0), _entry(3, 0.5)]
    kept = gs.visible_ghosts(entries, cull=False, sort_by_brightness=True)
    assert [e.number for e in kept] == [2, 3, 1]


def test_sort_by_brightness_is_a_no_op_before_metering():
    """Every entry at rel 0.0 — a stable sort must leave survey order alone
    rather than scrambling into an arbitrary one."""
    entries = [_entry(i, 0.0, metered=False) for i in (1, 2, 3)]
    kept = gs.visible_ghosts(entries, cull=False, sort_by_brightness=True)
    assert [e.number for e in kept] == [1, 2, 3]


def test_sort_applies_after_the_cull():
    entries = [_entry(1, 0.2), _entry(2, 1e-9), _entry(3, 1.0)]
    kept = gs.visible_ghosts(
        entries, cull=True, rel_threshold=0.1, sort_by_brightness=True
    )
    assert [e.number for e in kept] == [3, 1]


def test_find_by_number():
    entries = [_entry(3, 1.0), _entry(7, 0.5)]
    assert gs.find_by_number(entries, 7) == 1
    assert gs.find_by_number(entries, 4) is None
    assert gs.find_by_number(entries, None) is None


# ---------------------------------------------------------------------------
# Peak normalisation
# ---------------------------------------------------------------------------

def test_apply_peaks_normalises_to_the_brightest():
    entries = [_entry(1, 0.0, metered=False), _entry(2, 0.0, metered=False)]
    out = gs.apply_peaks(entries, {(1, 2): 4.0, (2, 3): 1.0})
    assert [e.rel for e in out] == [1.0, 0.25]
    assert all(e.metered for e in out)


def test_apply_peaks_treats_a_missing_pair_as_a_measured_zero():
    """``cull_dead_pairs`` drops pairs that reach no pixel — that is a result,
    not a gap, and exactly what the cull is for."""
    entries = [_entry(1, 0.0, metered=False), _entry(2, 0.0, metered=False)]
    out = gs.apply_peaks(entries, {(1, 2): 4.0})
    assert out[1].peak == 0.0 and out[1].metered is True


def test_apply_peaks_with_no_measurements_leaves_entries_unmetered():
    entries = [_entry(1, 0.0, metered=False)]
    assert gs.apply_peaks(entries, {}) == entries


@pytest.mark.parametrize(
    "key, expected",
    [
        ("ghost_s0_s4_r", (0, 4)),
        ("ghost_s12_s31_r", (12, 31)),
        ("ghost_bogus_r", None),
        ("ghost_r", None),
    ],
)
def test_parse_aov_key(key, expected):
    assert gs._parse_aov_key(key) == expected


# ---------------------------------------------------------------------------
# Surface labelling
# ---------------------------------------------------------------------------

def test_surface_labels_use_element_names(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    labels = gs.build_surface_labels(system)
    assert len(labels) == system.num_surfaces()
    # Every surface of the example doublet belongs to a named element, so none
    # should be left on the generic fallback.
    assert not any(lbl.startswith("surface ") for lbl in labels)


def test_surface_labels_name_a_two_surface_element_front_and_rear(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    labels = gs.build_surface_labels(system)
    two_surface = [el for el in system.elements if len(el.surface_ids) == 2]
    assert two_surface, "example lens should have a two-surface element"
    idx = two_surface[0].resolve_surfaces(system)
    assert labels[idx[0]].endswith("front")
    assert labels[idx[1]].endswith("rear")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def test_enumerate_entries_numbers_from_one(sample_lens_path):
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    cfg = ghostlight.PointFlareConfig()
    entries = gs.enumerate_entries(system, cfg, half_w=12.0, half_h=8.0)
    assert entries, "example doublet should have renderable ghosts"
    assert [e.number for e in entries] == list(range(1, len(entries) + 1))
    assert all(e.surf_a < e.surf_b for e in entries)
    assert not any(e.metered for e in entries)


def test_enumerate_entries_matches_the_renderer_prefilter(sample_lens_path):
    """The scrubber must not offer pairs the renderer would throw away."""
    system = ghostlight.OpticalSystem.load(str(sample_lens_path))
    cfg = ghostlight.PointFlareConfig()
    entries = gs.enumerate_entries(system, cfg, half_w=12.0, half_h=8.0)
    pairs, _boosts = ghostlight.filter_ghost_pairs(system, 12.0, 8.0, cfg)
    assert [e.pair for e in entries] == [(p.surf_a, p.surf_b) for p in pairs]


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def test_panel_pins_the_source(loaded_body):
    _project, body = loaded_body
    assert (SOURCE_X, SOURCE_Y) == (0.75, 0.25)
    assert body._canvas._sx == SOURCE_X
    assert body._canvas._sy == SOURCE_Y


def test_canvas_drag_does_not_move_the_source(loaded_body):
    """The pinned source must not be draggable — a control that silently did
    nothing would read as a bug."""
    _project, body = loaded_body
    canvas = body._canvas
    canvas.resize(200, 200)
    event = QMouseEvent(
        QEvent.MouseButtonPress, QPointF(20.0, 20.0),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
    )
    canvas.mousePressEvent(event)
    assert (canvas._sx, canvas._sy) == (SOURCE_X, SOURCE_Y)


@pytest.mark.parametrize(
    "preset", ["apply_preset_draft", "apply_preset_mid", "apply_preset_high"]
)
def test_every_preset_lands_with_the_additive_layers_off(loaded_body, preset):
    _project, body = loaded_body
    getattr(body, preset)()
    assert body.settings.starburst is False
    assert body.settings.veil is False


def test_defaults_to_the_high_preset_with_the_layers_stripped(loaded_body):
    _project, body = loaded_body
    from ghostlight_designer.render_common import HIGH_PRESET

    assert body.settings == body_mod._strip_layers(HIGH_PRESET)
    assert body.settings.ray_grid == HIGH_PRESET.ray_grid


def test_no_source_shape_controls(loaded_body):
    """Point source only — an extended source would blur every ghost by the
    same shape and hide the ghost's own footprint."""
    _project, body = loaded_body
    for attr in ("_shape_combo", "_spin_size_w", "_spin_size_h",
                 "_spin_rot", "_spin_sides", "_spin_samples"):
        assert not hasattr(body, attr), f"{attr} should be gone"


def test_desqueeze_defaults_on(loaded_body):
    _project, body = loaded_body
    assert body.desqueeze is True


def test_panel_lists_the_lens_ghosts(loaded_body):
    _project, body = loaded_body
    assert body.ghosts, "example doublet should produce ghosts"
    assert body.visible_ghosts == body.ghosts  # cull off by default
    assert body.selected_ghost() is not None
    assert body.selected_ghost().number == 1


def test_readout_names_both_surfaces_and_the_ghost_number(loaded_body):
    _project, body = loaded_body
    entry = body.selected_ghost()
    text = body._ghost_readout.text()
    assert f"Ghost {entry.number} of {len(body.ghosts)}" in text
    assert f"surface {entry.surf_a}" in text
    assert f"surface {entry.surf_b}" in text


def test_slider_range_tracks_the_offered_ghosts(loaded_body):
    _project, body = loaded_body
    assert body._ghost_slider.maximum() == len(body.visible_ghosts) - 1
    assert body._ghost_slider.value() == 0


def test_stepping_moves_the_selection_and_clamps(loaded_body):
    _project, body = loaded_body
    first = body.selected_ghost().number
    body.select_next_ghost()
    assert body.selected_ghost().number != first
    for _ in range(len(body.ghosts) + 5):
        body.select_next_ghost()
    assert body.selected_ghost().number == body.visible_ghosts[-1].number
    for _ in range(len(body.ghosts) + 5):
        body.select_previous_ghost()
    assert body.selected_ghost().number == body.visible_ghosts[0].number


def test_slider_drives_the_selection(loaded_body):
    _project, body = loaded_body
    body._ghost_slider.setValue(2)
    assert body.selected_ghost().number == body.visible_ghosts[2].number


def _fake_measure(body, brightest_number: int = 1) -> None:
    """Stand in for the GPU rough pass: mark one ghost bright, the rest dim."""
    body._ghosts = gs.apply_peaks(
        body._ghosts,
        {e.pair: (1.0 if e.number == brightest_number else 1e-9)
         for e in body._ghosts},
    )
    body._rough_dirty = False
    body._rebuild_view()


def test_cull_is_on_by_default(loaded_body):
    _project, body = loaded_body
    assert body.cull_dim_ghosts is True
    assert body.cull_threshold == pytest.approx(0.01)


def test_nothing_is_culled_before_a_measurement_lands(loaded_body):
    """Culling is on out of the box, but it must not hide anything until the
    rough pass has actually measured something."""
    _project, body = loaded_body
    assert not any(e.metered for e in body.ghosts)
    assert body.visible_ghosts == body.ghosts


def test_cull_toggle_does_not_renumber_survivors(loaded_body):
    """Ghost numbers identify a pair; hiding entries must not shift them."""
    _project, body = loaded_body
    before = {e.number: e.pair for e in body.ghosts}
    _fake_measure(body)
    assert len(body.visible_ghosts) == 1
    assert body.visible_ghosts[0].number == 1
    body.set_cull_dim_ghosts(False)
    assert sorted(e.number for e in body.visible_ghosts) == sorted(before)
    assert {e.number: e.pair for e in body.ghosts} == before


def test_cull_reseats_the_selection_onto_the_nearest_survivor(loaded_body):
    _project, body = loaded_body
    body.set_cull_dim_ghosts(False)
    body.select_ghost_number(body.ghosts[-1].number)
    _fake_measure(body)
    body.set_cull_dim_ghosts(True)
    assert body.selected_ghost().number == 1


def test_sort_defaults_on_and_reorders_the_slider_without_renumbering(loaded_body):
    _project, body = loaded_body
    assert body.sort_by_brightness is True
    body.set_cull_dim_ghosts(False)
    n = len(body.ghosts)
    # Nothing measured yet, so the stable sort leaves survey order alone.
    assert [e.number for e in body.visible_ghosts] == list(range(1, n + 1))
    # Brightness ascending with ghost number, so the sort should reverse it.
    body._ghosts = gs.apply_peaks(
        body._ghosts, {e.pair: float(e.number) for e in body._ghosts}
    )
    body._rough_dirty = False
    body._rebuild_view()
    assert [e.number for e in body.visible_ghosts] == list(range(n, 0, -1))
    # Numbers still name the same pairs, only the slider order moved.
    assert {e.number: e.pair for e in body.visible_ghosts} == \
        {e.number: e.pair for e in body.ghosts}
    body.set_sort_by_brightness(False)
    assert [e.number for e in body.visible_ghosts] == list(range(1, n + 1))


def test_panel_opens_on_the_brightest_ghost(loaded_body):
    """Nobody has chosen a ghost yet, so the selection follows the head of the
    scrubber as the brightest-first sort reorders it under us."""
    _project, body = loaded_body
    body._ghosts = gs.apply_peaks(
        body._ghosts, {e.pair: float(e.number) for e in body._ghosts}
    )
    body._rough_dirty = False
    body._rebuild_view()
    assert body.selected_ghost().number == len(body.ghosts)  # the brightest
    assert body._ghost_slider.value() == 0


def test_a_chosen_ghost_survives_a_reorder(loaded_body):
    """Once the user picks one, the panel stops following the head."""
    _project, body = loaded_body
    body.set_cull_dim_ghosts(False)
    body.select_ghost_number(2)
    body._ghosts = gs.apply_peaks(
        body._ghosts, {e.pair: float(e.number) for e in body._ghosts}
    )
    body._rough_dirty = False
    body._rebuild_view()
    assert body.selected_ghost().number == 2


def test_sort_keeps_the_selected_ghost_and_moves_the_slider(loaded_body):
    _project, body = loaded_body
    body.set_cull_dim_ghosts(False)
    body.set_sort_by_brightness(False)
    n = len(body.ghosts)
    body._ghosts = gs.apply_peaks(
        body._ghosts, {e.pair: float(e.number) for e in body._ghosts}
    )
    body._rough_dirty = False
    body._rebuild_view()
    body.select_ghost_number(1)
    assert body._ghost_slider.value() == 0
    body.set_sort_by_brightness(True)
    # Ghost 1 is now the dimmest, so it lands at the far right of the slider.
    assert body.selected_ghost().number == 1
    assert body._ghost_slider.value() == n - 1


def test_selecting_a_ghost_leaves_the_exposure_alone(loaded_body):
    """Exposure describes the lens, not the pair — so a dim ghost reads dim."""
    _project, body = loaded_body
    body.set_cull_dim_ghosts(False)
    body._exposure_stops = 7.5
    body.select_next_ghost()
    assert body._exposure_stops == 7.5


def test_rough_pass_only_asks_for_per_pair_layers_when_culling(
    loaded_body, monkeypatch
):
    """Per-pair layers are what make the rough pass scale with the ghost
    count; with the cull off it must stay a single cheap render."""
    seen = []

    def fake_survey(lens, calib, cfg, *, width, height, want_peaks=False):
        seen.append(want_peaks)
        return None, {}

    class InlineThread:
        def __init__(self, target=None, args=(), **_kw):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    _project, body = loaded_body
    monkeypatch.setattr(body_mod.threading, "Thread", InlineThread)
    monkeypatch.setattr(gs, "render_rough_survey", fake_survey)

    body._is_active = True
    body.set_cull_dim_ghosts(False)
    body.set_sort_by_brightness(False)
    body._rough_dirty = True
    body._maybe_launch_rough()
    assert seen[-1] is False

    body.set_cull_dim_ghosts(True)
    body._rough_dirty = True
    body._maybe_launch_rough()
    assert seen[-1] is True

    # Sorting by brightness needs the same measurement, so it keeps the layers
    # on even with the cull off.
    body.set_cull_dim_ghosts(False)
    body.set_sort_by_brightness(True)
    body._rough_dirty = True
    body._maybe_launch_rough()
    assert seen[-1] is True


# ---------------------------------------------------------------------------
# Registration + menus
# ---------------------------------------------------------------------------

def test_panel_type_registers(isolated_settings):
    register_ghost_explorer_panel_type(isolated_settings)
    panel_type = registry.get(GHOST_EXPLORER_TYPE_ID)
    assert panel_type is not None
    assert panel_type.display_name == "Ghost Explorer"


def test_menus_expose_the_cull_checkbox(loaded_body):
    project, body = loaded_body
    menus = build_menus(body, project)
    titles = [m.title() for m in menus]
    assert titles == ["&View", "&Settings"]
    view = menus[0]
    cull = [a for a in view.actions() if a.text() == "&Cull Dim Ghosts"]
    assert len(cull) == 1
    assert cull[0].isCheckable()
    assert cull[0].isChecked() is True
    cull[0].setChecked(False)
    assert body.cull_dim_ghosts is False
    cull[0].setChecked(True)
    assert body.cull_dim_ghosts is True


def test_view_menu_has_no_step_actions_or_per_ghost_exposure(loaded_body):
    """Stepping lives on the buttons flanking the slider; exposure is metered
    from the whole flare, so neither belongs in the menu."""
    project, body = loaded_body
    # Hold the menu list: build_menus returns unparented QMenus, which Python
    # would otherwise collect before the actions are read.
    menus = build_menus(body, project)
    texts = [a.text() for a in menus[0].actions()]
    assert "&Next Ghost" not in texts
    assert "&Previous Ghost" not in texts
    assert not any("Each" in t for t in texts)


def test_menus_expose_the_brightness_sort(loaded_body):
    project, body = loaded_body
    menus = build_menus(body, project)
    sort = [a for a in menus[0].actions() if a.text() == "Sort by &Brightness"]
    assert len(sort) == 1
    assert sort[0].isCheckable()
    assert sort[0].isChecked() is True
    sort[0].setChecked(False)
    assert body.sort_by_brightness is False
    sort[0].setChecked(True)
    assert body.sort_by_brightness is True


def test_menus_expose_desqueeze_checked_by_default(loaded_body):
    project, body = loaded_body
    view = build_menus(body, project)[0]
    act = [a for a in view.actions() if a.text() == "&Desqueeze"]
    assert len(act) == 1
    assert act[0].isCheckable() and act[0].isChecked()
