from __future__ import annotations

import pytest

from ghostlight_designer.project import Project


def _collect(signal):
    """Return a (list, slot) pair; calling the slot appends args to the list."""
    received: list = []

    def slot(*args):
        received.append(args if len(args) != 1 else args[0])

    signal.connect(slot)
    return received


def test_defaults_to_empty_untitled(qapp):
    p = Project()
    assert p.path is None
    assert p.is_dirty is False
    assert p.display_name == "Untitled"
    assert len(p.system.surfaces) == 0


def test_new_emits_system_replaced(qapp):
    p = Project()
    replaced = _collect(p.systemReplaced)
    dirty = _collect(p.dirtyChanged)

    p.new()

    assert len(replaced) == 1
    assert replaced[0] is p.system
    assert dirty == []  # stayed False, no transition
    assert p.is_dirty is False
    assert p.path is None


def test_mark_modified_sets_dirty_and_emits(qapp):
    p = Project()
    modified = _collect(p.systemModified)
    dirty = _collect(p.dirtyChanged)

    p.mark_modified()

    # systemModified collects an empty-tuple for a zero-arg signal
    assert len(modified) == 1
    assert dirty == [True]
    assert p.is_dirty is True

    p.mark_modified()
    assert len(modified) == 2
    assert dirty == [True]  # already True, no second transition


def test_load_clears_dirty_and_sets_path(qapp, sample_lens_path):
    p = Project()
    p.mark_modified()
    assert p.is_dirty is True

    replaced = _collect(p.systemReplaced)
    path_changed = _collect(p.pathChanged)
    dirty = _collect(p.dirtyChanged)

    p.load(str(sample_lens_path))

    assert p.path == str(sample_lens_path)
    assert p.is_dirty is False
    assert len(p.system.surfaces) > 0
    assert len(replaced) == 1
    assert path_changed == [str(sample_lens_path)]
    assert dirty == [False]


def test_save_as_round_trip(qapp, sample_lens_path, tmp_path):
    src = Project()
    src.load(str(sample_lens_path))
    original_count = len(src.system.surfaces)
    src.mark_modified()
    assert src.is_dirty

    out = tmp_path / "out.lens"
    src.save_as(str(out))
    assert src.is_dirty is False
    assert src.path == str(out)
    assert out.exists()

    reloaded = Project()
    reloaded.load(str(out))
    assert len(reloaded.system.surfaces) == original_count


def test_save_without_path_raises(qapp):
    with pytest.raises(ValueError):
        Project().save()


def test_display_name_dirty_marker(qapp, sample_lens_path):
    p = Project()
    p.load(str(sample_lens_path))
    assert p.display_name == sample_lens_path.name

    p.mark_modified()
    assert p.display_name == f"{sample_lens_path.name}*"
