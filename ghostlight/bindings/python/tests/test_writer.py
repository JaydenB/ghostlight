"""Round-trip tests for :mod:`ghostlight.writer`."""
from __future__ import annotations

import json

import ghostlight


def test_build_optical_system_doc_matches_disk(loaded_lens, tmp_path):
    """The in-memory dict from build_optical_system_doc must equal the JSON
    that write_optical_system / OpticalSystem.save would put on disk.
    Guarantees the dict builder is the single source of truth used by both
    file save and the in-memory undo snapshots."""
    out_path = tmp_path / "round_trip.lens"
    loaded_lens.save(str(out_path))

    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    in_memory = ghostlight.build_optical_system_doc(
        system=loaded_lens,
        metadata=loaded_lens._raw_metadata,
        glass_catalogue=loaded_lens._raw_glass_catalogue,
    )

    assert in_memory == on_disk


def test_build_optical_system_doc_reloads_to_same_geometry(loaded_lens, tmp_path):
    """Dump the in-memory doc back to disk and reload — the resulting
    geometry must match the original. Mirrors the undo restore path
    (snapshot dict -> temp file -> OpticalSystem.load)."""
    doc = ghostlight.build_optical_system_doc(
        system=loaded_lens,
        metadata=loaded_lens._raw_metadata,
        glass_catalogue=loaded_lens._raw_glass_catalogue,
    )
    target = tmp_path / "snap.lens"
    target.write_text(json.dumps(doc), encoding="utf-8")

    reloaded = ghostlight.OpticalSystem.load(str(target))
    assert len(reloaded.surfaces) == len(loaded_lens.surfaces)
    for a, b in zip(reloaded.surfaces, loaded_lens.surfaces):
        assert a.radius == b.radius
        assert a.thickness == b.thickness
        assert a.semi_aperture == b.semi_aperture
        assert int(a.form) == int(b.form)
        assert bool(a.is_stop) == bool(b.is_stop)
