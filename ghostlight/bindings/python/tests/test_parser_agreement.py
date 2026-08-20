"""The three parses of a ``.lens`` file must agree, across the whole corpus.

A lens file is read three times by three different implementations:

1. ``ghostlight/src/optical_system.cpp`` -- the C++ loader, which resolves each
   element's transform and bakes it into the flat surface array
   (``decenter_x/y``, ``z``, ``rot``);
2. ``ghostlight/bindings/python/ghostlight/element.py`` -- a second, independent
   JSON parse that recovers the element/pivot grouping the C++ discards;
3. ``ghostlight/bindings/python/ghostlight/pose.py`` -- a Python re-implementation
   of the C++ bake, whose own docstring says "the two implementations must
   agree".

``test_element_pose_bake.py`` guards (1) against (3), but only on a single
synthetic fixture. These tests run the same comparison over every lens in the
repo -- the rotated, pivoted and anamorphic design studies included, which are
the files where a divergence would actually show up.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import ghostlight
from ghostlight.element import load_elements_and_pivots
from ghostlight.pose import bake_system_poses

_REPO = pathlib.Path(__file__).resolve().parents[4]
# Every root that holds .lens files. validation/ carries the golden fixtures,
# which are real lens files and must pass the same gates as the library —
# they are migrated by the same tooling, so a corpus that skipped them would
# let a migration bug through.
_CORPUS_ROOTS = (
    _REPO / "lenses",
    _REPO / "ghostlight" / "bindings" / "python" / "tests" / "fixtures",
    _REPO / "validation",
)

# Deliberately malformed negative fixture (dangling pivot element reference).
EXCLUDED_NAMES = {"bad_pivot_dangling.lens"}


def _lens_files() -> list[pathlib.Path]:
    return sorted(
        p
        for root in _CORPUS_ROOTS
        for p in root.rglob("*.lens")
        if p.name not in EXCLUDED_NAMES
    )


def _poses(system):
    return [
        (float(s.decenter_x), float(s.decenter_y), float(s.z),
         tuple(float(v) for v in s.rot))
        for s in system.surfaces
    ]


def test_the_corpus_is_not_empty():
    assert len(_lens_files()) >= 25


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_python_bake_matches_cpp_loader(path):
    """pose.py must reproduce the C++ loader's baked surface poses."""
    system = ghostlight.OpticalSystem.load(str(path))
    before = _poses(system)
    assert bake_system_poses(system) is True
    after = _poses(system)

    assert len(before) == len(after)
    for i, (a, b) in enumerate(zip(before, after)):
        assert a[0] == pytest.approx(b[0], abs=2e-4), f"surface {i} decenter_x"
        assert a[1] == pytest.approx(b[1], abs=2e-4), f"surface {i} decenter_y"
        assert a[2] == pytest.approx(b[2], abs=2e-4), f"surface {i} z"
        for k in range(9):
            assert a[3][k] == pytest.approx(b[3][k], abs=2e-4), \
                f"surface {i} rot[{k}]"


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_python_element_parse_matches_the_file_and_the_cpp_surface_count(path):
    """element.py's independent parse must line up with the C++ load.

    Element count, per-element surface membership and the total surface count
    all have to agree, or the writer will reassemble the flat C++ array into
    the wrong elements on save.
    """
    elements, pivots = load_elements_and_pivots(str(path))
    system = ghostlight.OpticalSystem.load(str(path))

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc_elements = [e for e in doc.get("optical_system", [])
                    if e.get("type") == "element"]

    assert len(elements) == len(doc_elements)
    assert len(pivots) == len(doc.get("pivots") or [])
    assert sum(len(e.surface_ids) for e in elements) == system.num_surfaces()

    # Surface UUIDs must resolve, in order, against the flat C++ array.
    flat = list(system.surface_ids)
    walked = [sid for el in elements for sid in el.surface_ids]
    assert walked == flat

    # The loader enforces surfaces == materials + 1.
    for el in elements:
        assert len(el.surface_ids) == len(el.material_glasses) + 1


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_round_trip_preserves_geometry_and_version(path, tmp_path):
    """load -> save -> load must be a fixed point for geometry."""
    original = ghostlight.OpticalSystem.load(str(path))
    out = tmp_path / "rt.lens"
    original.save(str(out))

    with open(out, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["version"] == {"major": ghostlight.LENS_FORMAT_MAJOR,
                              "minor": ghostlight.LENS_FORMAT_MINOR}

    reloaded = ghostlight.OpticalSystem.load(str(out))
    assert reloaded.num_surfaces() == original.num_surfaces()
    for i, (a, b) in enumerate(zip(reloaded.surfaces, original.surfaces)):
        assert a.radius == pytest.approx(b.radius, rel=1e-6), f"surface {i}"
        assert a.thickness == pytest.approx(b.thickness, rel=1e-6, abs=1e-9), \
            f"surface {i}"
        assert a.semi_aperture == pytest.approx(b.semi_aperture, rel=1e-6), \
            f"surface {i}"
        assert int(a.form) == int(b.form), f"surface {i}"
        assert bool(a.is_stop) == bool(b.is_stop), f"surface {i}"
        assert bool(a.is_active) == bool(b.is_active), f"surface {i}"
        assert int(a.aperture_shape) == int(b.aperture_shape), f"surface {i}"
        assert int(a.coating.model) == int(b.coating.model), f"surface {i}"


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_save_is_idempotent(path, tmp_path):
    """A second save must reproduce the first byte for byte.

    A writer that derived the version from content would let a file change
    shape on a save that changed nothing.
    """
    system = ghostlight.OpticalSystem.load(str(path))
    first = tmp_path / "a.lens"
    system.save(str(first))
    second = tmp_path / "b.lens"
    ghostlight.OpticalSystem.load(str(first)).save(str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(
        encoding="utf-8")
