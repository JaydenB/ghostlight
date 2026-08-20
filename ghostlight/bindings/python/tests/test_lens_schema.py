"""Every ``.lens`` file in the repo must validate against the one schema.

This is the anti-drift mechanism.  Left unchecked, the schemas
were documentation only -- nothing validated against them -- which is exactly
how ``is_active`` came to be read and written by the loader while being absent
from a ``additionalProperties: false`` surface definition, making every
muted-surface file schema-invalid without anyone noticing.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import ghostlight

jsonschema = pytest.importorskip("jsonschema")

_REPO = pathlib.Path(__file__).resolve().parents[4]
_GHOSTLIGHT = _REPO / "ghostlight"
SCHEMA_PATH = _REPO / "lenses" / "schema" / "lens.schema.json"

# Every root that holds .lens files. validation/ carries the golden fixtures,
# which are real lens files written by the same tooling as the library, so a
# corpus that skipped them would let a writer bug through.
_CORPUS_ROOTS = (
    _REPO / "lenses",
    _GHOSTLIGHT / "bindings" / "python" / "tests" / "fixtures",
    _REPO / "validation",
)


def _lens_files() -> list[pathlib.Path]:
    return sorted(
        p
        for root in _CORPUS_ROOTS
        for p in root.rglob("*.lens")
    )


@pytest.fixture(scope="module")
def validator():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return jsonschema.Draft202012Validator(json.load(fh))


def test_there_is_exactly_one_schema():
    found = sorted(p.name for p in SCHEMA_PATH.parent.glob("*.schema.json"))
    assert found == ["lens.schema.json"], (
        f"expected a single lens schema, found {found}"
    )


def test_the_corpus_is_not_empty():
    # A glob that silently matches nothing would make every check below pass.
    assert len(_lens_files()) >= 25


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_lens_file_validates(path, validator):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, "\n".join(
        f"  {list(e.path)}: {e.message}" for e in errors[:5])


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_lens_file_declares_the_current_version(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["version"] == {"major": ghostlight.LENS_FORMAT_MAJOR,
                              "minor": ghostlight.LENS_FORMAT_MINOR}


def test_schema_version_matches_the_cpp_constant(validator):
    """The schema pins the same major the loader gates on."""
    major = validator.schema["$defs"]["version"]["properties"]["major"]
    assert major["const"] == ghostlight.LENS_FORMAT_MAJOR


@pytest.mark.parametrize("mutation,description", [
    (lambda d: d["version"].__setitem__("major", 2), "superseded major 2"),
    (lambda d: d["version"].__setitem__("major", 3), "future major 3"),
    (lambda d: d.pop("pivots"), "missing pivots section"),
    (lambda d: d.pop("metadata"), "missing metadata section"),
    (lambda d: d.pop("glass_catalogue"), "missing glass_catalogue section"),
    (lambda d: d["optical_system"][0]["surfaces"][0].__setitem__("bogus", 1),
     "unknown surface key"),
    (lambda d: d["optical_system"][0]["surfaces"][0].__setitem__(
        "modifiers", [{"type": "coating", "layers": [
            {"thickness_nm": 99.6,
             "nk_table": [{"lambda_um": 0.55, "n": 1.38, "k": 0.0}]}]}]),
     "layer-stack coating with no explicit model"),
    (lambda d: d["optical_system"][0]["surfaces"][0].__setitem__(
        "modifiers", [{"type": "aperture", "shape": "polygon", "blades": 6,
                       "semi_diameter": 12.0}]),
     "legacy semi_diameter on a polygon aperture"),
])
def test_schema_rejects(mutation, description, validator, example_lens_path):
    """The schema must actually reject things.

    Without these, a validator that accepted everything would still pass the
    whole-corpus check above.
    """
    import copy
    with open(example_lens_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert not list(validator.iter_errors(doc)), "baseline must be valid"

    broken = copy.deepcopy(doc)
    mutation(broken)
    assert list(validator.iter_errors(broken)), (
        f"schema failed to reject: {description}")


@pytest.mark.parametrize(
    "path", _lens_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_writer_output_validates(path, validator, tmp_path):
    """What the writer emits must satisfy the schema too.

    Validating only the files on disk would miss a writer that emits something
    the schema rejects -- and every designer save and every undo step goes
    through the writer.
    """
    if path.name in {"bad_pivot_dangling.lens"}:
        pytest.skip("deliberate negative fixture; does not load")
    system = ghostlight.OpticalSystem.load(str(path))
    out = tmp_path / "written.lens"
    system.save(str(out))
    with open(out, encoding="utf-8") as fh:
        doc = json.load(fh)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, "\n".join(
        f"  {list(e.path)}: {e.message}" for e in errors[:5])


def test_schema_accepts_is_active_false(validator, example_lens_path):
    """`is_active` is read and written by the loader; it must be schema-legal."""
    import copy
    with open(example_lens_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc = copy.deepcopy(doc)
    doc["optical_system"][0]["surfaces"][0]["is_active"] = False
    assert not list(validator.iter_errors(doc))
