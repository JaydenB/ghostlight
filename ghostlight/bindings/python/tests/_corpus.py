"""Locations of the lens files the suite reads.

Lives in its own module rather than in ``conftest``: a bare ``import conftest``
resolves to whichever suite pytest loaded first, so importing corpus helpers
that way breaks as soon as this suite is collected alongside another one.
"""

import pathlib

#: The shipped lens library.
LENSES_DIR = pathlib.Path(__file__).resolve().parents[4] / "lenses"

#: Narrow parser fixtures, plus prescriptions the library does not ship but
#: whose measured values several suites are pinned against.
FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"

_LIBRARY_SUBDIRS = ("spherical", "anamorphic", "experiments")


def lens_path(name: str) -> pathlib.Path:
    """Resolve a lens by bare file name: library root, then its category
    subdirectories, then the fixture corpus.

    Matching is case-exact even on a case-insensitive filesystem, so a fixture
    cannot silently shadow a library prescription that differs only in case.
    """
    for d in (LENSES_DIR, *(LENSES_DIR / s for s in _LIBRARY_SUBDIRS), FIXTURES_DIR):
        if d.is_dir() and name in {e.name for e in d.iterdir()}:
            return d / name
    return LENSES_DIR / name
