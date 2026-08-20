"""Repository paths shared by the validation harness."""
import pathlib
import sys

# validation/_paths.py -> validation/ -> repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Directory holding the importable package (and the built extension).
PKG_DIR = REPO_ROOT / "ghostlight" / "bindings" / "python"

#: Lens prescription library.
LENSES = REPO_ROOT / "lenses"

#: Prescriptions the library does not ship, kept as a test corpus because
#: several gates below are pinned against their measured values.
FIXTURE_LENSES = PKG_DIR / "tests" / "fixtures"

#: Saved .npy value goldens, beside this file.
GOLDENS = REPO_ROOT / "validation" / "goldens"

#: Generated validation reports and images.
ARTIFACTS = REPO_ROOT / "validation" / "artifacts"

_LIBRARY_SUBDIRS = ("spherical", "anamorphic", "experiments")


def lens_file(name: str) -> pathlib.Path:
    """Resolve a lens by bare file name: library root, then its category
    subdirectories, then the fixture corpus.

    Matching is case-exact even on a case-insensitive filesystem, so a fixture
    cannot silently shadow a library prescription that differs only in case.
    """
    for d in (LENSES, *(LENSES / s for s in _LIBRARY_SUBDIRS), FIXTURE_LENSES):
        if d.is_dir() and name in {e.name for e in d.iterdir()}:
            return d / name
    return LENSES / name


if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
