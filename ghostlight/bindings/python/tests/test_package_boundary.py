"""The core package must not depend on the UI packages.

Ghostlight ships three distributions from one repo:

    ghostlight-optics  ->  ghostlight            (raytracing / lens / loading)
    ghostlight-viewport ->  ghostlight_viewport  (PySide6 3D viewport)
    ghostlight-designer ->  ghostlight_designer  (PySide6 application)

The dependency chain runs one way only: designer -> viewport -> core. Nothing
enforces that at runtime, and the failure is silent -- a stray import in the
core package still works on a dev machine where all three are present, and only
breaks for someone who installed ghostlight-optics on its own, without PySide6.

This is a static scan rather than an import check on purpose: it catches
imports inside functions and `TYPE_CHECKING` blocks, which never execute during
a test run.
"""
import ast
import pathlib

import pytest

# tests/ -> python/ -> bindings/ -> ghostlight/ -> repo root
_REPO = pathlib.Path(__file__).resolve().parents[4]
_CORE_PKG = _REPO / "ghostlight" / "bindings" / "python" / "ghostlight"
_VIEWPORT_PKG = _REPO / "ghostlight_viewport" / "ghostlight_viewport"

FORBIDDEN_IN_CORE = ("ghostlight_viewport", "ghostlight_designer")
FORBIDDEN_IN_VIEWPORT = ("ghostlight_designer",)


def _imported_modules(path: pathlib.Path):
    """Every module name imported by a source file, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def _offenders(pkg_dir: pathlib.Path, forbidden):
    bad = []
    for src in sorted(pkg_dir.rglob("*.py")):
        for mod in _imported_modules(src):
            root = mod.split(".")[0]
            if root in forbidden:
                bad.append(f"{src.relative_to(_REPO)} imports {mod}")
    return bad


def test_core_package_exists():
    # A wrong path here would make every check below vacuously pass.
    assert _CORE_PKG.is_dir(), _CORE_PKG
    assert list(_CORE_PKG.rglob("*.py")), "no sources found in the core package"


def test_core_does_not_import_the_ui_packages():
    bad = _offenders(_CORE_PKG, FORBIDDEN_IN_CORE)
    assert not bad, (
        "the core package must not depend on the UI packages:\n  "
        + "\n  ".join(bad)
    )


@pytest.mark.skipif(not _VIEWPORT_PKG.is_dir(),
                    reason="viewport package not present in this checkout")
def test_viewport_does_not_import_the_designer():
    bad = _offenders(_VIEWPORT_PKG, FORBIDDEN_IN_VIEWPORT)
    assert not bad, (
        "the viewport must not depend on the designer:\n  " + "\n  ".join(bad)
    )


def test_core_cpp_does_not_include_the_ui_packages():
    """Only `#include` lines count.

    Prose references are fine and sometimes valuable: newton_aim.h cites
    ghostlight_designer/ray_tracing.py::_find_chief_ray_center as the
    implementation whose guards it mirrors. That is documentation, not a
    dependency, and flagging it would push people to delete a useful comment.
    """
    src = _REPO / "ghostlight" / "src"
    assert src.is_dir(), src
    bad = []
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in {".cpp", ".cu", ".h", ".hpp", ".cuh"}:
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.lstrip().startswith("#include"):
                continue
            for name in FORBIDDEN_IN_CORE:
                if name in line:
                    bad.append(f"{path.relative_to(_REPO)}:{lineno} includes {name}")
    assert not bad, "\n  ".join(bad)
