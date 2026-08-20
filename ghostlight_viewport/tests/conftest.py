"""Path bootstrap so the suite runs from a clean checkout.

The viewport package and the compiled `ghostlight` extension both live outside
this directory, and several modules import them at module scope. Doing the
insert here guarantees it happens before any test module is imported, rather
than depending on which module pytest collects first.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# Prefer an in-tree build when build_dev.ps1 has copied the compiled extension
# into the source package. Without that extension the source package is a
# stub, so inserting it unconditionally would shadow an installed
# ghostlight-optics wheel and break `import ghostlight._ghostlight`.
def _has_built_extension(package_root):
    pkg = package_root / "ghostlight"
    return any(pkg.glob("_ghostlight*.pyd")) or any(pkg.glob("_ghostlight*.so"))


_BINDINGS = _ROOT / "ghostlight" / "bindings" / "python"

_paths = [_ROOT / "ghostlight_viewport"]
if _has_built_extension(_BINDINGS):
    _paths.insert(0, _BINDINGS)

for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
