"""Path bootstrap so the suite runs from a clean checkout.

The viewport package and the compiled `ghostlight` extension both live outside
this directory, and several modules import them at module scope. Doing the
insert here guarantees it happens before any test module is imported, rather
than depending on which module pytest collects first.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

for _p in (_ROOT / "ghostlight" / "bindings" / "python", _ROOT / "ghostlight_viewport"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
