"""Location of the lens files the designer suite reads.

Lives in its own module rather than in ``conftest``: a bare ``import conftest``
resolves to whichever suite pytest loaded first, so importing corpus helpers
that way breaks as soon as this suite is collected alongside another one.
"""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: Compact worked example exercising every structural feature of the format.
#: It is a format specimen, not a design, so it lives in the fixture corpus
#: rather than the shipped library under ``lenses/``.
EXAMPLE_DOUBLET = (_ROOT / "ghostlight" / "bindings" / "python" / "tests"
                   / "fixtures" / "example_doublet.lens")
