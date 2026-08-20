"""Bundled example texture images for the Textures panel.

Two families, both plain 8-bit greyscale PNGs consumed by the *same*
``APERTURE_IMAGE`` mechanism:

``aperture_*.png``
    Hard mattes for :class:`ghostlight.ApertureShape.IMAGE`. The tracer thresholds
    at 0.5, so these are black/white with an anti-aliased edge.

``dirt_*.png``
    Graded front-glass transmission maps folded into the diffraction pupil when
    ``DiffractionConfig.use_surface_textures`` is on. Near-white = clean glass.

See ``README.md`` here for the conventions, and ``generate_textures.py`` to
regenerate or add to the set.
"""

from __future__ import annotations

import logging
from importlib import resources
from typing import List, Optional

_log = logging.getLogger("ghostlight_designer.resources.textures")

_PACKAGE = "ghostlight_designer.resources.textures"


def textures_dir() -> Optional[str]:
    """Filesystem path of this package, or ``None`` if it isn't unpacked.

    Used to seed the Textures-panel file dialog. Returns ``None`` rather than
    raising when the package is zipped or missing, so callers can just fall
    back to the platform default directory.
    """
    try:
        path = resources.files(_PACKAGE)
    except Exception:  # noqa: BLE001 - importlib raises several unrelated types
        _log.debug("bundled textures package not importable", exc_info=True)
        return None
    try:
        # Zip-imported packages have no real path; str() would still succeed on
        # some backends, so probe for a directory instead.
        if not path.is_dir():
            return None
        return str(path)
    except Exception:  # noqa: BLE001
        _log.debug("bundled textures package has no filesystem path", exc_info=True)
        return None


def example_paths(prefix: str = "") -> List[str]:
    """Sorted paths of the bundled PNGs, optionally filtered by name prefix.

    ``example_paths("aperture_")`` gives the mattes, ``example_paths("dirt_")``
    the transmission maps.
    """
    root = textures_dir()
    if root is None:
        return []
    import os

    try:
        names = sorted(
            n for n in os.listdir(root)
            if n.lower().endswith(".png") and n.startswith(prefix)
        )
    except OSError:
        return []
    return [os.path.join(root, n) for n in names]
