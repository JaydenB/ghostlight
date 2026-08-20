"""Icon lookup for tree-node kinds.

Maps the node-kind strings stored on :class:`TreeNode` (``"node-element"``,
``"node-material"``, ``"node-surface"``) to the painted-glyph icons defined
in :mod:`ghostlight_viewport.icons`, so the optical-editor tree shows the same
artwork as the viewport's selection-mode toolbar.

Form-modifier kinds (``"node-asphere"``, ``"node-cylindrical"``) and unknown
names fall through to a blank :class:`QIcon`.
"""
from __future__ import annotations

from PySide6.QtGui import QIcon

from ghostlight_viewport.icons import make_icon


_KIND_TO_GLYPH: dict[str, str] = {
    "node-element": "elem",
    "node-element-muted": "elem-muted",
    "node-stop": "stop",
    "node-material": "material",
    "node-surface": "surf",
    "node-surface-solo": "surf-solo",
}


def icon_for(name: str, size: int = 18) -> QIcon:
    glyph = _KIND_TO_GLYPH.get(name)
    if glyph is None:
        return QIcon()
    return make_icon(glyph, size=size)
