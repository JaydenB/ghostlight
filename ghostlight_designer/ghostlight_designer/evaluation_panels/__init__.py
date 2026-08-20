"""Evaluation panels — family of read-only chart/plot panels.

Currently:
* ``spot_diagram`` — image-plane spot pattern for a defined set of
  fields / wavelengths / pupil samples.

All panels share :class:`EvaluationPanelBody`, which provides:

* Edit-settle debounce + visibility-gated background compute.
* Two-layer auto-update gate (per-panel × global View → Auto-Update).
* Status label + threaded compute / GUI-thread result apply.
* The "Sync from System Setup" affordance — by default each evaluation
  panel carries its own spec (wavelengths, fields, pupil sampling…) so
  artists can probe a specific case without disturbing the live setup.
  One menu action pulls the current setup values in as defaults.

User-facing actions (Refresh, Sync, Auto-Update toggle, panel-specific
display toggles) live on the panel type's View menu — same convention
as ``sourceflare_panel``, ``ghost_explorer_panel`` and ``psf_panel``. No
in-panel toolbar.

To add a new evaluation panel:

1. Create a sibling subpackage (e.g. ``mtf/``) with a ``body.py`` that
   subclasses :class:`EvaluationPanelBody` and implements ``compute()``
   plus ``apply_result()``.
2. Add a ``menus.py`` building the panel's View menu — match the
   order/grouping the other render panels use (Auto / Refresh / data-
   refresh action, then display toggles, then dialogs / resets).
3. Add a ``type.py`` registering a :class:`PanelType` whose ``build_body``
   closure feeds the body the ``AppSettings`` it needs to honour the
   global toggle.
4. Register the type from ``main_window.py``.
"""
from __future__ import annotations

from .base import EvaluationPanelBody

__all__ = ["EvaluationPanelBody"]
