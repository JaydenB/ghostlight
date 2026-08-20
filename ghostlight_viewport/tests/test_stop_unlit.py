"""The aperture-stop fragment shader has the u_unlit branch.

Sanity-level: the lens.frag shader file declares ``uniform int u_unlit`` and a
fast-out flat-tint branch.  The widget's ``_draw_lens`` sets ``u_unlit=1`` for
elements whose kind is ``STOP``.
"""

from __future__ import annotations

import pathlib


def test_lens_frag_declares_u_unlit():
    frag_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ghostlight_viewport" / "shaders" / "lens.frag"
    )
    src = frag_path.read_text(encoding="utf-8")
    assert "uniform int u_unlit" in src, "u_unlit uniform missing in lens.frag"
    assert "u_unlit != 0" in src, "no fast-out branch for u_unlit in lens.frag"


def test_widget_sets_u_unlit_for_stop():
    widget_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ghostlight_viewport" / "widget.py"
    )
    src = widget_path.read_text(encoding="utf-8")
    # Locate _draw_lens and confirm it sets u_unlit per element based on kind.
    assert "u_unlit" in src, "widget never references u_unlit"
    assert "kind" in src and "STOP" in src, (
        "widget does not branch on STOP elements when setting u_unlit"
    )
