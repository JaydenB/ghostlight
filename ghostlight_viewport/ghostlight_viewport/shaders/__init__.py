"""Shader file loader using importlib.resources."""

from __future__ import annotations

from importlib import resources


def load(name: str) -> str:
    """Read a packaged shader source file as a string.

    Pass the filename including extension, e.g. ``load("lens.vert")``.
    """
    pkg = resources.files(__package__)
    with (pkg / name).open("r", encoding="utf-8") as fh:
        return fh.read()
