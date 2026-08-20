"""Optical Design Editor panel — the schema-driven element/surface tree."""
from __future__ import annotations

from .body import OpticalEditorBody
from .type import OPTICAL_EDITOR_TYPE_ID, register_optical_editor_panel_type

__all__ = ["OpticalEditorBody", "OPTICAL_EDITOR_TYPE_ID", "register_optical_editor_panel_type"]
