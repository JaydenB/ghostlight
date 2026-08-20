"""System Setup panel type — fields + image sensor configuration."""
from __future__ import annotations

from .body import SystemSetupBody
from .type import SYSTEM_SETUP_TYPE_ID, register_system_setup_panel_type

__all__ = [
    "SystemSetupBody",
    "SYSTEM_SETUP_TYPE_ID",
    "register_system_setup_panel_type",
]
