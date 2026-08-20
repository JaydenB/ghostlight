"""Menus contributed by the ``system_setup`` panel type (none yet)."""
from __future__ import annotations

from typing import List

from PySide6.QtWidgets import QMenu, QWidget

from ..project import Project


def build_menus(body: QWidget, project: Project) -> List[QMenu]:
    return []
