from __future__ import annotations

import os
import pathlib
import sys

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed; designer tests skipped")

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "ghostlight" / "bindings" / "python"))
sys.path.insert(0, str(_ROOT / "ghostlight_designer"))
sys.path.insert(0, str(_ROOT / "ghostlight_viewport"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ghostlight_designer.settings import AppSettings  # noqa: E402

from _corpus import EXAMPLE_DOUBLET  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("GhostlightTest")
    app.setApplicationName("GhostlightDesignerTest")
    return app


@pytest.fixture
def sample_lens_path() -> pathlib.Path:
    return EXAMPLE_DOUBLET


@pytest.fixture
def isolated_settings(qapp, tmp_path) -> AppSettings:
    qs = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    return AppSettings(qsettings=qs)
