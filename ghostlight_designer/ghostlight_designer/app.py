"""Application entry point. ``python -m ghostlight_designer [path]``."""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from PySide6.QtWidgets import QApplication, QMessageBox

from .errors import install_global_error_logging
from .main_window import MainWindow
from .project import Project
from .settings import AppSettings
from .splash import build_splash


def run(argv: Optional[Sequence[str]] = None) -> int:
    install_global_error_logging()
    argv = list(sys.argv if argv is None else argv)

    from ghostlight_viewport.widget import set_default_surface_format
    set_default_surface_format()
    app = QApplication.instance() or QApplication(argv)
    app.setOrganizationName("Ghostlight")
    app.setApplicationName("Ghostlight Designer")

    splash = build_splash()
    splash.show()
    app.processEvents()

    settings = AppSettings()
    project = Project()

    initial_path: Optional[str] = None
    if len(argv) > 1 and not argv[1].startswith("-"):
        initial_path = argv[1]

    if initial_path:
        try:
            project.load(initial_path)
            settings.add_recent_file(initial_path)
        except Exception as exc:
            splash.close()
            QMessageBox.warning(
                None,
                "Ghostlight Designer",
                f"Could not open {initial_path}:\n{exc}\n\nStarting with empty project.",
            )
            project.new()

    window = MainWindow(project=project, settings=settings)
    window.show()
    splash.finish(window)
    return app.exec()
