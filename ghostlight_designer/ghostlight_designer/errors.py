"""Route Python exceptions + Qt warnings to ``~/.ghostlight_designer_errors.log``.

Ported from ``ghostlight_viewport.widget.install_global_error_logging``. Qt's
event loop silently swallows exceptions raised from Python overrides of
virtual methods — this hook captures them post-mortem and also routes
``qDebug``/``qWarning``/``qCritical`` through Qt's message handler.
"""
from __future__ import annotations

import os
import sys
import traceback


_ERR_LOG_PATH = os.path.join(
    os.path.expanduser("~"), ".ghostlight_designer_errors.log"
)


def install_global_error_logging() -> None:
    """Idempotent: only the first call installs the hooks."""
    if getattr(install_global_error_logging, "_installed", False):
        return
    install_global_error_logging._installed = True  # type: ignore[attr-defined]

    previous_excepthook = sys.excepthook

    def hook(exc_type, exc, tb):
        try:
            msg = "".join(traceback.format_exception(exc_type, exc, tb))
            with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write("[unhandled exception]\n" + msg + "\n")
            sys.stderr.write("[ghostlight_designer] " + msg)
        except Exception:
            pass
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = hook

    try:
        from PySide6.QtCore import qInstallMessageHandler

        def qt_msg_handler(mode, ctx, msg):
            try:
                with open(_ERR_LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(f"[qt:{mode}] {msg}\n")
            except Exception:
                pass

        qInstallMessageHandler(qt_msg_handler)
    except Exception:
        pass
