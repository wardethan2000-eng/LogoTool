"""Application entry-point for the QuickLayer GUI.

Sets up the Fusion theme with a modern light palette, applies the
central QSS stylesheet, and launches the main window.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow
from .style import (
    ACCENT,
    BG,
    CARD,
    STYLESHEET,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    TEXT_SEC,
)


def _apply_light_palette(app: QApplication) -> None:
    """Set a light QPalette as baseline for widgets that ignore QSS."""
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(BG))
    p.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.Base, QColor(CARD))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(CARD))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    p.setColor(QPalette.ColorRole.Button, QColor(CARD))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))

    # Disabled colours
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(TEXT_MUTED),
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(TEXT_MUTED),
    )
    p.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(TEXT_MUTED),
    )
    app.setPalette(p)


def run_gui(file_path: str | None = None) -> int:
    """Launch the QuickLayer GUI.  Optionally open *file_path* immediately."""
    # Tell Windows this is its own app so the taskbar icon works
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "quicklayer.gui.1"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("QuickLayer")
    app.setOrganizationName("QuickLayer")
    app.setStyle("Fusion")

    _apply_light_palette(app)
    app.setStyleSheet(STYLESHEET)

    # Set application icon (taskbar / title-bar)
    icon_path = Path(__file__).resolve().parent.parent / "icons" / "quicklayer.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()

    if file_path:
        window.open_file(file_path)

    return app.exec()


def run_gui_cli() -> None:
    """Console-script entry point for ``quicklayer-gui``.

    Parses an optional positional file-path argument from *sys.argv*
    and delegates to :func:`run_gui`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="quicklayer-gui",
        description="Launch the QuickLayer graphical interface.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional image file to open on launch.",
    )
    args = parser.parse_args()
    raise SystemExit(run_gui(args.file))
