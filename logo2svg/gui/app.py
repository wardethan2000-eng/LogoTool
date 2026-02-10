"""Application entry-point for the logo2svg GUI.

Sets up the Fusion theme with a modern light palette, applies the
central QSS stylesheet, and launches the main window.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
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
    """Launch the logo2svg GUI.  Optionally open *file_path* immediately."""
    app = QApplication(sys.argv)
    app.setApplicationName("logo2svg")
    app.setOrganizationName("logo2svg")
    app.setStyle("Fusion")

    _apply_light_palette(app)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    if file_path:
        window.open_file(file_path)

    return app.exec()


def run_gui_cli() -> None:
    """Console-script entry point for ``logo2svg-gui``.

    Parses an optional positional file-path argument from *sys.argv*
    and delegates to :func:`run_gui`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="logo2svg-gui",
        description="Launch the logo2svg graphical interface.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional image file to open on launch.",
    )
    args = parser.parse_args()
    raise SystemExit(run_gui(args.file))
