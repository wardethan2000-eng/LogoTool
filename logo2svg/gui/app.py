"""Application entry-point for the logo2svg GUI."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def run_gui(file_path: str | None = None) -> int:
    """Launch the logo2svg GUI.  Optionally open *file_path* immediately."""
    app = QApplication(sys.argv)
    app.setApplicationName("logo2svg")
    app.setOrganizationName("logo2svg")

    window = MainWindow()
    window.show()

    if file_path:
        window.open_file(file_path)

    return app.exec()
