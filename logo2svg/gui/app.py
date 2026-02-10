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
        "file", nargs="?", default=None,
        help="Optional image file to open on launch.",
    )
    args = parser.parse_args()
    raise SystemExit(run_gui(args.file))
