"""logo2svg GUI package — PyQt6 desktop interface wrapping :class:`Session`."""

from .main_window import MainWindow
from .app import run_gui

__all__ = ["MainWindow", "run_gui"]
