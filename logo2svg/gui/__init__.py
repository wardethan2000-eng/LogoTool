"""QuickLayer GUI package — PyQt6 desktop interface wrapping :class:`Session`.

Heavy imports (MainWindow) are deferred so that ``from logo2svg.gui.app
import run_gui`` does not trigger OpenCV / scikit-learn loading.
"""


def __getattr__(name: str):
    if name == "MainWindow":
        from .main_window import MainWindow
        return MainWindow
    if name == "run_gui":
        from .app import run_gui
        return run_gui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MainWindow", "run_gui"]
