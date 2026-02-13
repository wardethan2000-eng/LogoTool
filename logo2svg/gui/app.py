"""Application entry-point for the QuickLayer GUI.

Shows a lightweight splash screen instantly, then lazy-loads the heavy
pipeline dependencies (OpenCV, scikit-learn, etc.) so the user sees
feedback within a second of launch.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ── Splash screen (only needs basic PyQt6 — no heavy deps) ──────────

def _create_splash_pixmap():
    """Create a simple branded splash pixmap."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap

    w, h = 400, 200
    pm = QPixmap(w, h)
    pm.fill(QColor("#ffffff"))

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Subtle border
    painter.setPen(QColor("#d1d5db"))
    painter.drawRoundedRect(1, 1, w - 2, h - 2, 10, 10)

    # App name
    font = QFont("Segoe UI", 26, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#111827"))
    painter.drawText(pm.rect().adjusted(0, -18, 0, 0),
                     Qt.AlignmentFlag.AlignCenter, "QuickLayer")

    # Loading text
    font = QFont("Segoe UI", 11)
    painter.setFont(font)
    painter.setPen(QColor("#9ca3af"))
    painter.drawText(pm.rect().adjusted(0, 36, 0, 0),
                     Qt.AlignmentFlag.AlignCenter, "Loading components\u2026")

    painter.end()
    return pm


# ── Palette (needs style tokens — still lightweight) ─────────────────

def _apply_light_palette(app) -> None:
    """Set a light QPalette as baseline for widgets that ignore QSS."""
    from PyQt6.QtGui import QColor, QPalette
    from .style import ACCENT, BG, CARD, SURFACE, TEXT, TEXT_MUTED, TEXT_SEC

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


# ── Main entry point ─────────────────────────────────────────────────

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

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QSplashScreen

    app = QApplication(sys.argv)
    app.setApplicationName("QuickLayer")
    app.setOrganizationName("QuickLayer")
    app.setStyle("Fusion")

    # ── Show splash IMMEDIATELY, before heavy imports ──
    splash = QSplashScreen(_create_splash_pixmap(),
                           Qt.WindowType.WindowStaysOnTopHint
                           | Qt.WindowType.FramelessWindowHint)
    # Centre the splash on the primary screen so it doesn't briefly
    # appear at (0, 0) before the window manager moves it.
    screen_geo = app.primaryScreen().geometry()
    splash_size = splash.size()
    splash.move(
        screen_geo.x() + (screen_geo.width() - splash_size.width()) // 2,
        screen_geo.y() + (screen_geo.height() - splash_size.height()) // 2,
    )
    splash.show()
    app.processEvents()

    # Close PyInstaller native splash (if present) now that Qt splash is up
    try:
        import pyi_splash  # type: ignore[import-not-found]
        pyi_splash.close()
    except ImportError:
        pass

    # ── Heavy imports happen here (cv2, sklearn, etc.) ──
    from PyQt6.QtGui import QColor as _QColor

    splash.showMessage(
        "Loading interface\u2026",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        _QColor("#9ca3af"),
    )
    app.processEvents()

    from .main_window import MainWindow
    app.processEvents()

    from .style import STYLESHEET
    app.processEvents()

    _apply_light_palette(app)

    # Determine base path for bundled resources (handles frozen PyInstaller builds)
    if getattr(sys, "frozen", False):
        _base = Path(sys._MEIPASS) / "logo2svg"
    else:
        _base = Path(__file__).resolve().parent.parent

    # Inject correct icons directory into the stylesheet
    icons_dir = str(_base / "icons").replace("\\", "/")
    app.setStyleSheet(STYLESHEET.replace("__ICONS_DIR__", icons_dir))

    # Set application icon (taskbar / title-bar)
    icon_path = _base / "icons" / "quicklayer.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    splash.finish(window)

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
