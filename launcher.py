"""Entry-point for the frozen (PyInstaller) GUI executable."""

import sys
from logo2svg.gui.app import run_gui

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(run_gui(file_path))
