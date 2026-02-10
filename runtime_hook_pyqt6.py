"""PyInstaller runtime hook: ensure Qt can find its platform plugins
when running from a frozen (one-folder / one-file) bundle.

Without this, the frozen executable often fails with:
  "qt.qpa.plugin: Could not find the Qt platform plugin 'windows'"
or PyQt6 import errors.
"""

import os
import sys

if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    base = sys._MEIPASS  # type: ignore[attr-defined]

    # Point Qt to the bundled plugins directory
    plugin_path = os.path.join(base, "PyQt6", "Qt6", "plugins")
    if os.path.isdir(plugin_path):
        os.environ["QT_PLUGIN_PATH"] = plugin_path

    # Also check the flat layout PyInstaller sometimes uses
    flat_plugin_path = os.path.join(base, "qt6_plugins")
    if os.path.isdir(flat_plugin_path):
        os.environ.setdefault("QT_PLUGIN_PATH", flat_plugin_path)
