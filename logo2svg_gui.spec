# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the logo2svg GUI executable."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

# Collect all submodules that PyInstaller might miss
hidden_imports = (
    collect_submodules("logo2svg")
    + collect_submodules("sklearn")
    + collect_submodules("PyQt6")
    + [
        "PIL",
        "PIL.Image",
        "numpy",
        "cv2",
        "svgwrite",
        "svgwrite.shapes",
        "svgwrite.path",
        "svgwrite.container",
        "webcolors",
        "potrace",
        "click",
        # Explicit PyQt6 modules that collect_submodules sometimes misses
        "PyQt6.sip",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        # GUI style module added recently
        "logo2svg.gui.style",
    ]
)

# PyQt6 needs its DLLs / .pyd files and Qt plugin data explicitly collected
pyqt6_binaries = collect_dynamic_libs("PyQt6")
pyqt6_datas = collect_data_files("PyQt6")

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=pyqt6_binaries,
    datas=pyqt6_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["runtime_hook_pyqt6.py"],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="logo2svg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed app — no console flash
    icon=None,              # set to "icon.ico" if you add one later
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="logo2svg",
)
