# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the QuickLayer GUI executable.

Tuned for minimal bundle size and fast startup:
- Only the Qt modules actually used are kept (Core, Gui, Widgets, Svg).
- Large unused Qt subsystems and numpy/scipy extras are excluded.
- UPX compression is enabled for binaries.
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)

# Keep hidden imports explicit and minimal for faster startup.
hidden_imports = [
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
    "PyQt6.sip",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    # Explicit GUI modules for robust frozen imports
    "logo2svg.gui.style",
    "logo2svg.gui.preview_panel",
    "logo2svg.gui.layer_panel",
    "logo2svg.gui.main_window",
    "logo2svg.gui.source_panel",
    "logo2svg.gui.settings_dialog",
]

# PyQt6 needs its DLLs / .pyd files and Qt plugin data explicitly collected
pyqt6_binaries = collect_dynamic_libs("PyQt6")
pyqt6_datas = collect_data_files("PyQt6")

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=pyqt6_binaries,
    datas=pyqt6_datas + [("logo2svg/icons", "logo2svg/icons")],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["runtime_hook_pyqt6.py"],
    excludes=[
        # Stdlib modules not used
        "tkinter", "unittest", "pydoc", "doctest", "difflib",
        "xmlrpc", "ftplib", "imaplib", "smtplib", "poplib", "nntplib",
        "telnetlib", "cgi", "cgitb", "webbrowser", "turtle",
        "turtledemo", "ensurepip", "venv", "lib2to3", "idlelib",
        "test", "distutils", "setuptools", "pip",

        # Large science/data libs we don't use
        "matplotlib", "IPython", "jupyter", "scipy", "pandas",
        "sklearn", "scikit_learn", "h5py", "lxml", "zmq",
        "tornado", "notebook", "sphinx",

        # Qt subsystems not used (huge savings)
        "PyQt6.Qt3D",           "PyQt6.Qt3DAnimation",
        "PyQt6.Qt3DCore",       "PyQt6.Qt3DExtras",
        "PyQt6.Qt3DInput",      "PyQt6.Qt3DLogic",
        "PyQt6.Qt3DRender",     "PyQt6.QtBluetooth",
        "PyQt6.QtCharts",       "PyQt6.QtDataVisualization",
        "PyQt6.QtDBus",         "PyQt6.QtDesigner",
        "PyQt6.QtHelp",         "PyQt6.QtHttpServer",
        "PyQt6.QtLocation",     "PyQt6.QtMultimedia",
        "PyQt6.QtMultimediaWidgets",
        "PyQt6.QtNetworkAuth",  "PyQt6.QtNfc",
        "PyQt6.QtOpenGL",       "PyQt6.QtOpenGLWidgets",
        "PyQt6.QtPdf",          "PyQt6.QtPdfWidgets",
        "PyQt6.QtPositioning",  "PyQt6.QtPrintSupport",
        "PyQt6.QtQml",          "PyQt6.QtQuick",
        "PyQt6.QtQuick3D",      "PyQt6.QtQuickWidgets",
        "PyQt6.QtRemoteObjects","PyQt6.QtScxml",
        "PyQt6.QtSensors",      "PyQt6.QtSerialPort",
        "PyQt6.QtSpatialAudio", "PyQt6.QtSql",
        "PyQt6.QtTest",         "PyQt6.QtTextToSpeech",
        "PyQt6.QtWebChannel",   "PyQt6.QtWebEngine",
        "PyQt6.QtWebEngineCore","PyQt6.QtWebEngineWidgets",
        "PyQt6.QtWebSockets",   "PyQt6.QtXml",

        # CLI-only dep — not needed in GUI exe
        "click",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# Native splash screen — displayed by the bootloader before Python starts,
# so the user sees immediate feedback on launch.
splash = Splash(
    "logo2svg/icons/quicklayer_256.png",
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    text_color="black",
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    getattr(splash, "binaries", []),
    [],
    exclude_binaries=True,
    name="QuickLayer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed app — no console flash
    icon="logo2svg/icons/quicklayer.ico",
)

# ---------------------------------------------------------------------------
# Filter out large Qt DLLs / data we don't need (translations, qml, etc.)
# ---------------------------------------------------------------------------
import re

_EXCLUDE_BINARIES_RE = re.compile(
    r"(?i)"
    r"(Qt6Web|Qt6Quick|Qt6Qml|Qt6Designer|Qt6Help|Qt63D|Qt6Bluetooth"
    r"|Qt6Charts|Qt6DataVis|Qt6Location|Qt6Multimedia|Qt6Nfc"
    r"|Qt6Pdf|Qt6Positioning|Qt6RemoteObjects|Qt6Scxml"
    r"|Qt6Sensors|Qt6Serial|Qt6Spatial|Qt6Sql|Qt6Test"
    r"|Qt6TextToSpeech|Qt6Xml|Qt6HttpServer|Qt6OpenGL"
    r"|Qt6PrintSupport|Qt6Labs|Qt6Pdf|Qt6Virtual"
    r"|opengl32sw|d3dcompiler|libGLESv2|libEGL"
    r"|Qt6Network(?!Auth)"  # keep Qt6Network only if needed
    r")"
)
_EXCLUDE_DATAS_RE = re.compile(
    r"(?i)"
    r"(translations[/\\]|qml[/\\]|QtWebEngine"
    r"|Qt6Web|Qt6Quick|Qt63D|Qt6Charts|Qt6DataVis"
    r"|Qt6Designer|Qt6Help|Qt6Pdf"
    r"|doc[/\\]|examples[/\\]|include[/\\]"
    r"|\.qm$|LICENSE|NOTICE)"
)

filtered_binaries = [b for b in a.binaries if not _EXCLUDE_BINARIES_RE.search(b[0])]
filtered_datas = [d for d in a.datas + getattr(splash, "datas", [])
                  if not _EXCLUDE_DATAS_RE.search(d[0])]

coll = COLLECT(
    exe,
    filtered_binaries,
    a.zipfiles,
    filtered_datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="QuickLayer",
)
