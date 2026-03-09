"""Main application window for the QuickLayer GUI.

Layout
------
+-- Menu bar -----------------------------------------------+
+-- Header bar ---------------------------------------------+
|  QuickLayer | [Open] [Import SVG] [Export SVGs] | ... | G |
+-----------------------------+----------------------------+
|  SOURCE IMAGE               |                            |
|  (large view of original)   |       PREVIEW AREA         |
|                             |  (composite with layers)   |
+-----------------------------+  (drag and drop to open)   |
|  LAYERS            [3]      |  (zoom/pan support)        |
|  +----------------------+   |                            |
|  | V . Layer 1          |   |                            |
|  | V . Layer 2          |   |                            |
|  +----------------------+   |                            |
|  [Merge Selected]           |                            |
+-----------------------------+----------------------------+
|  Ready                                         xxxxxxxx  |
+----------------------------------------------------------+
"""

from __future__ import annotations

import sys
from pathlib import Path

from logo2svg import __version__

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..session import Session
from .layer_panel import LayerPanel
from .preview_panel import PreviewPanel
from .settings_dialog import SettingsDialog
from .source_panel import SourcePanel
from .style import ACCENT, BG, BORDER, CARD, SURFACE, TEXT, TEXT_SEC
from .workers import (
    ExportWorker,
    ImportSvgWorker,
    LoadWorker,
    PreprocessWorker,
    QuantizeWorker,
    TraceWorker,
)

IMAGE_FILTER = (
    "All Supported Images (*.png *.jpg *.jpeg *.webp *.bmp *.svg);;"
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;BMP (*.bmp);;SVG (*.svg)"
)
SVG_FILTER = "SVG Files (*.svg)"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}
SVG_EXTENSIONS = {".svg"}


class MainWindow(QMainWindow):
    """Top-level window with modern light-themed layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"QuickLayer v{__version__}")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        # Window icon (taskbar + title bar) -- handle frozen PyInstaller builds
        if getattr(sys, "frozen", False):
            _base = Path(sys._MEIPASS) / "logo2svg"
        else:
            _base = Path(__file__).resolve().parent.parent
        icon_path = _base / "icons" / "quicklayer.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Session & settings
        self._session = Session()
        self._bg_color: str = ""
        self._remove_tm: bool = True
        self._worker = None  # keep worker alive while running
        self._selected_indices: list[int] = []

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()

    # =================================================================
    #  UI Construction
    # =================================================================

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # -- File menu --
        file_menu = mb.addMenu("&File")

        open_act = QAction("&Open\u2026", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        import_svg_act = QAction("Import &SVG\u2026", self)
        import_svg_act.setShortcut("Ctrl+I")
        import_svg_act.triggered.connect(self._on_import_svg)
        file_menu.addAction(import_svg_act)

        export_act = QAction("&Export SVGs\u2026", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

        export_png_act = QAction("Export &PNG\u2026", self)
        export_png_act.triggered.connect(self._on_export_png)
        file_menu.addAction(export_png_act)

        file_menu.addSeparator()

        save_project_act = QAction("&Save Project\u2026", self)
        save_project_act.setShortcut("Ctrl+S")
        save_project_act.triggered.connect(self._on_save_project)
        file_menu.addAction(save_project_act)

        load_project_act = QAction("&Load Project\u2026", self)
        load_project_act.setShortcut("Ctrl+Shift+O")
        load_project_act.triggered.connect(self._on_load_project)
        file_menu.addAction(load_project_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # -- Edit menu --
        edit_menu = mb.addMenu("&Edit")

        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_act.triggered.connect(self._on_undo)
        self._undo_act.setEnabled(False)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("&Redo", self)
        self._redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_act.triggered.connect(self._on_redo)
        self._redo_act.setEnabled(False)
        edit_menu.addAction(self._redo_act)

        edit_menu.addSeparator()

        settings_act = QAction("&Settings\u2026", self)
        settings_act.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_act)

        # -- Tools menu --
        tools_menu = mb.addMenu("&Tools")

        add_text_act = QAction("Add &Text\u2026", self)
        add_text_act.setShortcut("Ctrl+T")
        add_text_act.triggered.connect(self._on_add_text)
        tools_menu.addAction(add_text_act)

        add_outline_act = QAction("Add &Outline to Layer\u2026", self)
        add_outline_act.triggered.connect(self._on_add_outline)
        tools_menu.addAction(add_outline_act)

        add_obj_border_act = QAction("Add &Border to Selection\u2026", self)
        add_obj_border_act.setShortcut("Ctrl+B")
        add_obj_border_act.triggered.connect(self._on_add_object_border)
        tools_menu.addAction(add_obj_border_act)

        add_border_act = QAction("Add Ca&nvas Border\u2026", self)
        add_border_act.triggered.connect(self._on_add_canvas_border)
        tools_menu.addAction(add_border_act)

        change_color_act = QAction("&Change Layer Color\u2026", self)
        change_color_act.triggered.connect(self._on_change_color_tool)
        tools_menu.addAction(change_color_act)

        tools_menu.addSeparator()

        preprocess_act = QAction("&Image Preprocessing\u2026", self)
        preprocess_act.triggered.connect(self._on_preprocess)
        tools_menu.addAction(preprocess_act)

        remove_bg_act = QAction("Remove &Background\u2026", self)
        remove_bg_act.triggered.connect(self._on_remove_background)
        tools_menu.addAction(remove_bg_act)

        # -- View menu --
        view_menu = mb.addMenu("&View")

        zoom_in_act = QAction("Zoom &In", self)
        zoom_in_act.setShortcut("Ctrl+=")
        zoom_in_act.triggered.connect(lambda: self._preview.zoom_in())
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("Zoom &Out", self)
        zoom_out_act.setShortcut("Ctrl+-")
        zoom_out_act.triggered.connect(lambda: self._preview.zoom_out())
        view_menu.addAction(zoom_out_act)

        zoom_reset_act = QAction("&Reset Zoom", self)
        zoom_reset_act.setShortcut("Ctrl+0")
        zoom_reset_act.triggered.connect(lambda: self._preview.zoom_reset())
        view_menu.addAction(zoom_reset_act)

        # -- Help menu --
        help_menu = mb.addMenu("&Help")

        guide_act = QAction("&Quick Guide\u2026", self)
        guide_act.setShortcut("F1")
        guide_act.triggered.connect(self._on_quick_guide)
        help_menu.addAction(guide_act)

        help_menu.addSeparator()

        about_act = QAction("&About QuickLayer", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

    def _on_quick_guide(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("QuickLayer \u2014 Quick Guide")
        dlg.resize(520, 560)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)

        from PyQt6.QtWidgets import QTextBrowser

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet("font-size: 13px; border: none;")
        browser.setHtml(
            "<style>"
            "h2 { margin-top: 14px; margin-bottom: 4px; }"
            "table { border-collapse: collapse; width: 100%; margin: 6px 0; }"
            "th, td { text-align: left; padding: 3px 8px; border-bottom: 1px solid #ddd; }"
            "th { font-weight: 600; }"
            "code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }"
            "</style>"
            "<h2>Keyboard Shortcuts</h2>"
            "<table>"
            "<tr><th>Action</th><th>Shortcut</th></tr>"
            "<tr><td>Open image</td><td><code>Ctrl+O</code></td></tr>"
            "<tr><td>Import SVG</td><td><code>Ctrl+I</code></td></tr>"
            "<tr><td>Export SVGs</td><td><code>Ctrl+E</code></td></tr>"
            "<tr><td>Save project</td><td><code>Ctrl+S</code></td></tr>"
            "<tr><td>Load project</td><td><code>Ctrl+Shift+O</code></td></tr>"
            "<tr><td>Undo / Redo</td><td><code>Ctrl+Z</code> / <code>Ctrl+Y</code></td></tr>"
            "<tr><td>Add text</td><td><code>Ctrl+T</code></td></tr>"
            "<tr><td>Add border to selection</td><td><code>Ctrl+B</code></td></tr>"
            "<tr><td>Zoom in / out / reset</td><td><code>Ctrl+=</code> / <code>Ctrl+-</code> / <code>Ctrl+0</code></td></tr>"
            "<tr><td>Filter layers (focus search)</td><td><code>Ctrl+F</code></td></tr>"
            "<tr><td>Quick Guide (this dialog)</td><td><code>F1</code></td></tr>"
            "</table>"

            "<h2>Layer Selection</h2>"
            "<table>"
            "<tr><th>Action</th><th>How</th></tr>"
            "<tr><td>Select one layer</td><td>Click it</td></tr>"
            "<tr><td>Add/remove from selection</td><td><code>Ctrl+Click</code></td></tr>"
            "<tr><td>Select a range</td><td><code>Shift+Click</code></td></tr>"
            "<tr><td>Add range to selection</td><td><code>Ctrl+Shift+Click</code></td></tr>"
            "<tr><td>Select all</td><td><code>Ctrl+A</code> or <b>Select All</b> button</td></tr>"
            "<tr><td>Invert selection</td><td><b>Invert</b> button</td></tr>"
            "<tr><td>Select by color</td><td>Select a layer, then click <b>Select Same Color</b></td></tr>"
            "<tr><td>Select small objects</td><td>Adjust the <b>Min px</b> slider, click <b>Select Small</b></td></tr>"
            "<tr><td>Filter by name / color</td><td>Type in the search box (<code>Ctrl+F</code> to focus)</td></tr>"
            "</table>"

            "<h2>Layer Actions</h2>"
            "<table>"
            "<tr><th>Action</th><th>How</th></tr>"
            "<tr><td>Delete selected</td><td><code>Delete</code> key or <b>Delete Selected</b> button</td></tr>"
            "<tr><td>Merge selected</td><td><b>Merge Selected</b> button (2+ layers)</td></tr>"
            "<tr><td>Merge by color</td><td><b>Merge by Color</b> \u2014 consolidates all same-colour layers</td></tr>"
            "<tr><td>Recolor selected</td><td><b>Recolor Selected</b> \u2014 change colour of all selected layers</td></tr>"
            "<tr><td>Duplicate</td><td>Select a layer, click <b>Duplicate</b></td></tr>"
            "<tr><td>Rename</td><td>Double-click the layer name</td></tr>"
            "<tr><td>Change color</td><td>Click the colour swatch</td></tr>"
            "<tr><td>Toggle visibility</td><td>Checkbox on each layer</td></tr>"
            "<tr><td>Reorder</td><td>\u25B2 / \u25BC buttons on each layer</td></tr>"
            "<tr><td>Sort layers</td><td>Use the <b>Sort</b> dropdown (by color, size, or position)</td></tr>"
            "<tr><td>Compact view</td><td>Click the <b>\u2630</b> toggle button to halve row height</td></tr>"
            "</table>"

            "<h2>Right-Click Menu</h2>"
            "<p>Right-click any layer card for a context menu with:</p>"
            "<ul>"
            "<li>Change Color, Select Same Color</li>"
            "<li>Delete, Rename</li>"
            "<li>Hide Others (solo this layer), Show All</li>"
            "<li>Move Up / Move Down</li>"
            "</ul>"

            "<h2>Preview Canvas</h2>"
            "<table>"
            "<tr><th>Action</th><th>How</th></tr>"
            "<tr><td>Select layer on canvas</td><td>Click on a shape in the preview</td></tr>"
            "<tr><td>Move layer</td><td>Drag a shape in the preview</td></tr>"
            "<tr><td>Edit text layer</td><td>Double-click text in the preview</td></tr>"
            "<tr><td>Zoom</td><td>Scroll wheel, or <code>Ctrl+=</code> / <code>Ctrl+-</code></td></tr>"
            "</table>"

            "<h2>Tips for Many Objects</h2>"
            "<p>When using <b>By Object</b> mode, each connected component becomes its own layer. "
            "This can result in dozens of layers. Here's how to manage them efficiently:</p>"
            "<ul>"
            "<li><b>Compact view</b> (\u2630 toggle) halves row height so you can see more layers at once.</li>"
            "<li><b>Filter</b> by typing in the search box (<code>Ctrl+F</code>) to narrow the list by name or hex colour.</li>"
            "<li><b>Sort</b> by color to group same-colour objects, or by size to find small fragments.</li>"
            "<li><b>Select Small</b> + adjust the <b>Min px</b> slider to find and remove tiny noise objects.</li>"
            "<li><b>Merge by Color</b> recombines same-colour objects into single layers.</li>"
            "<li>Use <b>Select Same Color</b> to quickly select all pieces of one colour, "
            "then merge, delete, or recolor them together.</li>"
            "<li><b>Shift+Click</b> to select contiguous ranges in the list.</li>"
            "<li><b>Invert</b> selection to keep a few layers and delete the rest.</li>"
            "<li>The <b>Delete</b> key removes all selected layers at once (with confirmation).</li>"
            "<li><b>Right-click</b> any layer for quick actions: Hide Others, Select Same Color, etc.</li>"
            "</ul>"
        )
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dlg.accept)
        from PyQt6.QtWidgets import QHBoxLayout as _HBox

        btn_row = _HBox()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.exec()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About QuickLayer",
            f"<h2>QuickLayer</h2>"
            f"<p>Version {__version__}</p>"
            f"<p>Convert logos into color-separated SVGs<br>"
            f"for multi-color 3D printing.</p>"
            f"<p>\u00a9 2026 QuickLayer</p>",
        )

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- Header toolbar --
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(52)

        hbar = QHBoxLayout(header)
        hbar.setContentsMargins(16, 0, 16, 0)
        hbar.setSpacing(8)

        # App title
        title = QLabel("QuickLayer")
        title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {TEXT}; "
            "margin-right: 12px;"
        )
        hbar.addWidget(title)

        # Separator
        hbar.addWidget(self._vsep())
        hbar.addSpacing(8)

        # Open button (primary)
        open_btn = QPushButton("Open")
        open_btn.setProperty("cssClass", "primary")
        open_btn.setToolTip("Open an image file  (Ctrl+O)")
        open_btn.setFixedHeight(32)
        open_btn.clicked.connect(self._on_open)
        hbar.addWidget(open_btn)

        # Import SVG button
        import_svg_btn = QPushButton("Import SVG")
        import_svg_btn.setToolTip("Import an SVG file as layers  (Ctrl+I)")
        import_svg_btn.setFixedHeight(32)
        import_svg_btn.clicked.connect(self._on_import_svg)
        hbar.addWidget(import_svg_btn)

        # Export button
        export_btn = QPushButton("Export SVGs")
        export_btn.setToolTip("Export SVG files  (Ctrl+E)")
        export_btn.setFixedHeight(32)
        export_btn.clicked.connect(self._on_export)
        hbar.addWidget(export_btn)

        remove_bg_btn = QPushButton("Remove Background")
        remove_bg_btn.setToolTip("Rebuild the foreground mask from the original image")
        remove_bg_btn.setFixedHeight(32)
        remove_bg_btn.clicked.connect(self._on_remove_background)
        hbar.addWidget(remove_bg_btn)

        hbar.addSpacing(12)
        hbar.addWidget(self._vsep())
        hbar.addSpacing(8)

        # Colors control with Auto toggle + manual spin box
        clr_label = QLabel("Colors")
        clr_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px;")
        hbar.addWidget(clr_label)

        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setChecked(True)
        self._auto_btn.setFixedHeight(32)
        self._auto_btn.setToolTip("Auto-detect number of colors")
        self._auto_btn.toggled.connect(self._on_auto_toggled)
        hbar.addWidget(self._auto_btn)

        self._color_spin = QSpinBox()
        self._color_spin.setRange(2, 20)
        self._color_spin.setValue(4)
        self._color_spin.setToolTip(
            "Number of colors for quantization"
        )
        self._color_spin.setFixedWidth(68)
        self._color_spin.setFixedHeight(32)
        self._color_spin.setEnabled(False)
        self._color_spin.valueChanged.connect(self._on_color_count_changed)
        hbar.addWidget(self._color_spin)

        hbar.addSpacing(12)
        hbar.addWidget(self._vsep())
        hbar.addSpacing(8)

        # Separation mode toggle
        sep_label = QLabel("Separate")
        sep_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px;")
        hbar.addWidget(sep_label)

        self._mode_color_btn = QPushButton("By Color")
        self._mode_color_btn.setCheckable(True)
        self._mode_color_btn.setChecked(True)
        self._mode_color_btn.setFixedHeight(32)
        self._mode_color_btn.setToolTip("One layer per color (default)")
        hbar.addWidget(self._mode_color_btn)

        self._mode_object_btn = QPushButton("By Object")
        self._mode_object_btn.setCheckable(True)
        self._mode_object_btn.setChecked(False)
        self._mode_object_btn.setFixedHeight(32)
        self._mode_object_btn.setToolTip("One layer per connected object")
        hbar.addWidget(self._mode_object_btn)

        # Mutual exclusivity via QButtonGroup
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_color_btn, 0)
        self._mode_group.addButton(self._mode_object_btn, 1)
        self._mode_group.idToggled.connect(self._on_separation_mode_changed)

        hbar.addSpacing(12)
        hbar.addWidget(self._vsep())
        hbar.addSpacing(8)

        # Undo / Redo buttons
        self._undo_btn = QPushButton("\u21B6")
        self._undo_btn.setProperty("cssClass", "icon")
        self._undo_btn.setToolTip("Undo  (Ctrl+Z)")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_undo)
        hbar.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("\u21B7")
        self._redo_btn.setProperty("cssClass", "icon")
        self._redo_btn.setToolTip("Redo  (Ctrl+Y)")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_redo)
        hbar.addWidget(self._redo_btn)

        hbar.addSpacing(4)

        # Zoom buttons
        zoom_out_btn = QPushButton("\u2212")
        zoom_out_btn.setProperty("cssClass", "icon")
        zoom_out_btn.setToolTip("Zoom out  (Ctrl+-)")
        zoom_out_btn.clicked.connect(lambda: self._preview.zoom_out())
        hbar.addWidget(zoom_out_btn)

        zoom_reset_btn = QPushButton("Fit")
        zoom_reset_btn.setProperty("cssClass", "icon")
        zoom_reset_btn.setToolTip("Fit to window  (Ctrl+0)")
        zoom_reset_btn.setStyleSheet(
            f"QPushButton {{ font-size: 10px; font-weight: 600; background: transparent; "
            f"border: 1px solid transparent; border-radius: 4px; "
            f"min-width: 28px; max-width: 32px; min-height: 28px; max-height: 28px; "
            f"color: {TEXT_SEC}; padding: 2px; }}"
            f"QPushButton:hover {{ background-color: {CARD}; border-color: {BORDER}; color: {TEXT}; }}"
        )
        zoom_reset_btn.clicked.connect(lambda: self._preview.zoom_reset())
        hbar.addWidget(zoom_reset_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setProperty("cssClass", "icon")
        zoom_in_btn.setToolTip("Zoom in  (Ctrl+=)")
        zoom_in_btn.clicked.connect(lambda: self._preview.zoom_in())
        hbar.addWidget(zoom_in_btn)

        hbar.addStretch()

        # Settings gear
        settings_btn = QPushButton("\u2699")
        settings_btn.setToolTip("Pipeline settings")
        settings_btn.setFixedSize(36, 36)
        settings_btn.setStyleSheet(
            f"QPushButton {{ font-size: 18px; background: transparent; "
            f"border: 1px solid transparent; border-radius: 6px; "
            f"min-width: 36px; max-width: 36px; "
            f"min-height: 36px; max-height: 36px; color: {TEXT_SEC}; }}"
            f"QPushButton:hover {{ background-color: {CARD}; "
            f"border-color: {BORDER}; color: {TEXT}; }}"
        )
        settings_btn.clicked.connect(self._on_settings)
        hbar.addWidget(settings_btn)

        root.addWidget(header)

        # -- Main content area (splitter) --
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)

        # Left: sidebar with source image (top) + layers (bottom)
        left_sidebar = QFrame()
        left_sidebar.setObjectName("leftSidebar")
        left_sidebar.setMinimumWidth(380)
        left_sidebar.setMaximumWidth(600)

        left_outer = QVBoxLayout(left_sidebar)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(0)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setHandleWidth(1)

        # Source image viewer (top of left side, large)
        self._source_panel = SourcePanel()
        left_splitter.addWidget(self._source_panel)

        # Layer panel (bottom of left side)
        self._layer_panel = LayerPanel()
        self._layer_panel.visibility_toggled.connect(
            self._on_visibility_toggled
        )
        self._layer_panel.color_picker_requested.connect(
            self._on_color_picker
        )
        self._layer_panel.delete_requested.connect(self._on_delete_layer)
        self._layer_panel.delete_multiple_requested.connect(
            self._on_delete_layers
        )
        self._layer_panel.merge_requested.connect(self._on_merge_layers)
        self._layer_panel.merge_by_color_requested.connect(
            self._on_merge_by_color
        )
        self._layer_panel.recolor_selected_requested.connect(
            self._on_recolor_selected
        )
        self._layer_panel.move_up_requested.connect(self._on_move_layer_up)
        self._layer_panel.move_down_requested.connect(self._on_move_layer_down)
        self._layer_panel.duplicate_requested.connect(self._on_duplicate_layer)
        self._layer_panel.rename_requested.connect(self._on_rename_layer)
        self._layer_panel.selection_changed.connect(
            self._on_layer_selection_changed
        )
        self._layer_panel.hide_others_requested.connect(
            self._on_hide_others
        )
        self._layer_panel.show_all_requested.connect(
            self._on_show_all
        )
        self._layer_panel.hide_all_requested.connect(
            self._on_hide_all
        )
        left_splitter.addWidget(self._layer_panel)

        # Source image gets more space than layers
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        left_splitter.setSizes([350, 250])

        left_outer.addWidget(left_splitter)
        main_splitter.addWidget(left_sidebar)

        # Right: preview panel (main focus, stretches)
        self._preview = PreviewPanel()
        self._preview.set_hit_test_callback(self._hit_test_layer)
        self._preview.set_bbox_callback(self._get_layer_bbox)
        self._preview.layer_selected.connect(self._on_canvas_layer_selected)
        self._preview.layer_moved.connect(self._on_canvas_layer_moved)
        self._preview.layer_double_clicked.connect(self._on_canvas_layer_double_clicked)
        main_splitter.addWidget(self._preview)

        # Stretch factors: sidebar stays, preview stretches
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([560, 700])

        root.addWidget(main_splitter, stretch=1)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(
            f"color: {TEXT_SEC}; font-size: 12px;"
        )
        sb.addWidget(self._status_label, stretch=1)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setFixedHeight(6)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        sb.addPermanentWidget(self._progress)

    @staticmethod
    def _vsep() -> QFrame:
        """Return a thin vertical separator line."""
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(28)
        sep.setStyleSheet(f"background-color: {BORDER};")
        return sep

    # =================================================================
    #  Drag and Drop
    # =================================================================

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                suffix = Path(url.toLocalFile()).suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            suffix = Path(path).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                self.open_file(path)
                return

    # =================================================================
    #  Public API (for programmatic use / tests)
    # =================================================================

    def open_file(self, path: str) -> None:
        """Open and begin processing *path*."""
        self._set_busy(True, f"Loading {Path(path).name}\u2026")
        worker = LoadWorker(
            self._session,
            path,
            bg_color=self._bg_color or None,
            remove_tm=self._remove_tm,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_load_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    # =================================================================
    #  Toolbar / Menu Handlers
    # =================================================================

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", IMAGE_FILTER
        )
        if path:
            self.open_file(path)

    def _on_import_svg(self) -> None:
        """Import an external SVG as new layers."""
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Import SVG",
                "Open an image first to set the canvas size."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import SVG", "", SVG_FILTER
        )
        if path:
            self._import_svg_file(path)

    def _import_svg_file(self, path: str) -> None:
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Import SVG",
                "Open an image first to set the canvas size."
            )
            return
        self._set_busy(True, f"Importing {Path(path).name}\u2026")
        worker = ImportSvgWorker(self._session, path)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_import_svg_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _on_export(self) -> None:
        if not self._session.is_quantized:
            QMessageBox.information(
                self, "Export", "Open and quantize an image first."
            )
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export Directory")
        if not out_dir:
            return
        self._set_busy(True, "Exporting\u2026")
        worker = ExportWorker(self._session, out_dir, combined=True)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_export_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _on_settings(self) -> None:
        dlg = SettingsDialog(
            min_area=self._session.min_area,
            alphamax=self._session.alphamax,
            opttolerance=self._session.opttolerance,
            turdsize=self._session.turdsize,
            bg_color=self._bg_color,
            remove_tm=self._remove_tm,
            parent=self,
        )
        if dlg.exec() != SettingsDialog.DialogCode.Accepted:
            return

        changed = False
        if dlg.min_area != self._session.min_area:
            self._session.min_area = dlg.min_area
            changed = True
        if dlg.alphamax != self._session.alphamax:
            self._session.alphamax = dlg.alphamax
            changed = True
        if dlg.opttolerance != self._session.opttolerance:
            self._session.opttolerance = dlg.opttolerance
            changed = True
        if dlg.turdsize != self._session.turdsize:
            self._session.turdsize = dlg.turdsize
            changed = True
        if dlg.bg_color != self._bg_color:
            self._bg_color = dlg.bg_color
            if self._session.path:
                self.open_file(str(self._session.path))
                return
        if dlg.remove_tm != self._remove_tm:
            self._remove_tm = dlg.remove_tm
            if self._session.path:
                self.open_file(str(self._session.path))
                return

        if changed and self._session.is_loaded:
            self._run_quantize()

    def _on_auto_toggled(self, checked: bool) -> None:
        """Auto button toggled."""
        self._color_spin.setEnabled(not checked)
        if checked and self._session.is_loaded:
            self._run_quantize()

    def _on_color_count_changed(self, _value: int) -> None:
        """Manual color count spinner changed."""
        if not self._auto_btn.isChecked() and self._session.is_loaded:
            self._run_quantize()

    def _on_separation_mode_changed(self, button_id: int, checked: bool) -> None:
        """Separation mode toggle changed."""
        if not checked:
            return
        mode = "color" if button_id == 0 else "object"
        self._session.set_separation_mode(mode)
        if self._session.is_quantized:
            self._refresh_layers()
            self._run_trace()

    # =================================================================
    #  Undo / Redo
    # =================================================================

    def _on_undo(self) -> None:
        if self._session.undo():
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_redo(self) -> None:
        if self._session.redo():
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _update_undo_redo_state(self) -> None:
        self._undo_btn.setEnabled(self._session.can_undo)
        self._redo_btn.setEnabled(self._session.can_redo)
        self._undo_act.setEnabled(self._session.can_undo)
        self._redo_act.setEnabled(self._session.can_redo)

    # =================================================================
    #  Tools: Text, Outline, Border, Change Color, Preprocessing
    # =================================================================

    def _on_add_text(self) -> None:
        """Show a dialog to add text as a new layer."""
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Add Text", "Open an image first."
            )
            return

        dlg = _TextDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session.add_text(
            dlg.text_value,
            color=dlg.color_value,
            font_scale=dlg.font_scale_value,
            thickness=dlg.thickness_value,
        )
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_add_outline(self) -> None:
        """Add an outline to a selected layer."""
        if not self._session.is_quantized:
            QMessageBox.information(
                self, "Add Outline",
                "Quantize an image first to create layers."
            )
            return

        layers = self._session.get_layers()
        if not layers:
            return

        dlg = _OutlineDialog(len(layers), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session.add_outline(
            dlg.layer_index, dlg.width_value, dlg.color_value
        )
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_add_object_border(self) -> None:
        """Add a border around selected layers (visible/checked layers)."""
        if not self._session.is_quantized:
            QMessageBox.information(
                self, "Add Border",
                "Quantize an image first to create layers."
            )
            return

        # Use checked (visible) layers from the layer panel as selection
        selected = self._layer_panel.get_selected_indices()
        if not selected:
            QMessageBox.information(
                self, "Add Border",
                "Check (select) one or more layers in the layer panel first.\n\n"
                "The border will be drawn around the combined shape of all "
                "selected layers."
            )
            return

        dlg = _ObjectBorderDialog(selected, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session.add_object_border(
            selected, dlg.width_value, dlg.color_value
        )
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_add_canvas_border(self) -> None:
        """Add a border around the entire canvas."""
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Add Border", "Open an image first."
            )
            return

        dlg = _BorderDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session.add_canvas_border(
            dlg.width_value, dlg.color_value
        )
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_change_color_tool(self) -> None:
        """Change the color of a specific layer."""
        if not self._session.is_quantized:
            QMessageBox.information(
                self, "Change Color",
                "Quantize an image first to create layers."
            )
            return

        layers = self._session.get_layers()
        if not layers:
            return

        # Ask which layer
        names = [f"Layer {i + 1} ({l.hex_color})" for i, l in enumerate(layers)]
        name, ok = QInputDialog.getItem(
            self, "Change Color", "Select layer:", names, 0, False
        )
        if not ok:
            return
        idx = names.index(name)

        current = QColor(layers[idx].hex_color)
        color = QColorDialog.getColor(current, self, "Pick New Color")
        if color.isValid():
            self._session.change_color(idx, color.name())
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_preprocess(self) -> None:
        """Show preprocessing options dialog."""
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Preprocessing", "Open an image first."
            )
            return

        # Analyze first
        report = self._session.analyze_image()

        dlg = _PreprocessDialog(report, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if not dlg.contrast and not dlg.sharpen and not dlg.denoise:
            return

        self._set_busy(True, "Preprocessing\u2026")
        worker = PreprocessWorker(
            self._session,
            contrast=dlg.contrast,
            sharpen=dlg.sharpen,
            denoise=dlg.denoise,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_preprocess_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _on_remove_background(self) -> None:
        """Reload the current image and re-apply background detection."""
        if not self._session.is_loaded or self._session.path is None:
            QMessageBox.information(
                self, "Remove Background", "Open an image first."
            )
            return

        dlg = _BackgroundDialog(
            bg_color=self._bg_color,
            remove_tm=self._remove_tm,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._bg_color = dlg.bg_color
        self._remove_tm = dlg.remove_tm

        self._set_busy(True, "Removing background\u2026")
        worker = LoadWorker(
            self._session,
            str(self._session.path),
            bg_color=self._bg_color or None,
            remove_tm=self._remove_tm,
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_load_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    # =================================================================
    #  Layer-panel signal handlers
    # =================================================================

    def _on_visibility_toggled(self, index: int, visible: bool) -> None:
        self._session.set_layer_visibility(index, visible)
        self._refresh_preview()

    def _on_color_picker(self, index: int) -> None:
        layers = self._session.get_layers()
        if index >= len(layers):
            return
        current = QColor(layers[index].hex_color)
        color = QColorDialog.getColor(current, self, "Pick Layer Color")
        if color.isValid():
            self._session.change_color(index, color.name())
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_delete_layer(self, index: int) -> None:
        self._session.remove_color(index)
        self._selected_indices = []
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_delete_layers(self, indices: list[int]) -> None:
        """Bulk-delete multiple layers at once."""
        self._session.remove_colors(indices)
        self._selected_indices = []
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_merge_layers(self, indices: list[int]) -> None:
        self._session.merge_colors(indices)
        self._selected_indices = []
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_merge_by_color(self) -> None:
        """Merge all layers sharing the same colour into single layers."""
        removed = self._session.merge_by_color()
        if removed:
            self._selected_indices = []
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()
            self._status_label.setText(
                f"Merged {removed} duplicate-colour layers"
            )
        else:
            self._status_label.setText("No duplicate colours to merge")

    def _on_recolor_selected(self, indices: list[int]) -> None:
        """Change the colour of all selected layers at once."""
        layers = self._session.get_layers()
        if not indices or not layers:
            return
        # Use the colour of the first selected layer as the starting colour
        first_idx = indices[0]
        if first_idx >= len(layers):
            return
        current = QColor(layers[first_idx].hex_color)
        color = QColorDialog.getColor(current, self, "Pick Color for Selected Layers")
        if color.isValid():
            self._session.change_colors(indices, color.name())
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_move_layer_up(self, index: int) -> None:
        if self._session.move_layer_up(index):
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_move_layer_down(self, index: int) -> None:
        if self._session.move_layer_down(index):
            self._refresh_layers()
            self._refresh_preview()
            self._update_undo_redo_state()

    def _on_duplicate_layer(self, index: int) -> None:
        self._session.duplicate_layer(index)
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_rename_layer(self, index: int, name: str) -> None:
        self._session.rename_layer(index, name)

    def _on_layer_selection_changed(self, indices: list[int]) -> None:
        self._selected_indices = indices
        self._refresh_preview()

    def _on_hide_others(self, keep_index: int) -> None:
        """Hide all layers except the one at *keep_index*."""
        layers = self._session.get_layers()
        for layer in layers:
            self._session.set_layer_visibility(
                layer.index, layer.index == keep_index
            )
        self._refresh_layers()
        self._refresh_preview()

    def _on_show_all(self) -> None:
        """Make all layers visible."""
        layers = self._session.get_layers()
        for layer in layers:
            self._session.set_layer_visibility(layer.index, True)
        self._refresh_layers()
        self._refresh_preview()

    def _on_hide_all(self) -> None:
        """Hide all layers."""
        layers = self._session.get_layers()
        for layer in layers:
            self._session.set_layer_visibility(layer.index, False)
        self._refresh_layers()
        self._refresh_preview()

    # =================================================================
    #  Canvas interaction handlers (click-to-select, drag-to-move)
    # =================================================================

    def _hit_test_layer(self, x: int, y: int):
        """Callback for PreviewPanel: returns layer index at (x, y) or None."""
        return self._session.layer_at_pixel(x, y)

    def _get_layer_bbox(self, index: int):
        """Callback for PreviewPanel: returns (x, y, w, h) or None."""
        return self._session.get_layer_bbox(index)

    def _on_canvas_layer_selected(self, index: int) -> None:
        """A layer was selected (or deselected with -1) on the canvas."""
        if index >= 0:
            self._status_label.setText(
                f"Selected: Layer {index + 1}  "
                f"(drag to move, double-click to edit)"
            )
        else:
            layers = self._session.get_layers()
            if layers:
                self._status_label.setText("Click a layer on the canvas to select it")
            else:
                self._status_label.setText("Ready")

    def _on_canvas_layer_moved(self, index: int, dx: int, dy: int) -> None:
        """A layer was dragged on the canvas by (dx, dy) pixels."""
        self._session.move_layer_pixels(index, dx, dy)
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_canvas_layer_double_clicked(self, index: int) -> None:
        """A layer was double-clicked on the canvas — edit if it's a text layer."""
        layers = self._session.get_layers()
        if index >= len(layers):
            return
        layer_info = layers[index]

        # Check if it's a text layer
        if layer_info.name.startswith('Text: "'):
            self._edit_text_layer(index, layer_info)
        else:
            # For non-text layers, show info or open colour picker
            self._on_color_picker(index)

    def _edit_text_layer(self, index: int, layer_info) -> None:
        """Open the text edit dialog for an existing text layer."""
        # Extract current text from layer name
        current_text = ""
        if layer_info.name.startswith('Text: "') and layer_info.name.endswith('"'):
            current_text = layer_info.name[7:-1]

        dlg = _TextDialog(self)
        dlg.setWindowTitle("Edit Text")
        dlg._text_edit.setText(current_text)
        dlg._color_hex = layer_info.hex_color
        dlg._color_btn.setText(layer_info.hex_color)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._session.edit_text_layer(
            index,
            dlg.text_value,
            color=dlg.color_value,
            font_scale=dlg.font_scale_value,
            thickness=dlg.thickness_value,
        )
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    # =================================================================
    #  Save / Load / Export PNG
    # =================================================================

    def _on_save_project(self) -> None:
        if not self._session.is_loaded:
            QMessageBox.information(
                self, "Save Project", "Open an image first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "QuickLayer Project (*.qlp)"
        )
        if path:
            try:
                self._session.save_project(path)
                self._status_label.setText(f"Project saved: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _on_load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "QuickLayer Project (*.qlp)"
        )
        if path:
            try:
                self._session.load_project(path)
                self._source_panel.set_image(
                    self._session.image, self._session.path
                )
                self._refresh_layers()
                self._refresh_preview()
                self._update_undo_redo_state()
                self._status_label.setText(f"Project loaded: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _on_export_png(self) -> None:
        if not self._session.is_quantized:
            QMessageBox.information(
                self, "Export PNG", "Open and quantize an image first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Image (*.png)"
        )
        if path:
            try:
                self._session.export_png(path)
                self._status_label.setText(f"PNG exported: {Path(path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # =================================================================
    #  Worker callbacks
    # =================================================================

    def _on_load_done(self, _result) -> None:
        self._set_busy(False, "Loaded")
        self._selected_indices = []
        self._source_panel.set_image(
            self._session.image, self._session.path
        )
        h, w = self._session.image_size
        self._status_label.setText(
            f"{self._session.path.name}  \u2014  {w}\u00d7{h}"
        )
        self._run_quantize()

    def _on_quantize_done(self, _result) -> None:
        self._set_busy(False, "Quantized")
        # Show auto-detected count in the spin box (without re-triggering)
        if self._auto_btn.isChecked():
            count = self._session.get_layer_count()
            self._color_spin.blockSignals(True)
            self._color_spin.setValue(max(2, count))
            self._color_spin.blockSignals(False)
        self._refresh_layers()
        self._run_trace()
        self._update_undo_redo_state()

    def _on_trace_done(self, _result) -> None:
        self._set_busy(False, "Ready")
        self._refresh_preview()

    def _on_export_done(self, files) -> None:
        self._set_busy(False, "Exported")
        n = len(files) if files else 0
        QMessageBox.information(
            self, "Export Complete", f"{n} file(s) written."
        )

    def _on_import_svg_done(self, count) -> None:
        self._set_busy(False, f"Imported {count} layer(s)")
        self._refresh_layers()
        self._refresh_preview()
        self._update_undo_redo_state()

    def _on_preprocess_done(self, report) -> None:
        self._set_busy(False, "Preprocessed")
        self._source_panel.set_image(
            self._session.image, self._session.path
        )
        # Re-quantize with the enhanced image
        self._run_quantize()

        # Show report if there are recommendations
        if report and report.recommendations:
            msg = "\n".join(f"- {r}" for r in report.recommendations)
            QMessageBox.information(
                self, "Image Analysis", f"Recommendations:\n{msg}"
            )

    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False, "Error")
        QMessageBox.critical(self, "Error", message)

    # =================================================================
    #  Internal helpers
    # =================================================================

    def _run_quantize(self) -> None:
        if not self._session.is_loaded:
            return
        n_colors = None if self._auto_btn.isChecked() else self._color_spin.value()
        self._set_busy(True, "Quantizing\u2026")
        worker = QuantizeWorker(self._session, n_colors=n_colors)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_quantize_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _run_trace(self) -> None:
        if not self._session.is_quantized:
            return
        self._set_busy(True, "Tracing\u2026")
        worker = TraceWorker(self._session)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_trace_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _refresh_layers(self) -> None:
        self._layer_panel.set_layers(self._session.get_layers())
        self._layer_panel.set_selected_indices(self._selected_indices)

    def _refresh_preview(self) -> None:
        comp = self._session.get_composite_preview(self._selected_indices)
        if comp is not None:
            self._preview.set_composite(comp)
        else:
            self._preview.clear()

    def _on_progress(self, message: str) -> None:
        """Update both the status label and the preview overlay."""
        self._status_label.setText(message)
        self._preview.update_processing_message(message)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        if busy:
            self._preview.show_processing(status)
            self._progress.show()
            self._status_label.setText(status)
        else:
            self._preview.hide_processing()
            self._progress.hide()
            if status:
                self._status_label.setText(status)

    def _start_worker(self, worker) -> None:
        """Keep a reference so the worker isn't garbage-collected."""
        self._worker = worker
        worker.start()


# =====================================================================
#  Tool Dialogs
# =====================================================================

class _TextDialog(QDialog):
    """Dialog for adding text to the canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Text")
        self.setMinimumWidth(380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 16)

        form = QFormLayout()
        form.setSpacing(10)

        self._text_edit = QLineEdit()
        self._text_edit.setPlaceholderText("Enter text...")
        form.addRow("Text:", self._text_edit)

        self._font_scale = QDoubleSpinBox()
        self._font_scale.setRange(0.5, 20.0)
        self._font_scale.setValue(2.0)
        self._font_scale.setSingleStep(0.5)
        form.addRow("Font scale:", self._font_scale)

        self._thickness = QSpinBox()
        self._thickness.setRange(1, 20)
        self._thickness.setValue(3)
        form.addRow("Thickness:", self._thickness)

        self._color_btn = QPushButton("#000000")
        self._color_btn.setFixedHeight(32)
        self._color_hex = "#000000"
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color_hex), self)
        if color.isValid():
            self._color_hex = color.name()
            self._color_btn.setText(self._color_hex)

    @property
    def text_value(self) -> str:
        return self._text_edit.text()

    @property
    def font_scale_value(self) -> float:
        return self._font_scale.value()

    @property
    def thickness_value(self) -> int:
        return self._thickness.value()

    @property
    def color_value(self) -> str:
        return self._color_hex


class _BackgroundDialog(QDialog):
    """Dialog for re-running background detection on the current image."""

    def __init__(self, bg_color: str = "", remove_tm: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Background")
        self.setMinimumWidth(400)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 16)

        description = QLabel(
            "Reload the original image and rebuild the foreground mask. "
            "Leave the color blank to use automatic detection."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        form.setSpacing(10)

        self._bg_color = QLineEdit(bg_color)
        self._bg_color.setPlaceholderText("#FFFFFF or leave blank for auto")
        form.addRow("Background color:", self._bg_color)

        self._remove_tm = QCheckBox("Remove TM / ® symbols from margins")
        self._remove_tm.setChecked(remove_tm)
        form.addRow("", self._remove_tm)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def bg_color(self) -> str:
        return self._bg_color.text().strip()

    @property
    def remove_tm(self) -> bool:
        return self._remove_tm.isChecked()


class _OutlineDialog(QDialog):
    """Dialog for adding an outline to a layer."""

    def __init__(self, num_layers: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Outline")
        self.setMinimumWidth(380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 16)

        form = QFormLayout()
        form.setSpacing(10)

        self._layer_spin = QSpinBox()
        self._layer_spin.setRange(1, num_layers)
        self._layer_spin.setValue(1)
        form.addRow("Layer:", self._layer_spin)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 50)
        self._width_spin.setValue(3)
        form.addRow("Width (px):", self._width_spin)

        self._color_btn = QPushButton("#000000")
        self._color_btn.setFixedHeight(32)
        self._color_hex = "#000000"
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color_hex), self)
        if color.isValid():
            self._color_hex = color.name()
            self._color_btn.setText(self._color_hex)

    @property
    def layer_index(self) -> int:
        return self._layer_spin.value() - 1

    @property
    def width_value(self) -> int:
        return self._width_spin.value()

    @property
    def color_value(self) -> str:
        return self._color_hex


class _BorderDialog(QDialog):
    """Dialog for adding a canvas border."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Canvas Border")
        self.setMinimumWidth(380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 16)

        form = QFormLayout()
        form.setSpacing(10)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 200)
        self._width_spin.setValue(10)
        form.addRow("Width (px):", self._width_spin)

        self._color_btn = QPushButton("#000000")
        self._color_btn.setFixedHeight(32)
        self._color_hex = "#000000"
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color_hex), self)
        if color.isValid():
            self._color_hex = color.name()
            self._color_btn.setText(self._color_hex)

    @property
    def width_value(self) -> int:
        return self._width_spin.value()

    @property
    def color_value(self) -> str:
        return self._color_hex


class _ObjectBorderDialog(QDialog):
    """Dialog for adding a border around selected layers."""

    def __init__(self, selected_indices: list[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Border to Selection")
        self.setMinimumWidth(380)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 16)

        info = QLabel(
            f"A border will be drawn around {len(selected_indices)} "
            f"selected layer(s)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px; margin-bottom: 4px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(10)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 50)
        self._width_spin.setValue(3)
        form.addRow("Border width (px):", self._width_spin)

        self._color_btn = QPushButton("#000000")
        self._color_btn.setFixedHeight(32)
        self._color_hex = "#000000"
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._color_hex), self)
        if color.isValid():
            self._color_hex = color.name()
            self._color_btn.setText(self._color_hex)

    @property
    def width_value(self) -> int:
        return self._width_spin.value()

    @property
    def color_value(self) -> str:
        return self._color_hex


class _PreprocessDialog(QDialog):
    """Dialog for image preprocessing options."""

    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preprocessing")
        self.setMinimumWidth(440)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 20)

        title = QLabel("Smart Image Preprocessing")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {TEXT}; "
            "margin-bottom: 4px;"
        )
        layout.addWidget(title)

        subtitle = QLabel("Enhance image quality before color quantization")
        subtitle.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SEC}; margin-bottom: 8px;"
        )
        layout.addWidget(subtitle)

        # Show analysis results
        if report and report.recommendations:
            rec_group = QGroupBox("Analysis Results")
            rec_layout = QVBoxLayout(rec_group)
            for rec in report.recommendations:
                lbl = QLabel(f"- {rec}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 12px;")
                rec_layout.addWidget(lbl)
            layout.addWidget(rec_group)

        # Enhancement options
        enhance_group = QGroupBox("Enhancement Options")
        enhance_form = QFormLayout(enhance_group)
        enhance_form.setSpacing(12)

        self._contrast_cb = QCheckBox("Contrast enhancement (CLAHE)")
        self._contrast_cb.setToolTip(
            "Improve contrast for washed-out images"
        )
        if report and report.contrast_low:
            self._contrast_cb.setChecked(True)
        enhance_form.addRow(self._contrast_cb)

        self._sharpen_cb = QCheckBox("Edge sharpening")
        self._sharpen_cb.setToolTip(
            "Sharpen edges for cleaner vector tracing"
        )
        enhance_form.addRow(self._sharpen_cb)

        self._denoise_cb = QCheckBox("Noise reduction")
        self._denoise_cb.setToolTip(
            "Reduce photo noise for cleaner color separation"
        )
        if report and report.noise_level > 30.0:
            self._denoise_cb.setChecked(True)
        enhance_form.addRow(self._denoise_cb)

        layout.addWidget(enhance_group)

        # Background warning
        if report and report.bg_blur_detected:
            warn_label = QLabel(
                f"Warning: Background is not uniform "
                f"(uniformity: {report.bg_uniformity:.0%}). "
                f"Consider using a solid background colour override."
            )
            warn_label.setWordWrap(True)
            warn_label.setStyleSheet(
                "color: #ca8a04; font-size: 12px; padding: 8px; "
                "background: #fef3c7; border-radius: 4px;"
            )
            layout.addWidget(warn_label)

        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setProperty("cssClass", "primary")
            ok_btn.style().unpolish(ok_btn)
            ok_btn.style().polish(ok_btn)

        layout.addWidget(buttons)

    @property
    def contrast(self) -> bool:
        return self._contrast_cb.isChecked()

    @property
    def sharpen(self) -> bool:
        return self._sharpen_cb.isChecked()

    @property
    def denoise(self) -> bool:
        return self._denoise_cb.isChecked()
