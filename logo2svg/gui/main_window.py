"""Main application window for the QuickLayer GUI.

Layout
------
┌─ Menu bar ───────────────────────────────────────────────┐
├─ Header bar ─────────────────────────────────────────────┤
│  QuickLayer │ [Open] [Export SVGs] │ Colors [Auto]  │ ⚙ │
├──────────────────────────┬───────────────────────────────┤
│  SOURCE IMAGE            │                               │
│  (large view of original)│                               │
│                          │       PREVIEW AREA            │
├──────────────────────────┤  (composite with layers)      │
│  LAYERS            [3]   │                               │
│  ┌──────────────────┐   │  (drag and drop to open)      │
│  │ ☑ ■ Layer 1      │   │                               │
│  │ ☑ ■ Layer 2      │   │                               │
│  └──────────────────┘   │                               │
│  [Merge Selected]        │                               │
├──────────────────────────┴───────────────────────────────┤
│  Ready                                         ████████  │
└──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
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
from .workers import ExportWorker, LoadWorker, QuantizeWorker, TraceWorker

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg);;PNG (*.png);;JPEG (*.jpg *.jpeg)"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class MainWindow(QMainWindow):
    """Top-level window with modern light-themed layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QuickLayer")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        # Window icon (taskbar + title bar)
        icon_path = Path(__file__).resolve().parent.parent / "icons" / "quicklayer.ico"
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

        file_menu = mb.addMenu("&File")

        open_act = QAction("&Open\u2026", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        export_act = QAction("&Export SVGs\u2026", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = mb.addMenu("&Edit")
        settings_act = QAction("&Settings\u2026", self)
        settings_act.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_act)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header toolbar ──────────────────────────────────────
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

        # Export button
        export_btn = QPushButton("Export SVGs")
        export_btn.setToolTip("Export SVG files  (Ctrl+E)")
        export_btn.setFixedHeight(32)
        export_btn.clicked.connect(self._on_export)
        hbar.addWidget(export_btn)

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

        # ── Main content area (splitter) ────────────────────────
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
        self._layer_panel.merge_requested.connect(self._on_merge_layers)
        self._layer_panel.selection_changed.connect(
            self._on_layer_selection_changed
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
        self._preview.preview_clicked.connect(self._on_preview_clicked)
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
                if Path(url.toLocalFile()).suffix.lower() in IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
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

    def _on_delete_layer(self, index: int) -> None:
        self._session.remove_color(index)
        self._selected_indices = []
        self._refresh_layers()
        self._refresh_preview()

    def _on_merge_layers(self, indices: list[int]) -> None:
        self._session.merge_colors(indices)
        self._selected_indices = []
        self._refresh_layers()
        self._refresh_preview()

    def _on_layer_selection_changed(self, indices: list[int]) -> None:
        self._selected_indices = indices
        self._refresh_preview()

    def _on_preview_clicked(
        self, x: int, y: int, modifiers: Qt.KeyboardModifier
    ) -> None:
        if not self._session.is_quantized:
            return

        layer_idx = self._session.get_layer_at_pixel(x, y, visible_only=True)
        multi = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

        if layer_idx is None:
            if not multi:
                self._layer_panel.set_selected_indices([])
            return

        if multi:
            selected = set(self._selected_indices)
            if layer_idx in selected:
                selected.remove(layer_idx)
            else:
                selected.add(layer_idx)
            self._layer_panel.set_selected_indices(sorted(selected))
        else:
            self._layer_panel.set_selected_indices([layer_idx])

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

    def _on_trace_done(self, _result) -> None:
        self._set_busy(False, "Ready")
        self._refresh_preview()

    def _on_export_done(self, files) -> None:
        self._set_busy(False, "Exported")
        n = len(files) if files else 0
        QMessageBox.information(
            self, "Export Complete", f"{n} file(s) written."
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
