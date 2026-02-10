"""Main application window for the logo2svg GUI.

Layout
------
┌─ Menu bar ───────────────────────────────────────────────┐
├─ Header bar ─────────────────────────────────────────────┤
│  logo2svg   │ [Open] [Export SVGs] │ Colours [Auto] │ ⚙ │
├───────────────────────────────┬──────────────────────────┤
│                               │  SOURCE                  │
│                               │  [thumb]  file.png       │
│       PREVIEW AREA            │          800×600 · PNG   │
│  (drag and drop to open)      ├──────────────────────────┤
│                               │  LAYERS            [3]   │
│                               │  ┌──────────────────┐    │
│                               │  │ ☐ ■ Red  #FF0000 │    │
│                               │  │ ☐ ■ Blue #0000FF │    │
│                               │  └──────────────────┘    │
│                               │  [Merge Selected]        │
├───────────────────────────────┴──────────────────────────┤
│  Ready                                         ████████  │
└──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent
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
    """Top-level window with modern dark-themed layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("logo2svg")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        # Session & settings
        self._session = Session()
        self._bg_color: str = ""
        self._remove_tm: bool = True
        self._worker = None  # keep worker alive while running

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
        title = QLabel("logo2svg")
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

        # Colours control
        clr_label = QLabel("Colours")
        clr_label.setStyleSheet(f"color: {TEXT_SEC}; font-size: 13px;")
        hbar.addWidget(clr_label)

        self._color_spin = QSpinBox()
        self._color_spin.setRange(0, 20)
        self._color_spin.setSpecialValueText("Auto")
        self._color_spin.setValue(0)
        self._color_spin.setToolTip(
            "Number of colours for quantization (0 = auto-detect)"
        )
        self._color_spin.setFixedWidth(80)
        self._color_spin.setFixedHeight(32)
        self._color_spin.valueChanged.connect(self._on_requantize)
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

        # Left: preview panel (main focus, stretches)
        self._preview = PreviewPanel()
        main_splitter.addWidget(self._preview)

        # Right: sidebar with source + layers
        right_sidebar = QFrame()
        right_sidebar.setObjectName("rightSidebar")
        right_sidebar.setMinimumWidth(320)
        right_sidebar.setMaximumWidth(500)

        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Source info panel (compact, at top)
        self._source_panel = SourcePanel()
        right_layout.addWidget(self._source_panel)

        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        right_layout.addWidget(divider)

        # Layer panel (fills remaining space)
        self._layer_panel = LayerPanel()
        self._layer_panel.visibility_toggled.connect(
            self._on_visibility_toggled
        )
        self._layer_panel.color_picker_requested.connect(
            self._on_color_picker
        )
        self._layer_panel.delete_requested.connect(self._on_delete_layer)
        self._layer_panel.merge_requested.connect(self._on_merge_layers)
        right_layout.addWidget(self._layer_panel, stretch=1)

        main_splitter.addWidget(right_sidebar)

        # Stretch factors: preview stretches, sidebar stays
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([880, 380])

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
        worker.progress.connect(self._status_label.setText)
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
        worker.progress.connect(self._status_label.setText)
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

    def _on_requantize(self, value: int) -> None:
        """Spinner value changed -> re-quantize."""
        if not self._session.is_loaded:
            return
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
        colour = QColorDialog.getColor(current, self, "Pick Layer Colour")
        if colour.isValid():
            self._session.change_color(index, colour.name())
            self._refresh_layers()
            self._refresh_preview()

    def _on_delete_layer(self, index: int) -> None:
        self._session.remove_color(index)
        self._refresh_layers()
        self._refresh_preview()

    def _on_merge_layers(self, indices: list[int]) -> None:
        self._session.merge_colors(indices)
        self._refresh_layers()
        self._refresh_preview()

    # =================================================================
    #  Worker callbacks
    # =================================================================

    def _on_load_done(self, _result) -> None:
        self._set_busy(False, "Loaded")
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
        n = self._color_spin.value()
        n_colors = n if n > 0 else None
        self._set_busy(True, "Quantizing\u2026")
        worker = QuantizeWorker(self._session, n_colors=n_colors)
        worker.progress.connect(self._status_label.setText)
        worker.finished.connect(self._on_quantize_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _run_trace(self) -> None:
        if not self._session.is_quantized:
            return
        self._set_busy(True, "Tracing\u2026")
        worker = TraceWorker(self._session)
        worker.progress.connect(self._status_label.setText)
        worker.finished.connect(self._on_trace_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _refresh_layers(self) -> None:
        self._layer_panel.set_layers(self._session.get_layers())

    def _refresh_preview(self) -> None:
        comp = self._session.get_composite_preview()
        if comp is not None:
            self._preview.set_composite(comp)
        else:
            self._preview.clear()

    def _set_busy(self, busy: bool, status: str = "") -> None:
        if busy:
            self._progress.show()
            self._status_label.setText(status)
        else:
            self._progress.hide()
            if status:
                self._status_label.setText(status)

    def _start_worker(self, worker) -> None:
        """Keep a reference so the worker isn't garbage-collected."""
        self._worker = worker
        worker.start()
