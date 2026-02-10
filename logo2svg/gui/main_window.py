"""Main application window for logo2svg GUI."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..session import Session
from .layer_panel import LayerPanel
from .preview_panel import PreviewPanel
from .settings_dialog import SettingsDialog
from .source_panel import SourcePanel
from .workers import ExportWorker, LoadWorker, QuantizeWorker, TraceWorker

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg);;PNG (*.png);;JPEG (*.jpg *.jpeg)"


class MainWindow(QMainWindow):
    """Top-level window: toolbar, left source panel, centre preview, right
    colour-layer panel, bottom status bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("logo2svg")
        self.resize(1200, 750)

        # Session & settings
        self._session = Session()
        self._bg_color: str = ""
        self._remove_tm: bool = True
        self._worker = None  # current QThread reference (keep alive)

        self._build_menu_bar()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

    # =================================================================
    #  UI Construction
    # =================================================================

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        open_act = QAction("&Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open)
        file_menu.addAction(open_act)

        export_act = QAction("&Export SVGs…", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = mb.addMenu("&Edit")
        settings_act = QAction("&Settings…", self)
        settings_act.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_act)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        open_btn = QAction("📂 Open", self)
        open_btn.setToolTip("Open an image file")
        open_btn.triggered.connect(self._on_open)
        tb.addAction(open_btn)

        export_btn = QAction("💾 Export", self)
        export_btn.setToolTip("Export SVG files")
        export_btn.triggered.connect(self._on_export)
        tb.addAction(export_btn)

        settings_btn = QAction("⚙ Settings", self)
        settings_btn.setToolTip("Pipeline settings")
        settings_btn.triggered.connect(self._on_settings)
        tb.addAction(settings_btn)

        tb.addSeparator()

        # Number of colours spinner
        tb.addWidget(QLabel("  Colours: "))
        self._color_spin = QSpinBox()
        self._color_spin.setRange(0, 20)
        self._color_spin.setSpecialValueText("Auto")
        self._color_spin.setValue(0)
        self._color_spin.setToolTip(
            "Number of colours for quantization (0 = auto-detect)"
        )
        self._color_spin.valueChanged.connect(self._on_requantize)
        tb.addWidget(self._color_spin)

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(4, 4, 4, 4)

        # Left: source info
        self._source_panel = SourcePanel()
        h_layout.addWidget(self._source_panel)

        # Centre: preview
        self._preview = PreviewPanel()
        h_layout.addWidget(self._preview, stretch=1)

        # Right: colour layers
        self._layer_panel = LayerPanel()
        self._layer_panel.visibility_toggled.connect(self._on_visibility_toggled)
        self._layer_panel.color_picker_requested.connect(self._on_color_picker)
        self._layer_panel.delete_requested.connect(self._on_delete_layer)
        self._layer_panel.merge_requested.connect(self._on_merge_layers)
        h_layout.addWidget(self._layer_panel)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, stretch=1)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        sb.addPermanentWidget(self._progress)

    # =================================================================
    #  Public API (for programmatic use / tests)
    # =================================================================

    def open_file(self, path: str) -> None:
        """Open and begin processing *path*."""
        self._set_busy(True, f"Loading {Path(path).name}…")
        worker = LoadWorker(
            self._session, path, bg_color=self._bg_color or None,
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
            QMessageBox.information(self, "Export", "Open and quantize an image first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export Directory")
        if not out_dir:
            return
        self._set_busy(True, "Exporting…")
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
            # Need to reload with new bg colour
            if self._session.path:
                self.open_file(str(self._session.path))
                return
        if dlg.remove_tm != self._remove_tm:
            self._remove_tm = dlg.remove_tm
            # Need to reload to apply / undo TM removal
            if self._session.path:
                self.open_file(str(self._session.path))
                return

        if changed and self._session.is_loaded:
            self._run_quantize()

    def _on_requantize(self, value: int) -> None:
        """Spinner value changed → re-quantize."""
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
        # Show source thumbnail + info
        self._source_panel.set_image(self._session.image, self._session.path)
        h, w = self._session.image_size
        self._status_label.setText(f"{self._session.path.name}  —  {w}×{h}")
        # Automatically quantize
        self._run_quantize()

    def _on_quantize_done(self, _result) -> None:
        self._set_busy(False, "Quantized")
        self._refresh_layers()
        # Auto-trace for preview
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
        self._set_busy(True, "Quantizing…")
        worker = QuantizeWorker(self._session, n_colors=n_colors)
        worker.progress.connect(self._status_label.setText)
        worker.finished.connect(self._on_quantize_done)
        worker.error.connect(self._on_worker_error)
        self._start_worker(worker)

    def _run_trace(self) -> None:
        if not self._session.is_quantized:
            return
        self._set_busy(True, "Tracing…")
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
        """Keep a reference to the worker so it isn't garbage-collected,
        and start it."""
        self._worker = worker
        worker.start()
