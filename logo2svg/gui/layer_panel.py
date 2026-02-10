"""Right-panel colour-layer list widget.

Each row:  [swatch] hex  name  pixel-count  [eye toggle] [color-picker] [delete]
Bottom:    [Merge Selected]
"""

from __future__ import annotations

from typing import List

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..session import LayerInfo


def _swatch_pixmap(hex_color: str, size: int = 24) -> QPixmap:
    """Create a small square pixmap filled with *hex_color*."""
    pm = QPixmap(size, size)
    pm.fill(QColor(hex_color))
    return pm


class LayerRow(QWidget):
    """One row in the layer list."""

    visibility_changed = pyqtSignal(int, bool)
    color_change_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    selection_changed = pyqtSignal()

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self._index = info.index
        self._hex = info.hex_color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Selection checkbox (for merge)
        self.select_cb = QCheckBox()
        self.select_cb.stateChanged.connect(lambda: self.selection_changed.emit())
        layout.addWidget(self.select_cb)

        # Color swatch
        self.swatch_label = QLabel()
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color))
        self.swatch_label.setFixedSize(28, 28)
        layout.addWidget(self.swatch_label)

        # Hex value
        self.hex_label = QLabel(info.hex_color)
        self.hex_label.setFixedWidth(80)
        self.hex_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.hex_label)

        # Color name
        self.name_label = QLabel(info.color_name)
        self.name_label.setMinimumWidth(90)
        layout.addWidget(self.name_label, stretch=1)

        # Pixel count
        self.count_label = QLabel(f"{info.pixel_count:,} px")
        self.count_label.setFixedWidth(90)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.count_label)

        # Eye toggle (visibility)
        self.eye_btn = QPushButton("👁")
        self.eye_btn.setFixedWidth(32)
        self.eye_btn.setCheckable(True)
        self.eye_btn.setChecked(info.visible)
        self.eye_btn.setToolTip("Toggle layer visibility")
        self.eye_btn.toggled.connect(
            lambda checked: self.visibility_changed.emit(self._index, checked)
        )
        layout.addWidget(self.eye_btn)

        # Color picker button
        color_btn = QPushButton("🎨")
        color_btn.setFixedWidth(32)
        color_btn.setToolTip("Change output colour")
        color_btn.clicked.connect(
            lambda: self.color_change_requested.emit(self._index)
        )
        layout.addWidget(color_btn)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(32)
        del_btn.setToolTip("Remove this layer")
        del_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._index)
        )
        layout.addWidget(del_btn)

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_selected(self) -> bool:
        return self.select_cb.isChecked()

    def update_info(self, info: LayerInfo) -> None:
        """Refresh display from a layer snapshot."""
        self._index = info.index
        self._hex = info.hex_color
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color))
        self.hex_label.setText(info.hex_color)
        self.name_label.setText(info.color_name)
        self.count_label.setText(f"{info.pixel_count:,} px")
        self.eye_btn.setChecked(info.visible)


class LayerPanel(QWidget):
    """Scrollable colour-layer list with merge button."""

    # Signals that the main window connects to
    visibility_toggled = pyqtSignal(int, bool)
    color_picker_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    merge_requested = pyqtSignal(list)  # list[int]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(420)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Colour Layers")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 6px;")
        outer.addWidget(header)

        # Scrollable area for rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)
        self._layout.addStretch()
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

        # Merge button
        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setEnabled(False)
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        outer.addWidget(self._merge_btn)

        self._rows: list[LayerRow] = []

    # -- public API -------------------------------------------------------

    def set_layers(self, layers: list[LayerInfo]) -> None:
        """Rebuild the entire list from a fresh snapshot."""
        self._clear_rows()
        for info in layers:
            row = LayerRow(info)
            row.visibility_changed.connect(self.visibility_toggled.emit)
            row.color_change_requested.connect(self.color_picker_requested.emit)
            row.delete_requested.connect(self.delete_requested.emit)
            row.selection_changed.connect(self._update_merge_button)
            self._rows.append(row)
            # Insert before the terminal stretch
            self._layout.insertWidget(self._layout.count() - 1, row)
        self._update_merge_button()

    def get_selected_indices(self) -> list[int]:
        return [r.index for r in self._rows if r.is_selected]

    def clear(self) -> None:
        self._clear_rows()

    # -- internals --------------------------------------------------------

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _update_merge_button(self) -> None:
        self._merge_btn.setEnabled(len(self.get_selected_indices()) >= 2)

    def _on_merge_clicked(self) -> None:
        sel = self.get_selected_indices()
        if len(sel) >= 2:
            self.merge_requested.emit(sel)
