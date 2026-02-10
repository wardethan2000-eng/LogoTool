"""Right-sidebar colour-layer list with modern card-style rows.

Each row is a styled QFrame card containing a colour swatch, colour
name, hex value, pixel count, visibility toggle, and delete button.
The bottom of the panel has a Merge Selected button.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..session import LayerInfo
from .style import (
    ACCENT,
    BORDER,
    BORDER_LIGHT,
    CARD,
    CARD_HOVER,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    TEXT_SEC,
)


def _swatch_pixmap(hex_color: str, size: int = 20) -> QPixmap:
    """Create a rounded square colour swatch."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(hex_color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 4, 4)
    painter.end()
    return pm


class LayerRow(QFrame):
    """Single layer card row."""

    visibility_changed = pyqtSignal(int, bool)
    color_change_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    selection_changed = pyqtSignal()

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self._index = info.index
        self._hex = info.hex_color

        self.setProperty("cssClass", "layerCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.ArrowCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(8)

        # Selection checkbox (for merge)
        self.select_cb = QCheckBox()
        self.select_cb.setToolTip("Select for merge")
        self.select_cb.stateChanged.connect(lambda: self.selection_changed.emit())
        layout.addWidget(self.select_cb)

        # Colour swatch — click to change colour
        self.swatch_label = QLabel()
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.swatch_label.setFixedSize(26, 26)
        self.swatch_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch_label.setToolTip("Click to change colour")
        self.swatch_label.mousePressEvent = (
            lambda _e: self.color_change_requested.emit(self._index)
        )
        layout.addWidget(self.swatch_label)

        # Colour info column: name + hex on top, pixel count below
        info_col = QVBoxLayout()
        info_col.setSpacing(0)
        info_col.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        self.name_label = QLabel(info.color_name)
        self.name_label.setStyleSheet(
            f"font-size: 13px; font-weight: 500; color: {TEXT};"
        )
        name_row.addWidget(self.name_label)

        self.hex_label = QLabel(info.hex_color)
        self.hex_label.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', monospace; "
            f"font-size: 11px; color: {TEXT_MUTED};"
        )
        name_row.addWidget(self.hex_label)
        name_row.addStretch()
        info_col.addLayout(name_row)

        self.count_label = QLabel(f"{info.pixel_count:,} px")
        self.count_label.setStyleSheet(f"font-size: 11px; color: {TEXT_SEC};")
        info_col.addWidget(self.count_label)

        layout.addLayout(info_col, stretch=1)

        # Visibility toggle
        self.eye_btn = QPushButton("\u25cf")  # ● filled circle
        self.eye_btn.setProperty("cssClass", "icon")
        self.eye_btn.setCheckable(True)
        self.eye_btn.setChecked(info.visible)
        self.eye_btn.setToolTip("Toggle visibility")
        self._update_eye_icon(info.visible)
        self.eye_btn.toggled.connect(self._on_eye_toggled)
        layout.addWidget(self.eye_btn)

        # Delete button
        del_btn = QPushButton("\u2715")  # ✕
        del_btn.setProperty("cssClass", "danger")
        del_btn.setToolTip("Remove this layer")
        del_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._index)
        )
        layout.addWidget(del_btn)

    # -- properties -------------------------------------------------------

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_selected(self) -> bool:
        return self.select_cb.isChecked()

    # -- public -----------------------------------------------------------

    def update_info(self, info: LayerInfo) -> None:
        """Refresh display from a layer snapshot."""
        self._index = info.index
        self._hex = info.hex_color
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.hex_label.setText(info.hex_color)
        self.name_label.setText(info.color_name)
        self.count_label.setText(f"{info.pixel_count:,} px")
        self.eye_btn.setChecked(info.visible)
        self._update_eye_icon(info.visible)

    # -- internals --------------------------------------------------------

    def _on_eye_toggled(self, checked: bool) -> None:
        self._update_eye_icon(checked)
        self.visibility_changed.emit(self._index, checked)

    def _update_eye_icon(self, visible: bool) -> None:
        if visible:
            self.eye_btn.setText("\u25cf")  # ● filled
            self.eye_btn.setStyleSheet(
                f"color: {ACCENT}; font-size: 14px; "
                "background: transparent; border: 1px solid transparent; "
                "border-radius: 4px; min-width: 28px; max-width: 28px; "
                "min-height: 28px; max-height: 28px; padding: 2px;"
            )
        else:
            self.eye_btn.setText("\u25cb")  # ○ empty
            self.eye_btn.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 14px; "
                "background: transparent; border: 1px solid transparent; "
                "border-radius: 4px; min-width: 28px; max-width: 28px; "
                "min-height: 28px; max-height: 28px; padding: 2px;"
            )


class LayerPanel(QWidget):
    """Scrollable colour-layer list with merge button."""

    # Signals consumed by the main window
    visibility_toggled = pyqtSignal(int, bool)
    color_picker_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    merge_requested = pyqtSignal(list)  # list[int]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("layerPanel")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        # Header row with count badge
        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        header = QLabel("LAYERS")
        header.setObjectName("sectionHeader")
        header_row.addWidget(header)

        self._count_badge = QLabel("0")
        self._count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_badge.setStyleSheet(
            f"background-color: {BORDER}; color: {TEXT_SEC}; "
            "font-size: 11px; font-weight: 600; "
            "padding: 1px 7px; border-radius: 8px; min-width: 16px;"
        )
        header_row.addWidget(self._count_badge)
        header_row.addStretch()
        outer.addLayout(header_row)

        # Scrollable area for layer cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 4, 0)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

        # Merge button
        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setProperty("cssClass", "primary")
        self._merge_btn.setEnabled(False)
        self._merge_btn.setFixedHeight(36)
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
            row.color_change_requested.connect(
                self.color_picker_requested.emit
            )
            row.delete_requested.connect(self.delete_requested.emit)
            row.selection_changed.connect(self._update_merge_button)
            self._rows.append(row)
            self._layout.insertWidget(self._layout.count() - 1, row)
        self._count_badge.setText(str(len(layers)))
        self._update_merge_button()

    def get_selected_indices(self) -> list[int]:
        return [r.index for r in self._rows if r.is_selected]

    def clear(self) -> None:
        self._clear_rows()
        self._count_badge.setText("0")

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
