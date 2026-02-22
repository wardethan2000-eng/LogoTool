"""Left-sidebar color-layer list with modern card-style rows.

Each row is a styled QFrame card containing a visibility checkbox,
color swatch, layer label, hex value, pixel count, and delete button.
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
    """Create a rounded square color swatch."""
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
    selection_requested = pyqtSignal(int, Qt.KeyboardModifier)

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self._index = info.index
        self._hex = info.hex_color
        self._selected = False

        self.setProperty("cssClass", "layerCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setProperty("selected", "false")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(8)

        # Visibility checkbox (checked = visible in preview)
        self.select_cb = QCheckBox()
        self.select_cb.setChecked(info.visible)  # set before connecting signal
        self.select_cb.setToolTip("Show/hide this layer in preview")
        self.select_cb.stateChanged.connect(self._on_checkbox_changed)
        layout.addWidget(self.select_cb)

        # Color swatch — click to change color
        self.swatch_label = QLabel()
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.swatch_label.setFixedSize(26, 26)
        self.swatch_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch_label.setToolTip("Click to change color")
        self.swatch_label.mousePressEvent = (
            lambda _e: self.color_change_requested.emit(self._index)
        )
        layout.addWidget(self.swatch_label)

        # Layer info column: name + hex on top, pixel count below
        info_col = QVBoxLayout()
        info_col.setSpacing(0)
        info_col.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        self.name_label = QLabel(f"Layer {info.index + 1}")
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
        return self._selected

    @property
    def is_visible(self) -> bool:
        return self.select_cb.isChecked()

    # -- public -----------------------------------------------------------

    def update_info(self, info: LayerInfo) -> None:
        """Refresh display from a layer snapshot."""
        self._index = info.index
        self._hex = info.hex_color
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.hex_label.setText(info.hex_color)
        self.name_label.setText(f"Layer {info.index + 1}")
        self.count_label.setText(f"{info.pixel_count:,} px")
        self.select_cb.blockSignals(True)
        self.select_cb.setChecked(info.visible)
        self.select_cb.blockSignals(False)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.selection_requested.emit(self._index, event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    # -- internals --------------------------------------------------------

    def _on_checkbox_changed(self, state) -> None:
        checked = bool(state)
        self.visibility_changed.emit(self._index, checked)


class LayerPanel(QWidget):
    """Scrollable color-layer list with merge button."""

    # Signals consumed by the main window
    visibility_toggled = pyqtSignal(int, bool)
    color_picker_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    merge_requested = pyqtSignal(list)  # list[int]
    selection_changed = pyqtSignal(list)  # list[int]

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

        # Selection actions
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)

        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setFixedHeight(28)
        self._select_all_btn.clicked.connect(self.select_all)
        actions_row.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.setFixedHeight(28)
        self._deselect_all_btn.clicked.connect(self.deselect_all)
        actions_row.addWidget(self._deselect_all_btn)

        actions_row.addStretch()
        outer.addLayout(actions_row)

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
        selected_before = set(self.get_selected_indices())
        self._clear_rows()
        for info in layers:
            row = LayerRow(info)
            row.visibility_changed.connect(self.visibility_toggled.emit)
            row.color_change_requested.connect(
                self.color_picker_requested.emit
            )
            row.delete_requested.connect(self.delete_requested.emit)
            row.selection_requested.connect(self._on_row_selection_requested)
            self._rows.append(row)
            self._layout.insertWidget(self._layout.count() - 1, row)

            if info.index in selected_before:
                row.set_selected(True)
        self._count_badge.setText(str(len(layers)))
        self._update_merge_button()

    def get_selected_indices(self) -> list[int]:
        return [r.index for r in self._rows if r.is_selected]

    def select_all(self) -> None:
        for row in self._rows:
            row.set_selected(True)
        self._update_merge_button()
        self.selection_changed.emit(self.get_selected_indices())

    def deselect_all(self) -> None:
        for row in self._rows:
            row.set_selected(False)
        self._update_merge_button()
        self.selection_changed.emit(self.get_selected_indices())

    def set_selected_indices(self, indices: list[int]) -> None:
        selected = set(indices)
        for row in self._rows:
            row.set_selected(row.index in selected)
        self._update_merge_button()
        self.selection_changed.emit(self.get_selected_indices())

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

    def _on_row_selection_requested(
        self, index: int, modifiers: Qt.KeyboardModifier
    ) -> None:
        multi = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if multi:
            for row in self._rows:
                if row.index == index:
                    row.set_selected(not row.is_selected)
                    break
            self.selection_changed.emit(self.get_selected_indices())
        else:
            self.set_selected_indices([index])
            return
        self._update_merge_button()

    def _on_merge_clicked(self) -> None:
        sel = self.get_selected_indices()
        if len(sel) >= 2:
            self.merge_requested.emit(sel)
