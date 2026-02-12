"""Left-sidebar color-layer list with modern card-style rows.

Each row is a styled QFrame card containing a visibility checkbox,
color swatch, layer label, hex value, pixel count, move/delete buttons.
The bottom of the panel has Merge Selected and Duplicate buttons.

Supports: visibility toggle, color picker, delete, merge, move up/down,
duplicate, rename (double-click label), and Delete key shortcut.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    move_up_requested = pyqtSignal(int)
    move_down_requested = pyqtSignal(int)
    rename_requested = pyqtSignal(int, str)
    selection_changed = pyqtSignal()

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self._index = info.index
        self._hex = info.hex_color
        self._name = info.name or f"Layer {info.index + 1}"

        self.setProperty("cssClass", "layerCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(6)

        # Visibility checkbox (checked = visible in preview)
        self.select_cb = QCheckBox()
        self.select_cb.setChecked(info.visible)
        self.select_cb.setToolTip("Show/hide this layer in preview")
        self.select_cb.stateChanged.connect(self._on_checkbox_changed)
        layout.addWidget(self.select_cb)

        # Color swatch -- click to change color
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

        # Editable name label (double-click to rename)
        self.name_label = QLabel(self._name)
        self.name_label.setStyleSheet(
            f"font-size: 13px; font-weight: 500; color: {TEXT};"
        )
        self.name_label.setCursor(Qt.CursorShape.IBeamCursor)
        self.name_label.setToolTip("Double-click to rename")
        self.name_label.mouseDoubleClickEvent = lambda _e: self._start_rename()
        name_row.addWidget(self.name_label)

        # Inline rename editor (hidden by default)
        self._name_edit = QLineEdit()
        self._name_edit.setFixedHeight(20)
        self._name_edit.setStyleSheet("font-size: 12px; padding: 0 2px;")
        self._name_edit.hide()
        self._name_edit.returnPressed.connect(self._finish_rename)
        self._name_edit.editingFinished.connect(self._finish_rename)
        name_row.addWidget(self._name_edit)

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

        # Move up button
        up_btn = QPushButton("\u25B2")
        up_btn.setToolTip("Move layer up")
        up_btn.setFixedSize(22, 22)
        up_btn.setStyleSheet(
            f"QPushButton {{ font-size: 9px; background: transparent; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: 3px; "
            f"color: {TEXT_SEC}; padding: 0; }}"
            f"QPushButton:hover {{ background: {CARD}; color: {TEXT}; }}"
        )
        up_btn.clicked.connect(
            lambda: self.move_up_requested.emit(self._index)
        )
        layout.addWidget(up_btn)

        # Move down button
        down_btn = QPushButton("\u25BC")
        down_btn.setToolTip("Move layer down")
        down_btn.setFixedSize(22, 22)
        down_btn.setStyleSheet(
            f"QPushButton {{ font-size: 9px; background: transparent; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: 3px; "
            f"color: {TEXT_SEC}; padding: 0; }}"
            f"QPushButton:hover {{ background: {CARD}; color: {TEXT}; }}"
        )
        down_btn.clicked.connect(
            lambda: self.move_down_requested.emit(self._index)
        )
        layout.addWidget(down_btn)

        # Delete button
        del_btn = QPushButton("\u2715")  # x
        del_btn.setProperty("cssClass", "danger")
        del_btn.setToolTip("Remove this layer  (Delete)")
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
        self._name = info.name or f"Layer {info.index + 1}"
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.hex_label.setText(info.hex_color)
        self.name_label.setText(self._name)
        self.count_label.setText(f"{info.pixel_count:,} px")
        self.select_cb.blockSignals(True)
        self.select_cb.setChecked(info.visible)
        self.select_cb.blockSignals(False)

    # -- rename -----------------------------------------------------------

    def _start_rename(self) -> None:
        self.name_label.hide()
        self._name_edit.setText(self._name)
        self._name_edit.show()
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_rename(self) -> None:
        if not self._name_edit.isVisible():
            return
        new_name = self._name_edit.text().strip()
        self._name_edit.hide()
        self.name_label.show()
        if new_name and new_name != self._name:
            self._name = new_name
            self.name_label.setText(new_name)
            self.rename_requested.emit(self._index, new_name)

    # -- events -----------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit(self._index)
        else:
            super().keyPressEvent(event)

    # -- internals --------------------------------------------------------

    def _on_checkbox_changed(self, state) -> None:
        checked = bool(state)
        self.visibility_changed.emit(self._index, checked)
        self.selection_changed.emit()


class LayerPanel(QWidget):
    """Scrollable color-layer list with merge/duplicate/reorder buttons."""

    # Signals consumed by the main window
    visibility_toggled = pyqtSignal(int, bool)
    color_picker_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    merge_requested = pyqtSignal(list)  # list[int]
    move_up_requested = pyqtSignal(int)
    move_down_requested = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    rename_requested = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("layerPanel")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

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

        # Action buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setProperty("cssClass", "primary")
        self._merge_btn.setEnabled(False)
        self._merge_btn.setFixedHeight(32)
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        btn_row.addWidget(self._merge_btn)

        self._dup_btn = QPushButton("Duplicate")
        self._dup_btn.setFixedHeight(32)
        self._dup_btn.setEnabled(False)
        self._dup_btn.setToolTip("Duplicate the first selected layer")
        self._dup_btn.clicked.connect(self._on_duplicate_clicked)
        btn_row.addWidget(self._dup_btn)

        outer.addLayout(btn_row)

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
            row.move_up_requested.connect(self.move_up_requested.emit)
            row.move_down_requested.connect(self.move_down_requested.emit)
            row.rename_requested.connect(self.rename_requested.emit)
            row.selection_changed.connect(self._update_action_buttons)
            self._rows.append(row)
            self._layout.insertWidget(self._layout.count() - 1, row)
        self._count_badge.setText(str(len(layers)))
        self._update_action_buttons()

    def get_selected_indices(self) -> list[int]:
        return [r.index for r in self._rows if r.is_selected]

    def clear(self) -> None:
        self._clear_rows()
        self._count_badge.setText("0")

    # -- events -----------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle Delete key at the panel level."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sel = self.get_selected_indices()
            if sel:
                self.delete_requested.emit(sel[0])
                return
        super().keyPressEvent(event)

    # -- internals --------------------------------------------------------

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

    def _update_action_buttons(self) -> None:
        sel = self.get_selected_indices()
        self._merge_btn.setEnabled(len(sel) >= 2)
        self._dup_btn.setEnabled(len(sel) >= 1)

    def _on_merge_clicked(self) -> None:
        sel = self.get_selected_indices()
        if len(sel) >= 2:
            self.merge_requested.emit(sel)

    def _on_duplicate_clicked(self) -> None:
        sel = self.get_selected_indices()
        if sel:
            self.duplicate_requested.emit(sel[0])
