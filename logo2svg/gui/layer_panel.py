"""Left-sidebar color-layer list with modern card-style rows.

Each row is a styled QFrame card containing a visibility checkbox,
color swatch, layer label, hex value, pixel count, move/delete buttons.
The bottom of the panel has action buttons for merge, delete, duplicate, etc.

Supports: visibility toggle, color picker, delete (single & bulk), merge,
merge by color, select by color, invert selection, shift+click range
selection, move up/down, duplicate, rename (double-click label),
compact view toggle, sort, filter by size, right-click context menu,
and Delete key shortcut.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
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
    selection_requested = pyqtSignal(int, Qt.KeyboardModifier)

    # Signal for context menu actions that need panel coordination
    context_select_color = pyqtSignal(str)       # hex_color
    context_hide_others = pyqtSignal(int)         # keep this index visible
    context_show_all = pyqtSignal()

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self._index = info.index
        self._hex = info.hex_color
        self._name = info.name or f"Layer {info.index + 1}"
        self._pixel_count = info.pixel_count
        self._selected = False
        self._compact = False

        self.setProperty("cssClass", "layerCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setProperty("selected", "false")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

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
        self._up_btn = QPushButton("\u25B2")
        self._up_btn.setToolTip("Move layer up")
        self._up_btn.setFixedSize(22, 22)
        self._up_btn.setStyleSheet(
            f"QPushButton {{ font-size: 9px; background: transparent; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: 3px; "
            f"color: {TEXT_SEC}; padding: 0; }}"
            f"QPushButton:hover {{ background: {CARD}; color: {TEXT}; }}"
        )
        self._up_btn.clicked.connect(
            lambda: self.move_up_requested.emit(self._index)
        )
        layout.addWidget(self._up_btn)

        # Move down button
        self._down_btn = QPushButton("\u25BC")
        self._down_btn.setToolTip("Move layer down")
        self._down_btn.setFixedSize(22, 22)
        self._down_btn.setStyleSheet(
            f"QPushButton {{ font-size: 9px; background: transparent; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: 3px; "
            f"color: {TEXT_SEC}; padding: 0; }}"
            f"QPushButton:hover {{ background: {CARD}; color: {TEXT}; }}"
        )
        self._down_btn.clicked.connect(
            lambda: self.move_down_requested.emit(self._index)
        )
        layout.addWidget(self._down_btn)

        # Delete button
        self._del_btn = QPushButton("\u2715")  # x
        self._del_btn.setProperty("cssClass", "danger")
        self._del_btn.setToolTip("Remove this layer  (Delete)")
        self._del_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._index)
        )
        layout.addWidget(self._del_btn)

    # -- properties -------------------------------------------------------

    @property
    def index(self) -> int:
        return self._index

    @property
    def hex_color(self) -> str:
        return self._hex

    @property
    def pixel_count(self) -> int:
        return self._pixel_count

    @property
    def is_selected(self) -> bool:
        return self._selected

    @property
    def is_visible(self) -> bool:
        return self.select_cb.isChecked()

    # -- public -----------------------------------------------------------

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def update_info(self, info: LayerInfo) -> None:
        """Refresh display from a layer snapshot."""
        self._index = info.index
        self._hex = info.hex_color
        self._name = info.name or f"Layer {info.index + 1}"
        self._pixel_count = info.pixel_count
        self.swatch_label.setPixmap(_swatch_pixmap(info.hex_color, 22))
        self.hex_label.setText(info.hex_color)
        self.name_label.setText(self._name)
        self.count_label.setText(f"{info.pixel_count:,} px")
        self.select_cb.blockSignals(True)
        self.select_cb.setChecked(info.visible)
        self.select_cb.blockSignals(False)

    def set_compact(self, compact: bool) -> None:
        """Toggle between full (52px) and compact (28px) display."""
        if self._compact == compact:
            return
        self._compact = compact
        if compact:
            self.setFixedHeight(28)
            self.layout().setContentsMargins(6, 2, 4, 2)
            self.swatch_label.setFixedSize(18, 18)
            self.swatch_label.setPixmap(_swatch_pixmap(self._hex, 16))
            self.name_label.setStyleSheet(
                f"font-size: 11px; font-weight: 500; color: {TEXT};"
            )
            self.hex_label.hide()
            self.count_label.hide()
            self._up_btn.hide()
            self._down_btn.hide()
            self._del_btn.hide()
        else:
            self.setFixedHeight(52)
            self.layout().setContentsMargins(10, 6, 6, 6)
            self.swatch_label.setFixedSize(26, 26)
            self.swatch_label.setPixmap(_swatch_pixmap(self._hex, 22))
            self.name_label.setStyleSheet(
                f"font-size: 13px; font-weight: 500; color: {TEXT};"
            )
            self.hex_label.show()
            self.count_label.show()
            self._up_btn.show()
            self._down_btn.show()
            self._del_btn.show()

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

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.selection_requested.emit(self._index, event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    # -- context menu -----------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Change Color\u2026", lambda: self.color_change_requested.emit(self._index))
        menu.addAction("Select Same Color", lambda: self.context_select_color.emit(self._hex))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.delete_requested.emit(self._index))
        menu.addAction("Rename\u2026", self._start_rename)
        menu.addSeparator()
        menu.addAction("Hide Others", lambda: self.context_hide_others.emit(self._index))
        menu.addAction("Show All", lambda: self.context_show_all.emit())
        menu.addSeparator()
        menu.addAction("Move Up", lambda: self.move_up_requested.emit(self._index))
        menu.addAction("Move Down", lambda: self.move_down_requested.emit(self._index))
        menu.exec(self.mapToGlobal(pos))

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
    delete_multiple_requested = pyqtSignal(list)  # list[int]
    merge_requested = pyqtSignal(list)  # list[int]
    merge_by_color_requested = pyqtSignal()
    select_by_color_requested = pyqtSignal(str)  # hex_color
    recolor_selected_requested = pyqtSignal(list)  # list[int]
    move_up_requested = pyqtSignal(int)
    move_down_requested = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    rename_requested = pyqtSignal(int, str)
    selection_changed = pyqtSignal(list)  # list[int]
    hide_others_requested = pyqtSignal(int)  # keep this index visible
    show_all_requested = pyqtSignal()
    hide_all_requested = pyqtSignal()
    sort_changed = pyqtSignal(str)  # sort key name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("layerPanel")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._last_clicked_index: int | None = None  # for Shift+click range
        self._compact_mode = False
        self._current_sort = "default"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        # Header row with count badge + compact toggle
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

        # Compact view toggle
        self._compact_btn = QPushButton("\u2630")  # hamburger icon
        self._compact_btn.setCheckable(True)
        self._compact_btn.setFixedSize(28, 28)
        self._compact_btn.setToolTip("Toggle compact view")
        self._compact_btn.setStyleSheet(
            f"QPushButton {{ font-size: 14px; background: transparent; "
            f"border: 1px solid {BORDER_LIGHT}; border-radius: 4px; "
            f"color: {TEXT_SEC}; padding: 0; }}"
            f"QPushButton:hover {{ background: {CARD}; color: {TEXT}; }}"
            f"QPushButton:checked {{ background: {CARD}; color: {TEXT}; "
            f"border-color: {BORDER}; }}"
        )
        self._compact_btn.toggled.connect(self._on_compact_toggled)
        header_row.addWidget(self._compact_btn)
        outer.addLayout(header_row)

        # Search / filter by name
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("\U0001F50D  Filter layers by name or color\u2026")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setFixedHeight(28)
        self._search_edit.setStyleSheet(
            f"QLineEdit {{ font-size: 12px; padding: 2px 8px; "
            f"border: 1px solid {BORDER}; border-radius: 4px; "
            f"background: {SURFACE}; color: {TEXT}; }}"
            f"QLineEdit:focus {{ border-color: {BORDER_LIGHT}; }}"
        )
        self._search_edit.setToolTip(
            "Type to filter layers by name or hex color  (Ctrl+F)"
        )
        self._search_edit.textChanged.connect(self._apply_name_filter)
        outer.addWidget(self._search_edit)

        # Sort + filter row
        sort_filter_row = QHBoxLayout()
        sort_filter_row.setSpacing(6)

        sort_label = QLabel("Sort:")
        sort_label.setStyleSheet(f"font-size: 11px; color: {TEXT_SEC};")
        sort_filter_row.addWidget(sort_label)

        self._sort_combo = QComboBox()
        self._sort_combo.setFixedHeight(26)
        self._sort_combo.setFixedWidth(120)
        self._sort_combo.addItems([
            "Default",
            "Color",
            "Size \u2191",       # smallest first
            "Size \u2193",       # largest first
            "Position \u2193",   # top-to-bottom
        ])
        self._sort_combo.setToolTip("Sort layers")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_filter_row.addWidget(self._sort_combo)

        sort_filter_row.addStretch()

        # "Select Small" button
        self._select_small_btn = QPushButton("Select Small")
        self._select_small_btn.setFixedHeight(26)
        self._select_small_btn.setToolTip(
            "Select layers smaller than the threshold"
        )
        self._select_small_btn.clicked.connect(self._on_select_small)
        sort_filter_row.addWidget(self._select_small_btn)

        outer.addLayout(sort_filter_row)

        # Small-object threshold slider
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(6)

        thresh_label = QLabel("Min px:")
        thresh_label.setStyleSheet(f"font-size: 11px; color: {TEXT_SEC};")
        thresh_row.addWidget(thresh_label)

        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(0, 5000)
        self._thresh_slider.setValue(500)
        self._thresh_slider.setSingleStep(50)
        self._thresh_slider.setPageStep(500)
        self._thresh_slider.setToolTip("Pixel-count threshold for 'Select Small'")
        self._thresh_slider.valueChanged.connect(self._on_thresh_changed)
        thresh_row.addWidget(self._thresh_slider, stretch=1)

        self._thresh_value_label = QLabel("500")
        self._thresh_value_label.setFixedWidth(42)
        self._thresh_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._thresh_value_label.setStyleSheet(
            f"font-family: 'Cascadia Code', 'Consolas', monospace; "
            f"font-size: 11px; color: {TEXT_SEC};"
        )
        thresh_row.addWidget(self._thresh_value_label)

        outer.addLayout(thresh_row)

        # Selection actions row 1: Select All / Deselect All / Invert
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)

        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setFixedHeight(28)
        self._select_all_btn.setToolTip("Select all layers  (Ctrl+A)")
        self._select_all_btn.clicked.connect(self.select_all)
        actions_row.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.setFixedHeight(28)
        self._deselect_all_btn.clicked.connect(self.deselect_all)
        actions_row.addWidget(self._deselect_all_btn)

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setFixedHeight(28)
        self._invert_btn.setToolTip("Invert current selection")
        self._invert_btn.clicked.connect(self.invert_selection)
        actions_row.addWidget(self._invert_btn)

        actions_row.addStretch()
        outer.addLayout(actions_row)

        # Selection actions row 2: Select by Color / Recolor Selected
        actions_row2 = QHBoxLayout()
        actions_row2.setSpacing(6)

        self._select_color_btn = QPushButton("Select Same Color")
        self._select_color_btn.setFixedHeight(28)
        self._select_color_btn.setEnabled(False)
        self._select_color_btn.setToolTip(
            "Select all layers matching the selected layer's color"
        )
        self._select_color_btn.clicked.connect(self._on_select_by_color)
        actions_row2.addWidget(self._select_color_btn)

        self._recolor_btn = QPushButton("Recolor Selected")
        self._recolor_btn.setFixedHeight(28)
        self._recolor_btn.setEnabled(False)
        self._recolor_btn.setToolTip(
            "Change the color of all selected layers at once"
        )
        self._recolor_btn.clicked.connect(self._on_recolor_selected)
        actions_row2.addWidget(self._recolor_btn)

        actions_row2.addStretch()
        outer.addLayout(actions_row2)

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

        # Action buttons row 1: Merge Selected / Delete Selected
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.setProperty("cssClass", "primary")
        self._merge_btn.setEnabled(False)
        self._merge_btn.setFixedHeight(32)
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        btn_row.addWidget(self._merge_btn)

        self._delete_sel_btn = QPushButton("Delete Selected")
        self._delete_sel_btn.setProperty("cssClass", "danger")
        self._delete_sel_btn.setEnabled(False)
        self._delete_sel_btn.setFixedHeight(32)
        self._delete_sel_btn.setToolTip("Delete all selected layers  (Delete)")
        self._delete_sel_btn.clicked.connect(self._on_delete_selected_clicked)
        btn_row.addWidget(self._delete_sel_btn)

        outer.addLayout(btn_row)

        # Action buttons row 2: Merge by Color / Duplicate
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)

        self._merge_color_btn = QPushButton("Merge by Color")
        self._merge_color_btn.setFixedHeight(32)
        self._merge_color_btn.setToolTip(
            "Merge all layers that share the same color into single layers"
        )
        self._merge_color_btn.clicked.connect(self._on_merge_by_color_clicked)
        btn_row2.addWidget(self._merge_color_btn)

        self._dup_btn = QPushButton("Duplicate")
        self._dup_btn.setFixedHeight(32)
        self._dup_btn.setEnabled(False)
        self._dup_btn.setToolTip("Duplicate the first selected layer")
        self._dup_btn.clicked.connect(self._on_duplicate_clicked)
        btn_row2.addWidget(self._dup_btn)

        outer.addLayout(btn_row2)

        self._rows: list[LayerRow] = []
        self._layer_infos: list[LayerInfo] = []  # cached for sort/filter

    # -- public API -------------------------------------------------------

    def set_layers(self, layers: list[LayerInfo]) -> None:
        """Rebuild the entire list from a fresh snapshot."""
        self._layer_infos = list(layers)
        selected_before = set(self.get_selected_indices())
        self._clear_rows()

        # Apply current sort order for display
        display_order = self._sorted_infos(layers)

        for info in display_order:
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
            row.selection_requested.connect(self._on_row_selection_requested)
            # Context menu signals
            row.context_select_color.connect(self.select_by_color)
            row.context_hide_others.connect(self.hide_others_requested.emit)
            row.context_show_all.connect(self.show_all_requested.emit)
            if self._compact_mode:
                row.set_compact(True)
            self._rows.append(row)
            self._layout.insertWidget(self._layout.count() - 1, row)

            if info.index in selected_before:
                row.set_selected(True)
        self._count_badge.setText(str(len(layers)))
        self._update_action_buttons()
        # Re-apply any active name filter
        self._apply_name_filter(self._search_edit.text())

    def get_selected_indices(self) -> list[int]:
        return [r.index for r in self._rows if r.is_selected]

    def select_all(self) -> None:
        for row in self._rows:
            row.set_selected(True)
            row.select_cb.blockSignals(True)
            row.select_cb.setChecked(True)
            row.select_cb.blockSignals(False)
        self._update_action_buttons()
        self.selection_changed.emit(self.get_selected_indices())
        self.show_all_requested.emit()

    def deselect_all(self) -> None:
        for row in self._rows:
            row.set_selected(False)
            row.select_cb.blockSignals(True)
            row.select_cb.setChecked(False)
            row.select_cb.blockSignals(False)
        self._update_action_buttons()
        self.selection_changed.emit(self.get_selected_indices())
        self.hide_all_requested.emit()

    def invert_selection(self) -> None:
        """Flip the selection state of every row."""
        for row in self._rows:
            row.set_selected(not row.is_selected)
        self._update_action_buttons()
        self.selection_changed.emit(self.get_selected_indices())

    def select_by_color(self, hex_color: str) -> None:
        """Select all layers that share the given hex colour."""
        for row in self._rows:
            row.set_selected(row.hex_color == hex_color)
        self._update_action_buttons()
        self.selection_changed.emit(self.get_selected_indices())

    def set_selected_indices(self, indices: list[int]) -> None:
        selected = set(indices)
        for row in self._rows:
            row.set_selected(row.index in selected)
        self._update_action_buttons()
        self.selection_changed.emit(self.get_selected_indices())

    def clear(self) -> None:
        self._clear_rows()
        self._layer_infos.clear()
        self._count_badge.setText("0")

    # -- events -----------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle Delete key (bulk delete), Ctrl+A (select all), Ctrl+F (focus search)."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sel = self.get_selected_indices()
            if sel:
                self._confirm_and_delete(sel)
                return
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select_all()
            return
        elif event.key() == Qt.Key.Key_F and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._search_edit.setFocus()
            self._search_edit.selectAll()
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
        n = len(sel)
        self._merge_btn.setEnabled(n >= 2)
        self._dup_btn.setEnabled(n >= 1)
        self._delete_sel_btn.setEnabled(n >= 1)
        self._recolor_btn.setEnabled(n >= 1)
        # "Select Same Color" needs at least 1 selected
        self._select_color_btn.setEnabled(n >= 1)

    # -- name filter ------------------------------------------------------

    def _apply_name_filter(self, text: str = "") -> None:
        """Show/hide rows based on the search text.

        Matches against layer name and hex colour (case-insensitive).
        An empty query shows everything.
        """
        query = text.strip().lower() if text else ""
        visible_count = 0
        for row in self._rows:
            if not query:
                row.setVisible(True)
                visible_count += 1
            else:
                name_match = query in row._name.lower()
                hex_match = query in row.hex_color.lower()
                match = name_match or hex_match
                row.setVisible(match)
                if match:
                    visible_count += 1

    # -- compact view -----------------------------------------------------

    def _on_compact_toggled(self, checked: bool) -> None:
        self._compact_mode = checked
        for row in self._rows:
            row.set_compact(checked)

    # -- sort -------------------------------------------------------------

    def _on_sort_changed(self, combo_index: int) -> None:
        sort_keys = ["default", "color", "size_asc", "size_desc", "position"]
        self._current_sort = sort_keys[combo_index] if combo_index < len(sort_keys) else "default"
        # Re-display with new sort (re-use cached infos)
        if self._layer_infos:
            self.set_layers(self._layer_infos)
        self.sort_changed.emit(self._current_sort)

    def _sorted_infos(self, layers: list[LayerInfo]) -> list[LayerInfo]:
        """Return layers in display order based on current sort."""
        if self._current_sort == "color":
            return sorted(layers, key=lambda li: li.hex_color)
        elif self._current_sort == "size_asc":
            return sorted(layers, key=lambda li: li.pixel_count)
        elif self._current_sort == "size_desc":
            return sorted(layers, key=lambda li: li.pixel_count, reverse=True)
        elif self._current_sort == "position":
            # Sort by index (which reflects spatial order from separation)
            return sorted(layers, key=lambda li: li.index)
        return list(layers)  # default — keep session order

    # -- threshold filter -------------------------------------------------

    def _on_thresh_changed(self, value: int) -> None:
        self._thresh_value_label.setText(str(value))

    def _on_select_small(self) -> None:
        """Select all layers whose pixel count is below the threshold."""
        threshold = self._thresh_slider.value()
        to_select: list[int] = []
        for row in self._rows:
            if row.pixel_count < threshold:
                to_select.append(row.index)
        if to_select:
            self.set_selected_indices(to_select)
        else:
            QMessageBox.information(
                self,
                "Select Small",
                f"No layers found with fewer than {threshold:,} pixels.",
            )

    def _on_merge_clicked(self) -> None:
        sel = self.get_selected_indices()
        if len(sel) >= 2:
            self.merge_requested.emit(sel)

    def _on_merge_by_color_clicked(self) -> None:
        self.merge_by_color_requested.emit()

    def _on_delete_selected_clicked(self) -> None:
        sel = self.get_selected_indices()
        if sel:
            self._confirm_and_delete(sel)

    def _confirm_and_delete(self, indices: list[int]) -> None:
        """Delete selected layers, with confirmation if more than one."""
        if len(indices) == 1:
            self.delete_requested.emit(indices[0])
            return
        reply = QMessageBox.question(
            self,
            "Delete Layers",
            f"Delete {len(indices)} selected layers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_multiple_requested.emit(indices)

    def _on_duplicate_clicked(self) -> None:
        sel = self.get_selected_indices()
        if sel:
            self.duplicate_requested.emit(sel[0])

    def _on_select_by_color(self) -> None:
        """Select all layers sharing the same colour as the first selected."""
        sel = self.get_selected_indices()
        if not sel:
            return
        # Get the hex color of the first selected row
        for row in self._rows:
            if row.index == sel[0]:
                self.select_by_color(row.hex_color)
                return

    def _on_recolor_selected(self) -> None:
        """Emit signal to recolor all selected layers."""
        sel = self.get_selected_indices()
        if sel:
            self.recolor_selected_requested.emit(sel)

    def _on_row_selection_requested(
        self, index: int, modifiers: Qt.KeyboardModifier
    ) -> None:
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if shift and self._last_clicked_index is not None:
            # Shift+click: range selection
            row_indices = [r.index for r in self._rows]
            try:
                anchor_pos = row_indices.index(self._last_clicked_index)
                target_pos = row_indices.index(index)
            except ValueError:
                # Fallback to single select if anchor is gone
                self.set_selected_indices([index])
                self._last_clicked_index = index
                return
            lo = min(anchor_pos, target_pos)
            hi = max(anchor_pos, target_pos)
            range_indices = row_indices[lo : hi + 1]

            if ctrl:
                # Shift+Ctrl: add range to existing selection
                current = set(self.get_selected_indices())
                current.update(range_indices)
                self.set_selected_indices(list(current))
            else:
                # Shift only: replace selection with range
                self.set_selected_indices(range_indices)
            # Don't update _last_clicked_index on shift-clicks
            return
        elif ctrl:
            # Ctrl+click: toggle single
            for row in self._rows:
                if row.index == index:
                    row.set_selected(not row.is_selected)
                    break
            self._last_clicked_index = index
            self.selection_changed.emit(self.get_selected_indices())
        else:
            # Normal click: select one
            self.set_selected_indices([index])
            self._last_clicked_index = index
            return
        self._update_action_buttons()
