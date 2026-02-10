"""Right-sidebar source-image info card.

Displays a compact thumbnail and basic file metadata in the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .style import BORDER, CARD, TEXT, TEXT_MUTED, TEXT_SEC


class SourcePanel(QWidget):
    """Compact source-image card for the right sidebar."""

    THUMB_SIZE = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        # Section header
        header = QLabel("SOURCE")
        header.setObjectName("sectionHeader")
        outer.addWidget(header)

        # Content row: thumbnail + info
        row = QHBoxLayout()
        row.setSpacing(12)

        self._thumb = QLabel()
        self._thumb.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        row.addWidget(self._thumb)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        self._name_label = QLabel("No image loaded")
        self._name_label.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {TEXT_MUTED};"
        )
        self._name_label.setWordWrap(True)
        info_col.addWidget(self._name_label)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SEC};")
        info_col.addWidget(self._detail_label)
        info_col.addStretch()

        row.addLayout(info_col, stretch=1)
        outer.addLayout(row)

    # -- public API -------------------------------------------------------

    def set_image(self, rgb: np.ndarray, path: Path) -> None:
        """Show a thumbnail and file info."""
        h, w = rgb.shape[:2]
        qimg = QImage(
            rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        pm = QPixmap.fromImage(qimg).scaled(
            self.THUMB_SIZE,
            self.THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(pm)

        suffix = path.suffix.upper().lstrip(".")
        size_kb = path.stat().st_size / 1024
        self._name_label.setText(path.name)
        self._name_label.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {TEXT};"
        )
        self._detail_label.setText(
            f"{w} \u00d7 {h}  \u00b7  {suffix}  \u00b7  {size_kb:,.1f} KB"
        )

    def clear(self) -> None:
        self._thumb.clear()
        self._name_label.setText("No image loaded")
        self._name_label.setStyleSheet(
            f"font-weight: 600; font-size: 13px; color: {TEXT_MUTED};"
        )
        self._detail_label.setText("")
