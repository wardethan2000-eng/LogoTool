"""Left-panel source-image info widget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SourcePanel(QWidget):
    """Displays a thumbnail of the source image and basic file info."""

    THUMB_SIZE = 180

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Source Image")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        self._thumb = QLabel()
        self._thumb.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background: #2b2b2b; border: 1px solid #555;")
        layout.addWidget(self._thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("font-size: 12px; color: #aaa;")
        layout.addWidget(self._info)

        layout.addStretch()

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
        self._info.setText(
            f"<b>{path.name}</b><br>"
            f"{w} × {h} px<br>"
            f"{suffix} · {size_kb:,.1f} KB"
        )

    def clear(self) -> None:
        self._thumb.clear()
        self._info.setText("")
