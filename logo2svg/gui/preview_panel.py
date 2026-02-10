"""Centre-panel SVG / raster composite preview widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

import numpy as np


class PreviewPanel(QWidget):
    """Displays an RGBA composite of all visible layers, scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = QLabel("Load an image to begin")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "color: #888; font-size: 16px; background: #2b2b2b;"
        )
        self._scroll.setWidget(self._image_label)
        layout.addWidget(self._scroll)

        self._current_pixmap: QPixmap | None = None

    # -----------------------------------------------------------------

    def set_composite(self, rgba: np.ndarray) -> None:
        """Display an (H, W, 4) RGBA uint8 numpy array.

        The image is shown with a chequerboard background to indicate
        transparency, scaled to fit the available space.
        """
        h, w = rgba.shape[:2]
        # Build chequerboard background
        checker = self._checker_board(h, w)
        # Alpha-composite over the chequerboard
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = (rgba[:, :, :3].astype(np.float32) * alpha +
                   checker.astype(np.float32) * (1 - alpha))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Convert to QImage (RGB888)
        qimg = QImage(
            blended.data.tobytes(),
            w, h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._fit_pixmap()

    def set_source_image(self, rgb: np.ndarray) -> None:
        """Show the original RGB source thumbnail."""
        h, w = rgb.shape[:2]
        qimg = QImage(
            rgb.data.tobytes(),
            w, h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._fit_pixmap()

    def clear(self) -> None:
        self._current_pixmap = None
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("Load an image to begin")

    # -----------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        avail = self._scroll.size()
        scaled = self._current_pixmap.scaled(
            avail,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")

    @staticmethod
    def _checker_board(h: int, w: int, cell: int = 8) -> np.ndarray:
        """Return (H, W, 3) uint8 chequerboard pattern."""
        rows = np.arange(h) // cell
        cols = np.arange(w) // cell
        grid = (rows[:, None] + cols[None, :]) % 2
        light, dark = 204, 153
        board = np.where(grid[:, :, None] == 0, light, dark).astype(np.uint8)
        return np.broadcast_to(board, (h, w, 3)).copy()
