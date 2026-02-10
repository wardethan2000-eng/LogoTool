"""Centre preview panel with composite rendering and empty-state drop zone.

The preview occupies the main content area and shows the RGBA composite
of all visible layers over a light checkerboard pattern.  When no image
is loaded, a subtle drop-zone hint is displayed.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .style import BG, BORDER, SURFACE, TEXT_MUTED, TEXT_SEC


class PreviewPanel(QWidget):
    """Displays an RGBA composite of all visible layers, scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._image_label, stretch=1)

        self._current_pixmap: QPixmap | None = None
        self._show_empty_state()

    # -- public API -------------------------------------------------------

    def set_composite(self, rgba: np.ndarray) -> None:
        """Display an (H, W, 4) RGBA uint8 numpy array.

        Alpha-composited over a light checkerboard to indicate transparency.
        """
        h, w = rgba.shape[:2]
        checker = self._checker_board(h, w)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = (
            rgba[:, :, :3].astype(np.float32) * alpha
            + checker.astype(np.float32) * (1.0 - alpha)
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        qimg = QImage(
            blended.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._image_label.setStyleSheet(f"background-color: {BG};")
        self._fit_pixmap()

    def set_source_image(self, rgb: np.ndarray) -> None:
        """Show the original RGB source as a quick preview."""
        h, w = rgb.shape[:2]
        qimg = QImage(
            rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._image_label.setStyleSheet(f"background-color: {BG};")
        self._fit_pixmap()

    def clear(self) -> None:
        self._current_pixmap = None
        self._image_label.setPixmap(QPixmap())
        self._show_empty_state()

    # -- events -----------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    # -- internals --------------------------------------------------------

    def _show_empty_state(self) -> None:
        """Show the drag-and-drop / open hint."""
        self._image_label.setText(
            '<div style="text-align: center; padding: 48px;">'
            f'<p style="font-size: 40px; color: {TEXT_MUTED}; '
            f'margin-bottom: 16px;">\u25C7</p>'
            f'<p style="font-size: 15px; color: {TEXT_SEC}; '
            f'font-weight: 600; margin-bottom: 8px;">No image loaded</p>'
            f'<p style="font-size: 12px; color: {TEXT_MUTED};">'
            f"Open a file or drag and drop an image here</p>"
            "</div>"
        )
        self._image_label.setStyleSheet(
            f"QLabel {{ background-color: {SURFACE}; "
            f"border: 2px dashed {BORDER}; border-radius: 12px; "
            f"margin: 32px; }}"
        )

    def _fit_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        margin = 16
        target_w = max(1, self.width() - margin * 2)
        target_h = max(1, self.height() - margin * 2)
        scaled = self._current_pixmap.scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")

    @staticmethod
    def _checker_board(h: int, w: int, cell: int = 10) -> np.ndarray:
        """Return (H, W, 3) uint8 light checkerboard pattern."""
        rows = np.arange(h) // cell
        cols = np.arange(w) // cell
        grid = (rows[:, None] + cols[None, :]) % 2
        light, dark = 255, 204  # white / light-gray checker
        board = np.where(grid[:, :, None] == 0, light, dark).astype(np.uint8)
        return np.broadcast_to(board, (h, w, 3)).copy()
