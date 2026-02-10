"""Centre preview panel with composite rendering and empty-state drop zone.

The preview occupies the main content area and shows the RGBA composite
of all visible layers over a light checkerboard pattern.  When no image
is loaded, a subtle drop-zone hint is displayed.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .style import ACCENT, BG, BORDER, SURFACE, TEXT_MUTED, TEXT_SEC


class _ProcessingOverlay(QWidget):
    """Semi-transparent overlay with a spinning arc shown during processing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setVisible(False)

        self._angle = 0
        self._message = "Processing\u2026"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(25)

    # -- public -----------------------------------------------------------

    def show_message(self, msg: str) -> None:
        self._message = msg
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.setVisible(True)
        self.raise_()
        self._timer.start()

    def update_message(self, msg: str) -> None:
        self._message = msg
        self.update()

    def hide_overlay(self) -> None:
        self._timer.stop()
        self.setVisible(False)

    # -- internals --------------------------------------------------------

    def _tick(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent white overlay
        painter.fillRect(self.rect(), QColor(255, 255, 255, 190))

        cx = self.width() / 2
        cy = self.height() / 2
        r = 24
        arc_rect = QRectF(cx - r, cy - r - 12, r * 2, r * 2)

        # Light background ring
        bg_pen = QPen(QColor(BORDER), 3.0)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(arc_rect)

        # Spinning accent arc
        accent_pen = QPen(QColor(ACCENT), 3.0)
        accent_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(accent_pen)
        painter.drawArc(arc_rect, int(self._angle * 16), int(270 * 16))

        # Message text
        painter.setPen(QColor(TEXT_SEC))
        font = painter.font()
        font.setPointSize(12)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        text_rect = QRectF(0, cy + r + 2, self.width(), 30)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._message)

        painter.end()


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

        # Processing overlay (child of this widget, covers entire panel)
        self._overlay = _ProcessingOverlay(self)

    # -- public API -------------------------------------------------------

    def show_processing(self, message: str = "Processing\u2026") -> None:
        """Show a centered spinner overlay with *message*."""
        self._overlay.show_message(message)

    def update_processing_message(self, message: str) -> None:
        """Update the overlay text while it's visible."""
        if self._overlay.isVisible():
            self._overlay.update_message(message)

    def hide_processing(self) -> None:
        """Hide the spinner overlay."""
        self._overlay.hide_overlay()

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
        # Keep overlay sized to match panel
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.rect())

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
