"""Left-side source-image viewer.

Displays the original loaded image at a large size so it is clear what
is being processed.  File metadata is shown in a compact header row.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .style import BG, BORDER, SURFACE, TEXT_MUTED, TEXT_SEC


class SourcePanel(QWidget):
    """Large source-image viewer for the left panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {SURFACE};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Compact header row: "SOURCE · filename · dimensions"
        header_frame = QWidget()
        header_frame.setFixedHeight(36)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(8)

        header_label = QLabel("SOURCE")
        header_label.setObjectName("sectionHeader")
        header_layout.addWidget(header_label)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet(f"font-size: 12px; color: {TEXT_SEC};")
        header_layout.addWidget(self._info_label)
        header_layout.addStretch()

        layout.addWidget(header_frame)

        # Divider
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {BORDER};")
        layout.addWidget(divider)

        # Image display (fills remaining space)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(f"background-color: {BG};")
        layout.addWidget(self._image_label, stretch=1)

        self._current_pixmap: QPixmap | None = None
        self._source_rgb: np.ndarray | None = None
        self._source_fg_mask: np.ndarray | None = None
        self._show_mask_overlay = False
        self._show_empty()

    # -- public API -------------------------------------------------------

    def set_image(
        self,
        rgb: np.ndarray,
        path: Path,
        fg_mask: np.ndarray | None = None,
    ) -> None:
        """Show the source image scaled to fit the panel."""
        self._source_rgb = np.ascontiguousarray(rgb)
        self._source_fg_mask = None if fg_mask is None else np.asarray(fg_mask, dtype=bool)
        self._update_pixmap()

        h, w = rgb.shape[:2]

        suffix = path.suffix.upper().lstrip(".")
        size_kb = path.stat().st_size / 1024
        self._info_label.setText(
            f"{path.name}  \u00b7  {w}\u00d7{h}  \u00b7  {suffix}"
            f"  \u00b7  {size_kb:,.1f} KB"
        )
        self._image_label.setStyleSheet(f"background-color: {BG};")
        self._fit_pixmap()

    def clear(self) -> None:
        self._current_pixmap = None
        self._image_label.setPixmap(QPixmap())
        self._info_label.setText("")
        self._show_empty()
        self._source_rgb = None
        self._source_fg_mask = None

    def set_mask_overlay_visible(self, visible: bool) -> None:
        """Toggle the foreground-mask overlay on the source image."""
        self._show_mask_overlay = visible
        self._update_pixmap()
        self._fit_pixmap()

    # -- events -----------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    # -- internals --------------------------------------------------------

    def _show_empty(self) -> None:
        self._image_label.setText(
            '<div style="text-align: center; padding: 24px;">'
            f'<p style="font-size: 14px; color: {TEXT_MUTED};">'
            f"Source image will appear here</p>"
            "</div>"
        )
        self._image_label.setStyleSheet(f"background-color: {BG};")

    def _update_pixmap(self) -> None:
        if self._source_rgb is None:
            self._current_pixmap = None
            return

        display = self._source_rgb
        if self._show_mask_overlay and self._source_fg_mask is not None:
            display = _compose_mask_overlay(self._source_rgb, self._source_fg_mask)

        h, w = display.shape[:2]
        qimg = QImage(
            display.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qimg)

    def _fit_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        margin = 12
        target_w = max(1, self._image_label.width() - margin * 2)
        target_h = max(1, self._image_label.height() - margin * 2)
        scaled = self._current_pixmap.scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")


def _compose_mask_overlay(rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Return an RGB image with the foreground mask visualized on top."""
    display = rgb.astype(np.float32).copy()
    fg_mask = np.asarray(fg_mask, dtype=bool)

    background = ~fg_mask
    if np.any(background):
        display[background] = display[background] * 0.35 + np.array([245, 92, 92], dtype=np.float32) * 0.65

    if np.any(fg_mask):
        display[fg_mask] = display[fg_mask] * 0.9 + np.array([90, 215, 160], dtype=np.float32) * 0.1

    outline = cv2.morphologyEx(
        (fg_mask.astype(np.uint8) * 255),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    display[outline] = np.array([0, 255, 160], dtype=np.float32)

    return np.clip(display, 0, 255).astype(np.uint8)
