"""Centre preview panel with composite rendering, zoom/pan, layer selection,
drag-to-move, and alignment guides.

The preview occupies the main content area and shows the RGBA composite
of all visible layers over a light checkerboard pattern.  When no image
is loaded, a subtle drop-zone hint is displayed.

Interaction modes:
  - Left-click on a layer: select it (highlighted bounding box)
  - Drag a selected layer: move its pixels on the canvas
  - Double-click a text layer: request edit (signal emitted)
  - Left-click on empty space / Ctrl+drag: pan the view
  - Middle-click drag: pan the view (always)
  - Mouse wheel: zoom in/out (centred on cursor)
  - Double-click empty: reset zoom
  - Alignment guides appear when a moved layer's centre aligns with
    the canvas centre (horizontal or vertical).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
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


# Snap threshold in image-space pixels for alignment guides
_ALIGN_SNAP_PX = 4


class PreviewPanel(QWidget):
    """Displays an RGBA composite of all visible layers with zoom/pan support
    and interactive layer selection / drag-to-move.
    """

    # Zoom limits
    _MIN_ZOOM = 0.1
    _MAX_ZOOM = 20.0

    # Signals for layer interaction
    layer_selected = pyqtSignal(int)      # index (-1 = deselect)
    layer_moved = pyqtSignal(int, int, int)   # index, dx, dy (image-space)
    layer_double_clicked = pyqtSignal(int)    # index (for text edit)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG};")
        self.setMouseTracking(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._image_label, stretch=1)

        self._current_pixmap: QPixmap | None = None
        # Original image dimensions (needed for coordinate mapping)
        self._image_w: int = 0
        self._image_h: int = 0

        # Zoom/pan state
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF(0, 0)
        self._pan_offset_start = QPointF(0, 0)

        # Layer selection / move state
        self._selected_layer: int = -1
        self._moving_layer = False
        self._move_start_img = (0, 0)      # image-space start of drag
        self._move_accumulated = (0, 0)     # accumulated dx, dy in image-space
        self._layer_hit_fn = None           # callback: (x, y) -> int | None
        self._layer_bbox_fn = None          # callback: (index) -> (x, y, w, h) | None

        # Selection bbox overlay (image-space)
        self._sel_bbox: tuple[int, int, int, int] | None = None

        # Alignment guides state (shown during move)
        self._show_h_guide = False
        self._show_v_guide = False

        # Rendered image position/size in widget-space (set by _fit_pixmap)
        self._img_rect_x = 0
        self._img_rect_y = 0
        self._img_rect_w = 1
        self._img_rect_h = 1

        self._show_empty_state()

        # Processing overlay (child of this widget, covers entire panel)
        self._overlay = _ProcessingOverlay(self)

    # -- public API -------------------------------------------------------

    @property
    def zoom_level(self) -> float:
        """Current zoom factor (1.0 = fit to panel)."""
        return self._zoom

    @property
    def selected_layer(self) -> int:
        """Index of the currently selected layer, or -1."""
        return self._selected_layer

    def set_selected_layer(self, index: int) -> None:
        """Programmatically set the selected layer."""
        self._selected_layer = index
        self._update_selection_bbox()
        self._fit_pixmap()

    def set_hit_test_callback(self, fn) -> None:
        """Set callback ``fn(x, y) -> int | None`` for layer hit-testing."""
        self._layer_hit_fn = fn

    def set_bbox_callback(self, fn) -> None:
        """Set callback ``fn(index) -> (x, y, w, h) | None`` for layer bbox."""
        self._layer_bbox_fn = fn

    def zoom_in(self) -> None:
        """Zoom in by one step (25%)."""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Zoom out by one step (25%)."""
        self._set_zoom(self._zoom / 1.25)

    def zoom_reset(self) -> None:
        """Reset zoom to fit-in-panel."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._fit_pixmap()

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
        The checkerboard is cached and only regenerated when the image
        dimensions change.
        """
        h, w = rgba.shape[:2]
        self._image_h = h
        self._image_w = w

        # Reuse cached checkerboard when dimensions haven't changed
        cached = getattr(self, "_cached_checker", None)
        if cached is None or cached.shape[0] != h or cached.shape[1] != w:
            cached = self._checker_board(h, w)
            self._cached_checker = cached

        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = (
            rgba[:, :, :3].astype(np.float32) * alpha
            + cached.astype(np.float32) * (1.0 - alpha)
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        qimg = QImage(
            blended.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._image_label.setStyleSheet(f"background-color: {BG};")
        self._update_selection_bbox()
        self._fit_pixmap()

    def set_source_image(self, rgb: np.ndarray) -> None:
        """Show the original RGB source as a quick preview."""
        h, w = rgb.shape[:2]
        self._image_h = h
        self._image_w = w
        qimg = QImage(
            rgb.data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._image_label.setStyleSheet(f"background-color: {BG};")
        self._fit_pixmap()

    def clear(self) -> None:
        self._current_pixmap = None
        self._image_label.setPixmap(QPixmap())
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._selected_layer = -1
        self._sel_bbox = None
        self._show_empty_state()

    # -- coordinate mapping -----------------------------------------------

    def _screen_to_image(self, sx: float, sy: float) -> tuple[int, int]:
        """Convert widget-space (screen) coords to image-space coords."""
        if self._img_rect_w <= 0 or self._img_rect_h <= 0:
            return -1, -1
        ix = (sx - self._img_rect_x) * self._image_w / self._img_rect_w
        iy = (sy - self._img_rect_y) * self._image_h / self._img_rect_h
        return int(ix), int(iy)

    def _image_to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        """Convert image-space coords to widget-space (screen) coords."""
        sx = ix * self._img_rect_w / max(1, self._image_w) + self._img_rect_x
        sy = iy * self._img_rect_h / max(1, self._image_h) + self._img_rect_y
        return sx, sy

    # -- events -----------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()
        # Keep overlay sized to match panel
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.rect())

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out with mouse wheel, centered on cursor."""
        if self._current_pixmap is None:
            return

        # Get cursor position relative to the widget center
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))

        # Adjust pan so zoom centers on cursor position
        cursor = event.position()
        center = QPointF(self.width() / 2, self.height() / 2)
        cursor_offset = cursor - center - self._pan_offset

        scale_change = new_zoom / self._zoom
        self._pan_offset -= cursor_offset * (scale_change - 1)
        self._zoom = new_zoom
        self._fit_pixmap()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._current_pixmap is None:
            return

        pos = event.position()

        # Middle button or Ctrl+Left always pans
        if (event.button() == Qt.MouseButton.MiddleButton
                or (event.button() == Qt.MouseButton.LeftButton
                    and event.modifiers() & Qt.KeyboardModifier.ControlModifier)):
            self._start_pan(pos)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Try to hit-test a layer at this position
            ix, iy = self._screen_to_image(pos.x(), pos.y())
            hit_index = None
            if self._layer_hit_fn is not None:
                hit_index = self._layer_hit_fn(ix, iy)

            if hit_index is not None:
                # Select the layer
                if hit_index != self._selected_layer:
                    self._selected_layer = hit_index
                    self._update_selection_bbox()
                    self.layer_selected.emit(hit_index)
                    self._fit_pixmap()

                # Start move drag
                self._moving_layer = True
                self._move_start_img = (ix, iy)
                self._move_accumulated = (0, 0)
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                # Deselect and pan
                if self._selected_layer >= 0:
                    self._selected_layer = -1
                    self._sel_bbox = None
                    self.layer_selected.emit(-1)
                    self._fit_pixmap()
                self._start_pan(pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_offset = self._pan_offset_start + delta
            self._fit_pixmap()
        elif self._moving_layer and self._selected_layer >= 0:
            pos = event.position()
            ix, iy = self._screen_to_image(pos.x(), pos.y())
            dx = ix - self._move_start_img[0]
            dy = iy - self._move_start_img[1]
            self._move_accumulated = (dx, dy)
            # Check alignment with canvas centre
            self._check_alignment_guides()
            self._fit_pixmap()
        else:
            # Update cursor based on what's under the mouse
            if self._layer_hit_fn is not None and self._current_pixmap is not None:
                pos = event.position()
                ix, iy = self._screen_to_image(pos.x(), pos.y())
                hit = self._layer_hit_fn(ix, iy)
                if hit is not None:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            if self._panning:
                self._panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
            elif self._moving_layer and self._selected_layer >= 0:
                dx, dy = self._move_accumulated
                self._moving_layer = False
                self._show_h_guide = False
                self._show_v_guide = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                if dx != 0 or dy != 0:
                    self.layer_moved.emit(self._selected_layer, dx, dy)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._current_pixmap is None:
            return

        pos = event.position()
        ix, iy = self._screen_to_image(pos.x(), pos.y())

        # Check if double-clicking on a layer
        hit_index = None
        if self._layer_hit_fn is not None:
            hit_index = self._layer_hit_fn(ix, iy)

        if hit_index is not None:
            self._selected_layer = hit_index
            self._update_selection_bbox()
            self.layer_double_clicked.emit(hit_index)
            self._fit_pixmap()
        else:
            # Double-click on empty space resets zoom
            self.zoom_reset()

    # -- internals --------------------------------------------------------

    def _start_pan(self, pos: QPointF) -> None:
        self._panning = True
        self._pan_start = pos
        self._pan_offset_start = QPointF(self._pan_offset)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _update_selection_bbox(self) -> None:
        """Refresh the cached selection bounding box."""
        if self._selected_layer < 0 or self._layer_bbox_fn is None:
            self._sel_bbox = None
            return
        self._sel_bbox = self._layer_bbox_fn(self._selected_layer)

    def _check_alignment_guides(self) -> None:
        """Check if the selected layer's centre aligns with canvas centre."""
        self._show_h_guide = False
        self._show_v_guide = False
        if self._sel_bbox is None or self._image_w == 0 or self._image_h == 0:
            return

        bx, by, bw, bh = self._sel_bbox
        dx, dy = self._move_accumulated

        # Centre of the layer after the pending move
        layer_cx = bx + bw / 2 + dx
        layer_cy = by + bh / 2 + dy
        canvas_cx = self._image_w / 2
        canvas_cy = self._image_h / 2

        if abs(layer_cx - canvas_cx) < _ALIGN_SNAP_PX:
            self._show_v_guide = True
        if abs(layer_cy - canvas_cy) < _ALIGN_SNAP_PX:
            self._show_h_guide = True

    def _set_zoom(self, new_zoom: float) -> None:
        self._zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, new_zoom))
        self._fit_pixmap()

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

        # Apply zoom factor
        zoomed_w = int(target_w * self._zoom)
        zoomed_h = int(target_h * self._zoom)

        scaled = self._current_pixmap.scaled(
            QSize(zoomed_w, zoomed_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Create a canvas at the original target size for panning
        canvas_w = max(1, self.width())
        canvas_h = max(1, self.height())
        canvas = QPixmap(canvas_w, canvas_h)
        canvas.fill(QColor(BG))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Center the image + apply pan offset
        x = int((canvas_w - scaled.width()) / 2 + self._pan_offset.x())
        y = int((canvas_h - scaled.height()) / 2 + self._pan_offset.y())
        painter.drawPixmap(x, y, scaled)

        # Store the rendered image rect for coordinate mapping
        self._img_rect_x = x
        self._img_rect_y = y
        self._img_rect_w = scaled.width()
        self._img_rect_h = scaled.height()

        # Draw selection bounding box
        if self._sel_bbox is not None and self._selected_layer >= 0:
            bx, by, bw, bh = self._sel_bbox
            dx, dy = self._move_accumulated if self._moving_layer else (0, 0)

            sx1, sy1 = self._image_to_screen(bx + dx, by + dy)
            sx2, sy2 = self._image_to_screen(bx + bw + dx, by + bh + dy)

            sel_pen = QPen(QColor(ACCENT), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(sx1, sy1, sx2 - sx1, sy2 - sy1))

            # Corner handles
            handle_size = 5
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(ACCENT))
            for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                painter.drawRect(QRectF(
                    hx - handle_size / 2, hy - handle_size / 2,
                    handle_size, handle_size
                ))

        # Draw alignment guides during move
        if self._moving_layer:
            guide_pen = QPen(QColor("#f59e0b"), 1.0, Qt.PenStyle.DashDotLine)
            painter.setPen(guide_pen)

            if self._show_v_guide:
                # Vertical centre line
                sx, _ = self._image_to_screen(self._image_w / 2, 0)
                painter.drawLine(
                    int(sx), self._img_rect_y,
                    int(sx), self._img_rect_y + self._img_rect_h,
                )

            if self._show_h_guide:
                # Horizontal centre line
                _, sy = self._image_to_screen(0, self._image_h / 2)
                painter.drawLine(
                    self._img_rect_x, int(sy),
                    self._img_rect_x + self._img_rect_w, int(sy),
                )

        # Draw zoom indicator
        if abs(self._zoom - 1.0) > 0.01:
            zoom_text = f"{self._zoom * 100:.0f}%"
            painter.setPen(QColor(TEXT_SEC))
            font = painter.font()
            font.setPointSize(10)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.drawText(
                canvas_w - 70, canvas_h - 10, zoom_text
            )

        painter.end()

        self._image_label.setPixmap(canvas)
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
