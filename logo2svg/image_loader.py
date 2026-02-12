"""Image loading, background detection, and fringe pixel removal."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color_utils import hex_to_rgb

logger = logging.getLogger(__name__)


def load_image(
    path: Path, bg_color_override: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image file and separate foreground from background.

    Handles PNG, JPEG (with EXIF rotation and CMYK conversion), WebP, BMP,
    and SVG (rasterized at its native resolution or 1024px default).

    Background detection priority:
    1. If --bg-color is given, treat that color (with tolerance) as background.
    2. If the image has an alpha channel, treat transparent pixels as background.
    3. Otherwise, sample corner pixels and use the dominant corner color.

    Args:
        path: Path to the image file.
        bg_color_override: Optional hex color string (e.g. '#FFFFFF') to treat
            as background.

    Returns:
        image: (H, W, 3) uint8 RGB array.
        fg_mask: (H, W) bool array, True for foreground pixels.
    """
    # SVG primary input: rasterize to a bitmap, then process as RGBA
    if path.suffix.lower() == ".svg":
        return _load_svg_as_image(path, bg_color_override)

    pil_image = Image.open(path)

    # EXIF auto-rotation: honour EXIF Orientation tag (common in phone JPEGs).
    # This rotates the pixel data and strips the tag so downstream code
    # sees the image the way the user expects.
    pil_image = ImageOps.exif_transpose(pil_image)

    # CMYK handling: convert to RGB.  CMYK JPEGs are sometimes produced
    # by professional design tools — Pillow handles the conversion but
    # we log a note since the colour mapping may not be perfect without
    # an embedded ICC profile.
    if pil_image.mode == "CMYK":
        logger.info("CMYK image detected — converting to RGB")
        pil_image = pil_image.convert("RGB")

    has_alpha = pil_image.mode in ("RGBA", "LA", "PA")

    # Convert to RGBA to get alpha if present, then extract RGB
    if has_alpha:
        pil_rgba = pil_image.convert("RGBA")
        rgba_array = np.array(pil_rgba, dtype=np.uint8)
        image = rgba_array[:, :, :3]
        alpha = rgba_array[:, :, 3]
    else:
        pil_rgb = pil_image.convert("RGB")
        image = np.array(pil_rgb, dtype=np.uint8)
        alpha = None

    # Determine foreground mask
    if bg_color_override is not None:
        bg_rgb = np.array(hex_to_rgb(bg_color_override), dtype=np.uint8)
        fg_mask = _remove_bg_color(image, bg_rgb, tolerance=30)
    elif has_alpha:
        fg_mask = _detect_bg_from_alpha(alpha, threshold=10)
        # If the alpha channel is useless (all opaque or all transparent),
        # fall through to corner-based detection instead.
        fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        if fg_ratio > 0.99 or fg_ratio < 0.01:
            bg_rgb = _detect_bg_from_corners(image)
            if bg_rgb is not None:
                fg_mask = _remove_bg_color(image, bg_rgb, tolerance=30)
            elif fg_ratio > 0.99:
                # Alpha says everything is foreground and corners disagree —
                # keep the all-foreground mask as last resort.
                pass
    else:
        bg_rgb = _detect_bg_from_corners(image)
        if bg_rgb is not None:
            fg_mask = _remove_bg_color(image, bg_rgb, tolerance=30)
        else:
            # No clear background detected — treat everything as foreground
            fg_mask = np.ones(image.shape[:2], dtype=bool)

    return image, fg_mask


def _detect_bg_from_alpha(alpha: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Return foreground mask from alpha channel. Transparent = background."""
    return alpha > threshold


def _detect_bg_from_corners(
    image: np.ndarray, sample_size: int = 10
) -> np.ndarray | None:
    """Sample corner regions and return background color if corners agree.

    Samples pixels from all four corners of the image and checks if they
    share a common color. If >60% of sampled pixels are similar, that color
    is the background.

    Returns:
        Background RGB color as (3,) uint8 array, or None if corners disagree.
    """
    h, w = image.shape[:2]
    s = min(sample_size, h // 4, w // 4)
    if s < 1:
        return None

    # Gather corner pixels
    corners = np.concatenate([
        image[:s, :s].reshape(-1, 3),       # top-left
        image[:s, -s:].reshape(-1, 3),      # top-right
        image[-s:, :s].reshape(-1, 3),      # bottom-left
        image[-s:, -s:].reshape(-1, 3),     # bottom-right
    ])

    # Find the most common color among corner pixels via median
    median_color = np.median(corners, axis=0).astype(np.uint8)

    # Check what fraction of corner pixels are close to the median
    diffs = np.sqrt(np.sum((corners.astype(float) - median_color.astype(float)) ** 2, axis=1))
    close_fraction = np.mean(diffs < 30)

    if close_fraction > 0.6:
        return median_color
    return None


def _remove_bg_color(
    image: np.ndarray, bg_rgb: np.ndarray, tolerance: int = 30
) -> np.ndarray:
    """Create foreground mask by removing background regions connected to image edges.

    Uses region-based detection: only pixels that both match the background
    color AND belong to a connected component touching the image border are
    classified as background.  This preserves foreground elements that share
    the background color (e.g., white text inside a logo on a white background).
    """
    diff = np.sqrt(
        np.sum((image.astype(float) - bg_rgb.astype(float)) ** 2, axis=2)
    )
    potential_bg = (diff <= tolerance).astype(np.uint8)

    h, w = image.shape[:2]

    # Find connected components among pixels matching the background color
    num_labels, label_img = cv2.connectedComponents(potential_bg, connectivity=8)

    # Identify which components touch the image border
    border_labels = set()
    border_labels.update(label_img[0, :].tolist())       # top row
    border_labels.update(label_img[-1, :].tolist())      # bottom row
    border_labels.update(label_img[:, 0].tolist())       # left column
    border_labels.update(label_img[:, -1].tolist())      # right column
    border_labels.discard(0)  # label 0 = non-matching pixels in connectedComponents

    # Background = bg-colored pixels connected to the image border
    if border_labels:
        border_array = np.array(list(border_labels))
        bg_mask = np.isin(label_img, border_array)
    else:
        bg_mask = np.zeros((h, w), dtype=bool)

    return ~bg_mask


def _load_svg_as_image(
    path: Path, bg_color_override: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize an SVG file and return it as an RGB image with foreground mask.

    Uses the SVG's native dimensions scaled up to at least 1024px on the
    longest side, preserving aspect ratio.
    """
    from .svg_importer import rasterize_svg, _parse_svg_dimensions

    svg_w, svg_h = _parse_svg_dimensions(path)

    # Scale up to at least 1024px on the longest side for good detail
    min_size = 1024
    if max(svg_w, svg_h) < min_size:
        scale = min_size / max(svg_w, svg_h)
        target_w = int(svg_w * scale)
        target_h = int(svg_h * scale)
    else:
        target_w, target_h = svg_w, svg_h

    rgba = rasterize_svg(path, target_w, target_h)
    alpha = rgba[:, :, 3]
    image = rgba[:, :, :3]  # RGB

    if bg_color_override is not None:
        bg_rgb = np.array(hex_to_rgb(bg_color_override), dtype=np.uint8)
        fg_mask = _remove_bg_color(image, bg_rgb, tolerance=30)
    else:
        # Use alpha channel as foreground mask
        fg_mask = _detect_bg_from_alpha(alpha, threshold=10)
        fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        if fg_ratio > 0.99 or fg_ratio < 0.01:
            bg_rgb = _detect_bg_from_corners(image)
            if bg_rgb is not None:
                fg_mask = _remove_bg_color(image, bg_rgb, tolerance=30)
            elif fg_ratio > 0.99:
                pass

    return image, fg_mask

