"""Image loading, background detection, and fringe pixel removal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .color_utils import hex_to_rgb


def load_image(
    path: Path, bg_color_override: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load a PNG file and separate foreground from background.

    Background detection priority:
    1. If --bg-color is given, treat that color (with tolerance) as background.
    2. If the image has an alpha channel, treat transparent pixels as background.
    3. Otherwise, sample corner pixels and use the dominant corner color.

    Args:
        path: Path to the PNG file.
        bg_color_override: Optional hex color string (e.g. '#FFFFFF') to treat
            as background.

    Returns:
        image: (H, W, 3) uint8 RGB array.
        fg_mask: (H, W) bool array, True for foreground pixels.
    """
    pil_image = Image.open(path)

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
    bg_mask = np.zeros((h, w), dtype=bool)
    for label_id in border_labels:
        bg_mask |= (label_img == label_id)

    return ~bg_mask


def _remove_fringe_pixels(fg_mask: np.ndarray, fringe_width: int = 1) -> np.ndarray:
    """Erode the foreground mask to strip anti-aliased fringe at background boundary.

    Removes the outermost ring of pixels where the logo blends into the
    background. This prevents blended-color pixels from being included in
    the color quantization.
    """
    mask_uint8 = fg_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=fringe_width)
    return eroded.astype(bool)
