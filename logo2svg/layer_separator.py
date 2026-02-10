"""Per-color binary mask generation with morphological cleanup."""

from __future__ import annotations

import cv2
import numpy as np

from .color_utils import nearest_color_name, rgb_to_hex


def separate_layers(
    labels: np.ndarray,
    centers_rgb: np.ndarray,
    fg_mask: np.ndarray,
    min_area: int,
) -> list[dict]:
    """Create one layer dict per unique foreground cluster.

    For each cluster:
    1. Extract binary mask (label == k) & fg_mask
    2. Morphological close then open to clean up
    3. Remove connected components smaller than min_area

    Args:
        labels: (H, W) int32 label array, -1 for background.
        centers_rgb: (k, 3) uint8 cluster centers in RGB.
        fg_mask: (H, W) bool foreground mask.
        min_area: Minimum component area in pixels to retain.

    Returns:
        List of dicts with keys: 'rgb', 'hex_color', 'color_name', 'mask'.
        Mask is (H, W) uint8 with values 0 or 255.
    """
    layers = []
    n_clusters = len(centers_rgb)

    for k in range(n_clusters):
        # Binary mask for this cluster
        mask = ((labels == k) & fg_mask).astype(np.uint8) * 255

        # Morphological cleanup
        mask = _morphological_cleanup(mask)

        # Remove small connected components
        mask = _filter_small_components(mask, min_area)

        # Skip empty layers
        if np.count_nonzero(mask) == 0:
            continue

        r, g, b = int(centers_rgb[k, 0]), int(centers_rgb[k, 1]), int(centers_rgb[k, 2])
        hex_color = rgb_to_hex(r, g, b)
        color_name = nearest_color_name(r, g, b)

        layers.append({
            "rgb": (r, g, b),
            "hex_color": hex_color,
            "color_name": color_name,
            "mask": mask,
        })

    return layers


def _morphological_cleanup(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological close (fill tiny holes) then open (remove speckles).

    Uses a 5x5 kernel by default for effective cleanup at typical logo resolutions.
    Close uses 2 iterations to better fill gaps at color boundaries.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Close: dilate then erode — fills small gaps within the mask
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Open: erode then dilate — removes small noise specks
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    if min_area <= 0:
        return mask

    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Label 0 is the background in connectedComponents output
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[label_img == i] = 255

    return result
