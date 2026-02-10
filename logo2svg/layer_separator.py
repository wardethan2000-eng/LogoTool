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

        # Re-clamp to foreground: morph close can expand the mask beyond
        # the original foreground boundary, which would re-create background
        # rectangles in the SVG output.
        mask = mask & (fg_mask.astype(np.uint8) * 255)

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
            "cluster_idx": k,
        })

    # Resolve any pixel overlaps created by morphological expansion
    layers = _resolve_overlaps(layers, labels, fg_mask)

    return layers


def _morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological close to fill tiny holes in the mask.

    Uses only MORPH_CLOSE (dilate then erode) with a 3×3 kernel.  The
    previous close+open approach included an erode step (MORPH_OPEN) that
    destroyed thin features — eating 10-15 % of narrow strokes.  Speckle
    removal is handled separately by _filter_small_components.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Close: dilate then erode — fills small 1-2px gaps within the mask
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _resolve_overlaps(
    layers: list[dict],
    labels: np.ndarray,
    fg_mask: np.ndarray,
) -> list[dict]:
    """Ensure each pixel belongs to at most one layer.

    After morphological cleanup, layer masks can overlap.  For overlapping
    pixels, we keep the assignment that matches the original K-means label.
    This prevents one color from bleeding into another color's territory.
    """
    if len(layers) <= 1:
        return layers

    h, w = fg_mask.shape
    n_layers = len(layers)

    # Stack masks: (n_layers, h, w)
    mask_stack = np.stack([layer["mask"] > 0 for layer in layers], axis=0)

    # Coverage count per pixel
    coverage = mask_stack.sum(axis=0)

    # Find overlap pixels (coverage > 1)
    overlap_mask = coverage > 1

    if not np.any(overlap_mask):
        return layers

    # Build mapping: original cluster index → layer index
    cluster_to_layer = {}
    for li, layer in enumerate(layers):
        cluster_to_layer[layer["cluster_idx"]] = li

    # For overlapping pixels, resolve by original K-means label
    resolved = np.full((h, w), -1, dtype=np.int32)
    for cluster_idx, layer_idx in cluster_to_layer.items():
        resolved[labels == cluster_idx] = layer_idx

    # Remove overlapping pixels from layers that don't match the original label
    for li in range(n_layers):
        # Pixels where this layer's mask is set AND there's overlap
        overlap_in_layer = overlap_mask & mask_stack[li]
        # Remove pixel from this layer if the original label says a different layer
        should_remove = overlap_in_layer & (resolved != li)
        if np.any(should_remove):
            layers[li]["mask"][should_remove] = 0

    return layers


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
