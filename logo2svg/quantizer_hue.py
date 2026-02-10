"""Color quantization via hue-based grouping for solid-color logos.

Instead of treating all color variations as distinct, this approach:
1. Groups colors by their hue (all greys together, all reds together, etc.)
2. Within each hue group, uses a single representative color
3. Naturally collapses anti-aliasing artifacts for clean solid-color extraction
"""

from __future__ import annotations

import cv2
import numpy as np


def quantize_colors(
    image: np.ndarray,
    fg_mask: np.ndarray,
    n_colors: int | None = None,
    max_k: int = 6,
    sample_limit: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize foreground pixels by grouping colors into hue families.

    For solid-color logos, this groups colors by hue (ignoring saturation/value).
    All greys merge together, all reds merge together, etc. This naturally
    collapses anti-aliasing artifacts that would otherwise create fake color variants.

    Args:
        image: (H, W, 3) uint8 RGB image.
        fg_mask: (H, W) bool foreground mask.
        n_colors: Target number of color clusters. If None, uses auto-detection.
        max_k: Maximum number of color clusters.
        sample_limit: Unused in this implementation.

    Returns:
        labels: (H, W) int32 array. -1 for background, 0..k-1 for clusters.
        centers_rgb: (k, 3) uint8 array of cluster center colors in RGB.
    """
    h, w = image.shape[:2]

    # Convert to HSV for hue-based grouping
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 0] = image_hsv[:, :, 0] / 180.0  # Normalize H to [0, 1]

    # Extract foreground pixels
    fg_pixels_hsv = image_hsv[fg_mask]
    fg_pixels_rgb = image[fg_mask]
    n_fg = len(fg_pixels_hsv)

    if n_fg == 0:
        raise ValueError("No foreground pixels found. Check background detection settings.")

    # Group by hue: quantize hue into bins (achromatic/grey share bin 0)
    hue_bins = 16  # Number of hue bins (16 * 22.5° = 360°)
    hue_values = fg_pixels_hsv[:, 0]
    saturation_values = fg_pixels_hsv[:, 1]
    
    # All near-grey colors (low saturation) go to bin 0, regardless of hue
    hue_groups = np.where(
        saturation_values < 0.1,
        0,  # Grey bin
        (hue_values * hue_bins).astype(int) + 1  # Hue-based bin (offset by 1 to avoid collision with grey bin 0)
    )
    
    # Find unique hue groups and their representative RGB colors
    unique_groups = np.unique(hue_groups)
    group_colors_rgb = []
    group_id_map = {}  # Map from original hue group ID to output label index
    
    for output_idx, group_id in enumerate(unique_groups):
        mask = hue_groups == group_id
        group_pixels_rgb = fg_pixels_rgb[mask]
        # Use median color for this hue group (robust to outliers)
        representative_color = np.median(group_pixels_rgb, axis=0).astype(np.uint8)
        group_colors_rgb.append(representative_color)
        group_id_map[int(group_id)] = output_idx
    
    group_colors_rgb = np.array(group_colors_rgb, dtype=np.uint8)
    
    # If we have too many hue groups, merge the closest ones
    target_k = n_colors if n_colors is not None else min(len(group_colors_rgb), max_k)
    
    if len(group_colors_rgb) > target_k:
        # Merge groups by RGB distance until we hit target
        group_colors_rgb, merge_mapping = _merge_close_colors_rgb(group_colors_rgb, target_k)
        # Apply merge mapping to group_id_map
        new_group_id_map = {}
        for old_gid, old_idx in group_id_map.items():
            new_idx = merge_mapping[old_idx]
            new_group_id_map[old_gid] = new_idx
        group_id_map = new_group_id_map
    
    # Build full label image
    labels = np.full((h, w), -1, dtype=np.int32)
    # Map each foreground pixel's hue group to its output label
    for i, fg_idx in enumerate(np.where(fg_mask.flat)[0]):
        hue_group_id = hue_groups[i]
        output_label = group_id_map[int(hue_group_id)]
        labels.flat[fg_idx] = output_label
    
    return labels, group_colors_rgb


def _merge_close_colors_rgb(colors_rgb: np.ndarray, target_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge closest pairs of RGB colors until we have target_k colors.
    
    Returns:
        merged_colors: The final color array after merging.
        merge_mapping: Array where merge_mapping[i] = j means color i now maps to j.
    """
    n = len(colors_rgb)
    mapping = np.arange(n, dtype=int)
    
    while len(np.unique(mapping)) > target_k:
        # Find closest pair of distinct color groups
        unique_ids = np.unique(mapping)
        min_dist = np.inf
        merge_from, merge_to = None, None
        
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                id_i, id_j = unique_ids[i], unique_ids[j]
                # Average color of each group
                color_i = np.mean(colors_rgb[mapping == id_i], axis=0)
                color_j = np.mean(colors_rgb[mapping == id_j], axis=0)
                dist = np.linalg.norm(color_i - color_j)
                if dist < min_dist:
                    min_dist = dist
                    merge_from, merge_to = id_i, id_j
        
        if merge_from is not None:
            mapping[mapping == merge_from] = merge_to
    
    # Renumber mapping to 0..target_k-1
    unique_mapped = np.unique(mapping)
    final_mapping = np.zeros(n, dtype=int)
    for new_idx, old_id in enumerate(unique_mapped):
        final_mapping[mapping == old_id] = new_idx
    
    # Compute final color centers
    final_colors = []
    for idx in range(len(unique_mapped)):
        color = np.mean(colors_rgb[final_mapping == idx], axis=0).astype(np.uint8)
        final_colors.append(color)
    
    return np.array(final_colors), final_mapping
