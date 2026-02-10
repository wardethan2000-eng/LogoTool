"""Color quantization via K-means clustering in CIELAB color space."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def quantize_colors(
    image: np.ndarray,
    fg_mask: np.ndarray,
    n_colors: int | None = None,
    max_k: int = 8,
    sample_limit: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize foreground pixels into distinct color clusters.

    Operates in CIELAB color space for perceptually accurate grouping.
    If n_colors is None, auto-detects the optimal number of clusters
    using silhouette scoring.

    Args:
        image: (H, W, 3) uint8 RGB image.
        fg_mask: (H, W) bool foreground mask.
        n_colors: Fixed number of clusters, or None for auto-detection.
        max_k: Maximum k to try during auto-detection.
        sample_limit: Max pixels to subsample for silhouette scoring.

    Returns:
        labels: (H, W) int32 array. -1 for background, 0..k-1 for clusters.
        centers_rgb: (k, 3) uint8 array of cluster center colors in RGB.
    """
    h, w = image.shape[:2]

    # Convert to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Extract foreground pixels
    fg_pixels_lab = image_lab[fg_mask]
    n_fg = len(fg_pixels_lab)

    if n_fg == 0:
        raise ValueError("No foreground pixels found. Check background detection settings.")

    # Auto-detect number of colors if not specified
    if n_colors is None:
        n_colors = _auto_detect_k(fg_pixels_lab, max_k, sample_limit)

    # Clamp n_colors to valid range
    n_unique = len(np.unique(fg_pixels_lab.reshape(-1, 3), axis=0))
    n_colors = max(1, min(n_colors, n_unique, max_k))

    # Run K-means clustering
    # Use MiniBatchKMeans for large images for speed
    if n_fg > 100000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_colors, n_init=10, random_state=42, batch_size=10000
        )
    else:
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)

    cluster_labels = kmeans.fit_predict(fg_pixels_lab)
    centers_lab = kmeans.cluster_centers_

    # Build full label image (-1 for background)
    labels = np.full((h, w), -1, dtype=np.int32)
    labels[fg_mask] = cluster_labels

    # Reassign anti-aliased boundary pixels to the nearest cluster
    labels = _reassign_boundary_pixels(image_lab, labels, centers_lab, fg_mask)

    # Convert LAB cluster centers back to RGB
    centers_rgb = _lab_centers_to_rgb(centers_lab)

    return labels, centers_rgb


def _auto_detect_k(
    pixels_lab: np.ndarray, max_k: int, sample_limit: int
) -> int:
    """Auto-detect the number of distinct colours in a logo image.

    Uses a "start generous, merge similar" strategy that works much
    better for logos than plain silhouette scoring:

    1.  Cluster with a generous initial *k* (up to ``max_k``).
    2.  Discard negligibly small clusters (< 1 % of foreground).
    3.  Iteratively merge the two closest cluster centres until all
        remaining centres differ by at least ``MIN_DELTA_E`` in standard
        CIELAB space (≈ 15 ΔE, clearly distinguishable colours).
    """
    n = len(pixels_lab)

    # Pre-check: if color variance is very low, this is a single-color logo
    if n > 0 and np.all(np.std(pixels_lab, axis=0) < 5.0):
        return 1

    # Subsample if too many pixels
    if n > sample_limit:
        indices = np.random.RandomState(42).choice(n, sample_limit, replace=False)
        sample = pixels_lab[indices]
    else:
        sample = pixels_lab

    # --- Step 1: generous initial clustering ---
    n_unique = len(np.unique(sample.reshape(-1, 3), axis=0))
    initial_k = min(max_k, n_unique, max(8, max_k))
    initial_k = max(2, min(initial_k, len(sample) // 10))

    kmeans = MiniBatchKMeans(
        n_clusters=initial_k, n_init=10, random_state=42, batch_size=5000
    )
    labels = kmeans.fit_predict(sample)
    centers = kmeans.cluster_centers_.copy()

    # --- Step 2: drop negligibly small clusters ---
    cluster_sizes = np.bincount(labels, minlength=initial_k)
    min_pixels = max(50, int(0.01 * len(sample)))
    keep = cluster_sizes >= min_pixels
    if np.sum(keep) >= 1:
        centers = centers[keep]
    if len(centers) <= 1:
        return max(1, len(centers))

    # --- Step 3: convert centres to standard CIELAB for perceptual distance ---
    std = centers.copy()
    std[:, 0] = std[:, 0] * (100.0 / 255.0)   # L: 0-255 → 0-100
    std[:, 1] = std[:, 1] - 128.0               # a: 0-255 → −128..127
    std[:, 2] = std[:, 2] - 128.0               # b: 0-255 → −128..127

    # --- Step 4: merge until all pairs are ≥ MIN_DELTA_E apart ---
    MIN_DELTA_E = 15.0

    while len(std) > 1:
        # Find the two closest centres
        min_dist = float("inf")
        merge_i, merge_j = 0, 1
        nc = len(std)
        for i in range(nc):
            for j in range(i + 1, nc):
                d = float(np.linalg.norm(std[i] - std[j]))
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j

        if min_dist >= MIN_DELTA_E:
            break  # all remaining centres are distinct enough

        # Merge: weighted average (approximate — just average here)
        std[merge_i] = (std[merge_i] + std[merge_j]) / 2.0
        std = np.delete(std, merge_j, axis=0)

    return max(1, len(std))


def _reassign_boundary_pixels(
    image_lab: np.ndarray,
    labels: np.ndarray,
    centers_lab: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Reassign anti-aliased boundary pixels to their nearest LAB cluster center.

    Boundary pixels are foreground pixels whose 8-connected neighborhood
    contains a pixel with a different label. These tend to be blended colors
    from anti-aliasing between two solid color regions.
    """
    h, w = labels.shape
    kernel = np.ones((3, 3), np.uint8)

    # Find boundary pixels: pixels where a neighbor has a different label
    # We detect this by dilating each cluster region and checking overlap
    boundary_mask = np.zeros((h, w), dtype=bool)

    unique_labels = [l for l in np.unique(labels) if l >= 0]

    for k in unique_labels:
        region = (labels == k).astype(np.uint8)
        dilated = cv2.dilate(region, kernel, iterations=1)
        # Pixels in dilated region that belong to a different foreground cluster
        # are at the boundary
        boundary_mask |= (dilated > 0) & (labels != k) & fg_mask

    if not np.any(boundary_mask):
        return labels

    # Reassign boundary pixels to nearest cluster center in LAB space
    boundary_pixels_lab = image_lab[boundary_mask]
    # Compute distances to each cluster center
    distances = np.linalg.norm(
        boundary_pixels_lab[:, np.newaxis, :] - centers_lab[np.newaxis, :, :],
        axis=2,
    )
    new_labels = distances.argmin(axis=1).astype(np.int32)

    labels = labels.copy()
    labels[boundary_mask] = new_labels

    return labels


def _lab_centers_to_rgb(centers_lab: np.ndarray) -> np.ndarray:
    """Convert LAB cluster centers to RGB uint8 values.

    Keeps the data in floating-point through the LAB→RGB conversion to
    avoid quantisation error from premature uint8 rounding.  The input
    *centers_lab* values are in the OpenCV uint8 LAB range (L: 0-255,
    a/b: 0-255) because they originate from ``cv2.cvtColor`` on a uint8
    image.  We convert to the float LAB range (L: 0-100, a/b: −128..127)
    before calling ``cv2.cvtColor`` with float32 input, then clip and
    cast the final RGB result to uint8.
    """
    # Convert from uint8 LAB range to float LAB range
    centers = centers_lab.astype(np.float32).reshape(1, -1, 3)
    centers[:, :, 0] = centers[:, :, 0] * (100.0 / 255.0)   # L: 0-255 → 0-100
    centers[:, :, 1] = centers[:, :, 1] - 128.0               # a: 0-255 → -128..127
    centers[:, :, 2] = centers[:, :, 2] - 128.0               # b: 0-255 → -128..127

    # cv2.cvtColor with float32 LAB input returns float32 RGB in [0, 1]
    rgb_float = cv2.cvtColor(centers, cv2.COLOR_Lab2RGB)

    # Convert [0, 1] float RGB to uint8
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8).reshape(-1, 3)
