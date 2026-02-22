"""Color quantization via K-means clustering in CIELAB color space."""

from __future__ import annotations

import cv2
import numpy as np


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

    # Run K-means clustering in LAB space.
    cluster_labels, centers_lab = _kmeans_lab(
        fg_pixels_lab,
        n_clusters=n_colors,
        attempts=10,
    )

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
    """Try k=2..max_k and return k with the highest silhouette score.

    Uses subsampling for performance since silhouette scoring is O(n^2).
    Returns 1 if the foreground is essentially a single color.
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

    if len(sample) < 2:
        return 1

    best_k = 2
    best_score = -1.0

    for k in range(2, min(max_k + 1, len(sample))):
        cluster_labels, _ = _kmeans_lab(sample, n_clusters=k, attempts=5)

        # Need at least 2 distinct labels for silhouette score.
        if len(np.unique(cluster_labels)) < 2:
            continue

        score = _silhouette_score_approx(
            sample,
            cluster_labels,
            max_points=1200,
        )

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _kmeans_lab(
    pixels_lab: np.ndarray,
    *,
    n_clusters: int,
    attempts: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster LAB pixels with OpenCV K-means.

    Returns integer labels and LAB centers in float64.
    """
    data = np.asarray(pixels_lab, dtype=np.float32)
    if len(data) == 0:
        raise ValueError("Cannot cluster an empty pixel set.")

    n_clusters = max(1, min(int(n_clusters), len(data)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.2,
    )

    cv2.setRNGSeed(42)
    _compactness, labels, centers = cv2.kmeans(
        data,
        n_clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1).astype(np.int32)
    centers = centers.astype(np.float64)
    return labels, centers


def _silhouette_score_approx(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 1200,
) -> float:
    """Approximate silhouette score using bounded random subsampling."""
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")

    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if len(points) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        labels = labels[idx]

    # Pairwise Euclidean distances.
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    silhouettes = np.zeros(len(points), dtype=np.float64)
    for i in range(len(points)):
        this_label = labels[i]

        same = labels == this_label
        same_count = int(np.count_nonzero(same))
        if same_count <= 1:
            silhouettes[i] = 0.0
            continue

        # Intra-cluster distance (exclude self).
        a = (np.sum(dmat[i, same]) - 0.0) / (same_count - 1)

        # Nearest other-cluster mean distance.
        b = np.inf
        for other in unique:
            if other == this_label:
                continue
            mask = labels == other
            if not np.any(mask):
                continue
            dist = float(np.mean(dmat[i, mask]))
            if dist < b:
                b = dist

        denom = max(a, b)
        silhouettes[i] = 0.0 if denom <= 1e-12 else (b - a) / denom

    return float(np.mean(silhouettes))


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
