"""Contour tracing: OpenCV contours -> bezier curves -> SVG path strings."""

from __future__ import annotations

import cv2
import numpy as np

from . import bezier_fit


def find_contours(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Find contours in a binary mask using 2-level hierarchy.

    Uses RETR_CCOMP which gives outer contours and their direct holes.

    Args:
        mask: (H, W) uint8 binary mask with values 0 or 255.

    Returns:
        contours: List of OpenCV contour arrays, each (N, 1, 2).
        hierarchy: (1, N, 4) array with [Next, Prev, FirstChild, Parent],
                   or None if no contours found.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours), hierarchy


def trace_to_svg_paths(
    contours: list[np.ndarray],
    hierarchy: np.ndarray | None,
    tolerance: float = 2.0,
    simplify: float | None = None,
) -> list[str]:
    """Convert OpenCV contours to SVG path 'd' attribute strings.

    Groups outer contours with their holes and produces compound paths
    using fill-rule="evenodd" for correct hole rendering.

    Args:
        contours: List of OpenCV contour arrays.
        hierarchy: Hierarchy from findContours, or None.
        tolerance: Max bezier fitting error in pixels.
        simplify: Optional simplification factor (0.0-1.0) for approxPolyDP.

    Returns:
        List of SVG path 'd' strings. Each may contain multiple sub-paths
        for shapes with holes.
    """
    if not contours or hierarchy is None:
        return []

    groups = _group_contours_by_hierarchy(contours, hierarchy)
    svg_paths = []

    for outer_idx, hole_indices in groups:
        parts = []

        # Outer contour
        outer_path = _contour_to_svg_subpath(contours[outer_idx], tolerance, simplify)
        if outer_path:
            parts.append(outer_path)

        # Hole contours
        for hole_idx in hole_indices:
            hole_path = _contour_to_svg_subpath(contours[hole_idx], tolerance, simplify)
            if hole_path:
                parts.append(hole_path)

        if parts:
            svg_paths.append(" ".join(parts))

    return svg_paths


def _group_contours_by_hierarchy(
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
) -> list[tuple[int, list[int]]]:
    """Group contours into (outer_index, [hole_indices]) using RETR_CCOMP hierarchy.

    In RETR_CCOMP hierarchy:
    - Outer contours have parent == -1 (hierarchy[0][i][3] == -1)
    - Holes have parent pointing to their outer contour
    """
    h = hierarchy[0]  # Shape: (N, 4)
    groups = []

    for i in range(len(contours)):
        if h[i][3] == -1:  # No parent => outer contour
            holes = []
            child = h[i][2]  # First child
            while child != -1:
                holes.append(child)
                child = h[child][0]  # Next sibling
            groups.append((i, holes))

    return groups


def _contour_to_svg_subpath(
    contour: np.ndarray,
    tolerance: float,
    simplify: float | None,
) -> str:
    """Convert a single OpenCV contour to an SVG sub-path string.

    Steps:
    1. Extract (x, y) points from the contour's (N, 1, 2) shape
    2. Apply approxPolyDP for initial point reduction
    3. Fit cubic bezier curves for smooth output
    4. Build SVG path string: 'M x y C cx1 cy1 cx2 cy2 x y ... Z'

    Falls back to straight-line segments (L commands) for very small contours.
    """
    # Extract points from OpenCV's (N, 1, 2) format
    points = contour.reshape(-1, 2).astype(np.float64)

    if len(points) < 3:
        # Too few points for bezier fitting — use straight lines
        return _points_to_line_path(points)

    # Simplify with approxPolyDP
    epsilon = _compute_epsilon(contour, simplify)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    points = approx.reshape(-1, 2).astype(np.float64)

    if len(points) < 3:
        return _points_to_line_path(points)

    # Fit cubic bezier curves to the closed contour
    try:
        segments = bezier_fit.fit_curve_closed(points, max_error=tolerance)
    except Exception:
        # Fallback to line segments if bezier fitting fails
        return _points_to_line_path(points)

    if not segments:
        return _points_to_line_path(points)

    return _beziers_to_svg_subpath(segments)


def _compute_epsilon(contour: np.ndarray, simplify: float | None) -> float:
    """Compute the epsilon value for approxPolyDP.

    Default: 0.1% of contour perimeter (gentle smoothing).
    With --simplify: scaled up for more aggressive point reduction.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    if simplify is not None:
        return simplify * 0.01 * perimeter
    return 0.001 * perimeter


def _beziers_to_svg_subpath(segments: list[np.ndarray]) -> str:
    """Convert cubic bezier segments to an SVG sub-path string.

    Each segment is a (4, 2) array: [start, ctrl1, ctrl2, end].
    Consecutive segments share endpoints (end of one = start of next).
    """
    if not segments:
        return ""

    parts = []
    # Move to the start of the first segment
    p0 = segments[0][0]
    parts.append(f"M {p0[0]:.1f} {p0[1]:.1f}")

    for seg in segments:
        cp1 = seg[1]
        cp2 = seg[2]
        end = seg[3]
        parts.append(
            f"C {cp1[0]:.1f} {cp1[1]:.1f} {cp2[0]:.1f} {cp2[1]:.1f} "
            f"{end[0]:.1f} {end[1]:.1f}"
        )

    parts.append("Z")
    return " ".join(parts)


def _points_to_line_path(points: np.ndarray) -> str:
    """Convert points to a simple SVG path using line segments (M/L/Z)."""
    if len(points) < 2:
        return ""

    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for pt in points[1:]:
        parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")
    parts.append("Z")
    return " ".join(parts)
