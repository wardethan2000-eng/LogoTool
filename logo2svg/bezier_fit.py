"""Cubic Bezier curve generation via Catmull-Rom spline interpolation.

Converts a sequence of 2D points into smooth cubic Bezier curves by treating
the points as a Catmull-Rom spline and converting each segment to its cubic
Bezier equivalent.  This approach is unconditionally stable — control points
are always bounded near the data points, with no iterative fitting that can
diverge.

For a Catmull-Rom segment through P_{i-1}, P_i, P_{i+1}, P_{i+2}, the
equivalent cubic Bezier from P_i to P_{i+1} has control points:

    B0 = P_i
    B1 = P_i   + (P_{i+1} - P_{i-1}) / 6
    B2 = P_{i+1} - (P_{i+2} - P_i)   / 6
    B3 = P_{i+1}

This gives C1-continuous curves that pass exactly through every input point.
"""

from __future__ import annotations

import numpy as np


def fit_curve(points: np.ndarray, max_error: float = 2.0) -> list[np.ndarray]:
    """Fit cubic Bezier curves to an open polyline via Catmull-Rom.

    Args:
        points: (N, 2) array of 2D points.
        max_error: Unused (kept for API compatibility). Smoothness is
                   controlled by the density of input points.

    Returns:
        List of (4, 2) arrays, each a cubic Bezier segment
        [start, ctrl1, ctrl2, end].
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        return []
    if n == 2:
        return [_two_point_bezier(points[0], points[1])]

    segments: list[np.ndarray] = []

    for i in range(n - 1):
        # For endpoints, reflect the neighboring point to create a virtual
        # Catmull-Rom neighbor so the curve starts/ends with the right slope.
        if i == 0:
            p_prev = 2.0 * points[0] - points[1]
        else:
            p_prev = points[i - 1]

        p_curr = points[i]
        p_next = points[i + 1]

        if i + 2 >= n:
            p_next2 = 2.0 * points[-1] - points[-2]
        else:
            p_next2 = points[i + 2]

        cp1 = p_curr + (p_next - p_prev) / 6.0
        cp2 = p_next - (p_next2 - p_curr) / 6.0

        segments.append(np.array([p_curr, cp1, cp2, p_next]))

    return segments


def fit_curve_closed(points: np.ndarray, max_error: float = 2.0) -> list[np.ndarray]:
    """Fit cubic Bezier curves to a closed contour via Catmull-Rom.

    Each input point becomes a knot that the curve passes through exactly.
    The contour wraps around so there is no seam artefact.

    Args:
        points: (N, 2) array of 2D points forming a closed contour.
                The first and last points should NOT be duplicated.
        max_error: Unused (kept for API compatibility).

    Returns:
        List of (4, 2) arrays, each a cubic Bezier segment.
        The last segment's endpoint equals the first segment's start point.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        return []
    if n == 2:
        return [
            _two_point_bezier(points[0], points[1]),
            _two_point_bezier(points[1], points[0]),
        ]
    if n == 3:
        # Still use Catmull-Rom — wrapping gives good results even for 3 pts.
        pass

    segments: list[np.ndarray] = []

    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]
        p_next2 = points[(i + 2) % n]

        cp1 = p_curr + (p_next - p_prev) / 6.0
        cp2 = p_next - (p_next2 - p_curr) / 6.0

        segments.append(np.array([p_curr, cp1, cp2, p_next]))

    return segments


def _two_point_bezier(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Create a cubic Bezier that represents a straight line between two points."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    # Place control points at 1/3 and 2/3 along the line
    return np.array([p0, p0 + (p1 - p0) / 3.0, p1 - (p1 - p0) / 3.0, p1])


# ---------------------------------------------------------------------------
# Keep _evaluate_bezier so existing test imports still work.
# ---------------------------------------------------------------------------

def _evaluate_bezier(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a cubic Bezier at parameter *t*."""
    p0, p1, p2, p3 = control_points
    s = 1.0 - t
    return s*s*s * p0 + 3*s*s*t * p1 + 3*s*t*t * p2 + t*t*t * p3
