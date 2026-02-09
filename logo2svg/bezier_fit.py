"""Cubic Bezier curve fitting via Philip J. Schneider's algorithm.

Adapted from the algorithm described in:
    "An Algorithm for Automatically Fitting Digitized Curves"
    from Graphics Gems (Academic Press, 1990).

Given a sequence of 2D points, fits one or more cubic Bezier curves such that
the maximum distance from any original point to the fitted curve is below
a specified error threshold.
"""

from __future__ import annotations

import numpy as np

# Maximum iterations for Newton-Raphson reparameterization
_MAX_ITERATIONS = 4


def fit_curve(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    """Fit cubic bezier curves to an open polyline.

    Args:
        points: (N, 2) array of 2D points.
        max_error: Maximum allowed fitting error in pixels.

    Returns:
        List of (4, 2) arrays, each a cubic bezier segment
        [start, ctrl1, ctrl2, end].
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return []
    if len(points) == 2:
        return [_two_point_bezier(points[0], points[1])]

    # Compute left and right endpoint tangents
    left_tangent = _normalize(points[1] - points[0])
    right_tangent = _normalize(points[-2] - points[-1])

    return _fit_cubic(points, left_tangent, right_tangent, max_error)


def fit_curve_closed(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    """Fit cubic bezier curves to a closed contour.

    Finds the sharpest corner in the contour and uses it as the split point
    so tangent computation at the seam is natural. If no sharp corner exists,
    uses an overlap strategy to avoid a kink at the closure.

    Args:
        points: (N, 2) array of 2D points forming a closed contour.
                The first and last points should NOT be the same.
        max_error: Maximum allowed fitting error in pixels.

    Returns:
        List of (4, 2) arrays, each a cubic bezier segment.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)

    if n < 2:
        return []
    if n == 2:
        return [_two_point_bezier(points[0], points[1]),
                _two_point_bezier(points[1], points[0])]

    # Find the sharpest corner to use as split point
    split_idx = _find_sharpest_corner(points)

    # Rotate points so the split point is at the start/end
    rotated = np.roll(points, -split_idx, axis=0)

    # Close the polyline by appending the first point
    closed = np.vstack([rotated, rotated[0:1]])

    # Compute tangents at the split point
    # Use the vectors from the corner to its neighbors
    left_tangent = _normalize(closed[1] - closed[0])
    right_tangent = _normalize(closed[-2] - closed[-1])

    return _fit_cubic(closed, left_tangent, right_tangent, max_error)


def _find_sharpest_corner(points: np.ndarray) -> int:
    """Find the index of the sharpest corner (smallest angle) in a closed polygon."""
    n = len(points)
    min_cos = 2.0  # cos(angle), smaller = sharper corner
    best_idx = 0

    for i in range(n):
        prev_pt = points[(i - 1) % n]
        curr_pt = points[i]
        next_pt = points[(i + 1) % n]

        v1 = _normalize(prev_pt - curr_pt)
        v2 = _normalize(next_pt - curr_pt)

        cos_angle = np.dot(v1, v2)
        if cos_angle < min_cos:
            min_cos = cos_angle
            best_idx = i

    return best_idx


def _fit_cubic(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    error: float,
) -> list[np.ndarray]:
    """Recursively fit cubic bezier(s) to a sequence of points.

    If a single bezier doesn't fit within the error threshold, the point
    sequence is split at the point of maximum error and each half is
    fitted recursively.
    """
    n = len(points)

    if n == 2:
        return [_two_point_bezier(points[0], points[1])]

    # Parameterize points by chord length
    u = _chord_length_parameterize(points)

    # Generate a bezier curve from the parameterization
    bezier = _generate_bezier(points, u, left_tangent, right_tangent)
    max_err, split_point = _compute_max_error(points, bezier, u)

    if max_err < error:
        return [bezier]

    # Try reparameterization if error is not too large
    if max_err < error * error:
        for _ in range(_MAX_ITERATIONS):
            u_prime = _reparameterize(bezier, points, u)
            bezier = _generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_err, split_point = _compute_max_error(points, bezier, u_prime)
            if max_err < error:
                return [bezier]
            u = u_prime

    # Split at point of max error and fit each half
    # Ensure split point is not at the endpoints
    split_point = max(1, min(split_point, n - 2))

    center_tangent = _normalize(points[split_point + 1] - points[split_point - 1])

    left_segments = _fit_cubic(
        points[: split_point + 1], left_tangent, -center_tangent, error
    )
    right_segments = _fit_cubic(
        points[split_point:], center_tangent, right_tangent, error
    )

    return left_segments + right_segments


def _generate_bezier(
    points: np.ndarray,
    parameters: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> np.ndarray:
    """Compute optimal control points for a single cubic bezier via least-squares.

    Returns a (4, 2) array: [P0, P1, P2, P3].
    """
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    # Compute the A matrix: A[i] = [B1(u[i]) * left_tangent, B2(u[i]) * right_tangent]
    a = np.zeros((n, 2, 2))
    for i, u in enumerate(parameters):
        a[i, 0] = left_tangent * _b1(u)
        a[i, 1] = right_tangent * _b2(u)

    # Build the C and X matrices for the least-squares system
    c = np.zeros((2, 2))
    x = np.zeros(2)

    for i in range(n):
        c[0, 0] += np.dot(a[i, 0], a[i, 0])
        c[0, 1] += np.dot(a[i, 0], a[i, 1])
        c[1, 0] = c[0, 1]
        c[1, 1] += np.dot(a[i, 1], a[i, 1])

        u = parameters[i]
        tmp = points[i] - (
            p0 * _b0(u) + p0 * _b1(u) + p3 * _b2(u) + p3 * _b3(u)
        )
        x[0] += np.dot(a[i, 0], tmp)
        x[1] += np.dot(a[i, 1], tmp)

    # Solve for alpha_l and alpha_r (control point distances along tangents)
    det = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    if abs(det) < 1e-12:
        # Degenerate case: use heuristic distances
        dist = np.linalg.norm(p3 - p0) / 3.0
        return np.array([p0, p0 + left_tangent * dist, p3 + right_tangent * dist, p3])

    alpha_l = (x[0] * c[1, 1] - x[1] * c[0, 1]) / det
    alpha_r = (c[0, 0] * x[1] - c[1, 0] * x[0]) / det

    # If alphas are negative or zero, use heuristic
    seg_length = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_length

    if alpha_l < epsilon or alpha_r < epsilon:
        dist = seg_length / 3.0
        return np.array([p0, p0 + left_tangent * dist, p3 + right_tangent * dist, p3])

    return np.array([
        p0,
        p0 + left_tangent * alpha_l,
        p3 + right_tangent * alpha_r,
        p3,
    ])


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values [0, 1] based on cumulative chord length."""
    diffs = np.diff(points, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = cumulative[-1]
    if total < 1e-12:
        return np.linspace(0, 1, len(points))
    return cumulative / total


def _reparameterize(
    bezier_pts: np.ndarray, points: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Newton-Raphson refinement of parameter values."""
    new_params = np.copy(parameters)
    for i, (point, u) in enumerate(zip(points, parameters)):
        # Q(u)
        q_u = _evaluate_bezier(bezier_pts, u)
        # Q'(u)
        q1_u = _evaluate_bezier_derivative(bezier_pts, u)
        # Q''(u)
        q2_u = _evaluate_bezier_second_derivative(bezier_pts, u)

        # Newton step: u' = u - f(u)/f'(u)
        # where f(u) = (Q(u) - P) . Q'(u)
        # and f'(u) = Q'(u) . Q'(u) + (Q(u) - P) . Q''(u)
        diff = q_u - point
        numerator = np.dot(diff, q1_u)
        denominator = np.dot(q1_u, q1_u) + np.dot(diff, q2_u)

        if abs(denominator) > 1e-12:
            new_params[i] = u - numerator / denominator

    # Clamp to [0, 1]
    new_params = np.clip(new_params, 0.0, 1.0)
    return new_params


def _compute_max_error(
    points: np.ndarray, bezier_pts: np.ndarray, parameters: np.ndarray
) -> tuple[float, int]:
    """Find the maximum distance from any point to the bezier curve.

    Returns (max_error, index_of_max_error_point).
    """
    max_err = 0.0
    split_point = len(points) // 2

    for i, (point, u) in enumerate(zip(points, parameters)):
        q_u = _evaluate_bezier(bezier_pts, u)
        dist_sq = np.sum((point - q_u) ** 2)
        if dist_sq > max_err:
            max_err = dist_sq
            split_point = i

    return max_err, split_point


def _evaluate_bezier(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a cubic bezier at parameter t using the direct formula."""
    p0, p1, p2, p3 = control_points
    s = 1.0 - t
    return s * s * s * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t * t * t * p3


def _evaluate_bezier_derivative(control_points: np.ndarray, t: float) -> np.ndarray:
    """First derivative of cubic bezier at parameter t."""
    p0, p1, p2, p3 = control_points
    s = 1.0 - t
    return (
        3 * s * s * (p1 - p0)
        + 6 * s * t * (p2 - p1)
        + 3 * t * t * (p3 - p2)
    )


def _evaluate_bezier_second_derivative(
    control_points: np.ndarray, t: float
) -> np.ndarray:
    """Second derivative of cubic bezier at parameter t."""
    p0, p1, p2, p3 = control_points
    s = 1.0 - t
    return 6 * s * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)


# Bernstein basis functions for cubic bezier
def _b0(t: float) -> float:
    s = 1.0 - t
    return s * s * s


def _b1(t: float) -> float:
    s = 1.0 - t
    return 3.0 * s * s * t


def _b2(t: float) -> float:
    s = 1.0 - t
    return 3.0 * s * t * t


def _b3(t: float) -> float:
    return t * t * t


def _two_point_bezier(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Create a bezier for just two points (a straight line segment)."""
    dist = np.linalg.norm(p1 - p0) / 3.0
    tangent = _normalize(p1 - p0)
    return np.array([p0, p0 + tangent * dist, p1 - tangent * dist, p1])


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length. Returns zero vector if input is zero."""
    mag = np.linalg.norm(v)
    if mag < 1e-12:
        return np.zeros_like(v)
    return v / mag
