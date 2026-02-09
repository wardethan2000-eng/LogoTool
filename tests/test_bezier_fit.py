"""Tests for bezier_fit module."""

import numpy as np

from logo2svg.bezier_fit import fit_curve, fit_curve_closed


def test_fit_curve_two_points():
    """Two points should produce one bezier segment (a line)."""
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    segments = fit_curve(points, max_error=1.0)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.shape == (4, 2)
    # Start and end should match input points
    np.testing.assert_allclose(seg[0], [0, 0], atol=0.1)
    np.testing.assert_allclose(seg[3], [10, 0], atol=0.1)


def test_fit_curve_straight_line():
    """Collinear points should produce a near-straight bezier."""
    points = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    segments = fit_curve(points, max_error=1.0)
    assert len(segments) >= 1
    # All control points should have y close to 0
    for seg in segments:
        for pt in seg:
            assert abs(pt[1]) < 1.0


def test_fit_curve_returns_segments():
    """A curved polyline should produce multiple bezier segments."""
    t = np.linspace(0, np.pi, 20)
    points = np.column_stack([t * 10, np.sin(t) * 10])
    segments = fit_curve(points, max_error=0.5)
    assert len(segments) >= 1
    # First segment starts near the first point
    np.testing.assert_allclose(segments[0][0], points[0], atol=0.5)
    # Last segment ends near the last point
    np.testing.assert_allclose(segments[-1][3], points[-1], atol=0.5)


def test_fit_curve_closed_square():
    """A closed square should be fitted with bezier curves."""
    points = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]
    ])
    segments = fit_curve_closed(points, max_error=1.0)
    assert len(segments) >= 4  # At least one segment per side
    # The path should close: last endpoint should be near first start point
    np.testing.assert_allclose(segments[-1][3], segments[0][0], atol=1.0)


def test_fit_curve_closed_circle():
    """A closed circle approximation should produce smooth beziers."""
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    points = np.column_stack([np.cos(t) * 50 + 50, np.sin(t) * 50 + 50])
    segments = fit_curve_closed(points, max_error=2.0)
    assert len(segments) >= 3
    # Verify closure
    np.testing.assert_allclose(segments[-1][3], segments[0][0], atol=2.0)


def test_fit_curve_empty_input():
    """Empty or single-point input should return empty list."""
    assert fit_curve(np.array([]).reshape(0, 2), max_error=1.0) == []
    assert fit_curve(np.array([[5.0, 5.0]]), max_error=1.0) == []


def test_fit_curve_max_error_respected():
    """Fitted curve should stay within max_error of the original points."""
    from logo2svg.bezier_fit import _evaluate_bezier

    t = np.linspace(0, np.pi, 30)
    points = np.column_stack([t * 20, np.sin(t) * 15])
    max_error = 1.0
    segments = fit_curve(points, max_error=max_error)

    # For each original point, find the closest point on the fitted curve
    for pt in points:
        min_dist = float("inf")
        for seg in segments:
            for t_val in np.linspace(0, 1, 50):
                curve_pt = _evaluate_bezier(seg, t_val)
                dist = np.linalg.norm(pt - curve_pt)
                min_dist = min(min_dist, dist)
        # Allow some margin since we're sampling the curve
        assert min_dist < max_error * 3, f"Point {pt} too far from curve: {min_dist}"
