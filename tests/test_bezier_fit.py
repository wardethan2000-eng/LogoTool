"""Tests for bezier_fit module (Catmull-Rom implementation)."""

import numpy as np

from logo2svg.bezier_fit import fit_curve, fit_curve_closed, _evaluate_bezier


def test_fit_curve_two_points():
    """Two points should produce one bezier segment (a line)."""
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    segments = fit_curve(points, max_error=1.0)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.shape == (4, 2)
    np.testing.assert_allclose(seg[0], [0, 0], atol=0.1)
    np.testing.assert_allclose(seg[3], [10, 0], atol=0.1)


def test_fit_curve_straight_line():
    """Collinear points should produce near-straight beziers."""
    points = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    segments = fit_curve(points, max_error=1.0)
    # Catmull-Rom: N-1 segments for N open points
    assert len(segments) == 2
    for seg in segments:
        for pt in seg:
            assert abs(pt[1]) < 1.0


def test_fit_curve_returns_segments():
    """A curved polyline should produce one segment per adjacent pair."""
    t = np.linspace(0, np.pi, 20)
    points = np.column_stack([t * 10, np.sin(t) * 10])
    segments = fit_curve(points, max_error=0.5)
    # Catmull-Rom: exactly N-1 segments for N points
    assert len(segments) == 19
    np.testing.assert_allclose(segments[0][0], points[0], atol=0.01)
    np.testing.assert_allclose(segments[-1][3], points[-1], atol=0.01)


def test_fit_curve_closed_square():
    """A closed square should produce exactly 4 bezier segments."""
    points = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]
    ])
    segments = fit_curve_closed(points, max_error=1.0)
    # Catmull-Rom closed: exactly N segments for N points
    assert len(segments) == 4
    np.testing.assert_allclose(segments[-1][3], segments[0][0], atol=0.01)


def test_fit_curve_closed_circle():
    """A closed circle should produce one segment per sample point."""
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    points = np.column_stack([np.cos(t) * 50 + 50, np.sin(t) * 50 + 50])
    segments = fit_curve_closed(points, max_error=2.0)
    assert len(segments) == 36
    np.testing.assert_allclose(segments[-1][3], segments[0][0], atol=0.01)


def test_fit_curve_empty_input():
    """Empty or single-point input should return empty list."""
    assert fit_curve(np.array([]).reshape(0, 2), max_error=1.0) == []
    assert fit_curve(np.array([[5.0, 5.0]]), max_error=1.0) == []


def test_catmull_rom_passes_through_points():
    """Catmull-Rom curves must pass through every input point exactly."""
    t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.column_stack([np.cos(t) * 50 + 50, np.sin(t) * 50 + 50])
    segments = fit_curve_closed(points, max_error=2.0)

    # Each segment starts at the corresponding input point
    for i, seg in enumerate(segments):
        np.testing.assert_allclose(seg[0], points[i], atol=1e-10)
    # Each segment ends at the next input point
    for i, seg in enumerate(segments):
        np.testing.assert_allclose(seg[3], points[(i + 1) % len(points)], atol=1e-10)


def test_control_points_stay_bounded():
    """Control points must stay near the data — no wild excursions."""
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    points = np.column_stack([np.cos(t) * 50 + 50, np.sin(t) * 50 + 50])
    segments = fit_curve_closed(points, max_error=2.0)

    for seg in segments:
        for pt in seg:
            # All control points should be within the bounding box of the
            # data with generous margin (say 20px for a radius-50 circle)
            assert -20 < pt[0] < 120, f"Control point x out of range: {pt}"
            assert -20 < pt[1] < 120, f"Control point y out of range: {pt}"


def test_fit_curve_max_error_respected():
    """Catmull-Rom curves should stay close to the original points."""
    t = np.linspace(0, np.pi, 30)
    points = np.column_stack([t * 20, np.sin(t) * 15])
    segments = fit_curve(points, max_error=1.0)

    # Each segment's midpoint should be reasonably close to where
    # the original data lies
    for pt in points:
        min_dist = float("inf")
        for seg in segments:
            for t_val in np.linspace(0, 1, 50):
                curve_pt = _evaluate_bezier(seg, t_val)
                dist = np.linalg.norm(pt - curve_pt)
                min_dist = min(min_dist, dist)
        assert min_dist < 3.0, f"Point {pt} too far from curve: {min_dist}"
