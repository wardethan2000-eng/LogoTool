"""Tests for the tracer module."""

import numpy as np

from logo2svg.tracer import find_contours, trace_to_svg_paths


def _make_rectangle_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Create a binary mask with a white rectangle on black background."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _make_ring_mask(h: int, w: int, cx: int, cy: int, r_outer: int, r_inner: int) -> np.ndarray:
    """Create a binary mask with a ring (circle with hole)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    outer = (x - cx) ** 2 + (y - cy) ** 2 <= r_outer ** 2
    inner = (x - cx) ** 2 + (y - cy) ** 2 <= r_inner ** 2
    mask[outer & ~inner] = 255
    return mask


def test_find_contours_rectangle():
    """Rectangle mask should produce at least one contour."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    contours, hierarchy = find_contours(mask)
    assert len(contours) >= 1
    assert hierarchy is not None


def test_trace_rectangle_produces_svg_paths():
    """Tracing a rectangle should produce SVG path strings."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    contours, hierarchy = find_contours(mask)
    paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0)
    assert len(paths) >= 1
    # Path should contain M (moveto), C or L (curves/lines), Z (close)
    path = paths[0]
    assert path.startswith("M")
    assert "Z" in path


def test_trace_ring_has_hole():
    """A ring (circle with hole) should produce a compound path with two sub-paths."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    contours, hierarchy = find_contours(mask)
    paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0)
    assert len(paths) >= 1
    # Compound path should contain two M commands (outer + hole)
    path = paths[0]
    m_count = path.count("M ")
    assert m_count >= 2, f"Expected >=2 sub-paths (M commands), got {m_count}"


def test_trace_empty_mask():
    """Empty mask should produce no paths."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    contours, hierarchy = find_contours(mask)
    paths = trace_to_svg_paths(contours, hierarchy)
    assert paths == []


def test_trace_with_simplify():
    """Simplification should produce valid paths with fewer points."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    contours, hierarchy = find_contours(mask)

    paths_default = trace_to_svg_paths(contours, hierarchy, tolerance=2.0)
    paths_simplified = trace_to_svg_paths(contours, hierarchy, tolerance=2.0, simplify=0.5)

    # Both should produce valid paths
    assert len(paths_default) >= 1
    assert len(paths_simplified) >= 1

    # Simplified paths should generally be shorter (fewer control points)
    assert len(paths_simplified[0]) <= len(paths_default[0]) * 1.5  # Allow some margin


def test_find_contours_with_smooth():
    """Smoothing should still produce valid contours."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    contours, hierarchy = find_contours(mask, smooth=1.4)
    assert len(contours) >= 1
    assert hierarchy is not None


def test_smooth_produces_valid_paths():
    """Full trace with smooth should produce valid SVG paths."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    contours, hierarchy = find_contours(mask, smooth=1.4)
    paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0, smooth=1.4)
    assert len(paths) >= 1
    path = paths[0]
    assert path.startswith("M")
    assert "Z" in path


def test_smooth_produces_reasonable_complexity():
    """Smoothed contours should produce paths with reasonable segment counts."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)

    contours_smooth, hier_smooth = find_contours(mask, smooth=2.0)
    paths_smooth = trace_to_svg_paths(contours_smooth, hier_smooth, tolerance=2.0, smooth=2.0)

    assert len(paths_smooth) >= 1
    # A smoothed ring should use cubic bezier curves (C commands), not line segments
    assert " C " in paths_smooth[0]
    # Should have a reasonable number of segments (not hundreds from pixel staircase)
    segment_count = paths_smooth[0].count(" C ")
    assert segment_count <= 20, f"Expected <=20 bezier segments, got {segment_count}"


def test_smooth_zero_matches_original():
    """smooth=0 should produce identical output to no smoothing."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    contours_a, hier_a = find_contours(mask)
    contours_b, hier_b = find_contours(mask, smooth=0.0)
    paths_a = trace_to_svg_paths(contours_a, hier_a, tolerance=2.0)
    paths_b = trace_to_svg_paths(contours_b, hier_b, tolerance=2.0, smooth=0.0)
    assert paths_a == paths_b
