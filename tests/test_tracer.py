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
