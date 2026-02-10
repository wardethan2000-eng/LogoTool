"""Tests for the tracer module (potrace-based)."""

import numpy as np

from logo2svg.tracer import find_contours, trace_to_svg_paths, trace_mask_to_svg_paths


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
    paths = trace_mask_to_svg_paths(mask)
    assert len(paths) >= 1
    path = paths[0]
    assert path.startswith("M")
    assert "Z" in path


def test_trace_rectangle_has_corners():
    """A rectangle should be traced with corner segments (L commands)."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    paths = trace_mask_to_svg_paths(mask, alphamax=1.0)
    assert len(paths) >= 1
    path = paths[0]
    # Potrace should detect the rectangle's corners
    assert "L " in path


def test_trace_ring_has_hole():
    """A ring (circle with hole) should produce a compound path with two sub-paths."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    paths = trace_mask_to_svg_paths(mask)
    assert len(paths) >= 1
    # Compound path should contain two M commands (outer + hole)
    path = paths[0]
    m_count = path.count("M ")
    assert m_count >= 2, f"Expected >=2 sub-paths (M commands), got {m_count}"


def test_trace_empty_mask():
    """Empty mask should produce no paths."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    paths = trace_mask_to_svg_paths(mask)
    assert paths == []


def test_trace_ring_uses_bezier_curves():
    """A circle should be traced with smooth bezier curves (C commands)."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    paths = trace_mask_to_svg_paths(mask)
    assert len(paths) >= 1
    # Circles should use C (cubic bezier) commands, not just L (lines)
    assert " C " in paths[0]


def test_turdsize_filters_small_speckles():
    """Small isolated regions should be suppressed by turdsize."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Add a large rectangle
    mask[20:180, 20:180] = 255
    # Add a tiny 2x2 speckle far from the rectangle
    mask[5:7, 5:7] = 255

    # With high turdsize, the speckle should be filtered
    paths_filtered = trace_mask_to_svg_paths(mask, turdsize=10)
    # With turdsize=0, even tiny bits are kept
    paths_all = trace_mask_to_svg_paths(mask, turdsize=0)

    # The filtered version should have fewer sub-paths
    m_filtered = paths_filtered[0].count("M ") if paths_filtered else 0
    m_all = paths_all[0].count("M ") if paths_all else 0
    assert m_filtered <= m_all


def test_alphamax_controls_corner_detection():
    """Lower alphamax should produce more corners (L commands)."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)

    # alphamax=0: force all corners
    paths_sharp = trace_mask_to_svg_paths(mask, alphamax=0.0)
    # alphamax=1.334: force all curves
    paths_smooth = trace_mask_to_svg_paths(mask, alphamax=1.334)

    if paths_sharp and paths_smooth:
        l_count_sharp = paths_sharp[0].count("L ")
        l_count_smooth = paths_smooth[0].count("L ")
        # Sharp version should have at least as many L commands
        assert l_count_sharp >= l_count_smooth


def test_legacy_trace_to_svg_paths():
    """Legacy trace_to_svg_paths should produce valid SVG paths."""
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    contours, hierarchy = find_contours(mask)
    paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0)
    assert len(paths) >= 1
    path = paths[0]
    assert path.startswith("M")
    assert "Z" in path


def test_legacy_trace_to_svg_paths_with_simplify():
    """Legacy trace_to_svg_paths accepts a simplify parameter without crashing.

    The *simplify* parameter is accepted for backward compatibility but is
    ignored (Potrace handles simplification internally).  This test documents
    that calling with extra positional args doesn't raise.
    """
    mask = _make_rectangle_mask(100, 100, 20, 20, 80, 80)
    contours, hierarchy = find_contours(mask)
    # Call with explicit simplify= kwarg — should not crash
    paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0, simplify=0.5)
    assert len(paths) >= 1
    assert paths[0].startswith("M")
    assert "Z" in paths[0]


def test_find_contours_with_smooth():
    """Smoothing should still produce valid contours."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    contours, hierarchy = find_contours(mask, smooth=1.4)
    assert len(contours) >= 1
    assert hierarchy is not None


def test_potrace_circle_is_smooth():
    """Potrace should trace a circle with smooth bezier curves, not jaggy lines."""
    mask = _make_ring_mask(200, 200, 100, 100, 80, 40)
    paths = trace_mask_to_svg_paths(mask, alphamax=1.0, opticurve=True)
    assert len(paths) >= 1
    # Should have bezier curves
    c_count = paths[0].count(" C ")
    assert c_count >= 4, f"Expected >= 4 bezier segments for a circle, got {c_count}"
    # Should NOT have many line segments (circles are smooth)
    l_count = paths[0].count(" L ")
    assert l_count <= c_count, f"More lines ({l_count}) than curves ({c_count}) for a circle"
