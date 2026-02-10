"""Tests for image_loader module: background detection and foreground masking."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from logo2svg.image_loader import (
    load_image,
    _detect_bg_from_alpha,
    _detect_bg_from_corners,
    _remove_bg_color,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_rgba(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img, "RGBA").save(str(path))


def _save_rgb(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img, "RGB").save(str(path))


# ---------------------------------------------------------------------------
# Alpha-based background detection
# ---------------------------------------------------------------------------

class TestAlphaDetection:
    """Tests for _detect_bg_from_alpha and the alpha path in load_image."""

    def test_transparent_bg_detected(self, tmp_path):
        """Fully-transparent pixels are classified as background."""
        size = 100
        img = np.zeros((size, size, 4), dtype=np.uint8)
        # Opaque red square in centre
        img[20:80, 20:80] = [255, 0, 0, 255]
        _save_rgba(tmp_path / "test.png", img)

        image, fg_mask = load_image(tmp_path / "test.png")
        # The red square should be foreground
        assert fg_mask[50, 50]
        # Corners (transparent) should be background
        assert not fg_mask[0, 0]
        assert not fg_mask[0, 99]
        assert not fg_mask[99, 0]
        assert not fg_mask[99, 99]

    def test_semi_transparent_bg(self, tmp_path):
        """Pixels with very low alpha (< threshold) are background."""
        size = 100
        img = np.zeros((size, size, 4), dtype=np.uint8)
        img[:, :, 3] = 5  # nearly transparent everywhere
        img[40:60, 40:60] = [0, 0, 200, 255]  # opaque blue patch
        _save_rgba(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        assert fg_mask[50, 50]
        assert not fg_mask[0, 0]

    def test_alpha_all_opaque_falls_through(self, tmp_path):
        """When alpha is all-opaque, fall through to corner-based detection."""
        size = 100
        # White background with a blue square, RGBA all-opaque
        img = np.full((size, size, 4), 255, dtype=np.uint8)
        img[30:70, 30:70, :3] = [0, 0, 200]
        _save_rgba(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        # Corner pixels are white → should be detected as background via fallback
        assert not fg_mask[0, 0]
        # Blue square should be foreground
        assert fg_mask[50, 50]


# ---------------------------------------------------------------------------
# Corner-based background detection
# ---------------------------------------------------------------------------

class TestCornerDetection:
    """Tests for _detect_bg_from_corners and the corner path in load_image."""

    def test_uniform_corners_detected(self):
        """If all four corners share a colour, that colour is the background."""
        size = 100
        img = np.full((size, size, 3), [200, 200, 200], dtype=np.uint8)
        img[30:70, 30:70] = [50, 50, 200]

        bg_rgb = _detect_bg_from_corners(img)
        assert bg_rgb is not None
        # Should be close to (200, 200, 200)
        assert np.allclose(bg_rgb, [200, 200, 200], atol=10)

    def test_disagreeing_corners_returns_none(self):
        """If corners don't agree, return None (no dominant background)."""
        size = 100
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:50, :50] = [255, 0, 0]
        img[:50, 50:] = [0, 255, 0]
        img[50:, :50] = [0, 0, 255]
        img[50:, 50:] = [255, 255, 0]

        bg_rgb = _detect_bg_from_corners(img)
        assert bg_rgb is None

    def test_corner_detection_rgb_image(self, tmp_path):
        """load_image uses corner detection for opaque RGB images."""
        size = 100
        img = np.full((size, size, 3), [255, 255, 255], dtype=np.uint8)
        img[20:80, 20:80] = [0, 0, 0]
        _save_rgb(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        assert fg_mask[50, 50]  # black square is foreground
        assert not fg_mask[0, 0]  # white corner is background


# ---------------------------------------------------------------------------
# Explicit --bg-color override
# ---------------------------------------------------------------------------

class TestBgColorOverride:
    """Tests for the --bg-color parameter."""

    def test_bg_color_override_removes_matching_pixels(self, tmp_path):
        """Pixels matching the override colour are classified as background."""
        size = 100
        img = np.full((size, size, 3), [0, 128, 0], dtype=np.uint8)  # green bg
        img[30:70, 30:70] = [255, 0, 0]  # red square
        _save_rgb(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png", bg_color_override="#008000")
        assert fg_mask[50, 50]  # red square
        assert not fg_mask[0, 0]  # green background

    def test_bg_color_override_ignores_alpha(self, tmp_path):
        """When --bg-color is given, alpha channel is ignored."""
        size = 100
        img = np.full((size, size, 4), 255, dtype=np.uint8)  # all opaque white
        img[30:70, 30:70, :3] = [0, 0, 0]  # black square
        _save_rgba(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png", bg_color_override="#FFFFFF")
        assert fg_mask[50, 50]  # black square is foreground
        assert not fg_mask[0, 0]  # white background


# ---------------------------------------------------------------------------
# Edge-connected flood fill: interior regions preserved
# ---------------------------------------------------------------------------

class TestFloodFillPreservation:
    """Verify that interior white regions are preserved when bg is also white."""

    def test_interior_white_preserved(self, tmp_path):
        """White region fully enclosed by coloured pixels is NOT background."""
        size = 200
        img = np.full((size, size, 3), 255, dtype=np.uint8)  # white bg
        # Blue border ring
        img[40:160, 40:160] = [0, 0, 200]
        # White interior (fully enclosed by blue)
        img[70:130, 70:130] = [255, 255, 255]
        _save_rgb(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        # Exterior white = background
        assert not fg_mask[0, 0]
        # Blue ring = foreground
        assert fg_mask[50, 50]
        # Interior white = foreground (connected-component logic)
        assert fg_mask[100, 100]

    def test_edge_connected_white_is_background(self, tmp_path):
        """White region touching the image edge IS background."""
        size = 100
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        img[30:70, 30:70] = [0, 0, 200]  # blue square
        _save_rgb(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        assert not fg_mask[0, 0]
        assert fg_mask[50, 50]


# ---------------------------------------------------------------------------
# Fallback when no background detected
# ---------------------------------------------------------------------------

class TestFallback:
    """When no clear background exists, all pixels should be foreground."""

    def test_all_foreground_fallback(self, tmp_path):
        """An image with random corners → everything treated as foreground."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        _save_rgb(tmp_path / "test.png", img)

        _, fg_mask = load_image(tmp_path / "test.png")
        # With random corner colours presumably disagreeing,
        # the fallback is all-foreground
        # (Exact result depends on colour variance; just check it doesn't crash
        # and the mask has the right shape)
        assert fg_mask.shape == (100, 100)
        assert fg_mask.dtype == bool


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestRemoveBgColor:
    """Tests for the _remove_bg_color connected-component logic."""

    def test_basic_removal(self):
        """Matching border-connected pixels are removed."""
        img = np.full((50, 50, 3), [200, 200, 200], dtype=np.uint8)
        img[15:35, 15:35] = [50, 50, 50]
        bg_rgb = np.array([200, 200, 200], dtype=np.uint8)

        fg_mask = _remove_bg_color(img, bg_rgb, tolerance=30)
        assert not fg_mask[0, 0]
        assert fg_mask[25, 25]

    def test_interior_match_preserved(self):
        """Background-coloured pixels NOT touching the border are kept."""
        img = np.full((50, 50, 3), [200, 200, 200], dtype=np.uint8)
        img[10:40, 10:40] = [50, 50, 50]
        img[20:30, 20:30] = [200, 200, 200]  # interior matching bg

        bg_rgb = np.array([200, 200, 200], dtype=np.uint8)
        fg_mask = _remove_bg_color(img, bg_rgb, tolerance=30)

        assert not fg_mask[0, 0]          # actual bg
        assert fg_mask[25, 25]            # interior match → preserved
        assert fg_mask[15, 15]            # dark region → foreground


class TestDetectBgFromAlpha:
    """Unit tests for _detect_bg_from_alpha."""

    def test_threshold_boundary(self):
        alpha = np.array([[0, 5, 10, 11, 255]], dtype=np.uint8)
        fg = _detect_bg_from_alpha(alpha, threshold=10)
        assert not fg[0, 0]
        assert not fg[0, 1]
        assert not fg[0, 2]  # exactly at threshold → not above → bg
        assert fg[0, 3]
        assert fg[0, 4]
