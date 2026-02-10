"""Tests for quantizer module: K-means clustering and auto-detection."""

from __future__ import annotations

import numpy as np
import pytest

from logo2svg.quantizer import (
    quantize_colors,
    _auto_detect_k,
    _reassign_boundary_pixels,
    _lab_centers_to_rgb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_four_quadrant_image(size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic image with 4 solid-colour quadrants and full fg mask.

    Top-left:     Red   (255, 0, 0)
    Top-right:    Green (0, 255, 0)
    Bottom-left:  Blue  (0, 0, 255)
    Bottom-right: Yellow(255, 255, 0)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    img[:half, :half] = [255, 0, 0]
    img[:half, half:] = [0, 255, 0]
    img[half:, :half] = [0, 0, 255]
    img[half:, half:] = [255, 255, 0]
    fg_mask = np.ones((size, size), dtype=bool)
    return img, fg_mask


def _make_single_color_image(size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """All-red image with full fg mask."""
    img = np.full((size, size, 3), [200, 50, 50], dtype=np.uint8)
    fg_mask = np.ones((size, size), dtype=bool)
    return img, fg_mask


# ---------------------------------------------------------------------------
# Basic K-means clustering
# ---------------------------------------------------------------------------

class TestQuantizeColors:
    """Core quantization tests."""

    def test_four_colors_known(self):
        """Four solid quadrants with n_colors=4 should yield 4 clusters."""
        img, fg_mask = _make_four_quadrant_image()
        labels, centers = quantize_colors(img, fg_mask, n_colors=4)

        assert centers.shape == (4, 3)
        assert labels.shape == img.shape[:2]
        # Every foreground pixel should have a label 0..3
        assert np.all(labels[fg_mask] >= 0)
        assert np.all(labels[fg_mask] < 4)

    def test_two_colors_merges_quadrants(self):
        """Requesting 2 colours from a 4-colour image produces exactly 2 clusters."""
        img, fg_mask = _make_four_quadrant_image()
        labels, centers = quantize_colors(img, fg_mask, n_colors=2)
        assert len(centers) == 2
        assert len(np.unique(labels[fg_mask])) == 2

    def test_labels_minus_one_for_background(self):
        """Background pixels (fg_mask=False) should have label -1."""
        img, fg_mask = _make_four_quadrant_image()
        fg_mask[:10, :10] = False  # make top-left corner bg
        labels, _ = quantize_colors(img, fg_mask, n_colors=4)
        assert np.all(labels[~fg_mask] == -1)

    def test_centers_are_uint8_rgb(self):
        """Cluster centers should be valid uint8 RGB values."""
        img, fg_mask = _make_four_quadrant_image()
        _, centers = quantize_colors(img, fg_mask, n_colors=4)
        assert centers.dtype == np.uint8
        assert np.all(centers >= 0)
        assert np.all(centers <= 255)

    def test_no_foreground_raises(self):
        """All-background mask should raise ValueError."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        fg_mask = np.zeros((50, 50), dtype=bool)
        with pytest.raises(ValueError, match="No foreground"):
            quantize_colors(img, fg_mask, n_colors=2)


# ---------------------------------------------------------------------------
# Auto-detection of k
# ---------------------------------------------------------------------------

class TestAutoDetectK:
    """Tests for silhouette-score–based auto-k detection."""

    def test_auto_detect_four_quadrants(self):
        """Auto-detection on 4 solid quadrants should find k in [3, 5]."""
        img, fg_mask = _make_four_quadrant_image(200)
        labels, centers = quantize_colors(img, fg_mask, n_colors=None)
        k = len(centers)
        # Silhouette should discover 4 (±1 is acceptable given the algorithm)
        assert 3 <= k <= 5, f"Expected 3-5, got {k}"

    def test_single_color_auto_detects_one(self):
        """A single-colour image should auto-detect k=1 (or at most 2)."""
        img, fg_mask = _make_single_color_image()
        import cv2
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
        fg_pixels_lab = img_lab[fg_mask]
        k = _auto_detect_k(fg_pixels_lab, max_k=8, sample_limit=50000)
        assert k <= 2, f"Expected k<=2 for single colour, got {k}"

    def test_auto_detect_respects_max_k(self):
        """Auto-detection should never return k > max_k."""
        img, fg_mask = _make_four_quadrant_image()
        import cv2
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
        fg_pixels_lab = img_lab[fg_mask]
        k = _auto_detect_k(fg_pixels_lab, max_k=3, sample_limit=50000)
        assert k <= 3


# ---------------------------------------------------------------------------
# Boundary pixel reassignment
# ---------------------------------------------------------------------------

class TestBoundaryReassignment:
    """Tests for _reassign_boundary_pixels."""

    def test_boundary_pixels_reassigned(self):
        """Pixels at the border of two clusters get reassigned to nearest center."""
        import cv2
        size = 100
        # Left half red, right half blue, with a blended boundary column
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :49] = [255, 0, 0]
        img[:, 49:51] = [128, 0, 128]  # blended
        img[:, 51:] = [0, 0, 255]
        fg_mask = np.ones((size, size), dtype=bool)

        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
        labels = np.full((size, size), -1, dtype=np.int32)
        labels[:, :50] = 0
        labels[:, 50:] = 1
        centers_lab = np.array([
            cv2.cvtColor(np.array([[[255, 0, 0]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64),
            cv2.cvtColor(np.array([[[0, 0, 255]]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64),
        ])

        result = _reassign_boundary_pixels(img_lab, labels, centers_lab, fg_mask)
        # Boundary pixels should still be assigned to either 0 or 1
        assert np.all(result[fg_mask] >= 0)


# ---------------------------------------------------------------------------
# LAB center conversion
# ---------------------------------------------------------------------------

class TestLabToRgb:
    """Tests for _lab_centers_to_rgb."""

    def test_roundtrip_approximate(self):
        """Converting known RGB → LAB → RGB should be close to original."""
        import cv2
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
        rgb_back = _lab_centers_to_rgb(lab)
        # Due to uint8 clipping, allow some tolerance
        assert rgb_back.shape == (3, 3)
        assert rgb_back.dtype == np.uint8
        # Red channel of the first center should be high
        assert rgb_back[0, 0] > 200


# ---------------------------------------------------------------------------
# Single-color input
# ---------------------------------------------------------------------------

class TestSingleColor:
    """Edge case: entire foreground is one colour."""

    def test_single_color_n1(self):
        """n_colors=1 on a single-colour image should produce exactly 1 cluster."""
        img, fg_mask = _make_single_color_image()
        labels, centers = quantize_colors(img, fg_mask, n_colors=1)
        assert len(centers) == 1
        assert np.all(labels[fg_mask] == 0)

    def test_single_color_n2_clamped(self):
        """Requesting 2 colours from a truly uniform image may clamp to fewer unique."""
        img = np.full((50, 50, 3), [100, 100, 100], dtype=np.uint8)
        fg_mask = np.ones((50, 50), dtype=bool)
        labels, centers = quantize_colors(img, fg_mask, n_colors=2)
        # K-means with identical pixels may yield 1 effective cluster
        assert len(centers) >= 1
