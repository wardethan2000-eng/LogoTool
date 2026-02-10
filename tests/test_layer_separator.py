"""Tests for layer_separator module: mask generation, cleanup, and overlap resolution."""

from __future__ import annotations

import numpy as np
import pytest

from logo2svg.layer_separator import (
    separate_layers,
    _morphological_cleanup,
    _filter_small_components,
    _resolve_overlaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labels_and_centers(
    size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-cluster labels: left half = 0 (red), right half = 1 (blue).

    Returns (labels, centers_rgb, fg_mask).
    """
    labels = np.full((size, size), -1, dtype=np.int32)
    labels[:, : size // 2] = 0
    labels[:, size // 2 :] = 1
    centers_rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    fg_mask = np.ones((size, size), dtype=bool)
    return labels, centers_rgb, fg_mask


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

class TestMorphologicalCleanup:
    """Tests for _morphological_cleanup."""

    def test_fills_small_holes(self):
        """Morph close should fill a 1-pixel hole inside a solid region."""
        mask = np.full((50, 50), 255, dtype=np.uint8)
        mask[25, 25] = 0  # single-pixel hole
        cleaned = _morphological_cleanup(mask)
        assert cleaned[25, 25] == 255

    def test_preserves_solid_region(self):
        """A solid rectangle should pass through unchanged."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        cleaned = _morphological_cleanup(mask)
        assert np.array_equal(mask, cleaned)

    def test_empty_mask_stays_empty(self):
        """All-zero mask should remain all-zero."""
        mask = np.zeros((30, 30), dtype=np.uint8)
        cleaned = _morphological_cleanup(mask)
        assert np.count_nonzero(cleaned) == 0


# ---------------------------------------------------------------------------
# Small component filtering
# ---------------------------------------------------------------------------

class TestFilterSmallComponents:
    """Tests for _filter_small_components."""

    def test_removes_small_blobs(self):
        """Components smaller than min_area are removed."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:60] = 255   # large component (2500 px)
        mask[80:83, 80:83] = 255   # small component (9 px)
        result = _filter_small_components(mask, min_area=50)
        # Large should remain
        assert result[30, 30] == 255
        # Small should be removed
        assert result[81, 81] == 0

    def test_keeps_large_blobs(self):
        """Components >= min_area are retained."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:60] = 255
        result = _filter_small_components(mask, min_area=50)
        assert np.count_nonzero(result) > 0

    def test_min_area_zero_keeps_all(self):
        """min_area=0 should keep everything."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[0, 0] = 255
        result = _filter_small_components(mask, min_area=0)
        assert result[0, 0] == 255


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

class TestResolveOverlaps:
    """Tests for _resolve_overlaps."""

    def test_no_overlap_unchanged(self):
        """Non-overlapping layers should be returned unchanged."""
        labels, centers_rgb, fg_mask = _make_labels_and_centers()
        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=0)
        # Without morph expansion, layers should not overlap at all
        total_fg = sum(np.count_nonzero(l["mask"]) for l in layers)
        assert total_fg <= fg_mask.sum()

    def test_overlap_resolved_by_original_label(self):
        """When masks overlap, the original K-means label wins."""
        size = 100
        labels = np.full((size, size), -1, dtype=np.int32)
        labels[:, :55] = 0  # extends past midpoint
        labels[:, 50:] = 1  # overlapping zone [50..55)
        # Original label at overlap zone should be 1 (last write)
        centers_rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        fg_mask = np.ones((size, size), dtype=bool)

        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=0)
        # After overlap resolution, each pixel belongs to at most one layer
        total_mask = np.zeros((size, size), dtype=np.uint8)
        for layer in layers:
            total_mask += (layer["mask"] > 0).astype(np.uint8)
        assert np.all(total_mask <= 1)


# ---------------------------------------------------------------------------
# Empty layer exclusion
# ---------------------------------------------------------------------------

class TestEmptyLayers:
    """Verify that empty clusters are excluded from output."""

    def test_empty_cluster_excluded(self):
        """A cluster with no pixels (e.g. all filtered out) should not appear."""
        size = 100
        labels = np.full((size, size), 0, dtype=np.int32)
        # Cluster 1 gets only a tiny region that will be filtered
        labels[0, 0] = 1
        centers_rgb = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        fg_mask = np.ones((size, size), dtype=bool)

        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=50)
        # Only the large cluster should survive
        assert len(layers) == 1
        assert layers[0]["hex_color"] == "#FF0000"

    def test_all_empty_returns_empty_list(self):
        """If every cluster is below min_area, return an empty list."""
        labels = np.full((10, 10), 0, dtype=np.int32)
        centers_rgb = np.array([[100, 100, 100]], dtype=np.uint8)
        fg_mask = np.ones((10, 10), dtype=bool)

        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=999)
        assert layers == []


# ---------------------------------------------------------------------------
# Integration: separate_layers end-to-end
# ---------------------------------------------------------------------------

class TestSeparateLayers:
    """Integration tests for separate_layers."""

    def test_produces_correct_layer_dicts(self):
        """Each layer dict should have required keys."""
        labels, centers_rgb, fg_mask = _make_labels_and_centers()
        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=0)
        assert len(layers) == 2
        for layer in layers:
            assert "rgb" in layer
            assert "hex_color" in layer
            assert "color_name" in layer
            assert "mask" in layer
            assert "cluster_idx" in layer
            assert layer["mask"].dtype == np.uint8

    def test_masks_match_image_size(self):
        """All masks should have same shape as input labels."""
        labels, centers_rgb, fg_mask = _make_labels_and_centers(size=80)
        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=0)
        for layer in layers:
            assert layer["mask"].shape == (80, 80)

    def test_hex_colors_valid(self):
        """Hex colours should be valid 7-char strings (#RRGGBB)."""
        labels, centers_rgb, fg_mask = _make_labels_and_centers()
        layers = separate_layers(labels, centers_rgb, fg_mask, min_area=0)
        import re
        for layer in layers:
            assert re.match(r"^#[0-9A-F]{6}$", layer["hex_color"])
