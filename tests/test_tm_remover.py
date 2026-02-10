"""Tests for tm_remover module: trademark symbol detection and removal."""

from __future__ import annotations

import numpy as np
import pytest

from logo2svg.tm_remover import remove_tm_symbols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(height: int, width: int, blobs: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create a boolean mask with rectangular blobs.

    Each blob is (y_start, y_end, x_start, x_end).
    """
    mask = np.zeros((height, width), dtype=bool)
    for y0, y1, x0, x1 in blobs:
        mask[y0:y1, x0:x1] = True
    return mask


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestRemoveTmSymbols:
    """Core tests for the remove_tm_symbols function."""

    def test_no_foreground(self):
        """Empty mask returns empty mask, zero removed."""
        mask = np.zeros((200, 200), dtype=bool)
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 0
        assert not np.any(cleaned)

    def test_large_centre_blob_kept(self):
        """A single large blob in the centre is never removed."""
        mask = _make_mask(200, 200, [(40, 160, 40, 160)])
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 0
        np.testing.assert_array_equal(cleaned, mask)

    def test_small_corner_blob_removed(self):
        """A small blob in the bottom-right corner (typical TM location) is removed."""
        # Large main logo in centre
        blobs = [
            (20, 180, 20, 180),   # main body
            (185, 195, 185, 195), # small TM in bottom-right corner
        ]
        mask = _make_mask(200, 200, blobs)
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 1
        # Main body pixels still present
        assert cleaned[100, 100]
        # TM corner pixels gone
        assert not cleaned[190, 190]

    def test_small_centre_blob_kept(self):
        """A small blob in the centre of the image is NOT removed (not in margin)."""
        blobs = [
            (20, 180, 20, 180),   # main body
            (95, 105, 95, 105),   # small element centred in image
        ]
        mask = _make_mask(200, 200, blobs)
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 0
        assert cleaned[100, 100]

    def test_large_corner_blob_kept(self):
        """A large blob in the corner should NOT be removed (too big)."""
        blobs = [
            (0, 100, 0, 100),     # large element in top-left
            (110, 200, 110, 200), # large element in bottom-right
        ]
        mask = _make_mask(200, 200, blobs)
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 0

    def test_multiple_corner_blobs_removed(self):
        """Multiple small blobs in different corners are all removed."""
        blobs = [
            (30, 170, 30, 170),   # main body
            (2, 8, 2, 8),         # tiny top-left
            (2, 8, 192, 198),     # tiny top-right
            (192, 198, 192, 198), # tiny bottom-right
        ]
        mask = _make_mask(200, 200, blobs)
        cleaned, removed = remove_tm_symbols(mask)
        assert removed == 3
        # Main body intact
        assert cleaned[100, 100]
        # Corners cleared
        assert not cleaned[5, 5]
        assert not cleaned[5, 195]
        assert not cleaned[195, 195]

    def test_custom_thresholds(self):
        """Adjusting max_area_pct and margin_pct changes detection behaviour."""
        # Small blob in a "near-edge" position, not touching main body
        blobs = [
            (20, 170, 20, 170),   # main body
            (180, 190, 180, 190), # 100 px blob near bottom-right (isolated)
        ]
        mask = _make_mask(200, 200, blobs)

        # With tight margin (5%), margin = 10px, blob centre at (185, 185)
        # 185 > 200-10=190? No. So it should NOT be removed with 5% margin.
        cleaned, removed = remove_tm_symbols(mask, margin_pct=5.0)
        assert removed == 0

        # With generous margin (15%), margin = 30px, blob centre at (185, 185)
        # 185 > 200-30=170? Yes. So it SHOULD be removed.
        cleaned, removed = remove_tm_symbols(mask, margin_pct=15.0)
        assert removed == 1

    def test_returns_copy(self):
        """The returned mask is a copy, not a view of the original."""
        mask = _make_mask(200, 200, [(50, 150, 50, 150)])
        cleaned, _ = remove_tm_symbols(mask)
        cleaned[0, 0] = True
        assert not mask[0, 0]
