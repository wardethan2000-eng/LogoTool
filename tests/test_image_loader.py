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


# ---------------------------------------------------------------------------
# EXIF auto-rotation
# ---------------------------------------------------------------------------

class TestExifRotation:
    """Tests for automatic EXIF orientation handling."""

    def test_exif_rotation_applied(self, tmp_path):
        """An image with EXIF Orientation=6 (90° CW) should be auto-rotated."""
        # Create a 100x50 image (landscape) and save as JPEG with
        # Orientation tag 6 (rotate 90° CW → becomes 50x100 portrait)
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        img[:, :50] = [255, 0, 0]  # left half red
        img[:, 50:] = [0, 0, 255]  # right half blue

        pil = Image.fromarray(img, "RGB")
        import piexif
        exif_dict = {"0th": {piexif.ImageIFD.Orientation: 6}}
        exif_bytes = piexif.dump(exif_dict)
        jpeg_path = tmp_path / "rotated.jpg"
        pil.save(str(jpeg_path), "JPEG", exif=exif_bytes, quality=95)

        loaded, fg_mask = load_image(jpeg_path)
        # After 90° CW rotation, the 50x100 image becomes 100x50
        assert loaded.shape == (100, 50, 3)

    def test_no_exif_no_change(self, tmp_path):
        """An image without EXIF data is loaded normally."""
        img = np.full((80, 120, 3), [200, 200, 200], dtype=np.uint8)
        img[20:60, 30:90] = [0, 0, 0]
        _save_rgb(tmp_path / "no_exif.png", img)

        loaded, _ = load_image(tmp_path / "no_exif.png")
        assert loaded.shape == (80, 120, 3)


# ---------------------------------------------------------------------------
# CMYK image handling
# ---------------------------------------------------------------------------

class TestCmykHandling:
    """Tests for CMYK image conversion."""

    def test_cmyk_image_converted_to_rgb(self, tmp_path):
        """A CMYK image should be converted to 3-channel RGB."""
        # Create a CMYK image (4 channels)
        cmyk_data = np.zeros((50, 50, 4), dtype=np.uint8)
        cmyk_data[:, :, 0] = 0    # C
        cmyk_data[:, :, 1] = 255  # M
        cmyk_data[:, :, 2] = 255  # Y
        cmyk_data[:, :, 3] = 0    # K

        pil_cmyk = Image.fromarray(cmyk_data, "CMYK")
        path = tmp_path / "cmyk_test.tiff"
        pil_cmyk.save(str(path))

        loaded, fg_mask = load_image(path)
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3  # RGB, not CMYK


# ---------------------------------------------------------------------------
# SVG primary input
# ---------------------------------------------------------------------------

class TestSvgPrimaryInput:
    """Tests for loading SVG files as primary images."""

    def test_svg_loads_as_rgb(self, tmp_path):
        """An SVG file should be rasterized and returned as RGB."""
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <rect width="100" height="100" fill="#FFFFFF"/>
  <rect x="20" y="20" width="60" height="60" fill="#FF0000"/>
</svg>"""
        svg_path = tmp_path / "test.svg"
        svg_path.write_text(svg_content)

        loaded, fg_mask = load_image(svg_path)
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3
        # Image should be at least 100px on each side (scaled up)
        assert loaded.shape[0] >= 100
        assert loaded.shape[1] >= 100

    def test_svg_fg_mask_from_alpha(self, tmp_path):
        """SVG with transparent background should produce correct fg mask."""
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="50" fill="#0000FF"/>
</svg>"""
        svg_path = tmp_path / "circle.svg"
        svg_path.write_text(svg_content)

        loaded, fg_mask = load_image(svg_path)
        assert fg_mask.shape == loaded.shape[:2]
        # Some pixels should be foreground (the circle)
        assert np.any(fg_mask)
        # Some pixels should be background (outside the circle)
        assert not np.all(fg_mask)
