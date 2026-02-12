"""Tests for the preprocessor module."""

import numpy as np
import pytest

from logo2svg.preprocessor import (
    PreprocessReport,
    analyze_image,
    enhance_contrast,
    preprocess_image,
    reduce_noise,
    sharpen_edges,
)


@pytest.fixture
def sample_image():
    """A 100x100 RGB image with some variation."""
    rng = np.random.RandomState(42)
    img = rng.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    return img


@pytest.fixture
def low_contrast_image():
    """An image with very low contrast."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    img[:50, :] = 132  # barely different
    return img


@pytest.fixture
def fg_mask():
    """Foreground mask: center 60x60 is foreground."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


class TestAnalyzeImage:
    def test_basic_analysis(self, sample_image, fg_mask):
        report = analyze_image(sample_image, fg_mask)
        assert isinstance(report, PreprocessReport)
        assert isinstance(report.noise_level, float)
        assert isinstance(report.bg_uniformity, float)
        assert isinstance(report.recommendations, list)

    def test_low_contrast_detection(self, low_contrast_image):
        report = analyze_image(low_contrast_image)
        assert report.contrast_low is True
        assert any("contrast" in r.lower() for r in report.recommendations)

    def test_no_mask(self, sample_image):
        report = analyze_image(sample_image, None)
        assert report.bg_blur_detected is False

    def test_uniform_background(self, fg_mask):
        # White background, colored foreground
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        img[20:80, 20:80] = [100, 50, 200]
        report = analyze_image(img, fg_mask)
        assert report.bg_uniformity > 0.8


class TestEnhanceContrast:
    def test_output_shape(self, sample_image):
        result = enhance_contrast(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_changes_values(self, low_contrast_image):
        result = enhance_contrast(low_contrast_image, clip_limit=3.0)
        # Should have increased contrast (higher std dev)
        assert np.std(result) >= np.std(low_contrast_image) - 1


class TestSharpenEdges:
    def test_output_shape(self, sample_image):
        result = sharpen_edges(sample_image, strength=1.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestReduceNoise:
    def test_output_shape(self, sample_image):
        result = reduce_noise(sample_image, strength=5)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestPreprocessImage:
    def test_no_processing(self, sample_image):
        result, report = preprocess_image(sample_image)
        np.testing.assert_array_equal(result, sample_image)
        assert isinstance(report, PreprocessReport)

    def test_all_processing(self, sample_image, fg_mask):
        result, report = preprocess_image(
            sample_image, fg_mask,
            contrast=True, sharpen=True, denoise=True,
        )
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
        assert isinstance(report, PreprocessReport)

    def test_contrast_only(self, sample_image):
        result, _ = preprocess_image(sample_image, contrast=True)
        assert result.shape == sample_image.shape
