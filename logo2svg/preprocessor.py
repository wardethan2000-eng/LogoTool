"""Smart image preprocessing applied before colour quantization.

Provides four enhancement steps that improve vector quality from
photos and imperfect source images:

1. **Contrast enhancement** for washed-out images (CLAHE in LAB space)
2. **Edge sharpening** to improve boundaries for cleaner vector tracing
3. **Noise reduction** for photos (non-local means denoising)
4. **Background blur detection** to warn when the background is not uniform

All functions operate on (H, W, 3) uint8 RGB numpy arrays and return
the same format.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PreprocessReport:
    """Report produced by :func:`analyze_image`."""

    contrast_low: bool = False
    noise_level: float = 0.0
    bg_blur_detected: bool = False
    bg_uniformity: float = 1.0
    recommendations: list[str] | None = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


# ---------------------------------------------------------------------------
# Analysis (non-destructive)
# ---------------------------------------------------------------------------

def analyze_image(
    image: np.ndarray,
    fg_mask: np.ndarray | None = None,
) -> PreprocessReport:
    """Analyze an image and produce preprocessing recommendations.

    Args:
        image: (H, W, 3) uint8 RGB array.
        fg_mask: Optional (H, W) bool foreground mask.

    Returns:
        A :class:`PreprocessReport` with flags and recommendations.
    """
    report = PreprocessReport()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # --- Contrast analysis ---
    # Use the standard deviation of the luminance channel.
    # Low std-dev means washed-out / low-contrast.
    std_val = float(np.std(gray))
    if std_val < 40.0:
        report.contrast_low = True
        report.recommendations.append(
            "Image appears low-contrast / washed out. "
            "Consider enabling contrast enhancement."
        )

    # --- Noise estimation ---
    # Laplacian variance method: high = sharp, low = blurry/noisy.
    # We look at a normalized noise metric.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_var = float(laplacian.var())
    # Normalize to a rough 0-100 scale
    report.noise_level = min(100.0, noise_var / 50.0)
    if report.noise_level > 30.0:
        report.recommendations.append(
            "Moderate noise detected. "
            "Consider enabling noise reduction for cleaner vectors."
        )

    # --- Background blur / uniformity detection ---
    if fg_mask is not None:
        bg_mask = ~fg_mask
        bg_pixels = image[bg_mask]
        if len(bg_pixels) > 100:
            bg_std = float(np.std(bg_pixels, axis=0).mean())
            report.bg_uniformity = max(0.0, 1.0 - bg_std / 50.0)
            if report.bg_uniformity < 0.7:
                report.bg_blur_detected = True
                report.recommendations.append(
                    "Background is not uniform (std={:.1f}). "
                    "This may cause stray colour clusters. "
                    "Consider using a solid background override.".format(bg_std)
                )

    return report


# ---------------------------------------------------------------------------
# Enhancement functions
# ---------------------------------------------------------------------------

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Enhance contrast using CLAHE in the LAB colour space.

    Operates on the L channel only, preserving colour fidelity.

    Args:
        image: (H, W, 3) uint8 RGB.
        clip_limit: CLAHE clip limit (higher = stronger contrast).

    Returns:
        Enhanced (H, W, 3) uint8 RGB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)

    enhanced_lab = cv2.merge([l_enhanced, a_chan, b_chan])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def sharpen_edges(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Sharpen edges using unsharp masking.

    Args:
        image: (H, W, 3) uint8 RGB.
        strength: Sharpening strength multiplier.

    Returns:
        Sharpened (H, W, 3) uint8 RGB.
    """
    # Gaussian blur for the unsharp mask
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(
        image, 1.0 + strength, blurred, -strength, 0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def reduce_noise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Reduce noise using non-local means denoising.

    Args:
        image: (H, W, 3) uint8 RGB.
        strength: Filter strength (higher = more denoising, more blur).

    Returns:
        Denoised (H, W, 3) uint8 RGB.
    """
    # OpenCV's fastNlMeansDenoisingColored works in BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        bgr, None, strength, strength, 7, 21
    )
    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image: np.ndarray,
    fg_mask: np.ndarray | None = None,
    *,
    contrast: bool = False,
    sharpen: bool = False,
    denoise: bool = False,
    contrast_strength: float = 2.0,
    sharpen_strength: float = 1.0,
    denoise_strength: int = 10,
) -> tuple[np.ndarray, PreprocessReport]:
    """Run selected preprocessing steps and return the result + report.

    Steps are applied in order: denoise -> contrast -> sharpen.
    Analysis is always performed on the *original* image.

    Args:
        image: (H, W, 3) uint8 RGB source.
        fg_mask: Optional foreground mask for background analysis.
        contrast: Enable contrast enhancement.
        sharpen: Enable edge sharpening.
        denoise: Enable noise reduction.
        contrast_strength: CLAHE clip limit.
        sharpen_strength: Unsharp mask strength.
        denoise_strength: Non-local means filter strength.

    Returns:
        Tuple of (processed_image, report).
    """
    report = analyze_image(image, fg_mask)

    result = image.copy()

    if denoise:
        result = reduce_noise(result, strength=denoise_strength)

    if contrast:
        result = enhance_contrast(result, clip_limit=contrast_strength)

    if sharpen:
        result = sharpen_edges(result, strength=sharpen_strength)

    return result, report
