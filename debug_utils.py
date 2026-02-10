"""Shared stubs and helpers for debug scripts.

The original pipeline used OpenCV contour smoothing and epsilon-based
simplification (approxPolyDP).  Those functions were removed from
``logo2svg.tracer`` when Potrace replaced the OpenCV tracing back-end.
The stubs below let debug scripts that still exercise the old OpenCV path
continue to run without importing deleted symbols.
"""

from __future__ import annotations

import cv2
import numpy as np


def smooth_contour(pts: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Stub: Gaussian smoothing of contour coordinates.

    Replaced by Potrace's built-in curve optimisation.
    """
    from scipy.ndimage import gaussian_filter1d

    if len(pts) < 3 or sigma <= 0:
        return pts
    smoothed = gaussian_filter1d(pts, sigma=sigma, axis=0, mode="wrap")
    return smoothed


def compute_epsilon(pts_cv: np.ndarray, factor: float | None = 0.005) -> float:
    """Stub: compute approxPolyDP epsilon from arc length.

    Replaced by Potrace's built-in corner / curve detection.
    """
    if factor is None:
        factor = 0.005
    perim = cv2.arcLength(pts_cv, closed=True)
    return factor * perim
