"""Contour tracing: binary mask -> potrace -> SVG path strings.

Uses the Potrace algorithm (pure-Python port) to convert binary masks
directly into optimised cubic Bezier curves.  Potrace handles:
  1. Optimal polygon decomposition of bitmap boundaries
  2. Corner vs. smooth-curve detection (alpha parameter)
  3. Mathematically optimal Bezier fitting with bounded deviation
  4. Curve optimisation that merges segments when possible

This replaces the previous OpenCV contour + approxPolyDP + Catmull-Rom
pipeline, which introduced cumulative distortion at every stage.
"""

from __future__ import annotations

import numpy as np
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY

def trace_mask_to_svg_paths(
    mask: np.ndarray,
    *,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opticurve: bool = True,
    opttolerance: float = 0.2,
) -> list[str]:
    """Trace a binary mask to SVG path 'd' strings via Potrace.

    Each returned path string represents one separate shape (curve)
    traced by Potrace.  This ensures slicers and SVG editors treat
    each shape as an independent object that can be individually
    colored or manipulated.

    Args:
        mask: (H, W) uint8 binary mask, 0 = background, 255 = foreground.
        turdsize: Suppress speckles of up to this many pixels.
            Acts as a built-in small-component filter.
        alphamax: Corner detection threshold (0.0 – 1.334).
            Lower = more corners detected (sharper output).
            Higher = more curves (smoother output).
            Default 1.0 is a good balance for logos.
        opticurve: Enable curve optimisation (merge adjacent Bezier
            segments when the error stays within *opttolerance*).
        opttolerance: Maximum deviation allowed when merging curves.
            Lower = more faithful, higher = fewer segments.

    Returns:
        List of SVG path 'd' strings.  Each string is one separate
        shape traced by Potrace.
    """
    if mask.size == 0 or not np.any(mask):
        return []

    # Convert uint8 mask to bool for Potrace
    bool_mask = mask > 127

    # Create bitmap and invert: Potrace's constructor calls invert()
    # internally, so we call invert() again to get foreground = True.
    bm = Bitmap(bool_mask)
    bm.invert()

    plist = bm.trace(
        turdsize=turdsize,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=alphamax,
        opticurve=opticurve,
        opttolerance=opttolerance,
    )

    if not plist:
        return []

    # Convert potrace curves into SVG path strings
    return _curves_to_svg_paths(plist)

# ------------------------------------------------------------------
# Legacy wrappers – kept so existing callers / debug scripts still work.
# ------------------------------------------------------------------

def find_contours(
    mask: np.ndarray,
    smooth: float = 0.0,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Legacy wrapper: find contours with OpenCV.

    Retained for callers that inspect raw contour arrays (debug scripts).
    The main pipeline no longer uses this – it calls
    trace_mask_to_svg_paths() directly.
    """
    import cv2

    if smooth > 0:
        blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=smooth)
        mask = (blurred > 127).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return list(contours), hierarchy

def trace_to_svg_paths(
    contours: list[np.ndarray],
    hierarchy: np.ndarray | None,
    tolerance: float = 2.0,
    simplify: float | None = None,
    smooth: float = 0.0,
) -> list[str]:
    """Legacy wrapper: convert OpenCV contours to SVG paths.

    Re-rasterises the contours into a mask and traces with potrace.
    New code should call trace_mask_to_svg_paths() directly.
    """
    if not contours or hierarchy is None:
        return []

    import cv2

    # Re-rasterise contours into a mask and trace with potrace
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    h = int(all_pts[:, 1].max()) + 2
    w = int(all_pts[:, 0].max()) + 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED, hierarchy=hierarchy)

    return trace_mask_to_svg_paths(mask)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _curve_to_svg_d(curve) -> str:
    """Convert a single potrace curve into an SVG path 'd' string."""
    parts: list[str] = []
    fs = curve.start_point
    parts.append(f"M {fs.x:.2f} {fs.y:.2f}")

    for segment in curve.segments:
        if segment.is_corner:
            a = segment.c
            b = segment.end_point
            parts.append(
                f"L {a.x:.2f} {a.y:.2f} L {b.x:.2f} {b.y:.2f}"
            )
        else:
            a = segment.c1
            b = segment.c2
            c = segment.end_point
            parts.append(
                f"C {a.x:.2f} {a.y:.2f} {b.x:.2f} {b.y:.2f} "
                f"{c.x:.2f} {c.y:.2f}"
            )

    parts.append("Z")
    return " ".join(parts)


def _curves_to_svg_paths(plist) -> list[str]:
    """Convert potrace path list into SVG path 'd' strings.

    Each Potrace curve becomes its own SVG path string so that
    downstream SVG writers emit separate <path> elements.  This
    allows slicers (e.g. Bambu Studio, PrusaSlicer) to treat each
    shape as an independent object for multi-color assignment.

    Shapes that contain holes (e.g. the letter 'O') still render
    correctly because each <path> element uses fill-rule="evenodd".
    """
    if not plist:
        return []

    return [_curve_to_svg_d(curve) for curve in plist]