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

    Each returned path string may contain multiple sub-paths (outer
    boundary + holes) using the SVG evenodd fill rule.

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
        List of SVG path 'd' strings.  Each string is a complete
        compound path with evenodd winding for proper hole rendering.
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
# Internal helpers
# ------------------------------------------------------------------

def _curves_to_svg_paths(plist) -> list[str]:
    """Convert potrace path list into SVG path 'd' strings.

    All curves are combined into a single compound SVG path that
    relies on fill-rule="evenodd" for correct hole rendering.
    """
    if not plist:
        return []

    parts: list[str] = []

    for curve in plist:
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

    # Return as a single compound path (multiple M...Z sub-paths)
    return [" ".join(parts)]
