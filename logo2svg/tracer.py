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

import cv2
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

    prepared_mask, trace_scale = _prepare_mask_for_tracing(mask)

    # Convert uint8 mask to bool for Potrace
    bool_mask = prepared_mask > 127

    # Potrace fills False pixels by default, so invert the bitmap to make the
    # logo mask the traced foreground.
    bm = Bitmap(bool_mask)
    bm.invert()

    effective_turdsize = max(0, int(round(turdsize * trace_scale * trace_scale)))

    plist = bm.trace(
        turdsize=effective_turdsize,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=alphamax,
        opticurve=opticurve,
        opttolerance=opttolerance,
    )

    if not plist:
        return []

    # Convert potrace curves into SVG path strings
    return _curves_to_svg_paths(plist, scale=1.0 / trace_scale)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _prepare_mask_for_tracing(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Upsample and lightly smooth a binary mask before tracing.

    Potrace works best when the bitmap boundary already approximates the
    intended silhouette. A small amount of supersampling plus blur reduces
    staircase artefacts from raster edges while keeping sharp corners intact.
    """
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))

    if not _needs_trace_smoothing(mask_u8):
        return mask_u8, 1

    longest_side = max(mask_u8.shape)
    if longest_side >= 1024:
        scale = 1
    elif longest_side >= 384:
        scale = 2
    else:
        scale = 4

    if scale == 1:
        return mask_u8, scale

    upscaled = cv2.resize(
        mask_u8,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=0.35 * scale)
    smoothed = (blurred >= 127).astype(np.uint8) * 255
    return smoothed, scale


def _needs_trace_smoothing(mask: np.ndarray) -> bool:
    """Return True when the mask boundary is dominated by staircase steps."""
    mask_bool = mask > 127
    if mask_bool.shape[0] < 3 or mask_bool.shape[1] < 3:
        return False

    tl = mask_bool[:-1, :-1]
    tr = mask_bool[:-1, 1:]
    bl = mask_bool[1:, :-1]
    br = mask_bool[1:, 1:]

    sums = tl.astype(np.uint8) + tr.astype(np.uint8) + bl.astype(np.uint8) + br.astype(np.uint8)
    uniform = (sums == 0) | (sums == 4)
    horizontal_split = (tl == tr) & (bl == br) & (tl != bl)
    vertical_split = (tl == bl) & (tr == br) & (tl != tr)

    mixed = ~uniform
    if not np.any(mixed):
        return False

    stair_steps = mixed & ~(horizontal_split | vertical_split)
    mixed_count = int(np.count_nonzero(mixed))
    stair_ratio = np.count_nonzero(stair_steps) / mixed_count
    return stair_ratio >= 0.12


def _curves_to_svg_paths(plist, *, scale: float = 1.0) -> list[str]:
    """Convert potrace path list into SVG path 'd' strings.

    All curves are combined into a single compound SVG path that
    relies on fill-rule="evenodd" for correct hole rendering.
    """
    if not plist:
        return []

    parts: list[str] = []

    for curve in plist:
        fs = curve.start_point
        parts.append(f"M {fs.x * scale:.2f} {fs.y * scale:.2f}")

        for segment in curve.segments:
            if segment.is_corner:
                a = segment.c
                b = segment.end_point
                parts.append(
                    f"L {a.x * scale:.2f} {a.y * scale:.2f} "
                    f"L {b.x * scale:.2f} {b.y * scale:.2f}"
                )
            else:
                a = segment.c1
                b = segment.c2
                c = segment.end_point
                parts.append(
                    f"C {a.x * scale:.2f} {a.y * scale:.2f} "
                    f"{b.x * scale:.2f} {b.y * scale:.2f} "
                    f"{c.x * scale:.2f} {c.y * scale:.2f}"
                )

        parts.append("Z")

    # Return as a single compound path (multiple M...Z sub-paths)
    return [" ".join(parts)]
