"""Detect and remove small trademark symbols (TM, ®, ©) from the foreground mask.

Sports logos and other branding images often include tiny TM / ® / © marks
tucked into a corner.  These are a nuisance for multi-colour 3D-printing
because they add unwanted fine detail that is too small to print cleanly.

The algorithm works on the **foreground mask** (before colour quantization):

1. Find all connected components.
2. For each component, check whether it is:
   a. **Small** — area is less than *max_area_pct* of total foreground area
      (default 1.5 %).
   b. **In the margin** — its bounding-box centre falls within *margin_pct*
      of the image edges (default 12 %).
   c. **Isolated** — no other large component overlaps or closely neighbours
      it (optional, strengthens confidence).
3. Components matching (a) AND (b) are erased from the mask.

This is intentionally conservative: if a small element is not near an edge it
is kept, and if a near-edge element is large it is kept.
"""

from __future__ import annotations

import cv2
import numpy as np


def remove_tm_symbols(
    fg_mask: np.ndarray,
    *,
    max_area_pct: float = 1.5,
    margin_pct: float = 12.0,
) -> tuple[np.ndarray, int]:
    """Remove small trademark-like components from the margin of *fg_mask*.

    Args:
        fg_mask: (H, W) boolean foreground mask.
        max_area_pct: Maximum component area as a percentage of total
            foreground pixels.  Components larger than this are never removed.
        margin_pct: Width of the edge margin zone as a percentage of the
            image dimension.  A component's bounding-box centre must fall
            in this zone to be considered a TM candidate.

    Returns:
        cleaned_mask: (H, W) boolean mask with TM-like components removed.
        removed_count: Number of components that were removed.
    """
    h, w = fg_mask.shape[:2]
    mask_u8 = fg_mask.astype(np.uint8) * 255

    # Connected components (8-connectivity) -----------------------------------
    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    total_fg = int(np.count_nonzero(fg_mask))
    if total_fg == 0:
        return fg_mask.copy(), 0

    max_area = total_fg * (max_area_pct / 100.0)

    # Margin boundaries (in pixels) ------------------------------------------
    margin_x = w * (margin_pct / 100.0)
    margin_y = h * (margin_pct / 100.0)

    remove_labels: list[int] = []

    for lbl in range(1, num_labels):  # skip label 0 (background)
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]

        # Condition 1: small enough
        if area > max_area:
            continue

        # Condition 2: in the margin zone (centre of bbox near any edge)
        in_margin = (
            cx < margin_x
            or cx > w - margin_x
            or cy < margin_y
            or cy > h - margin_y
        )
        if not in_margin:
            continue

        remove_labels.append(lbl)

    # Build cleaned mask ------------------------------------------------------
    if not remove_labels:
        return fg_mask.copy(), 0

    cleaned = fg_mask.copy()
    for lbl in remove_labels:
        cleaned[label_img == lbl] = False

    return cleaned, len(remove_labels)
