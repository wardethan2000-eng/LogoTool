"""Test smooth + approxPolyDP combinations."""
import numpy as np, cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from logo2svg.image_loader import load_image
from logo2svg.quantizer import quantize_colors
from logo2svg.layer_separator import separate_layers
from logo2svg.tracer import find_contours, _smooth_contour
from logo2svg.pipeline import _erode_mask, _recover_fringe_pixels

image, fg_mask = load_image(Path("white_sox.png"))
h, w = image.shape[:2]
fg_eroded = _erode_mask(fg_mask)
labels, centers = quantize_colors(image, fg_eroded)
labels = _recover_fringe_pixels(image, labels, centers, fg_mask, fg_eroded)
layers = separate_layers(labels, centers, fg_mask, min_area=100)

header = f"{'combo':>30} | pts | diff"
sep = f"{'-'*30} | --- | -----"

for li, layer in enumerate(layers):
    mask = layer["mask"]
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        continue

    biggest = max(contours, key=len)
    pts_raw = biggest.reshape(-1, 2).astype(np.float64)

    fill_ref = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fill_ref, [biggest], -1, 255, cv2.FILLED)

    print(f"\nLayer {li} ({layer['hex_color']}), biggest contour: {len(biggest)} pts")
    print(f"  {header}")
    print(f"  {sep}")

    for smooth_sigma in [0.0, 0.8, 1.2, 2.0]:
        pts = pts_raw.copy()
        if smooth_sigma > 0:
            pts = _smooth_contour(pts, sigma=smooth_sigma)

        for eps_cap in [0.5, 1.0, 1.5, 2.0, 3.0]:
            pcv = pts.reshape(-1, 1, 2).astype(np.float32)
            eps = min(0.005 * cv2.arcLength(pcv, closed=True), eps_cap)
            approx = cv2.approxPolyDP(pcv, eps, closed=True)
            n_pts = len(approx)

            fill_test = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(fill_test, [approx.astype(np.int32)], -1, 255, cv2.FILLED)
            diff = np.count_nonzero(fill_ref != fill_test)

            label = f"s={smooth_sigma:.1f} cap={eps_cap:.1f}"
            print(f"  {label:>30} | {n_pts:>3} | {diff:>5}")
