#!/usr/bin/env python3
"""Debug script: saves intermediate pipeline images to diagnose shape fidelity.

NOTE: This is a one-time diagnostic script from the Potrace migration.
It requires a test image (e.g. white_sox.png) placed in the project root
directory.  It is not part of the automated test suite.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from logo2svg.image_loader import load_image
from logo2svg.session import Session
from logo2svg.quantizer import quantize_colors
from logo2svg.layer_separator import separate_layers
from logo2svg.tracer import find_contours

# Legacy helpers — shared stubs for functions removed when Potrace replaced
# the OpenCV pipeline.
from debug_utils import smooth_contour as _smooth_contour, compute_epsilon as _compute_epsilon

def _svg_path_to_polygon(path_d: str, num_samples: int = 20) -> np.ndarray | None:
    """Parse SVG path d string and convert curves to polygon points."""
    points = []
    tokens = path_d.replace(',', ' ').split()
    i = 0
    
    while i < len(tokens):
        cmd = tokens[i]
        if cmd == 'M':
            x, y = float(tokens[i+1]), float(tokens[i+2])
            points.append([x, y])
            i += 3
        elif cmd == 'C':
            x1, y1 = float(tokens[i+1]), float(tokens[i+2])
            x2, y2 = float(tokens[i+3]), float(tokens[i+4])
            x3, y3 = float(tokens[i+5]), float(tokens[i+6])
            p0 = np.array(points[-1] if points else [x1, y1])
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])
            for t_i in range(1, num_samples + 1):
                t = t_i / num_samples
                s = 1 - t
                pt = s**3*p0 + 3*s**2*t*p1 + 3*s*t**2*p2 + t**3*p3
                points.append(pt.tolist())
            i += 7
        elif cmd == 'L':
            x, y = float(tokens[i+1]), float(tokens[i+2])
            points.append([x, y])
            i += 3
        elif cmd == 'Z':
            i += 1
        else:
            i += 1
    
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.float64)


INPUT = Path(__file__).resolve().parent.parent / "white_sox.png"
OUT = Path(__file__).resolve().parent.parent / "debug_output"
OUT.mkdir(exist_ok=True)

# ── Stage 1: Load image ────────────────────────────────────────────
print("Stage 1: Loading image...")
image, fg_mask = load_image(INPUT)
h, w = image.shape[:2]
fg_count = np.count_nonzero(fg_mask)
print(f"  Size: {w}x{h}, foreground pixels: {fg_count} ({100*fg_count/(h*w):.1f}%)")

# Save the foreground mask (before fringe removal we can't easily get, so just save fg_mask)
cv2.imwrite(str(OUT / "01_fg_mask.png"), fg_mask.astype(np.uint8) * 255)

# Also show what fringe removal does — reload without fringe removal
from PIL import Image as PILImage
pil_image = PILImage.open(INPUT).convert("RGBA")
rgba = np.array(pil_image, dtype=np.uint8)
rgb = rgba[:, :, :3]
from logo2svg.image_loader import _detect_bg_from_corners, _remove_bg_color
bg_rgb = _detect_bg_from_corners(rgb)
if bg_rgb is not None:
    fg_mask_no_fringe = _remove_bg_color(rgb, bg_rgb, tolerance=30)
    cv2.imwrite(str(OUT / "01b_fg_mask_before_fringe_removal.png"),
                fg_mask_no_fringe.astype(np.uint8) * 255)
    # Difference: pixels eaten by fringe removal
    fringe_diff = fg_mask_no_fringe.astype(np.uint8) - fg_mask.astype(np.uint8)
    fringe_diff = np.clip(fringe_diff, 0, 1) * 255
    cv2.imwrite(str(OUT / "01c_fringe_removed_pixels.png"), fringe_diff.astype(np.uint8))
    fringe_count = np.count_nonzero(fringe_diff)
    print(f"  Fringe removal ate {fringe_count} pixels")

# ── Stage 2: Quantize ──────────────────────────────────────────────
print("\nStage 2: Quantizing colors...")
labels, centers_rgb = quantize_colors(image, fg_mask)
n_colors = len(centers_rgb)
print(f"  Detected {n_colors} colors: {[tuple(c) for c in centers_rgb]}")

# Save quantized image
quantized = np.full((h, w, 3), 255, dtype=np.uint8)
for k in range(n_colors):
    quantized[labels == k] = centers_rgb[k]
cv2.imwrite(str(OUT / "02_quantized.png"), cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))

# ── Stage 3: Layer separation ──────────────────────────────────────
print("\nStage 3: Separating layers...")
layers = separate_layers(labels, centers_rgb, fg_mask, min_area=100)
print(f"  Created {len(layers)} layers")
for i, layer in enumerate(layers):
    cv2.imwrite(str(OUT / f"03_mask_{i}_{layer['hex_color'].lstrip('#')}.png"), layer["mask"])
    px = np.count_nonzero(layer["mask"])
    print(f"  Layer {i} ({layer['hex_color']} {layer['color_name']}): {px} pixels")

# ── Stage 4: Contour tracing analysis ──────────────────────────────
print("\nStage 4: Contour analysis...")

for li, layer in enumerate(layers):
    mask = layer["mask"]
    hex_clean = layer['hex_color'].lstrip('#')
    
    # 4a: Raw contours (no blur, no smooth)
    raw_contours, raw_hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    raw_img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(raw_img, raw_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(OUT / f"04a_contours_raw_{li}_{hex_clean}.png"), raw_img)
    
    total_raw_pts = sum(len(c) for c in raw_contours)
    print(f"\n  Layer {li} ({layer['hex_color']}):")
    print(f"    Raw contours: {len(raw_contours)}, total points: {total_raw_pts}")
    
    # 4b: Mask after GaussianBlur (what find_contours does with smooth=1.4)
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.4)
    mask_blurred = (blurred > 127).astype(np.uint8) * 255
    blur_contours, blur_hier = cv2.findContours(mask_blurred, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    # Show mask difference (original vs blurred)
    diff_img = np.zeros((h, w, 3), dtype=np.uint8)
    diff_img[:, :, 1] = mask  # green = original
    diff_img[:, :, 2] = mask_blurred  # red = blurred
    # yellow = overlap, green = lost, red = gained
    cv2.imwrite(str(OUT / f"04b_mask_vs_blurred_{li}_{hex_clean}.png"), diff_img)
    
    lost = np.count_nonzero((mask > 0) & (mask_blurred == 0))
    gained = np.count_nonzero((mask == 0) & (mask_blurred > 0))
    print(f"    Mask blur: lost {lost} px, gained {gained} px")
    
    # 4c: Contours after coordinate smoothing + approxPolyDP (full pipeline)
    smooth_sigma = 1.4
    simplify = None
    pipeline_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for ci, cnt in enumerate(blur_contours):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        
        # Coordinate smoothing
        smoothed_pts = _smooth_contour(pts, sigma=smooth_sigma)
        
        # approxPolyDP
        pts_cv = smoothed_pts.reshape(-1, 1, 2).astype(np.float32)
        epsilon = _compute_epsilon(pts_cv, simplify)
        approx = cv2.approxPolyDP(pts_cv, epsilon, closed=True)
        simplified = approx.reshape(-1, 2)
        
        if ci == 0:  # log for biggest contour
            print(f"    Contour 0: {len(pts)} → smooth → {len(smoothed_pts)} → approxPolyDP(eps={epsilon:.2f}) → {len(simplified)} pts")
        
        # Draw simplified contour
        approx_int = simplified.astype(np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0) if ci % 2 == 0 else (0, 0, 255)
        cv2.drawContours(pipeline_img, [approx_int], -1, color, 1)
    
    cv2.imwrite(str(OUT / f"04c_contours_simplified_{li}_{hex_clean}.png"), pipeline_img)
    
    # 4d: Compare point counts at different epsilon values
    print(f"    approxPolyDP sensitivity test (contour 0):")
    if len(blur_contours) > 0:
        cnt0 = blur_contours[0].reshape(-1, 2).astype(np.float64)
        if len(cnt0) >= 3:
            smoothed0 = _smooth_contour(cnt0, sigma=smooth_sigma)
            pts_cv0 = smoothed0.reshape(-1, 1, 2).astype(np.float32)
            perim = cv2.arcLength(pts_cv0, closed=True)
            for pct in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                eps = pct * 0.01 * perim
                approx = cv2.approxPolyDP(pts_cv0, eps, closed=True)
                print(f"      {pct:.1f}% → eps={eps:.1f}, points: {len(approx)}")

    # 4e: Test without any smoothing at all
    nosmooth_img = np.zeros((h, w, 3), dtype=np.uint8)
    for ci, cnt in enumerate(raw_contours):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
        epsilon = _compute_epsilon(pts_cv, simplify)
        approx = cv2.approxPolyDP(pts_cv, epsilon, closed=True)
        approx_int = approx.reshape(-1, 2).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0) if ci % 2 == 0 else (0, 0, 255)
        cv2.drawContours(nosmooth_img, [approx_int], -1, color, 1)
    cv2.imwrite(str(OUT / f"04e_contours_no_smooth_{li}_{hex_clean}.png"), nosmooth_img)

# ── Stage 5: Render SVG paths back to raster for comparison ─────────
print("\nStage 5: Rendering SVG paths back to raster for pixel diff...")
from logo2svg.tracer import trace_to_svg_paths
import re

rendered = np.zeros((h, w, 3), dtype=np.uint8)

for li, layer in enumerate(layers):
    contours, hierarchy = find_contours(layer["mask"], smooth=1.4)
    # NOTE: simplify= and smooth= are accepted by the legacy wrapper but
    # silently ignored — it re-rasterises contours and calls Potrace.
    svg_paths = trace_to_svg_paths(contours, hierarchy, tolerance=2.0, simplify=None, smooth=1.4)
    
    hex_clean = layer['hex_color'].lstrip('#')
    r, g, b = layer['rgb']
    
    # Parse SVG paths and render with OpenCV
    for path_d in svg_paths:
        # Parse M/C/L/Z commands to get all bezier curve points
        # Sample each bezier at many t values to approximate rendering
        poly_points = _svg_path_to_polygon(path_d, num_samples=20)
        if poly_points is not None and len(poly_points) >= 3:
            pts = poly_points.astype(np.int32)
            cv2.fillPoly(rendered, [pts], (b, g, r))

cv2.imwrite(str(OUT / "05_svg_rendered.png"), rendered)
print("  Saved 05_svg_rendered.png")

# Create pixel difference image
original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# Make background white in the original for comparison
orig_for_diff = np.full_like(original_bgr, 255)
orig_for_diff[fg_mask] = original_bgr[fg_mask]

# Quantized for comparison
quant_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
for layer in layers:
    m = layer["mask"] > 0
    r, g, b = layer['rgb']
    quant_bgr[m] = [b, g, r]

diff_quant = cv2.absdiff(orig_for_diff, quant_bgr)
diff_svg = cv2.absdiff(quant_bgr, rendered)
cv2.imwrite(str(OUT / "05b_diff_orig_vs_quantized.png"), diff_quant * 3)  # amplify
cv2.imwrite(str(OUT / "05c_diff_quantized_vs_svg.png"), diff_svg * 3)

# Count non-zero pixels in SVG render vs mask
svg_fg = np.any(rendered > 0, axis=2)
mask_fg = np.zeros((h, w), dtype=bool)
for layer in layers:
    mask_fg |= layer["mask"] > 0
    
overlap = np.count_nonzero(svg_fg & mask_fg)
svg_only = np.count_nonzero(svg_fg & ~mask_fg)
mask_only = np.count_nonzero(~svg_fg & mask_fg)
print(f"  SVG vs Mask: overlap={overlap}, svg_only={svg_only}, mask_only={mask_only}")
print(f"  Mask coverage: {100*overlap/(overlap+mask_only):.1f}% of mask pixels covered by SVG")

print(f"\nAll debug images saved to {OUT}/")
