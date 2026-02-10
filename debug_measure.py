"""Measure pixel-level impact of each pipeline stage."""
import numpy as np, cv2
from pathlib import Path
from logo2svg.image_loader import load_image
from logo2svg.quantizer import quantize_colors
from logo2svg.layer_separator import separate_layers
from logo2svg.tracer import find_contours, trace_to_svg_paths
from logo2svg.session import Session

image, fg_mask = load_image(Path("white_sox.png"))
h, w = image.shape[:2]
fg_eroded = Session._erode_mask(fg_mask)
labels, centers = quantize_colors(image, fg_eroded)
labels = Session._recover_fringe_pixels(image, labels, centers, fg_mask, fg_eroded)
layers = separate_layers(labels, centers, fg_mask, min_area=100)

# === 1. Morphological cleanup pixel loss ===
print("=== Morphological cleanup impact ===")
for k in range(len(centers)):
    raw_mask = ((labels == k) & fg_mask).astype(np.uint8) * 255
    raw_px = np.count_nonzero(raw_mask)
    # Find matching layer
    matching = [l for l in layers if l["cluster_idx"] == k]
    if matching:
        morph_px = np.count_nonzero(matching[0]["mask"])
        lost = raw_px - morph_px
        print(f"  Cluster {k}: raw={raw_px}, after_morph={morph_px}, lost={lost} ({100*lost/max(raw_px,1):.1f}%)")

# === 2. approxPolyDP pixel loss at different epsilon caps ===
print("\n=== approxPolyDP simplification: pixel difference from original contour fill ===")
for li, layer in enumerate(layers):
    mask = layer["mask"]
    contours, hier = find_contours(mask, smooth=0.0)
    hex_c = layer["hex_color"]
    print(f"\nLayer {li} ({hex_c}): {len(contours)} contours")
    
    for ci, cnt in enumerate(contours[:6]):
        n_raw = len(cnt)
        perim = cv2.arcLength(cnt, closed=True)
        
        fill_orig = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(fill_orig, [cnt], -1, 255, cv2.FILLED)
        orig_area = np.count_nonzero(fill_orig)
        
        if n_raw < 20:
            continue  # skip tiny contours
        
        results = []
        for cap in [1.0, 1.5, 2.0, 3.0, 999.0]:
            eps = min(0.005 * perim, cap)
            approx = cv2.approxPolyDP(cnt, eps, closed=True)
            n_simp = len(approx)
            
            fill_simp = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(fill_simp, [approx], -1, 255, cv2.FILLED)
            
            diff = np.count_nonzero(fill_orig != fill_simp)
            results.append(f"cap={cap if cap < 999 else 'none':>4}: {n_simp:>4}pts, diff={diff:>5}px")
        
        print(f"  cnt[{ci}] {n_raw}pts perim={perim:.0f}  |  " + "  |  ".join(results))

# === 3. Bezier curve vs polygon fill (legacy — requires legacy/bezier_fit.py) ===
print("\n=== Bezier curves vs simplified polygon: pixel difference ===")
import sys, importlib.util
_bf_path = str(Path(__file__).parent / "legacy" / "bezier_fit.py")
_bf_spec = importlib.util.spec_from_file_location("bezier_fit", _bf_path)
bezier_fit = importlib.util.module_from_spec(_bf_spec)
_bf_spec.loader.exec_module(bezier_fit)

for li, layer in enumerate(layers):
    mask = layer["mask"]
    contours, hier = find_contours(mask, smooth=0.0)
    hex_c = layer["hex_color"]
    
    for ci, cnt in enumerate(contours[:4]):
        n_raw = len(cnt)
        if n_raw < 20:
            continue
        
        perim = cv2.arcLength(cnt, closed=True)
        eps = min(0.005 * perim, 3.0)
        approx = cv2.approxPolyDP(cnt, eps, closed=True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        
        if len(pts) < 3:
            continue
        
        # Fill from polygon
        fill_poly = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(fill_poly, [approx], -1, 255, cv2.FILLED)
        
        # Fill from bezier curves
        segments = bezier_fit.fit_curve_closed(pts, max_error=2.0)
        bezier_pts = []
        for seg in segments:
            for ti in range(20):
                t = ti / 20
                s = 1 - t
                pt = s**3*seg[0] + 3*s**2*t*seg[1] + 3*s*t**2*seg[2] + t**3*seg[3]
                bezier_pts.append(pt.astype(np.int32))
        
        if bezier_pts:
            bezier_arr = np.array(bezier_pts).reshape(-1, 1, 2)
            fill_bezier = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(fill_bezier, [np.array(bezier_pts)])
            
            poly_area = np.count_nonzero(fill_poly)
            bezier_area = np.count_nonzero(fill_bezier)
            diff = np.count_nonzero(fill_poly != fill_bezier)
            print(f"  L{li} cnt[{ci}] {len(pts)}pts: poly={poly_area}, bezier={bezier_area}, diff={diff}")

# === 4. Full pipeline pixel accuracy ===
print("\n=== Full pipeline: mask vs SVG-rendered pixel accuracy ===")
for sigma in [0.0, 0.8, 1.4]:
    total_mask = 0
    total_overlap = 0
    total_mask_only = 0
    total_svg_only = 0
    
    for layer in layers:
        mask = layer["mask"]
        contours, hier = find_contours(mask, smooth=sigma)
        
        # Fill contours to get what the traced contours cover
        filled = np.zeros((h, w), dtype=np.uint8)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED, hierarchy=hier)
        
        # Now do approxPolyDP + bezier (full trace pipeline)
        # NOTE: simplify= and smooth= are accepted by the legacy wrapper but
        # silently ignored — it re-rasterises contours and calls Potrace.
        paths = trace_to_svg_paths(contours, hier, tolerance=2.0, smooth=sigma)
        
        # Re-render the SVG paths as polygons
        svg_filled = np.zeros((h, w), dtype=np.uint8)
        for path_d in paths:
            poly = _svg_path_to_polygon(path_d)
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(svg_filled, [poly.astype(np.int32)])
        
        m = mask > 0
        s = svg_filled > 0
        total_mask += np.count_nonzero(m)
        total_overlap += np.count_nonzero(m & s)
        total_mask_only += np.count_nonzero(m & ~s)
        total_svg_only += np.count_nonzero(~m & s)
    
    pct = 100 * total_overlap / max(total_mask, 1)
    print(f"  sigma={sigma:.1f}: coverage={pct:.2f}%, mask_only={total_mask_only}, svg_only={total_svg_only}")


def _svg_path_to_polygon(path_d, num_samples=20):
    points = []
    tokens = path_d.replace(",", " ").split()
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd == "M":
            points.append([float(tokens[i+1]), float(tokens[i+2])]); i += 3
        elif cmd == "C":
            p0 = np.array(points[-1] if points else [0,0])
            p1 = np.array([float(tokens[i+1]), float(tokens[i+2])])
            p2 = np.array([float(tokens[i+3]), float(tokens[i+4])])
            p3 = np.array([float(tokens[i+5]), float(tokens[i+6])])
            for ti in range(1, num_samples+1):
                t = ti/num_samples; s = 1-t
                points.append((s**3*p0 + 3*s**2*t*p1 + 3*s*t**2*p2 + t**3*p3).tolist())
            i += 7
        elif cmd == "L":
            points.append([float(tokens[i+1]), float(tokens[i+2])]); i += 3
        elif cmd == "Z": i += 1
        else: i += 1
    return np.array(points) if len(points) >= 3 else None
