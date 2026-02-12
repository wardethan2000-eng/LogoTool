"""SVG file import: parse an SVG and rasterize it as colour layer(s).

Allows users to import external SVG files (logos, icons, decorations
created in Inkscape / Illustrator / Figma) as new colour layers in
the pipeline.  Each distinct fill colour in the SVG becomes a
separate layer mask.

The import process:
1. Rasterize the SVG at the target canvas size using OpenCV / cairosvg
2. Extract unique fill colours from the rasterized output
3. Create one layer dict per colour (same format as pipeline layers)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from .color_utils import hex_to_rgb, nearest_color_name, rgb_to_hex


def _parse_svg_colors(svg_path: Path) -> list[str]:
    """Extract unique fill colours from an SVG file.

    Returns a list of hex colour strings found in fill attributes.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    colors = set()

    def _extract_fill(element):
        # Check fill attribute
        fill = element.get("fill", "")
        if fill and fill.lower() not in ("none", "transparent", ""):
            norm = _normalize_color(fill)
            if norm:
                colors.add(norm)

        # Check style attribute for fill
        style = element.get("style", "")
        if style:
            match = re.search(r"fill\s*:\s*([^;]+)", style)
            if match:
                val = match.group(1).strip()
                if val.lower() not in ("none", "transparent", ""):
                    norm = _normalize_color(val)
                    if norm:
                        colors.add(norm)

        for child in element:
            _extract_fill(child)

    _extract_fill(root)
    return sorted(colors)


def _normalize_color(color_str: str) -> str | None:
    """Normalize a CSS colour to uppercase hex."""
    color_str = color_str.strip().lower()

    # Already hex
    if color_str.startswith("#"):
        hex_val = color_str[1:]
        if len(hex_val) == 3:
            hex_val = "".join(c * 2 for c in hex_val)
        if len(hex_val) == 6:
            return f"#{hex_val.upper()}"
        return None

    # rgb(r, g, b)
    match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"

    # Named colours (basic set)
    _NAMED = {
        "black": "#000000", "white": "#FFFFFF", "red": "#FF0000",
        "green": "#008000", "blue": "#0000FF", "yellow": "#FFFF00",
        "cyan": "#00FFFF", "magenta": "#FF00FF", "orange": "#FFA500",
        "purple": "#800080", "gray": "#808080", "grey": "#808080",
        "pink": "#FFC0CB", "brown": "#A52A2A", "navy": "#000080",
        "lime": "#00FF00", "teal": "#008080", "maroon": "#800000",
    }
    if color_str in _NAMED:
        return _NAMED[color_str]

    return None


def _parse_svg_dimensions(svg_path: Path) -> tuple[int, int]:
    """Extract width and height from an SVG file.

    Returns (width, height) in pixels.  Falls back to viewBox parsing.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def _parse_dim(val: str | None) -> float | None:
        if val is None:
            return None
        # Strip units
        val = re.sub(r"(px|pt|em|rem|%|mm|cm|in)$", "", val.strip())
        try:
            return float(val)
        except ValueError:
            return None

    w = _parse_dim(root.get("width"))
    h = _parse_dim(root.get("height"))

    if w and h:
        return int(w), int(h)

    # Fall back to viewBox
    vb = root.get("viewBox", "")
    parts = re.split(r"[\s,]+", vb.strip())
    if len(parts) == 4:
        try:
            return int(float(parts[2])), int(float(parts[3]))
        except ValueError:
            pass

    return 512, 512  # sensible default


def rasterize_svg(
    svg_path: Path,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Rasterize an SVG file to an RGBA numpy array.

    Attempts cairosvg first (best quality), falls back to a simple
    OpenCV-based rasterization if cairosvg is not available.

    Args:
        svg_path: Path to the SVG file.
        target_width: Desired output width in pixels.
        target_height: Desired output height in pixels.

    Returns:
        (H, W, 4) uint8 RGBA array.
    """
    try:
        import cairosvg
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=target_width,
            output_height=target_height,
        )
        arr = np.frombuffer(png_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("Failed to decode rasterized SVG")
        # Ensure RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # Convert BGRA -> RGBA
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # Resize to exact target
        if rgba.shape[:2] != (target_height, target_width):
            rgba = cv2.resize(
                rgba, (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        return rgba
    except ImportError:
        pass

    # Fallback: parse SVG colours and create simple filled masks
    return _simple_svg_rasterize(svg_path, target_width, target_height)


def _simple_svg_rasterize(
    svg_path: Path,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Simple fallback SVG rasterizer using shape parsing.

    This is a basic fallback when cairosvg is not installed.
    It parses SVG path/rect/circle elements and renders them onto
    a canvas using OpenCV.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    svg_w, svg_h = _parse_svg_dimensions(svg_path)
    scale_x = target_width / max(1, svg_w)
    scale_y = target_height / max(1, svg_h)

    canvas = np.zeros((target_height, target_width, 4), dtype=np.uint8)

    def _get_fill_rgba(elem) -> tuple[int, int, int, int] | None:
        fill = elem.get("fill", "")
        style = elem.get("style", "")
        if style:
            match = re.search(r"fill\s*:\s*([^;]+)", style)
            if match:
                fill = match.group(1).strip()
        if not fill or fill.lower() in ("none", "transparent"):
            return None
        norm = _normalize_color(fill)
        if norm:
            r, g, b = hex_to_rgb(norm)
            opacity = float(elem.get("opacity", elem.get("fill-opacity", "1")))
            return (r, g, b, int(opacity * 255))
        return None

    def _process_rects(root_elem):
        for tag_name in ["rect", "{http://www.w3.org/2000/svg}rect"]:
            for rect in root_elem.iter(tag_name):
                rgba = _get_fill_rgba(rect)
                if rgba is None:
                    continue
                x = float(rect.get("x", "0")) * scale_x
                y = float(rect.get("y", "0")) * scale_y
                w = float(rect.get("width", "0")) * scale_x
                h = float(rect.get("height", "0")) * scale_y
                pt1 = (int(x), int(y))
                pt2 = (int(x + w), int(y + h))
                r, g, b, a = rgba
                cv2.rectangle(canvas, pt1, pt2, (r, g, b, a), -1)

    def _process_circles(root_elem):
        for tag_name in ["circle", "{http://www.w3.org/2000/svg}circle"]:
            for circ in root_elem.iter(tag_name):
                rgba = _get_fill_rgba(circ)
                if rgba is None:
                    continue
                cx = int(float(circ.get("cx", "0")) * scale_x)
                cy = int(float(circ.get("cy", "0")) * scale_y)
                r_val = int(float(circ.get("r", "0")) * min(scale_x, scale_y))
                r, g, b, a = rgba
                cv2.circle(canvas, (cx, cy), r_val, (r, g, b, a), -1)

    _process_rects(root)
    _process_circles(root)

    return canvas


def import_svg_as_layers(
    svg_path: str | Path,
    canvas_height: int,
    canvas_width: int,
    *,
    color_override: str | None = None,
) -> list[dict]:
    """Import an SVG file as one or more colour layers.

    Each distinct colour found in the rasterized SVG becomes a separate
    layer dict compatible with the pipeline's layer format.

    Args:
        svg_path: Path to the SVG file.
        canvas_height: Height of the target canvas.
        canvas_width: Width of the target canvas.
        color_override: If set, import the entire SVG as a single layer
            with this hex colour (ignoring original colours).

    Returns:
        List of layer dicts with keys: rgb, hex_color, color_name, mask,
        cluster_idx, visible.
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")

    rgba = rasterize_svg(svg_path, canvas_width, canvas_height)
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]

    # Foreground = where alpha > 127
    fg = alpha > 127

    if not np.any(fg):
        return []

    if color_override:
        # Single layer with override colour
        r, g, b = hex_to_rgb(color_override)
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        mask[fg] = 255
        return [{
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": mask,
            "cluster_idx": -1,
            "visible": True,
        }]

    # Group pixels by colour
    fg_pixels = rgb[fg]  # (N, 3)
    # Quantize to reduce near-duplicate colours (round to nearest 8)
    quantized = (fg_pixels // 8) * 8
    unique_colors = np.unique(quantized, axis=0)

    layers = []
    for i, color in enumerate(unique_colors):
        r, g, b = int(color[0]), int(color[1]), int(color[2])

        # Create mask for pixels close to this quantized colour
        diff = np.abs(fg_pixels.astype(int) - color.astype(int))
        close = np.all(diff <= 12, axis=1)  # tolerance for anti-aliasing

        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        fg_indices = np.argwhere(fg)
        matching_indices = fg_indices[close]
        if len(matching_indices) == 0:
            continue
        mask[matching_indices[:, 0], matching_indices[:, 1]] = 255

        # Skip tiny regions
        if np.count_nonzero(mask) < 50:
            continue

        layers.append({
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": mask,
            "cluster_idx": -1,
            "visible": True,
        })

    return layers
