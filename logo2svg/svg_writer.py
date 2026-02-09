"""SVG file output: per-color SVGs, combined SVG, and preview PNG."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import svgwrite


def write_svg_files(
    base_name: str,
    layers: list[dict],
    image_size: tuple[int, int],
    output_dir: Path,
    combined: bool = False,
) -> list[Path]:
    """Write one SVG per color layer, optionally a combined SVG.

    Args:
        base_name: Base filename (without extension) for output files.
        layers: List of layer dicts with 'hex_color', 'color_name', 'svg_paths'.
        image_size: (height, width) of the source image.
        output_dir: Directory to write SVG files to.
        combined: Whether to also write a combined SVG.

    Returns:
        List of output file paths created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_size
    output_files = []

    for layer in layers:
        hex_clean = layer["hex_color"].lstrip("#")
        filename = f"{base_name}_color_{hex_clean}.svg"
        filepath = output_dir / filename

        _write_single_color_svg(filepath, layer["svg_paths"], layer["hex_color"], width, height)
        output_files.append(filepath)

    if combined:
        combined_path = output_dir / f"{base_name}_combined.svg"
        _write_combined_svg(combined_path, layers, width, height)
        output_files.append(combined_path)

    return output_files


def _write_single_color_svg(
    path: Path,
    svg_paths: list[str],
    hex_color: str,
    width: int,
    height: int,
) -> None:
    """Write one SVG file containing all paths for a single color."""
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
    )
    dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

    for d in svg_paths:
        dwg.add(dwg.path(d=d, fill=hex_color, fill_rule="evenodd", stroke="none"))

    dwg.save(pretty=True)


def _write_combined_svg(
    path: Path,
    layers: list[dict],
    width: int,
    height: int,
) -> None:
    """Write a combined SVG with one <g> group per color."""
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
    )
    dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

    for layer in layers:
        hex_clean = layer["hex_color"].lstrip("#")
        # Use id to encode both hex and color name (data- attributes fail svgwrite validation)
        group = dwg.g(id=f"color_{hex_clean}_{layer['color_name']}")

        for d in layer["svg_paths"]:
            group.add(
                dwg.path(d=d, fill=layer["hex_color"], fill_rule="evenodd", stroke="none")
            )

        dwg.add(group)

    dwg.save(pretty=True)


def write_preview(
    base_name: str,
    original_image: np.ndarray,
    layers: list[dict],
    image_size: tuple[int, int],
    output_dir: Path,
) -> Path:
    """Generate a preview PNG showing original, quantized, and each color mask.

    Layout: [Original] [Quantized] [Color 1 mask] [Color 2 mask] ...
    """
    height, width = image_size
    n_panels = 2 + len(layers)  # original + quantized + one per color
    panel_width = width
    canvas_width = panel_width * n_panels
    canvas = np.zeros((height, canvas_width, 3), dtype=np.uint8)

    # Panel 1: Original image (convert RGB to BGR for OpenCV)
    canvas[:, :panel_width] = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Panel 2: Quantized image (reconstruct from layers)
    quantized = np.zeros((height, width, 3), dtype=np.uint8)
    for layer in layers:
        mask_bool = layer["mask"] > 0
        r, g, b = layer["rgb"]
        quantized[mask_bool] = [b, g, r]  # BGR for OpenCV
    canvas[:, panel_width : 2 * panel_width] = quantized

    # Panels 3+: Individual color masks
    for i, layer in enumerate(layers):
        offset = (2 + i) * panel_width
        mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
        mask_bool = layer["mask"] > 0
        r, g, b = layer["rgb"]
        mask_colored[mask_bool] = [b, g, r]  # BGR
        canvas[:, offset : offset + panel_width] = mask_colored

    preview_path = output_dir / f"{base_name}_preview.png"
    cv2.imwrite(str(preview_path), canvas)
    return preview_path
