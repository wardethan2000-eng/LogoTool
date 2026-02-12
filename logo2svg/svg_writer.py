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
    scale: float = 1.0,
) -> list[Path]:
    """Write one SVG per color layer, optionally a combined SVG.

    Files are written into a subfolder named *base_name* inside
    *output_dir*.  Individual layers are named ``layer_1.svg``,
    ``layer_2.svg``, etc.  The combined file is ``combined.svg``.

    Args:
        base_name: Base filename (without extension) — used as subfolder name.
        layers: List of layer dicts with 'hex_color', 'color_name', 'svg_paths'.
        image_size: (height, width) of the source image.
        output_dir: Parent directory; a subfolder *base_name* is created inside.
        combined: Whether to also write a combined SVG.
        scale: Multiplier applied to viewBox dimensions and path coordinates.

    Returns:
        List of output file paths created.
    """
    sub_dir = output_dir / base_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_size
    output_files = []

    for i, layer in enumerate(layers, start=1):
        filename = f"layer_{i}.svg"
        filepath = sub_dir / filename

        _write_single_color_svg(
            filepath, layer["svg_paths"], layer["hex_color"], width, height,
            scale=scale,
        )
        output_files.append(filepath)

    if combined:
        combined_path = sub_dir / "combined.svg"
        _write_combined_svg(combined_path, layers, width, height, scale=scale)
        output_files.append(combined_path)

    return output_files


def _fmt(val: float) -> str:
    """Format a dimension value: integer string if whole, else 2 decimals."""
    if val == int(val):
        return str(int(val))
    return f"{val:.2f}"


def _scale_path(d: str, scale: float) -> str:
    """Scale all numeric coordinates in an SVG path 'd' string."""
    if scale == 1.0:
        return d
    import re

    def _repl(m: re.Match) -> str:
        return f"{float(m.group()) * scale:.2f}"

    return re.sub(r"-?\d+(?:\.\d+)?", _repl, d)


def _write_single_color_svg(
    path: Path,
    svg_paths: list[str],
    hex_color: str,
    width: int,
    height: int,
    scale: float = 1.0,
) -> None:
    """Write one SVG file containing all paths for a single color."""
    sw = width * scale
    sh = height * scale
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{_fmt(sw)}px", f"{_fmt(sh)}px"),
        viewBox=f"0 0 {_fmt(sw)} {_fmt(sh)}",
    )
    dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

    # Invisible bounding rectangle forces slicers (e.g. Bambu Studio) to
    # register the full canvas, so all per-color SVGs align correctly when
    # imported as separate files.
    dwg.add(dwg.rect(insert=(0, 0), size=(_fmt(sw), _fmt(sh)),
                      fill="none", stroke="none"))

    for d in svg_paths:
        dwg.add(dwg.path(d=_scale_path(d, scale), fill=hex_color, fill_rule="evenodd", stroke="none"))

    dwg.save(pretty=True)


def _write_combined_svg(
    path: Path,
    layers: list[dict],
    width: int,
    height: int,
    scale: float = 1.0,
) -> None:
    """Write a combined SVG with one <g> group per color."""
    sw = width * scale
    sh = height * scale
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{_fmt(sw)}px", f"{_fmt(sh)}px"),
        viewBox=f"0 0 {_fmt(sw)} {_fmt(sh)}",
    )
    dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

    for i, layer in enumerate(layers, start=1):
        hex_clean = layer["hex_color"].lstrip("#")
        group = dwg.g(id=f"color_{hex_clean}")
        for d in layer["svg_paths"]:
            group.add(
                dwg.path(d=_scale_path(d, scale), fill=layer["hex_color"], fill_rule="evenodd", stroke="none")
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

    preview_path = output_dir / base_name / f"preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), canvas)
    return preview_path
