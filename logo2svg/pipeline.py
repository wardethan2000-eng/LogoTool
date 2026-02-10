"""Pipeline orchestrator: wires all processing stages together."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

from .image_loader import load_image
from .layer_separator import separate_layers
from .quantizer import quantize_colors
from .svg_writer import write_preview, write_svg_files
from .tracer import find_contours, trace_to_svg_paths


@dataclass
class PipelineConfig:
    """All user-configurable parameters for a single conversion run."""

    colors: int | None = None
    tolerance: float = 2.0
    output_dir: Path = field(default_factory=lambda: Path("."))
    min_area: int = 100
    combined: bool = False
    bg_color: str | None = None
    preview: bool = False
    simplify: float | None = None
    smooth: float = 1.4


def process_single(input_path: Path, config: PipelineConfig) -> list[Path]:
    """Full pipeline for one PNG file.

    Stages:
    1. Load image and detect background
    2. Quantize colors via K-means in CIELAB
    3. Generate per-color binary masks
    4. Trace contours to SVG path strings
    5. Write SVG files

    Returns:
        List of output file paths created.
    """
    click.echo(f"Processing: {input_path.name}")

    # Stage 1: Load image
    click.echo("  Loading image and detecting background...")
    image, fg_mask = load_image(input_path, config.bg_color)
    height, width = image.shape[:2]
    fg_count = np.count_nonzero(fg_mask)
    click.echo(f"  Image size: {width}x{height}, foreground pixels: {fg_count}")

    if fg_count == 0:
        click.echo("  Error: No foreground pixels detected. Try --bg-color to specify background.")
        return []

    # Stage 2: Quantize colors
    click.echo("  Quantizing colors...")
    labels, centers_rgb = quantize_colors(image, fg_mask, n_colors=config.colors)
    n_colors = len(centers_rgb)
    click.echo(f"  Detected {n_colors} colors")

    # Stage 3: Separate into per-color masks
    click.echo("  Separating color layers...")
    layers = separate_layers(labels, centers_rgb, fg_mask, config.min_area)
    click.echo(f"  Created {len(layers)} color layers")

    # Stage 4: Trace contours to SVG paths
    click.echo("  Tracing contours to vector paths...")
    for layer in layers:
        contours, hierarchy = find_contours(layer["mask"], smooth=config.smooth)
        layer["svg_paths"] = trace_to_svg_paths(
            contours, hierarchy,
            tolerance=config.tolerance,
            simplify=config.simplify,
            smooth=config.smooth,
        )
        n_paths = len(layer["svg_paths"])
        click.echo(f"    {layer['hex_color']} ({layer['color_name']}): {n_paths} paths")

    # Stage 5: Write SVGs
    click.echo("  Writing SVG files...")
    base_name = input_path.stem
    output_files = write_svg_files(
        base_name, layers, (height, width), config.output_dir, config.combined
    )

    # Optional preview
    if config.preview:
        click.echo("  Generating preview image...")
        preview_path = write_preview(
            base_name, image, layers, (height, width), config.output_dir
        )
        output_files.append(preview_path)

    # Print summary
    _print_summary(layers, output_files)

    return output_files


def process_batch(input_dir: Path, config: PipelineConfig) -> list[Path]:
    """Process all PNG files in a directory.

    Returns:
        Combined list of all output file paths created.
    """
    png_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.PNG"))
    if not png_files:
        click.echo(f"No PNG files found in {input_dir}")
        return []

    click.echo(f"Batch processing {len(png_files)} PNG files from {input_dir}")
    all_outputs: list[Path] = []

    for png_file in png_files:
        try:
            outputs = process_single(png_file, config)
            all_outputs.extend(outputs)
        except Exception as e:
            click.echo(f"  Error processing {png_file.name}: {e}")

    click.echo(f"\nBatch complete. {len(all_outputs)} files created.")
    return all_outputs


def _print_summary(layers: list[dict], output_files: list[Path]) -> None:
    """Print a summary of detected colors and output files."""
    click.echo("\n" + "=" * 50)
    click.echo("Summary")
    click.echo("=" * 50)

    click.echo(f"\nDetected colors ({len(layers)}):")
    for layer in layers:
        click.echo(f"  {layer['hex_color']}  →  {layer['color_name']}")

    click.echo(f"\nFiles created ({len(output_files)}):")
    for f in output_files:
        click.echo(f"  {f}")

    click.echo()
