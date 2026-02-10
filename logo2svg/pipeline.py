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
from .tracer import trace_mask_to_svg_paths


def _erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a boolean mask to strip anti-aliased fringe at background boundary."""
    import cv2
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    return eroded.astype(bool)


def _recover_fringe_pixels(
    image: np.ndarray,
    labels: np.ndarray,
    centers_rgb: np.ndarray,
    fg_mask: np.ndarray,
    fg_mask_eroded: np.ndarray,
) -> np.ndarray:
    """Assign fringe pixels (in fg_mask but not fg_mask_eroded) to nearest cluster.

    Fringe pixels are excluded from K-means to prevent anti-aliased colors
    from creating spurious clusters, but we recover them here by assigning
    each to the nearest cluster center in RGB space.
    """
    fringe = fg_mask & ~fg_mask_eroded
    if not np.any(fringe):
        return labels

    fringe_pixels = image[fringe].astype(np.float64)
    centers = centers_rgb.astype(np.float64)
    # Distance to each cluster center
    distances = np.linalg.norm(
        fringe_pixels[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2,
    )
    nearest = distances.argmin(axis=1).astype(np.int32)

    labels = labels.copy()
    labels[fringe] = nearest
    return labels


@dataclass
class PipelineConfig:
    """All user-configurable parameters for a single conversion run."""

    colors: int | None = None
    output_dir: Path = field(default_factory=lambda: Path("."))
    min_area: int = 100
    combined: bool = False
    bg_color: str | None = None
    preview: bool = False
    # Potrace parameters
    alphamax: float = 1.0
    opttolerance: float = 0.2


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
    # Erode the fg_mask for K-means only — this prevents anti-aliased
    # boundary pixels from creating spurious color clusters.
    click.echo("  Quantizing colors...")
    fg_mask_eroded = _erode_mask(fg_mask, iterations=1)
    labels, centers_rgb = quantize_colors(image, fg_mask_eroded, n_colors=config.colors)
    n_colors = len(centers_rgb)
    click.echo(f"  Detected {n_colors} colors")

    # Recover fringe pixels by assigning them to their nearest cluster
    labels = _recover_fringe_pixels(image, labels, centers_rgb, fg_mask, fg_mask_eroded)

    # Stage 3: Separate into per-color masks (using FULL fg_mask, not eroded)
    click.echo("  Separating color layers...")
    layers = separate_layers(labels, centers_rgb, fg_mask, config.min_area)
    click.echo(f"  Created {len(layers)} color layers")

    # Stage 4: Trace masks to SVG paths (Potrace)
    click.echo("  Tracing masks to vector paths (potrace)...")
    for layer in layers:
        layer["svg_paths"] = trace_mask_to_svg_paths(
            layer["mask"],
            turdsize=2,
            alphamax=config.alphamax,
            opticurve=True,
            opttolerance=config.opttolerance,
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


# Supported image file extensions (Pillow can load all of these)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def process_batch(input_dir: Path, config: PipelineConfig) -> list[Path]:
    """Process all image files (PNG, JPEG) in a directory.

    Returns:
        Combined list of all output file paths created.
    """
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        click.echo(f"No image files found in {input_dir}")
        return []

    click.echo(f"Batch processing {len(image_files)} image files from {input_dir}")
    all_outputs: list[Path] = []

    for image_file in image_files:
        try:
            outputs = process_single(image_file, config)
            all_outputs.extend(outputs)
        except Exception as e:
            click.echo(f"  Error processing {image_file.name}: {e}")

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
