"""Pipeline orchestrator: wires all processing stages together."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

from .session import Session


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
    turdsize: int = 2
    # Verbosity: 0 = quiet, 1 = normal (default), 2 = verbose
    verbosity: int = 1
    # Target colours — skip K-means and assign to these exact colours
    target_colors: list[str] | None = None
    # Scale multiplier for SVG viewBox / paths
    scale: float = 1.0
    # Target width in mm (overrides scale)
    width: float | None = None
    # Report mode — print colours and exit without writing SVGs
    report: bool = False
    # Automatically remove small TM / ® symbols from logo margins
    remove_tm: bool = True


def _log(msg: str, config: PipelineConfig, level: int = 1) -> None:
    """Print a message if verbosity >= level."""
    if config.verbosity >= level:
        click.echo(msg)


def process_single(input_path: Path, config: PipelineConfig) -> list[Path]:
    """Full pipeline for one PNG file, delegating to a :class:`Session`.

    Stages:
    1. Load image and detect background
    2. Quantize colors via K-means in CIELAB (or assign to target colours)
    3. Generate per-color binary masks
    4. Trace contours to SVG path strings
    5. Write SVG files

    Returns:
        List of output file paths created.
    """
    _log(f"Processing: {input_path.name}", config)

    # Warn if the file extension isn't in the tested set
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _log(
            f"  Warning: '{input_path.suffix}' is not a tested format. "
            "For best results use PNG input.",
            config,
            level=0,
        )

    # JPEG artifact handling
    is_jpeg = input_path.suffix.lower() in (".jpg", ".jpeg")
    if is_jpeg:
        _log(
            "  Note: JPEG input detected. Applying light denoising to reduce "
            "compression artefact colour clusters.",
            config,
        )

    # Create a Session with the tunable parameters
    session = Session(
        min_area=config.min_area,
        alphamax=config.alphamax,
        opttolerance=config.opttolerance,
        turdsize=config.turdsize,
        scale=config.scale,
        width=config.width,
    )

    # Stage 1: Load image
    _log("  Loading image and detecting background...", config)
    session.load(input_path, bg_color=config.bg_color)

    # Auto-denoise JPEG to mitigate compression artefacts before quantization
    if is_jpeg:
        session.run_preprocessing(denoise=True, denoise_strength=7)

    height, width = session.image_size
    fg_count = int(np.count_nonzero(session.fg_mask))
    _log(f"  Image size: {width}x{height}, foreground pixels: {fg_count}", config)

    if fg_count == 0:
        _log("  Error: No foreground pixels detected. Try --bg-color to specify background.", config, level=0)
        return []

    # Stage 1b: Remove TM / ® symbols from margins
    if config.remove_tm:
        removed = session.remove_tm()
        if removed:
            _log(f"  Removed {removed} small trademark symbol(s) from margins", config)
    else:
        session.ensure_square()

    # Stage 2: Quantize colors
    _log("  Quantizing colors...", config)
    session.quantize(
        n_colors=config.colors,
        target_colors=config.target_colors,
    )
    layers_info = session.get_layers()
    n_colors = len(layers_info)
    if config.target_colors:
        _log(f"  Using {n_colors} target colors", config)
    else:
        _log(f"  Detected {n_colors} colors", config)

    # --report: print colour table and exit
    if config.report:
        _print_report_from_session(session, config)
        return []

    _log(f"  Created {n_colors} color layers", config)

    # Verbose: per-pixel label distribution
    if config.verbosity >= 2:
        for info in layers_info:
            _log(f"    {info.hex_color} pixels: {info.pixel_count}", config, level=2)

    # Stage 4: Trace masks to SVG paths (Potrace)
    _log("  Tracing masks to vector paths (potrace)...", config)
    session.trace()
    for layer in session._layers:
        n_paths = len(layer.get("svg_paths", []))
        _log(f"    {layer['hex_color']} ({layer['color_name']}): {n_paths} paths", config)

    # Stage 5: Write SVGs
    _log("  Writing SVG files...", config)
    output_files = session.export(
        config.output_dir,
        combined=config.combined,
        preview=config.preview,
    )

    # Print summary
    if config.verbosity >= 1:
        _print_summary(session._layers, output_files)

    return output_files


# Supported image file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}


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
        _log(f"No image files found in {input_dir}", config, level=0)
        return []

    _log(f"Batch processing {len(image_files)} image files from {input_dir}", config)
    all_outputs: list[Path] = []

    for image_file in image_files:
        try:
            outputs = process_single(image_file, config)
            all_outputs.extend(outputs)
        except Exception as e:
            _log(f"  Error processing {image_file.name}: {e}", config, level=0)

    _log(f"\nBatch complete. {len(all_outputs)} files created.", config)
    return all_outputs


def _print_report_from_session(session: Session, config: PipelineConfig) -> None:
    """Print detected colours from a :class:`Session` (--report mode)."""
    layers_info = session.get_layers()
    total_fg = sum(l.pixel_count for l in layers_info)

    click.echo("\nColor Report")
    click.echo("=" * 40)
    click.echo(f"Total foreground pixels: {total_fg}")
    click.echo()
    for info in layers_info:
        pct = 100.0 * info.pixel_count / total_fg if total_fg > 0 else 0
        click.echo(
            f"  {info.hex_color}  {info.color_name:<20s}  "
            f"{info.pixel_count:>8d} px  ({pct:5.1f}%)"
        )
    click.echo()


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
