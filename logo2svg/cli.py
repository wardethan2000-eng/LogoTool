"""Click CLI interface for logo2svg."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .pipeline import PipelineConfig, process_batch, process_single


@click.command()
@click.argument("input_path", type=click.Path())
@click.option(
    "--colors", type=int, default=None,
    help="Number of colors to extract. Auto-detected if omitted (typically 2-5 for logos).",
)
@click.option(
    "--output-dir", type=click.Path(), default=None,
    help="Output directory for SVG files. Defaults to current directory.",
)
@click.option(
    "--min-area", type=int, default=100,
    help="Minimum contour area in pixels. Smaller regions are filtered as noise.",
)
@click.option(
    "--combined", is_flag=True, default=False,
    help="Also output a single combined SVG with named groups per color.",
)
@click.option(
    "--bg-color", type=str, default=None,
    help="Background color to remove as hex (e.g. '#FFFFFF'). Auto-detected if omitted.",
)
@click.option(
    "--preview", is_flag=True, default=False,
    help="Generate a preview PNG showing original, quantized, and per-color masks.",
)
@click.option(
    "--batch", is_flag=True, default=False,
    help="Treat INPUT_PATH as a directory and process all image files (PNG, JPEG) in it.",
)
@click.option(
    "--alphamax", type=float, default=1.0,
    help="Potrace corner detection threshold (0.0-1.334). "
         "Lower = more corners (sharper). Higher = more curves (smoother). Default: 1.0.",
)
@click.option(
    "--opttolerance", type=float, default=0.2,
    help="Potrace curve optimisation tolerance. "
         "Lower = more faithful. Higher = fewer bezier segments. Default: 0.2.",
)
def main(
    input_path: str,
    colors: int | None,
    output_dir: str | None,
    min_area: int,
    combined: bool,
    bg_color: str | None,
    preview: bool,
    batch: bool,
    alphamax: float,
    opttolerance: float,
) -> None:
    """Convert a PNG or JPEG logo into color-separated SVG files for 3D printing.

    Each distinct color in the logo is output as its own SVG file containing
    only the vector paths for that color. The SVGs share the same viewBox so
    they align perfectly when imported into Bambu Studio or other slicers.

    \b
    Examples:
      python logo2svg.py white_sox_logo.png
      python logo2svg.py cubs_logo.png --colors 3 --output-dir ./cubs_svgs
      python logo2svg.py logo.png --bg-color "#FFFFFF" --preview --combined
      python logo2svg.py ./logos/ --batch
    """
    path = Path(input_path)

    # Validate inputs
    if colors is not None and colors < 1:
        click.echo("Error: --colors must be at least 1.", err=True)
        sys.exit(1)



    # Resolve output directory
    if output_dir is not None:
        out_path = Path(output_dir)
    elif batch:
        out_path = path if path.is_dir() else path.parent
    else:
        out_path = Path(".")

    config = PipelineConfig(
        colors=colors,
        output_dir=out_path,
        min_area=min_area,
        combined=combined,
        bg_color=bg_color,
        preview=preview,
        alphamax=alphamax,
        opttolerance=opttolerance,
    )

    try:
        if batch:
            if not path.is_dir():
                click.echo(f"Error: {path} is not a directory. --batch requires a directory.", err=True)
                sys.exit(1)
            process_batch(path, config)
        else:
            if not path.is_file():
                click.echo(f"Error: {path} does not exist or is not a file.", err=True)
                sys.exit(1)
            process_single(path, config)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
