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
@click.option(
    "--turdsize", type=int, default=2,
    help="Potrace speckle suppression: discard components up to this many pixels. Default: 2.",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Print detailed per-stage diagnostics.",
)
@click.option(
    "--quiet", is_flag=True, default=False,
    help="Suppress all output except errors.",
)
@click.option(
    "--target-colors", type=str, default=None,
    help='Comma-separated hex colors (e.g. "#FF0000,#FFFFFF,#000000"). '
         "Assign each pixel to the nearest target color instead of auto-detecting.",
)
@click.option(
    "--scale", type=float, default=1.0,
    help="Scale SVG viewBox and path coordinates by this multiplier. Default: 1.0.",
)
@click.option(
    "--width", type=float, default=None,
    help="Set target SVG width in mm. Overrides --scale.",
)
@click.option(
    "--report", is_flag=True, default=False,
    help="Print detected colors with pixel counts, then exit without writing SVGs.",
)
@click.option(
    "--gui", is_flag=True, default=False,
    help="Launch the graphical interface instead of processing on the command line.",
)
@click.option(
    "--keep-tm", is_flag=True, default=False,
    help="Keep small trademark symbols (TM/®) instead of auto-removing them from margins.",
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
    turdsize: int,
    verbose: bool,
    quiet: bool,
    target_colors: str | None,
    scale: float,
    width: float | None,
    report: bool,
    gui: bool,
    keep_tm: bool,
) -> None:
    """Convert a PNG or JPEG logo into color-separated SVG files for 3D printing.

    Each distinct color in the logo is output as its own SVG file containing
    only the vector paths for that color. The SVGs share the same viewBox so
    they align perfectly when imported into Bambu Studio or other slicers.

    \b
    Examples:
      logo2svg white_sox_logo.png
      logo2svg cubs_logo.png --colors 3 --output-dir ./cubs_svgs
      logo2svg logo.png --bg-color "#FFFFFF" --preview --combined
      logo2svg logo.png --target-colors "#FF0000,#FFFFFF,#000000"
      logo2svg ./logos/ --batch
      logo2svg logo.png --report
    """
    path = Path(input_path)

    # GUI mode — launch graphical interface
    if gui:
        from .gui import run_gui
        file_arg = str(path) if path.is_file() else None
        raise SystemExit(run_gui(file_arg))

    # Validate inputs
    if colors is not None and colors < 1:
        click.echo("Error: --colors must be at least 1.", err=True)
        sys.exit(1)

    if verbose and quiet:
        click.echo("Error: --verbose and --quiet are mutually exclusive.", err=True)
        sys.exit(1)

    # Parse target colors
    parsed_target_colors: list[str] | None = None
    if target_colors is not None:
        parsed_target_colors = [c.strip() for c in target_colors.split(",") if c.strip()]
        if not parsed_target_colors:
            click.echo("Error: --target-colors requires at least one hex color.", err=True)
            sys.exit(1)
        # Validate each hex color early so users get a clear message
        from .color_utils import hex_to_rgb
        for hex_val in parsed_target_colors:
            try:
                hex_to_rgb(hex_val)
            except ValueError:
                click.echo(
                    f"Error: invalid hex color '{hex_val}' in --target-colors. "
                    "Expected format: '#RRGGBB' (e.g. '#FF0000').",
                    err=True,
                )
                sys.exit(1)

    # Determine verbosity level
    verbosity = 1
    if quiet:
        verbosity = 0
    elif verbose:
        verbosity = 2

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
        turdsize=turdsize,
        verbosity=verbosity,
        target_colors=parsed_target_colors,
        scale=scale,
        width=width,
        report=report,
        remove_tm=not keep_tm,
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
