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
    help="Treat INPUT_PATH as a directory and process all supported image files in it.",
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
@click.option(
    "--enhance-contrast", is_flag=True, default=False,
    help="Apply CLAHE contrast enhancement before quantization.",
)
@click.option(
    "--sharpen", is_flag=True, default=False,
    help="Apply edge sharpening before quantization.",
)
@click.option(
    "--denoise", is_flag=True, default=False,
    help="Apply noise reduction before quantization.",
)
@click.option(
    "--threemf", is_flag=True, default=False,
    help="Export a 3MF file for Bambu Studio multi-colour printing. "
         "Each colour becomes a separate 3D object with its own extruder.",
)
@click.option(
    "--import-svg", type=click.Path(exists=True), default=None,
    help="Import an SVG file as additional color layers after quantization.",
)
@click.option(
    "--add-text", type=str, default=None,
    help='Add text as a layer (e.g. --add-text "My Logo").',
)
@click.option(
    "--add-border", type=int, default=None,
    help="Add a canvas border of this width in pixels.",
)
@click.option(
    "--border-color", type=str, default="#000000",
    help="Color for the canvas border (hex). Default: #000000.",
)
@click.option(
    "--text-color", type=str, default="#000000",
    help="Color for the text layer (hex). Default: #000000.",
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
    enhance_contrast: bool,
    sharpen: bool,
    denoise: bool,
    import_svg: str | None,
    add_text: str | None,
    add_border: int | None,
    border_color: str,
    text_color: str,
    threemf: bool,
) -> None:
    """Convert a logo image into color-separated SVG files for 3D printing.

    Supports PNG, JPEG, WebP, BMP, and SVG input. JPEG images are
    automatically EXIF-rotated and lightly denoised. CMYK images are
    converted to RGB. SVG inputs are rasterized at high resolution.

    Each distinct color in the logo is output as its own SVG file containing
    only the vector paths for that color. The SVGs share the same viewBox so
    they align perfectly when imported into Bambu Studio or other slicers.

    \b
    Examples:
      logo2svg white_sox_logo.png
      logo2svg cubs_logo.png --colors 3 --output-dir ./cubs_svgs
      logo2svg logo.png --bg-color "#FFFFFF" --preview --combined
      logo2svg logo.png --target-colors "#FF0000,#FFFFFF,#000000"
      logo2svg logo.svg --colors 4
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
        threemf=threemf,
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

            # Use Session directly when extra features are requested
            if enhance_contrast or sharpen or denoise or import_svg or add_text or add_border:
                from .session import Session
                session = Session(
                    min_area=config.min_area,
                    alphamax=config.alphamax,
                    opttolerance=config.opttolerance,
                    turdsize=config.turdsize,
                    scale=config.scale,
                    width=config.width,
                )
                session.load(path, bg_color=config.bg_color)
                if config.remove_tm:
                    session.remove_tm()
                else:
                    session.ensure_square()

                # Preprocessing
                if enhance_contrast or sharpen or denoise:
                    if config.verbosity >= 1:
                        click.echo("  Applying image preprocessing...")
                    report = session.run_preprocessing(
                        contrast=enhance_contrast,
                        sharpen=sharpen,
                        denoise=denoise,
                    )
                    for rec in (report.recommendations or []):
                        if config.verbosity >= 1:
                            click.echo(f"    {rec}")

                # Quantize
                session.quantize(
                    n_colors=config.colors,
                    target_colors=config.target_colors,
                )

                # Post-quantize additions
                if import_svg:
                    count = session.import_svg(import_svg)
                    if config.verbosity >= 1:
                        click.echo(f"  Imported {count} layer(s) from SVG")

                if add_border:
                    session.add_canvas_border(width=add_border, color=border_color)
                    if config.verbosity >= 1:
                        click.echo(f"  Added {add_border}px canvas border")

                if add_text:
                    session.add_text(add_text, color=text_color)
                    if config.verbosity >= 1:
                        click.echo(f'  Added text layer: "{add_text}"')

                # Report mode
                if config.report:
                    for info in session.get_layers():
                        total_fg = sum(l.pixel_count for l in session.get_layers())
                        pct = 100.0 * info.pixel_count / total_fg if total_fg > 0 else 0
                        click.echo(
                            f"  {info.hex_color}  {info.color_name:<20s}  "
                            f"{info.pixel_count:>8d} px  ({pct:5.1f}%)"
                        )
                else:
                    output_files = session.export(
                        config.output_dir,
                        combined=config.combined,
                        preview=config.preview,
                        threemf=config.threemf,
                    )
                    if config.verbosity >= 1:
                        click.echo(f"  {len(output_files)} file(s) created.")
            else:
                process_single(path, config)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
