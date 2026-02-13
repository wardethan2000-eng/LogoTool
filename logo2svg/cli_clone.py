"""Click CLI for the clone-parts command."""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command("clone-parts")
@click.argument("project_3mf", type=click.Path(exists=True))
@click.argument("layers_dir", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_3mf", type=click.Path(), default=None,
    help="Output 3MF path. Defaults to <project>_multicolor.3mf.",
)
@click.option(
    "--part-id", type=int, default=None,
    help="Object ID of the positioned SVG part. Auto-detected if omitted.",
)
@click.option(
    "--thickness", type=float, default=2.0,
    help="Extrusion thickness in mm for colour meshes. Default: 2.0.",
)
@click.option(
    "--scale", type=float, default=1.0,
    help="XY scale factor (pixels → mm). Default: 1.0.",
)
def clone_parts_cli(
    project_3mf: str,
    layers_dir: str,
    output_3mf: str | None,
    part_id: int | None,
    thickness: float,
    scale: float,
) -> None:
    """Inject aligned colour layers into a Bambu Studio project.

    Opens PROJECT_3MF (a Bambu Studio project with one SVG part already
    positioned on a model), converts the SVG files in LAYERS_DIR into
    3D meshes, and injects them with the same transform as the positioned
    part — guaranteeing perfect alignment.

    \b
    Typical workflow:
      1. logo2svg logo.png --output-dir ./layers/
      2. Open Bambu Studio → import helmet
      3. Add Part → SVG → position ONE layer on the helmet
      4. Save Project As → helmet.3mf
      5. logo2svg-clone helmet.3mf ./layers/ -o helmet_final.3mf
      6. Open helmet_final.3mf → all colours aligned
    """
    from .clone_parts import clone_parts

    try:
        out = clone_parts(
            project_3mf,
            layers_dir,
            output_3mf,
            part_id=part_id,
            thickness=thickness,
            scale=scale,
        )
        click.echo(f"✓ Written: {out}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
