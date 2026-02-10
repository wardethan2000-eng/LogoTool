# logo2svg

Convert PNG and JPEG logos into color-separated SVG files for multi-color 3D printing.

## What & Why

When 3D-printing a multi-color logo (for example on a Bambu Studio or PrusaSlicer workflow), you need **one SVG per color** — each containing only the vector paths for that color, all sharing the same `viewBox` so they align perfectly when imported as separate layers in the slicer.

**logo2svg** automates this:

1. Load a raster logo image (PNG or JPEG).
2. Detect and remove the background (alpha, corner-sampling, or explicit `--bg-color`).
3. Quantize the remaining foreground colors via K-means clustering in CIELAB space.
4. Generate a clean binary mask for each detected color.
5. Trace each mask to optimised cubic Bézier paths using [Potrace](http://potrace.sourceforge.net/).
6. Write one SVG per color — ready to drag into your slicer.

## Installation

```bash
# From the project directory
pip install .

# Or in development/editable mode
pip install -e ".[dev]"
```

After installation the `logo2svg` command is available on your PATH.

## Supported Input Formats

- **PNG** (recommended) — supports transparency for clean background removal.
- **JPEG** — works but compression artefacts may create spurious color clusters. A warning is printed when JPEG input is detected.

> **Note:** `.webp` and `.bmp` files may work (Pillow can open them) but are untested. Convert to PNG for best results.

## CLI Usage

```
logo2svg INPUT_PATH [OPTIONS]
```

### Examples

```bash
# Basic usage — auto-detect colors, write SVGs to current directory
logo2svg my_logo.png

# Specify exactly 3 colors, output to a subdirectory
logo2svg cubs_logo.png --colors 3 --output-dir ./cubs_svgs

# Explicit background color, combined SVG, and preview image
logo2svg logo.png --bg-color "#FFFFFF" --preview --combined

# Use exact filament colors instead of auto-detection
logo2svg logo.png --target-colors "#FF0000,#FFFFFF,#000000"

# Batch-process every image in a directory
logo2svg ./logos/ --batch

# Preview detected colors without writing SVGs
logo2svg logo.png --report

# Scale output for a specific width (in mm)
logo2svg logo.png --width 80

# Quieter output (errors only)
logo2svg logo.png --quiet
```

### Key Options

| Option | Description |
|---|---|
| `--colors N` | Number of foreground colors to extract. Auto-detected via silhouette scoring if omitted (typically 2–5 for logos). |
| `--alphamax F` | Potrace corner detection threshold (0.0–1.334). Lower → more corners (sharper). Higher → more curves (smoother). Default: `1.0`. |
| `--opttolerance F` | Potrace curve optimisation tolerance. Lower → more faithful. Higher → fewer Bézier segments. Default: `0.2`. |
| `--bg-color HEX` | Background color to remove (e.g. `"#FFFFFF"`). Auto-detected if omitted. |
| `--combined` | Also output a single combined SVG with `<g>` groups per color. |
| `--preview` | Generate a preview PNG showing original, quantized, and per-color masks side-by-side. |
| `--target-colors` | Comma-separated hex colors (e.g. `"#FF0000,#FFFFFF,#000000"`). Assigns each pixel to the nearest target color instead of auto-detecting via K-means. |
| `--scale F` | Scale the SVG viewBox and path coordinates by a float multiplier (default `1.0`). |
| `--width F` | Set a target SVG width in mm. Overrides `--scale`. |
| `--report` | Print detected colors with pixel counts, then exit without writing SVGs. |
| `--verbose` | Print detailed per-stage diagnostics. |
| `--quiet` | Suppress all output except errors. |
| `--batch` | Treat `INPUT_PATH` as a directory and process all PNG/JPEG files in it. |
| `--turdsize N` | Potrace speckle suppression: discard components up to this many pixels. Default: `2`. |
| `--min-area N` | Minimum contour area in pixels. Smaller regions are filtered as noise. Default: `100`. |

## GUI

logo2svg includes an optional graphical interface built with PyQt6.

```bash
# Launch via the CLI flag
logo2svg logo.png --gui

# Or use the dedicated entry point (no image argument required)
logo2svg-gui
```

The GUI provides an interactive preview of the pipeline stages, lets you adjust
color count, Potrace parameters, and target colors, and writes SVGs on demand.
Install the GUI extra to get PyQt6:

```bash
pip install -e ".[gui]"
```

## Legacy Directory

The `legacy/` directory contains modules that were replaced during the migration
from OpenCV contour tracing to Potrace:

- **`bezier_fit.py`** — Catmull-Rom spline → cubic Bézier fitting (replaced by Potrace).
- **`test_bezier_fit.py`** — Tests for the above.
- **`quantizer_hue.py`** — Hue-based color quantizer (never shipped; CIELAB K-means proved more robust).

See [`legacy/README.md`](legacy/README.md) for details.

## How It Works

### Pipeline Stages

1. **Image Loading** (`image_loader.py`): Opens the image via Pillow and determines which pixels are foreground. Background detection uses alpha transparency, corner-pixel sampling, or an explicit `--bg-color` override. An edge-connected flood-fill ensures that interior regions matching the background color are preserved (e.g. white text inside a logo on a white background).

2. **Color Quantization** (`quantizer.py`): Foreground pixels are converted to CIELAB colour space and clustered with K-means. When `--colors` is omitted, the optimal *k* is auto-detected by sweeping k = 2…8 and picking the highest silhouette score. Anti-aliased fringe pixels at the foreground boundary are temporarily eroded before clustering, then recovered by nearest-centre assignment.

3. **Layer Separation** (`layer_separator.py`): Each K-means cluster becomes a binary mask. Morphological closing fills tiny holes; connected-component filtering removes speckles below `--min-area`. Overlap between layers (caused by morphological expansion) is resolved by deferring to the original K-means label.

4. **Vector Tracing** (`tracer.py`): Each binary mask is traced to cubic Bézier SVG paths via the Potrace algorithm. Potrace handles optimal polygon decomposition, corner vs. curve detection (`--alphamax`), mathematically optimal Bézier fitting, and segment merging (`--opttolerance`).

5. **SVG Output** (`svg_writer.py`): One SVG file per colour is written with a shared `viewBox` matching the source image dimensions. Optionally a combined SVG (one `<g>` per colour) and a preview PNG are generated.

## License

MIT — see [LICENSE](LICENSE).
