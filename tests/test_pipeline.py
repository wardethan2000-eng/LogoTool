"""End-to-end pipeline tests with synthetic images."""

import numpy as np
from pathlib import Path
from PIL import Image

from logo2svg.pipeline import PipelineConfig, process_single


def _create_two_color_logo(path: Path) -> None:
    """Create a synthetic 2-color logo PNG (red circle on blue square, transparent bg)."""
    size = 200
    img = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA, transparent bg

    # Blue square in center
    img[40:160, 40:160, 0] = 0    # R
    img[40:160, 40:160, 1] = 0    # G
    img[40:160, 40:160, 2] = 200  # B
    img[40:160, 40:160, 3] = 255  # A

    # Red circle in center
    y, x = np.ogrid[:size, :size]
    circle_mask = (x - 100) ** 2 + (y - 100) ** 2 <= 30 ** 2
    img[circle_mask, 0] = 200  # R
    img[circle_mask, 1] = 0    # G
    img[circle_mask, 2] = 0    # B
    img[circle_mask, 3] = 255  # A

    Image.fromarray(img, "RGBA").save(str(path))


def _create_three_color_logo(path: Path) -> None:
    """Create a synthetic 3-color logo PNG on white background."""
    size = 200
    img = np.full((size, size, 3), 255, dtype=np.uint8)  # White background

    # Green rectangle
    img[30:90, 30:170, :] = [0, 150, 0]

    # Blue rectangle
    img[110:170, 30:170, :] = [0, 0, 150]

    # Red square in the middle
    img[80:120, 80:120, :] = [200, 0, 0]

    Image.fromarray(img, "RGB").save(str(path))


def test_pipeline_two_color_logo(tmp_path):
    """End-to-end test: 2-color logo should produce 2 SVG files."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
    )

    output_files = process_single(png_path, config)

    # Should produce exactly 2 SVG files
    svg_files = [f for f in output_files if f.suffix == ".svg"]
    assert len(svg_files) == 2

    # Each file should exist and contain valid SVG
    for svg_file in svg_files:
        assert svg_file.exists()
        content = svg_file.read_text()
        assert "<svg" in content
        assert "viewBox" in content
        assert "<path" in content
        assert 'fill-rule="evenodd"' in content


def test_pipeline_with_combined(tmp_path):
    """Combined flag should produce an additional combined SVG."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
        combined=True,
    )

    output_files = process_single(png_path, config)

    svg_files = [f for f in output_files if f.suffix == ".svg"]
    # 2 per-color + 1 combined = 3
    assert len(svg_files) == 3

    combined_files = [f for f in svg_files if "combined" in f.name]
    assert len(combined_files) == 1

    # Combined SVG should have groups
    content = combined_files[0].read_text()
    assert "<g " in content


def test_pipeline_with_preview(tmp_path):
    """Preview flag should produce a preview PNG."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
        preview=True,
    )

    output_files = process_single(png_path, config)

    png_files = [f for f in output_files if f.suffix == ".png"]
    assert len(png_files) == 1
    assert png_files[0].exists()


def test_pipeline_three_color_auto_detect(tmp_path):
    """Three-color logo with auto-detection should find ~3 colors."""
    png_path = tmp_path / "test_logo.png"
    _create_three_color_logo(png_path)

    config = PipelineConfig(
        colors=3,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
    )

    output_files = process_single(png_path, config)

    svg_files = [f for f in output_files if f.suffix == ".svg"]
    assert len(svg_files) == 3


def test_pipeline_svgs_share_viewbox(tmp_path):
    """All SVGs from the same logo should have identical viewBox."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
    )

    output_files = process_single(png_path, config)
    svg_files = [f for f in output_files if f.suffix == ".svg"]

    viewboxes = set()
    for svg_file in svg_files:
        content = svg_file.read_text()
        # Extract viewBox value
        import re
        match = re.search(r'viewBox="([^"]+)"', content)
        assert match is not None
        viewboxes.add(match.group(1))

    # All SVGs should have the same viewBox
    assert len(viewboxes) == 1
    assert viewboxes.pop() == "0 0 200 200"


def test_pipeline_with_smooth_zero(tmp_path):
    """Pipeline should work with smoothing explicitly disabled."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
        smooth=0.0,
    )

    output_files = process_single(png_path, config)
    svg_files = [f for f in output_files if f.suffix == ".svg"]
    assert len(svg_files) == 2
    for svg_file in svg_files:
        content = svg_file.read_text()
        assert "<path" in content


def test_pipeline_with_high_smooth(tmp_path):
    """Pipeline should work with high smoothing value."""
    png_path = tmp_path / "test_logo.png"
    _create_two_color_logo(png_path)

    config = PipelineConfig(
        colors=2,
        tolerance=2.0,
        output_dir=tmp_path,
        min_area=50,
        smooth=3.0,
    )

    output_files = process_single(png_path, config)
    svg_files = [f for f in output_files if f.suffix == ".svg"]
    assert len(svg_files) >= 1
    for svg_file in svg_files:
        content = svg_file.read_text()
        assert "<svg" in content
