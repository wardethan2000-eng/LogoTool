"""Tests for svg_writer module: SVG structure, combined SVG, preview PNG."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from logo2svg.svg_writer import (
    write_svg_files,
    write_preview,
    _write_single_color_svg,
    _write_combined_svg,
    _scale_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(
    height: int = 100, width: int = 100
) -> list[dict]:
    """Create two fake layers with simple SVG path data."""
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask1[10:50, 10:50] = 255
    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2[50:90, 50:90] = 255

    return [
        {
            "rgb": (255, 0, 0),
            "hex_color": "#FF0000",
            "color_name": "red",
            "mask": mask1,
            "cluster_idx": 0,
            "svg_paths": ["M 10.00 10.00 L 50.00 10.00 L 50.00 50.00 L 10.00 50.00 Z"],
        },
        {
            "rgb": (0, 0, 255),
            "hex_color": "#0000FF",
            "color_name": "blue",
            "mask": mask2,
            "cluster_idx": 1,
            "svg_paths": ["M 50.00 50.00 L 90.00 50.00 L 90.00 90.00 L 50.00 90.00 Z"],
        },
    ]


# ---------------------------------------------------------------------------
# Per-color SVG files
# ---------------------------------------------------------------------------

class TestWriteSvgFiles:
    """Tests for write_svg_files — one SVG per colour."""

    def test_creates_one_file_per_layer(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        assert len(files) == 2
        for f in files:
            assert f.exists()
            assert f.suffix == ".svg"

    def test_svg_is_valid_xml(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        for f in files:
            tree = ET.parse(f)
            root = tree.getroot()
            # Should be an <svg> element
            assert root.tag.endswith("svg")

    def test_viewbox_matches_image_size(self, tmp_path):
        layers = _make_layers(height=200, width=300)
        files = write_svg_files("test", layers, (200, 300), tmp_path)
        for f in files:
            content = f.read_text()
            assert 'viewBox="0 0 300 200"' in content

    def test_fill_colors_match_input(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        # First file should have fill="#FF0000"
        content0 = files[0].read_text()
        assert 'fill="#FF0000"' in content0
        # Second should have fill="#0000FF"
        content1 = files[1].read_text()
        assert 'fill="#0000FF"' in content1

    def test_fill_rule_evenodd(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        for f in files:
            content = f.read_text()
            assert 'fill-rule="evenodd"' in content

    def test_paths_present(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        for f in files:
            content = f.read_text()
            assert "<path" in content

    def test_output_dir_created(self, tmp_path):
        layers = _make_layers()
        out = tmp_path / "nested" / "dir"
        files = write_svg_files("test", layers, (100, 100), out)
        assert out.exists()
        assert len(files) == 2

    def test_individual_svgs_have_bounding_rect(self, tmp_path):
        """Each per-color SVG should include an invisible bounding rectangle for slicer alignment."""
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for f in files:
            tree = ET.parse(f)
            root = tree.getroot()
            rects = root.findall(".//svg:rect", ns)
            assert len(rects) >= 1, "Expected an invisible bounding rect"
            r = rects[0]
            assert r.get("fill") == "none"
            assert r.get("stroke") == "none"


# ---------------------------------------------------------------------------
# Combined SVG
# ---------------------------------------------------------------------------

class TestCombinedSvg:
    """Tests for the combined SVG with <g> groups."""

    def test_combined_has_groups(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path, combined=True)
        combined_files = [f for f in files if "combined" in f.name]
        assert len(combined_files) == 1

        tree = ET.parse(combined_files[0])
        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        groups = root.findall(".//svg:g", ns)
        assert len(groups) == 2

    def test_combined_viewbox(self, tmp_path):
        layers = _make_layers(200, 300)
        files = write_svg_files("test", layers, (200, 300), tmp_path, combined=True)
        combined = [f for f in files if "combined" in f.name][0]
        content = combined.read_text()
        assert 'viewBox="0 0 300 200"' in content

    def test_combined_group_ids_contain_hex(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path, combined=True)
        combined = [f for f in files if "combined" in f.name][0]
        content = combined.read_text()
        assert "color_FF0000" in content
        assert "color_0000FF" in content

    def test_combined_paths_have_correct_fills(self, tmp_path):
        layers = _make_layers()
        files = write_svg_files("test", layers, (100, 100), tmp_path, combined=True)
        combined = [f for f in files if "combined" in f.name][0]
        content = combined.read_text()
        assert 'fill="#FF0000"' in content
        assert 'fill="#0000FF"' in content


# ---------------------------------------------------------------------------
# Preview PNG
# ---------------------------------------------------------------------------

class TestWritePreview:
    """Tests for preview PNG generation."""

    def test_preview_created(self, tmp_path):
        layers = _make_layers()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        preview_path = write_preview("test", img, layers, (100, 100), tmp_path)
        assert preview_path.exists()
        assert preview_path.suffix == ".png"

    def test_preview_dimensions(self, tmp_path):
        """Preview should be (height) × (width × n_panels) with n_panels = 2 + n_layers."""
        layers = _make_layers()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        preview_path = write_preview("test", img, layers, (100, 100), tmp_path)

        from PIL import Image
        pil = Image.open(preview_path)
        w, h = pil.size
        n_panels = 2 + len(layers)  # original + quantized + per-colour
        assert h == 100
        assert w == 100 * n_panels


# ---------------------------------------------------------------------------
# Shared viewBox across per-colour SVGs
# ---------------------------------------------------------------------------

class TestSharedViewbox:
    """All per-colour SVGs from the same run must share a viewBox."""

    def test_all_svgs_same_viewbox(self, tmp_path):
        layers = _make_layers(150, 250)
        files = write_svg_files("test", layers, (150, 250), tmp_path)
        viewboxes = set()
        for f in files:
            match = re.search(r'viewBox="([^"]+)"', f.read_text())
            assert match
            viewboxes.add(match.group(1))
        assert len(viewboxes) == 1
        assert viewboxes.pop() == "0 0 250 150"


# ---------------------------------------------------------------------------
# _scale_path
# ---------------------------------------------------------------------------

class TestScalePath:
    """Tests for _scale_path coordinate scaling."""

    def test_identity_scale(self):
        """scale=1.0 should return the path string unchanged."""
        d = "M 10.00 20.00 L 30.00 40.00 Z"
        assert _scale_path(d, 1.0) == d

    def test_scale_up(self):
        """scale=2.0 should double all coordinates."""
        d = "M 10.00 20.00 L 30.00 40.00 Z"
        result = _scale_path(d, 2.0)
        assert "20.00" in result
        assert "40.00" in result
        assert "60.00" in result
        assert "80.00" in result

    def test_scale_down_negative_coords(self):
        """scale=0.5 with negative coordinates."""
        d = "M -10.00 -20.00 L 30.00 40.00 Z"
        result = _scale_path(d, 0.5)
        assert "-5.00" in result
        assert "-10.00" in result
        assert "15.00" in result
        assert "20.00" in result

    def test_integers_without_decimals(self):
        """Integers without decimal points should also be scaled."""
        d = "M 10 20 L 30 40 Z"
        result = _scale_path(d, 2.0)
        assert "20.00" in result
        assert "40.00" in result
        assert "60.00" in result
        assert "80.00" in result
