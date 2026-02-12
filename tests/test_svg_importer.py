"""Tests for the SVG importer module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from logo2svg.svg_importer import (
    _normalize_color,
    _parse_svg_colors,
    _parse_svg_dimensions,
    import_svg_as_layers,
)


@pytest.fixture
def simple_svg(tmp_path):
    """Create a simple SVG with known colors."""
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <rect x="10" y="10" width="80" height="80" fill="#FF0000"/>
  <circle cx="150" cy="150" r="40" fill="#0000FF"/>
  <rect x="50" y="50" width="60" height="60" fill="green"/>
</svg>"""
    svg_path = tmp_path / "test.svg"
    svg_path.write_text(svg_content)
    return svg_path


@pytest.fixture
def empty_svg(tmp_path):
    """Create an empty SVG."""
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
</svg>"""
    svg_path = tmp_path / "empty.svg"
    svg_path.write_text(svg_content)
    return svg_path


class TestNormalizeColor:
    def test_hex_6_digit(self):
        assert _normalize_color("#FF0000") == "#FF0000"

    def test_hex_3_digit(self):
        assert _normalize_color("#F00") == "#FF0000"

    def test_lowercase(self):
        assert _normalize_color("#ff0000") == "#FF0000"

    def test_rgb_function(self):
        assert _normalize_color("rgb(255, 0, 0)") == "#FF0000"

    def test_named_color(self):
        assert _normalize_color("red") == "#FF0000"
        assert _normalize_color("blue") == "#0000FF"

    def test_invalid(self):
        assert _normalize_color("notacolor") is None

    def test_none_fill(self):
        assert _normalize_color("none") is None


class TestParseSvgColors:
    def test_simple_svg(self, simple_svg):
        colors = _parse_svg_colors(simple_svg)
        assert "#FF0000" in colors
        assert "#0000FF" in colors
        assert "#008000" in colors  # green

    def test_empty_svg(self, empty_svg):
        colors = _parse_svg_colors(empty_svg)
        assert len(colors) == 0


class TestParseSvgDimensions:
    def test_explicit_dimensions(self, simple_svg):
        w, h = _parse_svg_dimensions(simple_svg)
        assert w == 200
        assert h == 200

    def test_viewbox_fallback(self, tmp_path):
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 400">
</svg>"""
        svg_path = tmp_path / "vb.svg"
        svg_path.write_text(svg_content)
        w, h = _parse_svg_dimensions(svg_path)
        assert w == 300
        assert h == 400


class TestImportSvgAsLayers:
    def test_basic_import(self, simple_svg):
        layers = import_svg_as_layers(simple_svg, 200, 200)
        assert isinstance(layers, list)
        for layer in layers:
            assert "rgb" in layer
            assert "hex_color" in layer
            assert "mask" in layer
            assert layer["mask"].shape == (200, 200)

    def test_color_override(self, simple_svg):
        layers = import_svg_as_layers(
            simple_svg, 200, 200, color_override="#00FF00"
        )
        # With color override, should get exactly 1 layer
        assert len(layers) == 1
        assert layers[0]["hex_color"] == "#00FF00"

    def test_empty_svg_returns_empty(self, empty_svg):
        layers = import_svg_as_layers(empty_svg, 100, 100)
        assert layers == []

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            import_svg_as_layers("/nonexistent/path.svg", 100, 100)

    def test_rescaling(self, simple_svg):
        layers = import_svg_as_layers(simple_svg, 400, 400)
        for layer in layers:
            assert layer["mask"].shape == (400, 400)
