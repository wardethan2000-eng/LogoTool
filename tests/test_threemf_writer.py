"""Tests for threemf_writer module: 3MF structure, mesh data, extruder config."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

from logo2svg.threemf_writer import (
    write_3mf,
    _parse_svg_paths,
    _triangulate_polygon,
    _linearise_cubic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers() -> list[dict]:
    """Create two fake layers with simple SVG path data (rectangles)."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:50, 10:50] = 255
    mask2 = np.zeros((100, 100), dtype=np.uint8)
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
# SVG path parsing
# ---------------------------------------------------------------------------

class TestParseSvgPaths:
    """Tests for SVG path 'd' string parsing."""

    def test_simple_rectangle(self):
        paths = _parse_svg_paths(
            ["M 0 0 L 10 0 L 10 10 L 0 10 Z"]
        )
        assert len(paths) == 1
        assert len(paths[0]) == 4

    def test_multiple_subpaths(self):
        """Two rectangles in one compound path."""
        d = "M 0 0 L 10 0 L 10 10 L 0 10 Z M 20 20 L 30 20 L 30 30 L 20 30 Z"
        paths = _parse_svg_paths([d])
        assert len(paths) == 2

    def test_cubic_bezier(self):
        """Cubic Bézier should be linearised into multiple points."""
        d = "M 0 0 C 10 0 10 10 0 10 Z"
        paths = _parse_svg_paths([d])
        assert len(paths) == 1
        # Should have more than just start + end (linearised segments)
        assert len(paths[0]) > 2

    def test_empty_input(self):
        assert _parse_svg_paths([]) == []

    def test_float_coordinates(self):
        d = "M 10.50 20.75 L 30.25 40.10 L 50.00 60.00 Z"
        paths = _parse_svg_paths([d])
        assert len(paths) == 1
        assert len(paths[0]) == 3


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

class TestTriangulatePolygon:
    """Tests for ear-clipping triangulation."""

    def test_triangle(self):
        """A triangle should produce exactly one triangle."""
        poly = [(0, 0), (10, 0), (5, 10)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 1

    def test_square(self):
        """A square should produce exactly two triangles."""
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 2

    def test_pentagon(self):
        """A pentagon should produce exactly three triangles."""
        import math
        poly = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5))
                for i in range(5)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 3

    def test_degenerate_line(self):
        """Less than 3 points should produce no triangles."""
        assert _triangulate_polygon([(0, 0), (1, 1)]) == []
        assert _triangulate_polygon([(0, 0)]) == []
        assert _triangulate_polygon([]) == []


# ---------------------------------------------------------------------------
# Linearise cubic Bézier
# ---------------------------------------------------------------------------

class TestLineariseCubic:
    """Tests for cubic Bézier linearisation."""

    def test_endpoints(self):
        """Last point should match the endpoint of the Bézier."""
        pts = _linearise_cubic(0, 0, 5, 0, 10, 10, 10, 10, 4)
        assert len(pts) == 4
        assert abs(pts[-1][0] - 10) < 0.001
        assert abs(pts[-1][1] - 10) < 0.001

    def test_straight_line(self):
        """A 'Bézier' that's actually straight should produce collinear points."""
        pts = _linearise_cubic(0, 0, 3.33, 0, 6.67, 0, 10, 0, 4)
        for x, y in pts:
            assert abs(y) < 0.001  # all on the x-axis


# ---------------------------------------------------------------------------
# 3MF file output
# ---------------------------------------------------------------------------

class TestWrite3mf:
    """Tests for the complete 3MF file generation."""

    def test_creates_valid_zip(self, tmp_path):
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        result = write_3mf(out, layers, 100, 100)
        assert result.exists()
        assert zipfile.is_zipfile(result)

    def test_zip_entries(self, tmp_path):
        """3MF should contain the required structure files."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "[Content_Types].xml" in names
            assert "_rels/.rels" in names
            assert "3D/3dmodel.model" in names
            assert "Metadata/model_settings.config" in names

    def test_model_has_objects_per_layer(self, tmp_path):
        """Model XML should have one <object> per colour layer."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            model_xml = zf.read("3D/3dmodel.model").decode("utf-8")

        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        root = ET.fromstring(model_xml)
        objects = root.findall(".//m:object", ns)
        assert len(objects) == 2

    def test_objects_have_mesh_data(self, tmp_path):
        """Each object should have non-empty vertex and triangle data."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            model_xml = zf.read("3D/3dmodel.model").decode("utf-8")

        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        root = ET.fromstring(model_xml)

        for obj in root.findall(".//m:object", ns):
            mesh = obj.find("m:mesh", ns)
            assert mesh is not None
            vertices = mesh.find("m:vertices", ns)
            assert vertices is not None
            assert len(vertices) > 0
            triangles = mesh.find("m:triangles", ns)
            assert triangles is not None
            assert len(triangles) > 0

    def test_build_section_has_items(self, tmp_path):
        """Build section should reference all objects."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            model_xml = zf.read("3D/3dmodel.model").decode("utf-8")

        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        root = ET.fromstring(model_xml)
        items = root.findall(".//m:item", ns)
        assert len(items) == 2

    def test_settings_config_extruder_assignments(self, tmp_path):
        """Each object should be assigned a distinct extruder in the config."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            config_xml = zf.read("Metadata/model_settings.config").decode("utf-8")

        root = ET.fromstring(config_xml)
        objects = root.findall(".//object")
        assert len(objects) == 2

        extruders = set()
        for obj in objects:
            for meta in obj.findall("metadata"):
                if meta.get("key") == "extruder":
                    extruders.add(meta.get("value"))

        # Should have two different extruder values
        assert len(extruders) == 2
        assert "1" in extruders
        assert "2" in extruders

    def test_settings_config_has_names(self, tmp_path):
        """Object names should include colour info."""
        layers = _make_layers()
        out = tmp_path / "test.3mf"
        write_3mf(out, layers, 100, 100)

        with zipfile.ZipFile(out) as zf:
            config_xml = zf.read("Metadata/model_settings.config").decode("utf-8")

        assert "#FF0000" in config_xml
        assert "#0000FF" in config_xml
        assert "red" in config_xml
        assert "blue" in config_xml

    def test_scale_affects_coordinates(self, tmp_path):
        """Scale should multiply vertex coordinates."""
        layers = _make_layers()
        out_1x = tmp_path / "test_1x.3mf"
        out_2x = tmp_path / "test_2x.3mf"
        write_3mf(out_1x, layers, 100, 100, scale=1.0)
        write_3mf(out_2x, layers, 100, 100, scale=2.0)

        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

        def _get_first_vertex(path):
            with zipfile.ZipFile(path) as zf:
                model_xml = zf.read("3D/3dmodel.model").decode("utf-8")
            root = ET.fromstring(model_xml)
            vert = root.find(".//m:vertex", ns)
            return float(vert.get("x"))

        x1 = _get_first_vertex(out_1x)
        x2 = _get_first_vertex(out_2x)
        assert abs(x2 - 2 * x1) < 0.01

    def test_empty_layers_raises(self, tmp_path):
        """Should raise ValueError if no layers have path data."""
        layers = [{"hex_color": "#000000", "color_name": "black", "svg_paths": []}]
        with pytest.raises(ValueError, match="No valid geometry"):
            write_3mf(tmp_path / "test.3mf", layers, 100, 100)

    def test_output_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        layers = _make_layers()
        out = tmp_path / "nested" / "dir" / "test.3mf"
        result = write_3mf(out, layers, 100, 100)
        assert result.exists()
