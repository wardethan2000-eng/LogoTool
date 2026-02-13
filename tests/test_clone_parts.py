"""Tests for the clone_parts module."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from logo2svg.clone_parts import (
    _extract_hex_color,
    _find_reference_part,
    _inject_objects,
    _is_model_file,
    _is_settings_file,
    _max_object_id,
    _read_project_3mf,
    _svgs_to_meshes,
    _update_settings,
    clone_parts,
)
from logo2svg.threemf_writer import _MeshObject


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def _make_model_xml(
    objects: list[tuple[int, str]],  # (id, name)
    transforms: dict[int, str] | None = None,
) -> str:
    """Build a minimal model XML with given objects and build items."""
    transforms = transforms or {}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="millimeter" xmlns="{_NS}">',
        "  <resources>",
    ]
    for oid, name in objects:
        lines.append(f'    <object id="{oid}" type="model" name="{name}">')
        lines.append("      <mesh>")
        lines.append("        <vertices>")
        lines.append('          <vertex x="0" y="0" z="0"/>')
        lines.append('          <vertex x="1" y="0" z="0"/>')
        lines.append('          <vertex x="0" y="1" z="0"/>')
        lines.append("        </vertices>")
        lines.append("        <triangles>")
        lines.append('          <triangle v1="0" v2="1" v3="2"/>')
        lines.append("        </triangles>")
        lines.append("      </mesh>")
        lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    for oid, _ in objects:
        t = transforms.get(oid, "1 0 0 0 1 0 0 0 1 0 0 0")
        lines.append(f'    <item objectid="{oid}" transform="{t}"/>')
    lines.append("  </build>")
    lines.append("</model>")
    return "\n".join(lines)


def _make_settings_xml(objects: list[tuple[int, str, int]]) -> str:
    """Build a minimal settings config XML.  Items are (id, name, extruder)."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    for oid, name, ext in objects:
        lines.append(f'  <object id="{oid}">')
        lines.append(f'    <metadata key="name" value="{name}"/>')
        lines.append(f'    <metadata key="extruder" value="{ext}"/>')
        lines.append("  </object>")
    lines.append("</config>")
    return "\n".join(lines)


def _write_test_3mf(
    path: Path,
    model_xml: str,
    settings_xml: str | None = None,
) -> None:
    """Write a minimal test 3MF ZIP."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        zf.writestr("3D/3dmodel.model", model_xml)
        if settings_xml:
            zf.writestr("Metadata/model_settings.config", settings_xml)


def _write_test_svg(path: Path, hex_color: str = "#FF0000") -> None:
    """Write a minimal SVG file with a simple rect path."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<path d="M 10 10 L 90 10 L 90 90 L 10 90 Z" fill="{hex_color}"/>'
        "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


# -----------------------------------------------------------------------
# Unit tests: helpers
# -----------------------------------------------------------------------


class TestFileDetection:
    def test_model_file(self):
        assert _is_model_file("3D/3dmodel.model")
        assert _is_model_file("3D/MyModel.model")
        assert not _is_model_file("Metadata/model_settings.config")

    def test_settings_file(self):
        assert _is_settings_file("Metadata/model_settings.config")
        assert not _is_settings_file("3D/3dmodel.model")


class TestHexExtraction:
    def test_from_svg_fill(self):
        svg = '<path fill="#FF0000" d="M 0 0"/>'
        assert _extract_hex_color(svg, "test.svg") == "#FF0000"

    def test_from_filename(self):
        svg = "<svg><path d='M 0 0'/></svg>"
        assert _extract_hex_color(svg, "layer_1_AABB00.svg") == "#AABB00"

    def test_fallback(self):
        svg = "<svg><path d='M 0 0'/></svg>"
        assert _extract_hex_color(svg, "test.svg") == "#888888"


# -----------------------------------------------------------------------
# Unit tests: model XML manipulation
# -----------------------------------------------------------------------


class TestFindReferencePart:
    def test_auto_detect_highest_id(self):
        xml = _make_model_xml(
            [(1, "Helmet"), (5, "SVG Layer")],
            transforms={1: "1 0 0 0 1 0 0 0 1 0 0 0", 5: "0.5 0 0 0 0.5 0 0 0 0.5 10 20 30"},
        )
        root = ET.fromstring(xml)
        obj_id, transform = _find_reference_part(root)
        assert obj_id == 5
        assert transform == "0.5 0 0 0 0.5 0 0 0 0.5 10 20 30"

    def test_explicit_part_id(self):
        xml = _make_model_xml(
            [(1, "Helmet"), (5, "SVG Layer")],
            transforms={1: "1 0 0 0 1 0 0 0 1 100 200 300"},
        )
        root = ET.fromstring(xml)
        obj_id, transform = _find_reference_part(root, part_id=1)
        assert obj_id == 1
        assert "100 200 300" in transform

    def test_missing_part_id_raises(self):
        xml = _make_model_xml([(1, "Helmet")])
        root = ET.fromstring(xml)
        with pytest.raises(ValueError, match="not found"):
            _find_reference_part(root, part_id=99)


class TestMaxObjectId:
    def test_returns_highest(self):
        xml = _make_model_xml([(1, "A"), (3, "B"), (7, "C")])
        root = ET.fromstring(xml)
        assert _max_object_id(root) == 7

    def test_empty(self):
        root = ET.fromstring(
            f'<model xmlns="{_NS}"><resources/><build/></model>'
        )
        assert _max_object_id(root) == 0


class TestInjectObjects:
    def test_adds_objects_and_items(self):
        xml = _make_model_xml([(1, "Helmet")])
        root = ET.fromstring(xml)

        mesh = _MeshObject(
            obj_id=2,
            name="Red",
            hex_color="#FF0000",
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            triangles=[(0, 1, 2)],
            extruder=1,
        )
        _inject_objects(root, [mesh], "0.5 0 0 0 0.5 0 0 0 0.5 10 20 30")

        # Check that object was added
        ns = {"m": _NS}
        objs = root.findall(".//m:object", ns)
        assert len(objs) == 2

        # Check that item was added with correct transform
        items = root.findall(".//m:item", ns)
        assert len(items) == 2
        new_item = [i for i in items if i.get("objectid") == "2"][0]
        assert new_item.get("transform") == "0.5 0 0 0 0.5 0 0 0 0.5 10 20 30"


class TestUpdateSettings:
    def test_adds_extruder_entries(self):
        root = ET.Element("config")
        mesh = _MeshObject(
            obj_id=5,
            name="Blue",
            hex_color="#0000FF",
            vertices=[],
            triangles=[],
            extruder=2,
        )
        _update_settings(root, [mesh])

        obj_elems = root.findall("object")
        assert len(obj_elems) == 1
        assert obj_elems[0].get("id") == "5"

        metas = obj_elems[0].findall("metadata")
        extruder_meta = [m for m in metas if m.get("key") == "extruder"][0]
        assert extruder_meta.get("value") == "2"


# -----------------------------------------------------------------------
# SVG → mesh conversion
# -----------------------------------------------------------------------


class TestSvgsToMeshes:
    def test_converts_svg_files(self, tmp_path):
        svg1 = tmp_path / "layer_1_red.svg"
        svg2 = tmp_path / "layer_2_blue.svg"
        _write_test_svg(svg1, "#FF0000")
        _write_test_svg(svg2, "#0000FF")

        meshes = _svgs_to_meshes([svg1, svg2], start_id=10, thickness=2.0, scale=1.0)
        assert len(meshes) == 2
        assert meshes[0].obj_id == 10
        assert meshes[1].obj_id == 11
        assert len(meshes[0].vertices) > 0
        assert len(meshes[0].triangles) > 0

    def test_skips_empty_svg(self, tmp_path):
        svg = tmp_path / "empty.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
        meshes = _svgs_to_meshes([svg], start_id=1, thickness=2.0, scale=1.0)
        assert len(meshes) == 0


# -----------------------------------------------------------------------
# Integration test: clone_parts end-to-end
# -----------------------------------------------------------------------


class TestClonePartsEndToEnd:
    def test_full_round_trip(self, tmp_path):
        """Create a fake project, clone parts, verify output."""
        # Create a project 3MF with one "helmet" and one "SVG layer"
        transform = "0.5 0 0 0 0.5 0 0 0 1 10 20 30"
        model_xml = _make_model_xml(
            [(1, "Helmet"), (2, "SVG Layer 1")],
            transforms={1: "1 0 0 0 1 0 0 0 1 0 0 0", 2: transform},
        )
        settings_xml = _make_settings_xml([
            (1, "Helmet", 1),
            (2, "SVG Layer 1", 1),
        ])

        project = tmp_path / "project.3mf"
        _write_test_3mf(project, model_xml, settings_xml)

        # Create SVG layers directory
        layers = tmp_path / "layers"
        layers.mkdir()
        _write_test_svg(layers / "layer_1_FF0000.svg", "#FF0000")
        _write_test_svg(layers / "layer_2_00FF00.svg", "#00FF00")
        _write_test_svg(layers / "layer_3_0000FF.svg", "#0000FF")

        # Run clone_parts
        output = tmp_path / "output.3mf"
        result = clone_parts(project, layers, output)
        assert result == output
        assert output.exists()

        # Verify the output 3MF
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            # Model file exists
            model_files = [n for n in names if "model" in n.lower()]
            assert len(model_files) >= 1

            # Parse model and verify transforms
            model_data = zf.read(model_files[0])
            root = ET.fromstring(model_data)
            ns = {"m": _NS}

            # Should have original 2 objects + 3 new ones = 5
            objs = root.findall(".//m:object", ns)
            assert len(objs) == 5

            # All new items should have the same transform
            items = root.findall(".//m:item", ns)
            new_items = [i for i in items if int(i.get("objectid", "0")) > 2]
            assert len(new_items) == 3
            for item in new_items:
                assert item.get("transform") == transform

    def test_no_svgs_raises(self, tmp_path):
        model_xml = _make_model_xml([(1, "Helmet")])
        project = tmp_path / "project.3mf"
        _write_test_3mf(project, model_xml)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No SVG"):
            clone_parts(project, empty_dir, tmp_path / "out.3mf")

    def test_default_output_name(self, tmp_path):
        model_xml = _make_model_xml([(1, "Helmet"), (2, "Layer")])
        settings_xml = _make_settings_xml([(1, "Helmet", 1)])
        project = tmp_path / "helmet.3mf"
        _write_test_3mf(project, model_xml, settings_xml)

        layers = tmp_path / "layers"
        layers.mkdir()
        _write_test_svg(layers / "layer_1.svg")

        result = clone_parts(project, layers)
        assert result.name == "helmet_multicolor.3mf"
        assert result.exists()

    def test_read_project_3mf(self, tmp_path):
        model_xml = _make_model_xml([(1, "Test")])
        settings_xml = _make_settings_xml([(1, "Test", 1)])
        project = tmp_path / "test.3mf"
        _write_test_3mf(project, model_xml, settings_xml)

        entries, model_bytes, settings_bytes = _read_project_3mf(project)
        assert b"<model" in model_bytes
        assert settings_bytes is not None
        assert b"<config" in settings_bytes
        # Content types should be in entries (not model/settings)
        assert "[Content_Types].xml" in entries
