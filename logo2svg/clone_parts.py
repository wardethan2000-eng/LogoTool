"""Clone-parts tool: inject aligned colour layers into a Bambu Studio project.

Opens a ``.3mf`` project file saved from Bambu Studio (containing a helmet
or other model with **one** SVG-derived colour layer already positioned),
reads that layer's transform, then adds the remaining colour layers as new
mesh objects with the **identical** transform — guaranteeing perfect alignment
on curved surfaces.

Usage (CLI)::

    logo2svg clone-parts project.3mf ./layers/ -o project_multicolour.3mf

Usage (Python)::

    from logo2svg.clone_parts import clone_parts
    clone_parts("project.3mf", "layers/", "out.3mf")
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .threemf_writer import (
    _MeshObject,
    _extrude_polygon,
    _parse_svg_paths,
    _settings_config_xml,
)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def clone_parts(
    project_3mf: str | Path,
    layers_dir: str | Path,
    output_3mf: str | Path | None = None,
    *,
    part_id: int | None = None,
    thickness: float = 2.0,
    scale: float = 1.0,
) -> Path:
    """Inject remaining colour SVG layers into a Bambu Studio project.

    Args:
        project_3mf: Path to the ``.3mf`` project saved from Bambu Studio.
        layers_dir: Directory containing per-colour SVG files from LogoTool.
        output_3mf: Output path. Defaults to ``<input>_multicolor.3mf``.
        part_id: Optional object ID of the positioned SVG part. If ``None``,
            the tool auto-detects it (highest object ID, assumed to be the
            most recently added SVG part).
        thickness: Extrusion thickness in mm for new colour meshes.
        scale: XY scale factor (pixels → mm). Should match the SVG export.

    Returns:
        The output path that was written.
    """
    project_3mf = Path(project_3mf)
    layers_dir = Path(layers_dir)
    if output_3mf is None:
        output_3mf = project_3mf.with_stem(project_3mf.stem + "_multicolor")
    output_3mf = Path(output_3mf)

    # Collect colour-layer SVGs (sorted for deterministic extruder order)
    svg_files = sorted(layers_dir.glob("*.svg"))
    if not svg_files:
        raise FileNotFoundError(f"No SVG files found in {layers_dir}")

    # Read the project 3MF
    entries, model_xml_bytes, settings_bytes = _read_project_3mf(project_3mf)

    # Parse the model XML
    model_root = ET.fromstring(model_xml_bytes)

    # Find the positioned SVG part and its transform
    ref_obj_id, transform_str = _find_reference_part(model_root, part_id)

    # Determine the next available object ID
    max_id = _max_object_id(model_root)

    # Parse each SVG file into mesh data
    colour_meshes = _svgs_to_meshes(
        svg_files, start_id=max_id + 1, thickness=thickness, scale=scale,
    )
    if not colour_meshes:
        raise ValueError("No valid geometry found in SVG files.")

    # Inject the new objects into the model XML
    _inject_objects(model_root, colour_meshes, transform_str)

    # Build updated settings config
    settings_root = ET.fromstring(settings_bytes) if settings_bytes else ET.Element("config")
    _update_settings(settings_root, colour_meshes)

    # Write the output 3MF
    _write_output_3mf(
        output_3mf, entries, model_root, settings_root,
        project_3mf,
    )

    return output_3mf


# -----------------------------------------------------------------------
# Project 3MF reading
# -----------------------------------------------------------------------


def _read_project_3mf(
    path: Path,
) -> tuple[dict[str, bytes], bytes, bytes | None]:
    """Read a Bambu Studio project 3MF and return its contents.

    Returns:
        (entries, model_xml_bytes, settings_bytes)
        *entries* is a dict mapping ZIP entry names to their raw bytes
        (excluding the model and settings files, which are returned
        separately for modification).
    """
    entries: dict[str, bytes] = {}
    model_xml_bytes = b""
    settings_bytes: bytes | None = None

    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if _is_model_file(name):
                model_xml_bytes = data
            elif _is_settings_file(name):
                settings_bytes = data
            else:
                entries[name] = data

    if not model_xml_bytes:
        raise ValueError(
            f"No 3D model file found in {path}. "
            "Expected a file like '3D/3dmodel.model'."
        )

    return entries, model_xml_bytes, settings_bytes


def _is_model_file(name: str) -> bool:
    """Check if a ZIP entry is the 3D model file."""
    lower = name.lower()
    return lower.endswith(".model") and "3d" in lower


def _is_settings_file(name: str) -> bool:
    """Check if a ZIP entry is the Bambu Studio settings config."""
    lower = name.lower()
    return "model_settings" in lower and lower.endswith(".config")


# -----------------------------------------------------------------------
# Reference part detection
# -----------------------------------------------------------------------


def _find_reference_part(
    model_root: ET.Element,
    part_id: int | None = None,
) -> tuple[int, str]:
    """Find the user-positioned SVG part and return (object_id, transform_str).

    If *part_id* is given, use that object. Otherwise, auto-detect by
    choosing the object with the highest ID (most recently added part).
    The corresponding ``<item>`` in ``<build>`` provides the transform.
    """
    ns = {"m": _NS}

    # Collect all build items
    build = model_root.find("m:build", ns)
    if build is None:
        # Try without namespace
        build = model_root.find("build")
    if build is None:
        raise ValueError("No <build> section found in the 3MF model.")

    items = build.findall("m:item", ns)
    if not items:
        items = build.findall("item")
    if not items:
        raise ValueError("No <item> elements found in the <build> section.")

    if part_id is not None:
        # Find the specific item
        for item in items:
            oid = int(item.get("objectid", "0"))
            if oid == part_id:
                transform = item.get("transform", "1 0 0 0 1 0 0 0 1 0 0 0")
                return oid, transform
        raise ValueError(
            f"Object ID {part_id} not found in the <build> section. "
            f"Available IDs: {[item.get('objectid') for item in items]}"
        )

    # Auto-detect: highest object ID = most recently added
    best_item = max(items, key=lambda it: int(it.get("objectid", "0")))
    oid = int(best_item.get("objectid", "0"))
    transform = best_item.get("transform", "1 0 0 0 1 0 0 0 1 0 0 0")
    return oid, transform


def _max_object_id(model_root: ET.Element) -> int:
    """Return the highest object ID in the model."""
    ns = {"m": _NS}
    max_id = 0

    resources = model_root.find("m:resources", ns)
    if resources is None:
        resources = model_root.find("resources")
    if resources is None:
        return max_id

    for obj in resources:
        tag = obj.tag.split("}")[-1] if "}" in obj.tag else obj.tag
        if tag == "object":
            oid = int(obj.get("id", "0"))
            max_id = max(max_id, oid)

    return max_id


# -----------------------------------------------------------------------
# SVG → mesh conversion
# -----------------------------------------------------------------------

# Pattern to extract hex colour from SVG fill attributes
_FILL_RE = re.compile(r'fill\s*[:=]\s*["\']?(#[0-9a-fA-F]{3,6})', re.IGNORECASE)
# Fallback: extract from filename like "layer_1_FF0000.svg"
_HEX_NAME_RE = re.compile(r'([0-9a-fA-F]{6})')


def _svgs_to_meshes(
    svg_files: list[Path],
    start_id: int,
    thickness: float,
    scale: float,
) -> list[_MeshObject]:
    """Convert a list of SVG files into _MeshObject instances."""
    meshes: list[_MeshObject] = []

    for i, svg_path in enumerate(svg_files):
        svg_text = svg_path.read_text(encoding="utf-8")

        # Extract SVG path `d` attributes
        d_strings = re.findall(r'd\s*=\s*"([^"]+)"', svg_text)
        if not d_strings:
            continue

        # Parse into polygons and extrude
        polygons = _parse_svg_paths(d_strings)
        if not polygons:
            continue

        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []

        for poly in polygons:
            if len(poly) < 3:
                continue
            _extrude_polygon(poly, thickness, scale, vertices, triangles)

        if not triangles:
            continue

        # Extract hex colour from SVG content or filename
        hex_color = _extract_hex_color(svg_text, svg_path.name)
        name = svg_path.stem.replace("_", " ").title()

        meshes.append(_MeshObject(
            obj_id=start_id + i,
            name=f"{name} ({hex_color})",
            hex_color=hex_color,
            vertices=vertices,
            triangles=triangles,
            extruder=i + 1,
        ))

    return meshes


def _extract_hex_color(svg_text: str, filename: str) -> str:
    """Extract a hex colour from SVG content or filename."""
    # Try to find a fill colour in the SVG (skip "none")
    fills = _FILL_RE.findall(svg_text)
    for f in fills:
        if f.lower() not in ("#fff", "#ffffff", "#000", "#000000"):
            return f.upper()
    # Try filename
    m = _HEX_NAME_RE.search(filename)
    if m:
        return f"#{m.group(1).upper()}"
    return "#888888"


# -----------------------------------------------------------------------
# Model XML manipulation
# -----------------------------------------------------------------------


def _inject_objects(
    model_root: ET.Element,
    meshes: list[_MeshObject],
    transform_str: str,
) -> None:
    """Add new mesh objects to the model XML with the given transform."""
    ns = {"m": _NS}

    # Determine the namespace prefix used in the document
    tag_prefix = ""
    if model_root.tag.startswith("{"):
        tag_prefix = model_root.tag.split("}")[0] + "}"

    # Find or create <resources> and <build>
    resources = model_root.find(f"{tag_prefix}resources")
    if resources is None:
        resources = ET.SubElement(model_root, f"{tag_prefix}resources")

    build = model_root.find(f"{tag_prefix}build")
    if build is None:
        build = ET.SubElement(model_root, f"{tag_prefix}build")

    for obj in meshes:
        # Add <object> with <mesh>
        obj_elem = ET.SubElement(resources, f"{tag_prefix}object", attrib={
            "id": str(obj.obj_id),
            "type": "model",
            "name": obj.name,
        })
        mesh_elem = ET.SubElement(obj_elem, f"{tag_prefix}mesh")

        # Vertices
        verts_elem = ET.SubElement(mesh_elem, f"{tag_prefix}vertices")
        for x, y, z in obj.vertices:
            ET.SubElement(verts_elem, f"{tag_prefix}vertex", attrib={
                "x": f"{x:.4f}",
                "y": f"{y:.4f}",
                "z": f"{z:.4f}",
            })

        # Triangles
        tris_elem = ET.SubElement(mesh_elem, f"{tag_prefix}triangles")
        for v1, v2, v3 in obj.triangles:
            ET.SubElement(tris_elem, f"{tag_prefix}triangle", attrib={
                "v1": str(v1),
                "v2": str(v2),
                "v3": str(v3),
            })

        # Add <item> in <build> with the cloned transform
        ET.SubElement(build, f"{tag_prefix}item", attrib={
            "objectid": str(obj.obj_id),
            "transform": transform_str,
        })


def _update_settings(
    settings_root: ET.Element,
    meshes: list[_MeshObject],
) -> None:
    """Add extruder assignments for new objects to the settings config."""
    for obj in meshes:
        obj_elem = ET.SubElement(settings_root, "object", attrib={
            "id": str(obj.obj_id),
        })
        ET.SubElement(obj_elem, "metadata", attrib={
            "key": "name",
            "value": obj.name,
        })
        ET.SubElement(obj_elem, "metadata", attrib={
            "key": "extruder",
            "value": str(obj.extruder),
        })


# -----------------------------------------------------------------------
# Output writing
# -----------------------------------------------------------------------


def _write_output_3mf(
    output_path: Path,
    other_entries: dict[str, bytes],
    model_root: ET.Element,
    settings_root: ET.Element,
    source_path: Path,
) -> None:
    """Write the modified project as a new 3MF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize model XML
    _indent_xml(model_root)
    model_bytes = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(model_root, encoding="unicode")
    ).encode("utf-8")

    # Serialize settings XML
    _indent_xml(settings_root)
    settings_bytes = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(settings_root, encoding="unicode")
    ).encode("utf-8")

    # Find the original paths for model and settings
    model_entry_name = "3D/3dmodel.model"
    settings_entry_name = "Metadata/model_settings.config"

    # Check original ZIP for actual paths
    with zipfile.ZipFile(source_path, "r") as zf:
        for name in zf.namelist():
            if _is_model_file(name):
                model_entry_name = name
            elif _is_settings_file(name):
                settings_entry_name = name

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write all non-model/non-settings entries from original
        for name, data in other_entries.items():
            zf.writestr(name, data)
        # Write modified model and settings
        zf.writestr(model_entry_name, model_bytes)
        zf.writestr(settings_entry_name, settings_bytes)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to an ElementTree element for pretty printing."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
