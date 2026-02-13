"""Clone-parts tool: inject aligned colour layers into a Bambu Studio project.

Opens a ``.3mf`` project file saved from Bambu Studio (containing a helmet
or other model with **one** SVG-derived colour layer already positioned),
reads that layer's transform, then adds the remaining colour layers as new
mesh objects with the **identical** transform — guaranteeing perfect alignment
on curved surfaces.

Bambu Studio internally uses an **assembly** pattern:

* Individual meshes live as ``<object>`` elements in ``<resources>``.
* A **parent** object groups them via ``<components>``, each with a
  ``transform`` attribute.
* ``<build>`` has a single ``<item>`` pointing to the parent.
* ``model_settings.config`` holds per-part metadata (extruder, matrix,
  ``<BambuStudioShape>`` for SVGs).
* SVG source files live in ``3D/`` inside the ZIP archive.

This module reproduces that exact structure.

Usage (CLI)::

    logo2svg-clone project.3mf ./layers/ -o project_multicolour.3mf

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

# -----------------------------------------------------------------------
# Namespace handling
# -----------------------------------------------------------------------

_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_BS_NS = "http://schemas.bambulab.com/package/2021"

# Register namespaces so ElementTree preserves them during round-trip.
ET.register_namespace("", _NS)
ET.register_namespace("BambuStudio", _BS_NS)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


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
        part_id: Unused (reserved for future use).
        thickness: Extrusion thickness — ignored; Bambu re-creates SVG meshes.
        scale: Scale factor — ignored; Bambu re-creates SVG meshes.

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

    # --------------------------------------------------
    # 1. Read the original project 3MF
    # --------------------------------------------------
    entries, model_xml_bytes, settings_bytes = _read_project_3mf(project_3mf)

    # --------------------------------------------------
    # 2. Parse the model XML and understand the assembly
    # --------------------------------------------------
    model_root = ET.fromstring(model_xml_bytes)

    # Detect namespace prefix used in the XML
    tp = ""  # tag prefix
    if model_root.tag.startswith("{"):
        tp = model_root.tag.split("}")[0] + "}"

    ns = {"m": _NS}

    # Find the assembly parent (object with <components>)
    parent_obj, parent_id = _find_parent_object(model_root, tp, ns)

    # Find the SVG component — the one whose component objectid points to a
    # mesh object that is described with a <BambuStudioShape> in settings.
    components = parent_obj.find(f"{tp}components")
    if components is None:
        raise ValueError("Parent object has no <components> section.")

    # Parse settings to find the SVG part's component transform and metadata
    settings_root = ET.fromstring(settings_bytes) if settings_bytes else None
    if settings_root is None:
        raise ValueError("No model_settings.config found in the project.")

    svg_part_info = _find_svg_part(settings_root, parent_id)
    if svg_part_info is None:
        raise ValueError(
            "Could not find an SVG-derived part in the project. "
            "Make sure you have imported at least one SVG layer in Bambu Studio."
        )

    ref_part_id = svg_part_info["part_id"]
    ref_matrix = svg_part_info["matrix"]
    ref_extruder = svg_part_info["extruder"]
    ref_shape_elem = svg_part_info["shape_elem"]
    ref_svg_3mf_path = svg_part_info.get("svg_3mf_path", "")

    # Find the component transform for the SVG part's sub-object
    ref_component_transform = None
    ref_component_objectid = None
    for comp in components:
        comp_tag = comp.tag.split("}")[-1] if "}" in comp.tag else comp.tag
        if comp_tag == "component":
            coid = comp.get("objectid")
            # Match by checking if this component's object id corresponds
            # to the SVG part.  In Bambu Studio, part id == sub-object id.
            if coid == str(ref_part_id):
                ref_component_transform = comp.get("transform")
                ref_component_objectid = coid
                break

    # If we didn't match by part_id == objectid, try matching by position
    # (the SVG part is usually the last component)
    if ref_component_transform is None:
        comp_list = [
            c for c in components
            if (c.tag.split("}")[-1] if "}" in c.tag else c.tag) == "component"
        ]
        if comp_list:
            last_comp = comp_list[-1]
            ref_component_transform = last_comp.get("transform")
            ref_component_objectid = last_comp.get("objectid")

    if ref_component_transform is None:
        raise ValueError("Could not determine the SVG component's transform.")

    # --------------------------------------------------
    # 3. Find the existing SVG mesh object to use as template
    # --------------------------------------------------
    resources = model_root.find(f"{tp}resources")
    ref_mesh_obj = None
    for obj in resources:
        obj_tag = obj.tag.split("}")[-1] if "}" in obj.tag else obj.tag
        if obj_tag == "object" and obj.get("id") == ref_component_objectid:
            ref_mesh_obj = obj
            break

    # --------------------------------------------------
    # 4. Determine next available IDs
    # --------------------------------------------------
    max_obj_id = _max_object_id(model_root)

    # --------------------------------------------------
    # 5. For each SVG layer, clone the structure
    # --------------------------------------------------
    new_obj_id = max_obj_id + 1
    next_extruder = _max_extruder(settings_root, parent_id) + 1

    # Find the parent object's settings entry
    parent_settings_obj = _find_settings_object(settings_root, parent_id)
    if parent_settings_obj is None:
        parent_settings_obj = ET.SubElement(
            settings_root, "object", attrib={"id": str(parent_id)}
        )

    for i, svg_path in enumerate(svg_files):
        svg_text = svg_path.read_text(encoding="utf-8")

        # Skip SVGs with no paths
        d_strings = re.findall(r'd\s*=\s*"([^"]+)"', svg_text)
        if not d_strings:
            continue

        svg_filename = svg_path.name
        svg_3mf_entry = f"3D/{svg_filename}"

        # 5a. Store the SVG file in the ZIP entries
        entries[svg_3mf_entry] = svg_path.read_bytes()

        # 5b. Create a new mesh object from THIS SVG's geometry.
        #     Each layer must have its own unique mesh — not a clone of
        #     the reference.  We use the SVG→mesh pipeline from
        #     threemf_writer to parse the SVG path data and extrude it.
        from .threemf_writer import _parse_svg_paths, _extrude_polygon

        # Read scale and depth from the reference BambuStudioShape so the
        # new meshes match the coordinate system Bambu Studio expects.
        ref_scale = float(
            ref_shape_elem.get("scale", "1e-5") if ref_shape_elem is not None else "1e-5"
        )
        ref_depth = float(
            ref_shape_elem.get("depth", "10") if ref_shape_elem is not None else "10"
        )

        polygons = _parse_svg_paths(d_strings)

        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []
        for poly in polygons:
            if len(poly) < 3:
                continue
            # Scale factor: convert SVG pixels → mm using the same scale
            # Bambu Studio uses.  ref_scale is typically ~1e-5.
            _extrude_polygon(poly, ref_depth, ref_scale, vertices, triangles)

        new_obj = ET.SubElement(resources, f"{tp}object", attrib={
            "id": str(new_obj_id), "type": "model",
        })
        mesh_elem = ET.SubElement(new_obj, f"{tp}mesh")
        verts_elem = ET.SubElement(mesh_elem, f"{tp}vertices")
        for x, y, z in vertices:
            ET.SubElement(verts_elem, f"{tp}vertex", attrib={
                "x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}",
            })
        tris_elem = ET.SubElement(mesh_elem, f"{tp}triangles")
        for v1, v2, v3 in triangles:
            ET.SubElement(tris_elem, f"{tp}triangle", attrib={
                "v1": str(v1), "v2": str(v2), "v3": str(v3),
            })
        face_count = len(triangles)

        # 5c. Add a <component> to the parent object
        ET.SubElement(components, f"{tp}component", attrib={
            "objectid": str(new_obj_id),
            "transform": ref_component_transform,
        })

        # 5d. Add a <part> to model_settings.config
        # Part ID MUST match the object ID — Bambu Studio uses this to
        # correlate components with their settings.
        part_elem = ET.SubElement(parent_settings_obj, "part", attrib={
            "id": str(new_obj_id),
            "subtype": "normal_part",
        })
        ET.SubElement(part_elem, "metadata", attrib={
            "key": "name", "value": svg_filename,
        })
        ET.SubElement(part_elem, "metadata", attrib={
            "key": "matrix", "value": ref_matrix,
        })
        ET.SubElement(part_elem, "metadata", attrib={
            "key": "extruder", "value": str(next_extruder),
        })

        # Clone the BambuStudioShape element with updated paths
        shape_attribs = dict(ref_shape_elem.attrib) if ref_shape_elem is not None else {}
        shape_attribs["filepath"] = svg_filename
        shape_attribs["filepath3mf"] = svg_3mf_entry
        ET.SubElement(part_elem, "BambuStudioShape", attrib=shape_attribs)

        # Add mesh_stat placeholder
        ET.SubElement(part_elem, "mesh_stat", attrib={
            "face_count": str(face_count),
            "edges_fixed": "0",
            "degenerate_facets": "0",
            "facets_removed": "0",
            "facets_reversed": "0",
            "backwards_edges": "0",
        })

        new_obj_id += 1
        next_extruder += 1

    # Update the parent object's face_count metadata
    _update_face_count(parent_settings_obj, model_root, tp, ns)

    # --------------------------------------------------
    # 6. Write the output 3MF
    # --------------------------------------------------
    _write_output_3mf(output_3mf, entries, model_root, settings_root, project_3mf)

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
# Assembly structure helpers
# -----------------------------------------------------------------------


def _find_parent_object(
    model_root: ET.Element, tp: str, ns: dict,
) -> tuple[ET.Element, int]:
    """Find the assembly parent object (the one with <components>)."""
    resources = model_root.find(f"{tp}resources")
    if resources is None:
        raise ValueError("No <resources> section in the model.")

    for obj in resources:
        obj_tag = obj.tag.split("}")[-1] if "}" in obj.tag else obj.tag
        if obj_tag != "object":
            continue
        comps = obj.find(f"{tp}components")
        if comps is not None:
            return obj, int(obj.get("id", "0"))

    raise ValueError(
        "No parent/assembly object found in the 3MF model. "
        "Expected an <object> with <components>."
    )


def _find_svg_part(
    settings_root: ET.Element, parent_id: int,
) -> dict | None:
    """Find the SVG-derived part in settings config.

    Returns a dict with keys: part_id, matrix, extruder, shape_elem, svg_3mf_path.
    """
    for obj_elem in settings_root.findall("object"):
        if obj_elem.get("id") != str(parent_id):
            continue

        for part in obj_elem.findall("part"):
            shape = part.find("BambuStudioShape")
            if shape is not None:
                part_id = int(part.get("id", "0"))
                matrix = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
                extruder = "1"
                for meta in part.findall("metadata"):
                    if meta.get("key") == "matrix":
                        matrix = meta.get("value", matrix)
                    elif meta.get("key") == "extruder":
                        extruder = meta.get("value", extruder)
                return {
                    "part_id": part_id,
                    "matrix": matrix,
                    "extruder": extruder,
                    "shape_elem": shape,
                    "svg_3mf_path": shape.get("filepath3mf", ""),
                }

    return None


def _find_settings_object(
    settings_root: ET.Element, parent_id: int,
) -> ET.Element | None:
    """Find the <object> element in settings for the given parent ID."""
    for obj_elem in settings_root.findall("object"):
        if obj_elem.get("id") == str(parent_id):
            return obj_elem
    return None


def _max_object_id(model_root: ET.Element) -> int:
    """Return the highest object ID in the model."""
    tp = ""
    if model_root.tag.startswith("{"):
        tp = model_root.tag.split("}")[0] + "}"

    max_id = 0
    resources = model_root.find(f"{tp}resources")
    if resources is None:
        return max_id

    for obj in resources:
        tag = obj.tag.split("}")[-1] if "}" in obj.tag else obj.tag
        if tag == "object":
            oid = int(obj.get("id", "0"))
            max_id = max(max_id, oid)

    return max_id


def _max_part_id(settings_root: ET.Element, parent_id: int) -> int:
    """Return the highest part ID within the parent object's settings."""
    max_id = 0
    obj = _find_settings_object(settings_root, parent_id)
    if obj is None:
        return max_id
    for part in obj.findall("part"):
        pid = int(part.get("id", "0"))
        max_id = max(max_id, pid)
    return max_id


def _max_extruder(settings_root: ET.Element, parent_id: int) -> int:
    """Return the highest extruder number in the parent object."""
    max_ext = 0
    obj = _find_settings_object(settings_root, parent_id)
    if obj is None:
        return max_ext
    # Check object-level extruder
    for meta in obj.findall("metadata"):
        if meta.get("key") == "extruder":
            try:
                max_ext = max(max_ext, int(meta.get("value", "0")))
            except ValueError:
                pass
    # Check part-level extruders
    for part in obj.findall("part"):
        for meta in part.findall("metadata"):
            if meta.get("key") == "extruder":
                try:
                    max_ext = max(max_ext, int(meta.get("value", "0")))
                except ValueError:
                    pass
    return max_ext


def _update_face_count(
    settings_obj: ET.Element,
    model_root: ET.Element,
    tp: str,
    ns: dict,
) -> None:
    """Update the face_count metadata on the parent settings object."""
    total = 0
    for part in settings_obj.findall("part"):
        for ms in part.findall("mesh_stat"):
            try:
                total += int(ms.get("face_count", "0"))
            except ValueError:
                pass
    # Update or add the face_count metadata
    for meta in settings_obj.findall("metadata"):
        if meta.get("face_count") is not None:
            meta.set("face_count", str(total))
            return
    # If no face_count metadata exists, don't add one
    # (it's optional in Bambu Studio)


def _clone_element(elem: ET.Element) -> ET.Element:
    """Deep-clone an ElementTree element."""
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in elem:
        new.append(_clone_element(child))
    return new


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
    indent = "\n" + " " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# -----------------------------------------------------------------------
# Aliases for imports used by tests / CLI
# -----------------------------------------------------------------------

# Keep backward-compatible imports used by test_clone_parts.py
_find_reference_part = None  # removed — tests need updating
_inject_objects = None  # removed — tests need updating
_update_settings = None  # removed — tests need updating
_svgs_to_meshes = None  # removed — tests need updating
_extract_hex_color = None  # removed — tests need updating
