"""3MF file output: multi-colour 3D-printable archive for Bambu Studio.

Converts per-colour SVG path data into a 3MF archive where each colour
occupies its own ``<object>`` with a pre-assigned extruder slot.  Opening
the resulting ``.3mf`` in Bambu Studio shows every colour as a separate
part with its filament already selected — no manual splitting required.

The 3MF format is a ZIP archive with the following structure::

    output.3mf
    ├── [Content_Types].xml
    ├── _rels/.rels
    ├── 3D/3dmodel.model
    └── Metadata/model_settings.config

Only standard-library modules (``zipfile``, ``xml``, ``re``) and the
already-required ``numpy`` are used — no new dependencies.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def write_3mf(
    path: Path,
    layers: list[dict],
    width: int,
    height: int,
    *,
    scale: float = 1.0,
    thickness: float = 2.0,
) -> Path:
    """Write a 3MF file with one object per colour layer.

    Args:
        path: Output ``.3mf`` file path.
        layers: List of layer dicts with ``hex_color``, ``color_name``,
            ``svg_paths`` keys (same format as the SVG writer).
        width: Source image width in pixels.
        height: Source image height in pixels.
        scale: Multiplier applied to XY coordinates (pixels → mm).
        thickness: Extrusion thickness in mm (Z height of each part).

    Returns:
        The *path* that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    objects: list[_MeshObject] = []

    for i, layer in enumerate(layers):
        svg_paths = layer.get("svg_paths", [])
        if not svg_paths:
            continue

        # Parse SVG path strings into 2D polygon loops
        polygons = _parse_svg_paths(svg_paths)
        if not polygons:
            continue

        # Triangulate and extrude each polygon
        vertices: list[tuple[float, float, float]] = []
        triangles: list[tuple[int, int, int]] = []

        for poly in polygons:
            if len(poly) < 3:
                continue
            _extrude_polygon(poly, thickness, scale, vertices, triangles)

        if not triangles:
            continue

        objects.append(_MeshObject(
            obj_id=i + 1,
            name=f"{layer.get('color_name', 'Color')} ({layer['hex_color']})",
            hex_color=layer["hex_color"],
            vertices=vertices,
            triangles=triangles,
            extruder=i + 1,
        ))

    if not objects:
        raise ValueError("No valid geometry to write — all layers have empty paths.")

    _write_3mf_zip(path, objects)
    return path


# -----------------------------------------------------------------------
# Internal data structures
# -----------------------------------------------------------------------

class _MeshObject:
    """One colour's 3D mesh + metadata."""

    __slots__ = ("obj_id", "name", "hex_color", "vertices", "triangles", "extruder")

    def __init__(
        self,
        obj_id: int,
        name: str,
        hex_color: str,
        vertices: list[tuple[float, float, float]],
        triangles: list[tuple[int, int, int]],
        extruder: int,
    ):
        self.obj_id = obj_id
        self.name = name
        self.hex_color = hex_color
        self.vertices = vertices
        self.triangles = triangles
        self.extruder = extruder


# -----------------------------------------------------------------------
# SVG path parsing
# -----------------------------------------------------------------------

# Regex to tokenize SVG path 'd' strings.
_SVG_CMD_RE = re.compile(r"([MmLlCcHhVvZz])|(-?\d+(?:\.\d+)?)")


def _parse_svg_paths(svg_paths: list[str]) -> list[list[tuple[float, float]]]:
    """Parse SVG path ``d`` strings into lists of 2D polygon vertices.

    Handles M (moveto), L (lineto), C (cubic Bézier — linearised),
    H (horizontal line), V (vertical line), and Z (close).
    Both absolute and relative variants are supported.
    """
    polygons: list[list[tuple[float, float]]] = []

    for d_str in svg_paths:
        tokens = _SVG_CMD_RE.findall(d_str)
        nums: list[float] = []
        cmd = ""
        current_polygon: list[tuple[float, float]] = []
        cx, cy = 0.0, 0.0  # current point
        sx, sy = 0.0, 0.0  # sub-path start

        i = 0
        while i < len(tokens):
            letter, number = tokens[i]
            if letter:
                cmd = letter
                i += 1
                continue
            if number:
                nums.append(float(number))
            i += 1

            # Dispatch when enough numbers collected for the command
            if cmd in ("M", "m") and len(nums) >= 2:
                # Close previous sub-path if it has points
                if len(current_polygon) >= 3:
                    polygons.append(current_polygon)
                current_polygon = []
                if cmd == "M":
                    cx, cy = nums[0], nums[1]
                else:
                    cx += nums[0]
                    cy += nums[1]
                sx, sy = cx, cy
                current_polygon.append((cx, cy))
                nums.clear()
                # Subsequent coordinate pairs are implicit LineTo
                cmd = "L" if cmd == "M" else "l"

            elif cmd in ("L", "l") and len(nums) >= 2:
                if cmd == "L":
                    cx, cy = nums[0], nums[1]
                else:
                    cx += nums[0]
                    cy += nums[1]
                current_polygon.append((cx, cy))
                nums.clear()

            elif cmd in ("H", "h") and len(nums) >= 1:
                if cmd == "H":
                    cx = nums[0]
                else:
                    cx += nums[0]
                current_polygon.append((cx, cy))
                nums.clear()

            elif cmd in ("V", "v") and len(nums) >= 1:
                if cmd == "V":
                    cy = nums[0]
                else:
                    cy += nums[0]
                current_polygon.append((cx, cy))
                nums.clear()

            elif cmd in ("C", "c") and len(nums) >= 6:
                # Cubic Bézier — linearise with ~8 samples
                if cmd == "C":
                    x1, y1, x2, y2, x3, y3 = nums[:6]
                else:
                    x1, y1 = cx + nums[0], cy + nums[1]
                    x2, y2 = cx + nums[2], cy + nums[3]
                    x3, y3 = cx + nums[4], cy + nums[5]
                pts = _linearise_cubic(cx, cy, x1, y1, x2, y2, x3, y3, 8)
                current_polygon.extend(pts)
                cx, cy = x3, y3
                nums = nums[6:]  # keep leftover for implicit repeat
                continue  # don't clear — there may be more Bézier coords

            elif cmd in ("Z", "z"):
                cx, cy = sx, sy
                if len(current_polygon) >= 3:
                    polygons.append(current_polygon)
                current_polygon = []
                nums.clear()

        # Flush any remaining polygon
        if len(current_polygon) >= 3:
            polygons.append(current_polygon)

    return polygons


def _linearise_cubic(
    x0: float, y0: float,
    x1: float, y1: float,
    x2: float, y2: float,
    x3: float, y3: float,
    n: int,
) -> list[tuple[float, float]]:
    """Approximate a cubic Bézier with *n* line segments."""
    pts: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * x0 + 3 * u**2 * t * x1 + 3 * u * t**2 * x2 + t**3 * x3
        y = u**3 * y0 + 3 * u**2 * t * y1 + 3 * u * t**2 * y2 + t**3 * y3
        pts.append((x, y))
    return pts


# -----------------------------------------------------------------------
# Triangulation (ear-clipping)
# -----------------------------------------------------------------------

def _triangulate_polygon(
    polygon: list[tuple[float, float]],
) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon.

    Returns a list of (i, j, k) index triples into *polygon*.
    """
    n = len(polygon)
    if n < 3:
        return []

    # Work with a mutable index list
    indices = list(range(n))

    # Ensure counter-clockwise winding
    if _signed_area(polygon) < 0:
        indices.reverse()

    tris: list[tuple[int, int, int]] = []
    max_iter = n * n  # safety limit

    while len(indices) > 2 and max_iter > 0:
        max_iter -= 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_idx = indices[(i - 1) % m]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % m]

            ax, ay = polygon[prev_idx]
            bx, by = polygon[curr_idx]
            cx, cy = polygon[next_idx]

            # Check convexity (cross product > 0 for CCW)
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross <= 0:
                continue

            # Check that no other vertex lies inside this triangle
            is_ear = True
            for j in range(m):
                if j in (i, (i - 1) % m, (i + 1) % m):
                    continue
                px, py = polygon[indices[j]]
                if _point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
                    is_ear = False
                    break

            if is_ear:
                tris.append((prev_idx, curr_idx, next_idx))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            # Degenerate polygon — stop
            break

    return tris


def _signed_area(polygon: list[tuple[float, float]]) -> float:
    """Signed area of a 2D polygon (positive = CCW)."""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _point_in_triangle(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> bool:
    """Return True if point (px, py) is strictly inside triangle ABC."""
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# -----------------------------------------------------------------------
# 3D mesh extrusion
# -----------------------------------------------------------------------

def _extrude_polygon(
    polygon: list[tuple[float, float]],
    thickness: float,
    scale: float,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> None:
    """Extrude a 2D polygon into a 3D slab and append to *vertices* / *triangles*.

    Top face at z = thickness, bottom face at z = 0.
    """
    tris_2d = _triangulate_polygon(polygon)
    if not tris_2d:
        return

    base = len(vertices)
    n = len(polygon)

    # Add vertices: bottom (z=0) then top (z=thickness)
    for x, y in polygon:
        vertices.append((x * scale, y * scale, 0.0))
    for x, y in polygon:
        vertices.append((x * scale, y * scale, thickness))

    # Bottom face triangles (reversed winding for outward normals)
    for i, j, k in tris_2d:
        triangles.append((base + i, base + k, base + j))

    # Top face triangles
    for i, j, k in tris_2d:
        triangles.append((base + n + i, base + n + j, base + n + k))

    # Side walls
    for i in range(n):
        j = (i + 1) % n
        # Bottom-left, bottom-right, top-right, top-left
        bl = base + i
        br = base + j
        tr = base + n + j
        tl = base + n + i
        # Two triangles per quad
        triangles.append((bl, br, tr))
        triangles.append((bl, tr, tl))


# -----------------------------------------------------------------------
# 3MF ZIP writing
# -----------------------------------------------------------------------

def _write_3mf_zip(path: Path, objects: list[_MeshObject]) -> None:
    """Write the complete 3MF ZIP archive."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml())
        zf.writestr("_rels/.rels", _rels_xml())
        zf.writestr("3D/3dmodel.model", _model_xml(objects))
        zf.writestr("Metadata/model_settings.config", _settings_config_xml(objects))


def _content_types_xml() -> str:
    """``[Content_Types].xml`` — MIME type declarations."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" ContentType='
        '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        '</Types>\n'
    )


def _rels_xml() -> str:
    """``_rels/.rels`` — package relationship pointing to the model."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type='
        '"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>\n'
    )


def _model_xml(objects: list[_MeshObject]) -> str:
    """``3D/3dmodel.model`` — core model with mesh data."""
    NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    model = Element("model", attrib={
        "unit": "millimeter",
        "xmlns": NS,
    })

    resources = SubElement(model, "resources")

    for obj in objects:
        obj_elem = SubElement(resources, "object", attrib={
            "id": str(obj.obj_id),
            "type": "model",
            "name": obj.name,
        })
        mesh = SubElement(obj_elem, "mesh")

        # Vertices
        verts_elem = SubElement(mesh, "vertices")
        for x, y, z in obj.vertices:
            SubElement(verts_elem, "vertex", attrib={
                "x": f"{x:.4f}",
                "y": f"{y:.4f}",
                "z": f"{z:.4f}",
            })

        # Triangles
        tris_elem = SubElement(mesh, "triangles")
        for v1, v2, v3 in obj.triangles:
            SubElement(tris_elem, "triangle", attrib={
                "v1": str(v1),
                "v2": str(v2),
                "v3": str(v3),
            })

    build = SubElement(model, "build")
    for obj in objects:
        SubElement(build, "item", attrib={
            "objectid": str(obj.obj_id),
        })

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + _indent_xml(model)


def _settings_config_xml(objects: list[_MeshObject]) -> str:
    """``Metadata/model_settings.config`` — Bambu Studio extruder mapping."""
    config = Element("config")
    for obj in objects:
        obj_elem = SubElement(config, "object", attrib={"id": str(obj.obj_id)})
        SubElement(obj_elem, "metadata", attrib={
            "key": "name",
            "value": obj.name,
        })
        SubElement(obj_elem, "metadata", attrib={
            "key": "extruder",
            "value": str(obj.extruder),
        })
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + _indent_xml(config)


def _indent_xml(elem: Element, level: int = 0) -> str:
    """Pretty-print an ElementTree element to a string with indentation."""
    indent = "  " * level
    result = f"{indent}<{elem.tag}"
    for k, v in elem.attrib.items():
        result += f' {k}="{v}"'
    children = list(elem)
    if children:
        result += ">\n"
        for child in children:
            result += _indent_xml(child, level + 1)
        result += f"{indent}</{elem.tag}>\n"
    else:
        result += "/>\n"
    return result
