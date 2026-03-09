"""Interactive conversion session: holds intermediate state and supports
incremental re-processing.

A :class:`Session` wraps the existing pipeline stages so they can be driven
one-at-a-time from a GUI or any interactive caller, rather than as a single
end-to-end ``process_single`` call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .color_utils import hex_to_rgb, nearest_color_name, rgb_to_hex
from .image_loader import load_image
from .layer_separator import separate_layers, separate_objects
from .preprocessor import PreprocessReport, preprocess_image
from .quantizer import quantize_colors
from .svg_importer import import_svg_as_layers
from .svg_writer import write_preview, write_svg_files
from .tm_remover import remove_tm_symbols
from .tracer import trace_mask_to_svg_paths


# ---------------------------------------------------------------------------
# Layer data class
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Public snapshot of one colour layer — returned by :meth:`Session.get_layers`."""

    index: int
    rgb: tuple[int, int, int]
    hex_color: str
    color_name: str
    pixel_count: int
    visible: bool = True
    name: str = ""


# ---------------------------------------------------------------------------
# Undo / Redo snapshot
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    """Lightweight snapshot of mutable session state for undo/redo."""
    layers: list[dict] | None
    traced: bool
    description: str = ""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Holds the intermediate state of a single logo -> SVG conversion.

    Typical interactive flow::

        s = Session()
        s.load("logo.png")
        s.quantize(n_colors=4)
        layers = s.get_layers()
        s.change_color(0, "#FF0000")
        s.remove_color(3)
        s.trace()
        s.export("./output", combined=True)

    All heavy methods (:meth:`load`, :meth:`quantize`, :meth:`trace`,
    :meth:`export`) are synchronous.  The GUI wraps them in ``QThread``.
    """

    # -- internal state --------------------------------------------------
    _path: Path | None
    _image: np.ndarray | None           # (H, W, 3) uint8 RGB
    _fg_mask: np.ndarray | None         # (H, W) bool
    _labels: np.ndarray | None          # (H, W) int32, -1 = bg
    _centers_rgb: np.ndarray | None     # (k, 3) uint8
    _layers: list[dict] | None         # pipeline-style layer dicts
    _traced: bool

    # Potrace & pipeline tunables
    min_area: int
    alphamax: float
    opttolerance: float
    turdsize: int
    scale: float
    width: float | None

    # Undo/redo
    _undo_stack: list[_Snapshot]
    _redo_stack: list[_Snapshot]
    _max_undo: int

    def __init__(
        self,
        *,
        min_area: int = 100,
        alphamax: float = 1.0,
        opttolerance: float = 0.2,
        turdsize: int = 2,
        scale: float = 1.0,
        width: float | None = None,
    ) -> None:
        self._path = None
        self._image = None
        self._fg_mask = None
        self._labels = None
        self._centers_rgb = None
        self._layers = None
        self._traced = False
        self._separation_mode: str = "color"  # "color" or "object"

        self.min_area = min_area
        self.alphamax = alphamax
        self.opttolerance = opttolerance
        self.turdsize = turdsize
        self.scale = scale
        self.width = width

        # Undo/redo stacks
        self._undo_stack = []
        self._redo_stack = []
        self._max_undo = 20

        # Preprocessing report (last analysis)
        self._preprocess_report: PreprocessReport | None = None

    # -- public properties ------------------------------------------------

    @property
    def image(self) -> np.ndarray | None:
        """The loaded RGB image, or ``None``."""
        return self._image

    @property
    def fg_mask(self) -> np.ndarray | None:
        """Foreground mask, or ``None``."""
        return self._fg_mask

    @property
    def image_size(self) -> tuple[int, int] | None:
        """``(height, width)`` of the loaded image, or ``None``."""
        if self._image is None:
            return None
        return self._image.shape[:2]

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._image is not None

    @property
    def is_quantized(self) -> bool:
        return self._layers is not None

    @property
    def is_traced(self) -> bool:
        return self._traced

    @property
    def preprocess_report(self) -> PreprocessReport | None:
        """Last preprocessing analysis report, or ``None``."""
        return self._preprocess_report

    # -- stage 1: load ----------------------------------------------------

    def load(self, path: str | Path, bg_color: str | None = None) -> None:
        """Load an image and run background detection.

        Resets all downstream state (quantization, layers, trace).
        """
        self._path = Path(path)
        self._image, self._fg_mask = load_image(self._path, bg_color)
        # Reset downstream
        self._labels = None
        self._centers_rgb = None
        self._layers = None
        self._traced = False
        self._undo_stack.clear()
        self._redo_stack.clear()

    def remove_background(
        self,
        bg_color: str | None = None,
        *,
        remove_tm: bool = True,
    ) -> int:
        """Reload the source image and rebuild the foreground mask.

        This restores the explicit background-removal workflow for interactive
        callers. The source image is reloaded from disk so background detection
        runs against the original pixels rather than already-processed state.

        Returns:
            Number of TM-like components removed after reloading.
        """
        if self._path is None:
            raise RuntimeError("No image loaded. Call load() first.")

        self.load(self._path, bg_color=bg_color)
        if remove_tm:
            return self.remove_tm()

        self.ensure_square()
        return 0

    # -- preprocessing ----------------------------------------------------

    def run_preprocessing(
        self,
        *,
        contrast: bool = False,
        sharpen: bool = False,
        denoise: bool = False,
        contrast_strength: float = 2.0,
        sharpen_strength: float = 1.0,
        denoise_strength: int = 10,
    ) -> PreprocessReport:
        """Run smart preprocessing on the loaded image.

        Should be called after :meth:`load` and before :meth:`quantize`.
        Modifies ``self._image`` in place.  Returns the analysis report.
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        self._image, report = preprocess_image(
            self._image,
            self._fg_mask,
            contrast=contrast,
            sharpen=sharpen,
            denoise=denoise,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            denoise_strength=denoise_strength,
        )
        self._preprocess_report = report
        return report

    def analyze_image(self) -> PreprocessReport:
        """Analyze the loaded image without modifying it.

        Returns a :class:`PreprocessReport` with recommendations.
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")
        from .preprocessor import analyze_image
        report = analyze_image(self._image, self._fg_mask)
        self._preprocess_report = report
        return report

    # -- stage 2: quantize ------------------------------------------------

    def remove_tm(self) -> int:
        """Remove small TM / (R) symbols from the foreground mask margins.

        Should be called after :meth:`load` and before :meth:`quantize`.
        After TM removal, pads the image to a square canvas.
        Returns the number of components removed.
        """
        if self._fg_mask is None:
            raise RuntimeError("No image loaded. Call load() first.")
        self._fg_mask, removed = remove_tm_symbols(self._fg_mask)
        self._pad_to_square()
        return removed

    def ensure_square(self) -> None:
        """Pad image to a square canvas if not already square.

        Call this after :meth:`load` when TM removal is skipped.
        When :meth:`remove_tm` is used, padding is applied automatically.
        """
        if self._image is not None:
            self._pad_to_square()

    def quantize(
        self,
        n_colors: int | None = None,
        target_colors: list[str] | None = None,
    ) -> None:
        """Run (or re-run) colour quantization on the current image.

        After this call :meth:`get_layers` will return the new layer list.
        Tracing state is reset.
        """
        if self._image is None or self._fg_mask is None:
            raise RuntimeError("No image loaded. Call load() first.")

        image = self._image
        fg_mask = self._fg_mask

        if target_colors:
            labels, centers_rgb = self._assign_to_target_colors(
                image, fg_mask, target_colors
            )
        else:
            fg_mask_eroded = self._erode_mask(fg_mask)
            labels, centers_rgb = quantize_colors(
                image, fg_mask_eroded, n_colors=n_colors
            )
            labels = self._recover_fringe_pixels(
                image, labels, centers_rgb, fg_mask, fg_mask_eroded
            )

        self._labels = labels
        self._centers_rgb = centers_rgb
        self._build_layers()
        self._traced = False

    # -- stage 3: layer queries / mutations --------------------------------

    def get_layers(self) -> list[LayerInfo]:
        """Return a list of :class:`LayerInfo` snapshots for the current state."""
        if self._layers is None:
            return []
        result: list[LayerInfo] = []
        for i, layer in enumerate(self._layers):
            result.append(
                LayerInfo(
                    index=i,
                    rgb=layer["rgb"],
                    hex_color=layer["hex_color"],
                    color_name=layer["color_name"],
                    pixel_count=int(np.count_nonzero(layer["mask"])),
                    visible=layer.get("visible", True),
                    name=layer.get("name", f"Layer {i + 1}"),
                )
            )
        return result

    def get_layer_count(self) -> int:
        """Number of active colour layers."""
        return len(self._layers) if self._layers else 0

    def remove_color(self, index: int) -> None:
        """Remove a colour layer entirely.  Resets tracing."""
        self._require_layers("remove_color")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")
        self._save_undo("Remove layer")
        del self._layers[index]
        self._traced = False

    def remove_colors(self, indices: list[int]) -> None:
        """Remove multiple colour layers at once.  Resets tracing."""
        self._require_layers("remove_colors")
        indices_sorted = sorted(set(indices))
        for idx in indices_sorted:
            if not 0 <= idx < len(self._layers):
                raise IndexError(f"Layer index {idx} out of range.")
        self._save_undo(f"Remove {len(indices_sorted)} layers")
        for idx in reversed(indices_sorted):
            del self._layers[idx]
        self._traced = False

    def merge_by_color(self) -> int:
        """Merge all layers that share the same hex colour.

        For each unique colour, the first (lowest-index) layer with that
        colour keeps its position and absorbs the masks of all later layers
        with the same colour.  Those later layers are removed.

        Returns the number of layers that were removed.
        """
        self._require_layers("merge_by_color")
        if len(self._layers) < 2:
            return 0

        self._save_undo("Merge by color")
        seen: dict[str, int] = {}  # hex_color -> index of keep layer
        to_remove: list[int] = []
        for i, layer in enumerate(self._layers):
            hc = layer["hex_color"]
            if hc in seen:
                keep = seen[hc]
                self._layers[keep]["mask"] = np.maximum(
                    self._layers[keep]["mask"], layer["mask"]
                )
                to_remove.append(i)
            else:
                seen[hc] = i

        if not to_remove:
            # Nothing to merge — pop the undo snapshot we just saved
            self._undo_stack.pop()
            return 0

        for idx in reversed(to_remove):
            del self._layers[idx]
        self._traced = False
        return len(to_remove)

    def merge_colors(self, indices: list[int]) -> None:
        """Merge two or more colour layers into one.

        The first index in *indices* is the "keep" layer — its colour is
        preserved.  Masks from the other layers are OR-ed into it, and those
        layers are removed.  Resets tracing.
        """
        self._require_layers("merge_colors")
        if len(indices) < 2:
            raise ValueError("merge_colors requires at least two indices.")
        indices_sorted = sorted(set(indices))
        for idx in indices_sorted:
            if not 0 <= idx < len(self._layers):
                raise IndexError(f"Layer index {idx} out of range.")

        self._save_undo("Merge layers")
        keep_idx = indices_sorted[0]
        merge_idxs = indices_sorted[1:]

        for idx in merge_idxs:
            self._layers[keep_idx]["mask"] = np.maximum(
                self._layers[keep_idx]["mask"], self._layers[idx]["mask"]
            )

        for idx in reversed(merge_idxs):
            del self._layers[idx]

        self._traced = False

    def change_colors(self, indices: list[int], new_hex: str) -> None:
        """Change the colour of multiple layers at once."""
        self._require_layers("change_colors")
        indices_sorted = sorted(set(indices))
        for idx in indices_sorted:
            if not 0 <= idx < len(self._layers):
                raise IndexError(f"Layer index {idx} out of range.")
        self._save_undo(f"Change color of {len(indices_sorted)} layers")
        r, g, b = hex_to_rgb(new_hex)
        for idx in indices_sorted:
            self._layers[idx]["rgb"] = (r, g, b)
            self._layers[idx]["hex_color"] = rgb_to_hex(r, g, b)
            self._layers[idx]["color_name"] = nearest_color_name(r, g, b)

    def change_color(self, index: int, new_hex: str) -> None:
        """Change the display / output colour of a layer (no re-quantize)."""
        self._require_layers("change_color")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")
        self._save_undo("Change color")
        r, g, b = hex_to_rgb(new_hex)
        self._layers[index]["rgb"] = (r, g, b)
        self._layers[index]["hex_color"] = rgb_to_hex(r, g, b)
        self._layers[index]["color_name"] = nearest_color_name(r, g, b)

    @property
    def separation_mode(self) -> str:
        """Current layer separation mode: ``'color'`` or ``'object'``."""
        return self._separation_mode

    def set_separation_mode(self, mode: str) -> None:
        """Set the layer separation mode."""
        if mode not in ("color", "object"):
            raise ValueError(f"Invalid separation mode: {mode!r}")
        if mode == self._separation_mode:
            return
        self._separation_mode = mode
        if self._labels is not None:
            self._build_layers()
            self._traced = False

    def set_layer_visibility(self, index: int, visible: bool) -> None:
        """Toggle visibility (for preview composite only -- does not affect export)."""
        self._require_layers("set_layer_visibility")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")
        self._layers[index]["visible"] = visible

    # -- layer hit-test / move --------------------------------------------

    def layer_at_pixel(self, x: int, y: int) -> int | None:
        """Return the index of the topmost visible layer at pixel (x, y).

        Iterates layers from top (last) to bottom (first) and returns the
        first layer whose mask is non-zero at the given coordinates.
        Returns ``None`` if no layer covers that pixel.
        """
        if self._layers is None or self._image is None:
            return None
        h, w = self._image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        for i in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[i]
            if not layer.get("visible", True):
                continue
            if layer["mask"][y, x] > 0:
                return i
        return None

    def move_layer_pixels(self, index: int, dx: int, dy: int) -> None:
        """Translate a layer's mask by (dx, dy) pixels.

        Pixels that shift off-canvas are clipped.  Saves an undo snapshot.
        """
        self._require_layers("move_layer_pixels")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")

        self._save_undo("Move layer")
        mask = self._layers[index]["mask"]
        h, w = mask.shape

        # Use numpy roll + zero-fill for the translation
        new_mask = np.zeros_like(mask)

        # Compute source and destination slices
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)

        dst_y_start = max(0, dy)
        dst_y_end = min(h, h + dy)
        dst_x_start = max(0, dx)
        dst_x_end = min(w, w + dx)

        if (src_y_end > src_y_start and src_x_end > src_x_start
                and dst_y_end > dst_y_start and dst_x_end > dst_x_start):
            new_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                mask[src_y_start:src_y_end, src_x_start:src_x_end]

        self._layers[index]["mask"] = new_mask
        self._resolve_overlaps()
        self._traced = False

    def get_layer_bbox(self, index: int) -> tuple[int, int, int, int] | None:
        """Return the bounding box (x, y, w, h) of a layer's mask content.

        Returns ``None`` if the mask is empty.
        """
        self._require_layers("get_layer_bbox")
        if not 0 <= index < len(self._layers):
            return None
        mask = self._layers[index]["mask"]
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not np.any(rows):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

    # -- layer rename -----------------------------------------------------

    def rename_layer(self, index: int, name: str) -> None:
        """Rename a layer."""
        self._require_layers("rename_layer")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")
        self._layers[index]["name"] = name

    # -- layer reorder ----------------------------------------------------

    def move_layer_up(self, index: int) -> bool:
        """Move a layer one position up (towards index 0 = rendered first).

        Returns True if moved, False if already at top.
        Resolves overlaps after the move.
        """
        self._require_layers("move_layer_up")
        if index <= 0 or index >= len(self._layers):
            return False
        self._save_undo("Move layer up")
        self._layers[index], self._layers[index - 1] = (
            self._layers[index - 1], self._layers[index]
        )
        self._resolve_overlaps()
        self._traced = False
        return True

    def move_layer_down(self, index: int) -> bool:
        """Move a layer one position down (towards end = rendered last/on top).

        Returns True if moved, False if already at bottom.
        Resolves overlaps after the move.
        """
        self._require_layers("move_layer_down")
        if index < 0 or index >= len(self._layers) - 1:
            return False
        self._save_undo("Move layer down")
        self._layers[index], self._layers[index + 1] = (
            self._layers[index + 1], self._layers[index]
        )
        self._resolve_overlaps()
        self._traced = False
        return True

    # -- duplicate layer --------------------------------------------------

    def duplicate_layer(self, index: int) -> int:
        """Duplicate a layer and insert the copy immediately after it.

        The duplicate gets a copy of the mask and colour, with " (copy)"
        appended to the name.  Overlaps are resolved after duplication
        (the duplicate will have no pixels since they all belong to the
        original -- useful as a starting point for the user to modify).

        Returns the index of the new layer.
        """
        self._require_layers("duplicate_layer")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")

        self._save_undo("Duplicate layer")
        original = self._layers[index]
        dup = {
            "rgb": original["rgb"],
            "hex_color": original["hex_color"],
            "color_name": original["color_name"],
            "mask": original["mask"].copy(),
            "cluster_idx": original.get("cluster_idx", -1),
            "visible": original.get("visible", True),
            "name": original.get("name", f"Layer {index + 1}") + " (copy)",
        }
        insert_at = index + 1
        self._layers.insert(insert_at, dup)
        # Resolve overlaps: the duplicate shares all pixels with the original.
        # Since the original has a lower index, it "owns" those pixels,
        # so the duplicate's mask gets cleared of overlap.
        self._resolve_overlaps()
        self._traced = False
        return insert_at

    # -- SVG import -------------------------------------------------------

    def import_svg(
        self,
        svg_path: str | Path,
        color_override: str | None = None,
    ) -> int:
        """Import an external SVG file as new colour layer(s).

        Each distinct colour in the SVG becomes a separate layer appended
        to the current layer list.  Overlaps with existing layers are
        resolved (imported layers take priority since they are on top).

        Returns the number of layers added.
        """
        if self._image is None:
            raise RuntimeError("No image loaded. Call load() first.")

        h, w = self._image.shape[:2]
        new_layers = import_svg_as_layers(
            svg_path, h, w, color_override=color_override,
        )

        if not new_layers:
            return 0

        if self._layers is None:
            self._layers = []

        self._save_undo("Import SVG")
        self._layers.extend(new_layers)
        self._resolve_overlaps()
        self._traced = False
        return len(new_layers)

    # -- border / outline tool --------------------------------------------

    def add_outline(
        self,
        index: int,
        width: int = 3,
        color: str = "#000000",
    ) -> int:
        """Add an outline around a layer's objects as a new layer.

        Creates a new layer containing only the border pixels (dilated mask
        minus original mask).  The new layer is inserted directly after the
        source layer.  Overlaps with other layers are resolved automatically.

        Returns the index of the newly created outline layer.
        """
        self._require_layers("add_outline")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")

        self._save_undo("Add outline")
        source_mask = self._layers[index]["mask"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1)
        )
        dilated = cv2.dilate(source_mask, kernel, iterations=1)
        outline_mask = dilated - source_mask

        r, g, b = hex_to_rgb(color)
        new_layer = {
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": outline_mask,
            "cluster_idx": -1,
            "visible": True,
            "name": f"Outline ({self._layers[index].get('name', f'Layer {index + 1}')})",
        }

        insert_at = index + 1
        self._layers.insert(insert_at, new_layer)
        self._resolve_overlaps()
        self._traced = False
        return insert_at

    def add_canvas_border(
        self,
        width: int = 10,
        color: str = "#000000",
    ) -> int:
        """Add a rectangular border around the entire canvas as a new layer.

        The border is inserted at the bottom of the layer stack (index 0)
        so it doesn't obscure any existing layers.  Overlaps are resolved.

        Returns the index of the newly created border layer.
        """
        if self._image is None:
            raise RuntimeError("No image loaded.")

        if self._layers is None:
            self._layers = []

        self._save_undo("Add canvas border")
        h, w_px = self._image.shape[:2]
        mask = np.zeros((h, w_px), dtype=np.uint8)
        mask[:width, :] = 255
        mask[-width:, :] = 255
        mask[:, :width] = 255
        mask[:, -width:] = 255

        r, g, b = hex_to_rgb(color)
        new_layer = {
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": mask,
            "cluster_idx": -1,
            "visible": True,
            "name": "Canvas border",
        }

        # Insert at the bottom so it doesn't cover existing content
        self._layers.insert(0, new_layer)
        self._resolve_overlaps()
        self._traced = False
        return 0

    # -- object border tool -----------------------------------------------

    def add_object_border(
        self,
        indices: list[int],
        width: int = 3,
        color: str = "#000000",
    ) -> int:
        """Add a border/outline around one or more layers combined.

        Merges the masks of all specified layers, dilates the combined
        mask, then subtracts the original combined mask to get the border
        pixels.  The border layer is inserted after the last source layer.

        Returns the index of the newly created border layer.
        """
        self._require_layers("add_object_border")
        for idx in indices:
            if not 0 <= idx < len(self._layers):
                raise IndexError(f"Layer index {idx} out of range.")

        self._save_undo("Add object border")

        # Combine masks from all selected layers
        h, w_px = self._layers[0]["mask"].shape
        combined = np.zeros((h, w_px), dtype=np.uint8)
        for idx in indices:
            combined = np.maximum(combined, self._layers[idx]["mask"])

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1)
        )
        dilated = cv2.dilate(combined, kernel, iterations=1)
        border_mask = dilated - combined

        r, g, b = hex_to_rgb(color)
        layer_names = []
        for idx in indices:
            layer_names.append(
                self._layers[idx].get("name", f"Layer {idx + 1}")
            )
        name_desc = ", ".join(layer_names) if len(layer_names) <= 3 else f"{len(layer_names)} layers"

        new_layer = {
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": border_mask,
            "cluster_idx": -1,
            "visible": True,
            "name": f"Border ({name_desc})",
        }

        insert_at = max(indices) + 1
        self._layers.insert(insert_at, new_layer)
        self._resolve_overlaps()
        self._traced = False
        return insert_at

    # -- text tool --------------------------------------------------------

    def add_text(
        self,
        text: str,
        *,
        color: str = "#000000",
        font_scale: float = 2.0,
        thickness: int = 3,
        x: int | None = None,
        y: int | None = None,
    ) -> int:
        """Render text as a new colour layer.

        Uses OpenCV's built-in font rendering.  The text is positioned at
        (x, y) or centered on the canvas if not specified.

        The text layer is added on top (highest index) and overlaps with
        existing layers are resolved (text takes priority).

        Returns the index of the newly created text layer.
        """
        if self._image is None:
            raise RuntimeError("No image loaded.")

        if self._layers is None:
            self._layers = []

        self._save_undo("Add text")
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        if x is None:
            x = max(0, (w - tw) // 2)
        if y is None:
            y = max(th, (h + th) // 2)

        cv2.putText(mask, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)

        r, g, b = hex_to_rgb(color)
        new_layer = {
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": mask,
            "cluster_idx": -1,
            "visible": True,
            "name": f'Text: "{text}"',
        }

        self._layers.append(new_layer)
        self._resolve_overlaps()
        self._traced = False
        return len(self._layers) - 1

    def edit_text_layer(
        self,
        index: int,
        text: str,
        *,
        color: str | None = None,
        font_scale: float = 2.0,
        thickness: int = 3,
    ) -> None:
        """Re-render text for an existing text layer (in place).

        Preserves the layer's position by centering the new text at the
        same bounding-box centre as the old mask content.
        """
        self._require_layers("edit_text_layer")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")

        self._save_undo("Edit text")
        layer = self._layers[index]
        h, w = self._image.shape[:2]

        # Find current centre of the old text mask
        bbox = self.get_layer_bbox(index)
        if bbox:
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
        else:
            cx, cy = w // 2, h // 2

        # Render new text
        mask = np.zeros((h, w), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x = max(0, cx - tw // 2)
        y = max(th, cy + th // 2)

        cv2.putText(mask, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)
        layer["mask"] = mask
        layer["name"] = f'Text: "{text}"'

        if color is not None:
            r, g, b = hex_to_rgb(color)
            layer["rgb"] = (r, g, b)
            layer["hex_color"] = rgb_to_hex(r, g, b)
            layer["color_name"] = nearest_color_name(r, g, b)

        self._resolve_overlaps()
        self._traced = False

    # -- overlap resolution -----------------------------------------------

    def _resolve_overlaps(self) -> None:
        """Ensure no two layers share any pixels.

        For 3D printing, each pixel must belong to exactly one colour layer.
        When layers overlap, the *last* layer in the list (highest index)
        wins — it keeps its pixels, and those pixels are cleared from all
        earlier layers.

        This is called automatically after any layer addition, import,
        or reorder operation.
        """
        if self._layers is None or len(self._layers) < 2:
            return

        # Build a "claimed" mask, iterating from last (top) to first (bottom).
        # Each layer keeps only pixels not already claimed by a layer above it.
        h, w = self._layers[0]["mask"].shape
        claimed = np.zeros((h, w), dtype=bool)

        for layer in reversed(self._layers):
            layer_mask = layer["mask"] > 0
            # Remove pixels already claimed by a higher layer
            layer_mask_clean = layer_mask & ~claimed
            layer["mask"] = (layer_mask_clean.astype(np.uint8) * 255)
            # Mark these pixels as claimed
            claimed |= layer_mask_clean

    # -- undo / redo ------------------------------------------------------

    def _save_undo(self, description: str = "") -> None:
        """Save the current layer state to the undo stack.

        Uses a fast shallow snapshot instead of ``copy.deepcopy`` — we
        explicitly copy only the mask arrays (numpy ``copy()``) and the
        scalar metadata.  ``paths`` lists are tiny strings and are cheap
        to copy.  This avoids the huge overhead of generic deepcopy on
        nested numpy arrays (which would pickle/unpickle each mask).
        """
        if self._layers is not None:
            snapshot = _Snapshot(
                layers=_fast_copy_layers(self._layers),
                traced=self._traced,
                description=description,
            )
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo the last layer mutation.  Returns True if undo was performed."""
        if not self._undo_stack:
            return False

        if self._layers is not None:
            self._redo_stack.append(_Snapshot(
                layers=_fast_copy_layers(self._layers),
                traced=self._traced,
                description="redo",
            ))

        snapshot = self._undo_stack.pop()
        self._layers = snapshot.layers
        self._traced = snapshot.traced
        return True

    def redo(self) -> bool:
        """Redo the last undone action.  Returns True if redo was performed."""
        if not self._redo_stack:
            return False

        if self._layers is not None:
            self._undo_stack.append(_Snapshot(
                layers=_fast_copy_layers(self._layers),
                traced=self._traced,
                description="undo",
            ))

        snapshot = self._redo_stack.pop()
        self._layers = snapshot.layers
        self._traced = snapshot.traced
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # -- stage 4: trace ---------------------------------------------------

    def trace(self) -> None:
        """Run Potrace on all current layer masks."""
        self._require_layers("trace")
        for layer in self._layers:
            layer["svg_paths"] = trace_mask_to_svg_paths(
                layer["mask"],
                turdsize=self.turdsize,
                alphamax=self.alphamax,
                opticurve=True,
                opttolerance=self.opttolerance,
            )
        self._traced = True

    # -- stage 5: export --------------------------------------------------

    def export(
        self,
        output_dir: str | Path,
        combined: bool = False,
        preview: bool = False,
    ) -> list[Path]:
        """Write SVG files for all layers.  Calls :meth:`trace` if needed."""
        if self._image is None:
            raise RuntimeError("No image loaded.")
        self._require_layers("export")

        if not self._traced:
            self.trace()

        out = Path(output_dir)
        height, width = self._image.shape[:2]
        base_name = self._path.stem if self._path else "output"

        effective_scale = self.scale
        if self.width is not None:
            effective_scale = self.width / width

        output_files = write_svg_files(
            base_name,
            self._layers,
            (height, width),
            out,
            combined,
            scale=effective_scale,
        )

        if preview:
            preview_path = write_preview(
                base_name, self._image, self._layers, (height, width), out
            )
            output_files.append(preview_path)

        return output_files

    def export_png(self, output_path: str | Path) -> Path:
        """Export the composite preview as a PNG file.

        Renders all visible layers onto a white background and saves as PNG.

        Args:
            output_path: Destination file path.

        Returns:
            The output path as a Path object.
        """
        if self._image is None:
            raise RuntimeError("No image loaded.")
        self._require_layers("export_png")

        from PIL import Image

        rgba = self.get_composite_preview()
        if rgba is None:
            raise RuntimeError("No layers to export.")

        # Composite over white background
        h, w = rgba.shape[:2]
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = (
            rgba[:, :, :3].astype(np.float32) * alpha
            + white.astype(np.float32) * (1.0 - alpha)
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out = Path(output_path)
        Image.fromarray(blended, "RGB").save(out)
        return out

    # -- save / load project ----------------------------------------------

    def save_project(self, project_path: str | Path) -> Path:
        """Save the current session state to a JSON project file.

        Saves: image path, tunable parameters, layer data (colours, masks,
        names, visibility).  Masks are stored as run-length encoded data
        for efficiency.

        Args:
            project_path: Destination .qlp (QuickLayer Project) file path.

        Returns:
            The project file path.
        """
        if self._image is None:
            raise RuntimeError("No image loaded.")

        project_path = Path(project_path)

        layers_data = []
        if self._layers:
            for i, layer in enumerate(self._layers):
                layers_data.append({
                    "rgb": list(layer["rgb"]),
                    "hex_color": layer["hex_color"],
                    "color_name": layer["color_name"],
                    "cluster_idx": layer.get("cluster_idx", -1),
                    "visible": layer.get("visible", True),
                    "name": layer.get("name", f"Layer {i + 1}"),
                    "mask_rle": _rle_encode(layer["mask"]),
                    "mask_shape": list(layer["mask"].shape),
                })

        project = {
            "version": 2,
            "source_path": str(self._path) if self._path else None,
            "image_shape": list(self._image.shape),
            "min_area": self.min_area,
            "alphamax": self.alphamax,
            "opttolerance": self.opttolerance,
            "turdsize": self.turdsize,
            "scale": self.scale,
            "width": self.width,
            "separation_mode": self._separation_mode,
            "layers": layers_data,
        }

        project_path.write_text(json.dumps(project, indent=2))
        return project_path

    def load_project(self, project_path: str | Path) -> None:
        """Load a session from a project file.

        Restores the source image (must still exist at the original path)
        and all layer data.

        Args:
            project_path: Path to a .qlp project file.
        """
        project_path = Path(project_path)
        project = json.loads(project_path.read_text())

        # Restore source image
        source_path = project.get("source_path")
        if source_path and Path(source_path).exists():
            self.load(source_path)
            self.ensure_square()
        else:
            raise FileNotFoundError(
                f"Source image not found: {source_path}. "
                "The original image must still exist at its original path."
            )

        # Restore tunables
        self.min_area = project.get("min_area", 100)
        self.alphamax = project.get("alphamax", 1.0)
        self.opttolerance = project.get("opttolerance", 0.2)
        self.turdsize = project.get("turdsize", 2)
        self.scale = project.get("scale", 1.0)
        self.width = project.get("width")
        self._separation_mode = project.get("separation_mode", "color")

        # Restore layers
        layers_data = project.get("layers", [])
        if layers_data:
            self._layers = []
            for ld in layers_data:
                mask = _rle_decode(ld["mask_rle"], tuple(ld["mask_shape"]))
                self._layers.append({
                    "rgb": tuple(ld["rgb"]),
                    "hex_color": ld["hex_color"],
                    "color_name": ld["color_name"],
                    "cluster_idx": ld.get("cluster_idx", -1),
                    "visible": ld.get("visible", True),
                    "name": ld.get("name", ""),
                    "mask": mask,
                })

        self._traced = False
        self._undo_stack.clear()
        self._redo_stack.clear()

    # -- composite preview ------------------------------------------------

    def get_composite_preview(
        self, selected_indices: list[int] | None = None
    ) -> np.ndarray | None:
        """Return an (H, W, 4) RGBA uint8 image compositing all *visible* layers.

        If *selected_indices* is provided, accent-coloured outlines are drawn
        around the selected layers.

        Returns ``None`` if no layers exist yet.

        Reuses an internal buffer to avoid re-allocating a large RGBA array
        on every call.
        """
        if self._image is None or self._layers is None:
            return None

        h, w = self._image.shape[:2]

        # Reuse composite buffer when dimensions match
        buf = getattr(self, "_composite_buf", None)
        if buf is None or buf.shape[0] != h or buf.shape[1] != w:
            buf = np.zeros((h, w, 4), dtype=np.uint8)
            self._composite_buf = buf
        else:
            buf[:] = 0

        for layer in self._layers:
            if not layer.get("visible", True):
                continue
            mask_raw = layer["mask"]  # uint8, 0 or 255

            # Anti-alias mask edges: a small Gaussian blur softens the hard
            # 0→255 transitions at layer boundaries, producing smooth edges
            # in the preview.  Interior pixels stay fully opaque (255).
            mask_aa = cv2.GaussianBlur(mask_raw, (3, 3), 0.7)

            nz = mask_aa > 0
            if not np.any(nz):
                continue

            r, g, b = layer["rgb"]
            alpha = mask_aa[nz].astype(np.float32) / 255.0
            inv = 1.0 - alpha

            # Alpha-blend this layer onto the composite buffer
            buf[nz, 0] = (buf[nz, 0].astype(np.float32) * inv + r * alpha + 0.5).astype(np.uint8)
            buf[nz, 1] = (buf[nz, 1].astype(np.float32) * inv + g * alpha + 0.5).astype(np.uint8)
            buf[nz, 2] = (buf[nz, 2].astype(np.float32) * inv + b * alpha + 0.5).astype(np.uint8)
            buf[nz, 3] = np.minimum(
                255, buf[nz, 3].astype(np.int16) + mask_aa[nz].astype(np.int16)
            ).astype(np.uint8)

        if selected_indices:
            self._apply_selection_outline(buf, selected_indices)

        return buf

    def get_layer_at_pixel(
        self,
        x: int,
        y: int,
        *,
        visible_only: bool = True,
    ) -> int | None:
        """Return top-most layer index at pixel ``(x, y)``, or ``None``.

        When ``visible_only`` is true, hidden layers are ignored.
        """
        self._require_layers("get_layer_at_pixel")
        if self._image is None:
            return None

        h, w = self._image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None

        for idx in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[idx]
            if visible_only and not layer.get("visible", True):
                continue
            if layer["mask"][y, x] > 0:
                return idx
        return None

    # -- SVG string for preview (no file write) ---------------------------

    def get_composite_svg(self) -> str | None:
        """Return an SVG string compositing all visible, traced layers."""
        if not self._traced or self._layers is None or self._image is None:
            return None

        height, width = self._image.shape[:2]
        effective_scale = self.scale
        if self.width is not None:
            effective_scale = self.width / width

        sw = width * effective_scale
        sh = height * effective_scale

        from .svg_writer import _scale_path, _fmt

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{_fmt(sw)}px" height="{_fmt(sh)}px" '
            f'viewBox="0 0 {_fmt(sw)} {_fmt(sh)}">'
        ]

        for layer in self._layers:
            if not layer.get("visible", True):
                continue
            for d in layer.get("svg_paths", []):
                parts.append(
                    f'  <path d="{_scale_path(d, effective_scale)}" '
                    f'fill="{layer["hex_color"]}" fill-rule="evenodd" stroke="none"/>'
                )

        parts.append("</svg>")
        return "\n".join(parts)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _apply_selection_outline(
        self, composite: np.ndarray, selected_indices: list[int]
    ) -> None:
        """Draw an accent border around selected layer masks."""
        if self._layers is None:
            return

        outline_color = np.array([79, 70, 229], dtype=np.uint8)  # ACCENT
        for idx in set(selected_indices):
            if idx < 0 or idx >= len(self._layers):
                continue
            layer = self._layers[idx]
            if not layer.get("visible", True):
                continue

            mask = layer["mask"] > 0
            if not np.any(mask):
                continue

            padded = np.pad(mask, ((1, 1), (1, 1)), constant_values=False)
            up = padded[:-2, 1:-1]
            down = padded[2:, 1:-1]
            left = padded[1:-1, :-2]
            right = padded[1:-1, 2:]
            interior = mask & up & down & left & right
            border = mask & ~interior

            composite[border, 0:3] = outline_color
            composite[border, 3] = 255

    def _require_layers(self, method: str) -> None:
        if self._layers is None:
            raise RuntimeError(
                f"Cannot call {method}() before quantize()."
            )

    def _build_layers(self) -> None:
        """Build layer dicts from current labels / centres."""
        assert self._labels is not None
        assert self._centers_rgb is not None
        assert self._fg_mask is not None
        layers = separate_layers(
            self._labels,
            self._centers_rgb,
            self._fg_mask,
            self.min_area,
        )
        if self._separation_mode == "object":
            layers = separate_objects(layers, self.min_area)
        self._layers = layers
        for i, layer in enumerate(self._layers):
            layer.setdefault("visible", True)
            layer.setdefault("name", f"Layer {i + 1}")

    def _pad_to_square(self) -> None:
        """Pad image and mask to a square canvas (no cropping, no downscale)."""
        h, w = self._image.shape[:2]
        if h == w:
            return

        size = max(h, w)
        bg_pixels = self._image[~self._fg_mask]
        if len(bg_pixels) > 0:
            bg_color = np.median(bg_pixels, axis=0).astype(np.uint8)
        else:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        padded_image = np.full((size, size, 3), bg_color, dtype=np.uint8)
        padded_mask = np.zeros((size, size), dtype=bool)

        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded_image[y_off:y_off + h, x_off:x_off + w] = self._image
        padded_mask[y_off:y_off + h, x_off:x_off + w] = self._fg_mask

        self._image = padded_image
        self._fg_mask = padded_mask

    @staticmethod
    def _erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        mask_uint8 = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
        return eroded.astype(bool)

    @staticmethod
    def _recover_fringe_pixels(
        image: np.ndarray,
        labels: np.ndarray,
        centers_rgb: np.ndarray,
        fg_mask: np.ndarray,
        fg_mask_eroded: np.ndarray,
    ) -> np.ndarray:
        fringe = fg_mask & ~fg_mask_eroded
        if not np.any(fringe):
            return labels
        fringe_pixels = image[fringe].astype(np.float32)
        centers = centers_rgb.astype(np.float32)
        distances = np.linalg.norm(
            fringe_pixels[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2,
        )
        nearest = distances.argmin(axis=1).astype(np.int32)
        labels = labels.copy()
        labels[fringe] = nearest
        return labels

    @staticmethod
    def _assign_to_target_colors(
        image: np.ndarray,
        fg_mask: np.ndarray,
        target_hex: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        targets_rgb = np.array(
            [hex_to_rgb(c) for c in target_hex], dtype=np.uint8
        )
        targets_lab = (
            cv2.cvtColor(targets_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
            .reshape(-1, 3)
            .astype(np.float32)
        )
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        fg_pixels_lab = image_lab[fg_mask]
        distances = np.linalg.norm(
            fg_pixels_lab[:, np.newaxis, :] - targets_lab[np.newaxis, :, :],
            axis=2,
        )
        nearest = distances.argmin(axis=1).astype(np.int32)
        labels = np.full((h, w), -1, dtype=np.int32)
        labels[fg_mask] = nearest
        return labels, targets_rgb


# =========================================================================
# Fast layer snapshot helper (replaces slow copy.deepcopy for undo)
# =========================================================================

def _fast_copy_layers(layers: list[dict]) -> list[dict]:
    """Return a cheap independent copy of the layer list.

    Mask arrays are copied with ``np.ndarray.copy()`` (a single memcpy);
    all other values are small scalars / short lists of strings and are
    shallow-copied.  This is **orders of magnitude** faster than
    ``copy.deepcopy`` which would pickle each numpy array.
    """
    out: list[dict] = []
    for layer in layers:
        d = dict(layer)                       # shallow dict copy
        d["mask"] = layer["mask"].copy()      # fast memcpy of the array
        if "paths" in d and d["paths"]:
            d["paths"] = list(d["paths"])     # copy the string list
        out.append(d)
    return out


# =========================================================================
# RLE encoding/decoding for mask serialization (numpy-vectorised)
# =========================================================================

def _rle_encode(mask: np.ndarray) -> list[int]:
    """Run-length encode a uint8 mask (0 or 255) as a list of run lengths.

    The encoding alternates between runs of 0s and runs of 255s,
    always starting with a 0-run (which may be length 0).

    Uses numpy diff-based detection instead of a Python for-loop,
    giving ~50-100× speedup on large masks.
    """
    flat = (mask.ravel() > 0).astype(np.uint8)
    n = len(flat)
    if n == 0:
        return [0]

    # Find positions where the value changes
    diff = np.diff(flat)
    change_idx = np.flatnonzero(diff)  # indices where flat[i] != flat[i+1]

    # Build run lengths from change indices
    if len(change_idx) == 0:
        # Entire mask is one value
        if flat[0] == 0:
            return [n]
        else:
            return [0, n]

    runs: list[int] = []
    # First run
    first_len = int(change_idx[0]) + 1
    if flat[0] == 1:
        runs.append(0)  # start with zero-length 0-run
    runs.append(first_len)

    # Middle runs
    for i in range(1, len(change_idx)):
        runs.append(int(change_idx[i] - change_idx[i - 1]))

    # Last run
    runs.append(n - int(change_idx[-1]) - 1)

    return runs


def _rle_decode(runs: list[int], shape: tuple[int, ...]) -> np.ndarray:
    """Decode an RLE-encoded mask back to a uint8 array."""
    total = shape[0] * shape[1]
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    current = 0
    for length in runs:
        if current == 1 and length > 0:
            end = min(pos + length, total)
            flat[pos:end] = 255
        pos += length
        current = 1 - current
    return flat.reshape(shape)
