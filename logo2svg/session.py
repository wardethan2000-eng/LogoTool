"""Interactive conversion session: holds intermediate state and supports
incremental re-processing.

A :class:`Session` wraps the existing pipeline stages so they can be driven
one-at-a-time from a GUI or any interactive caller, rather than as a single
end-to-end ``process_single`` call.
"""

from __future__ import annotations

import copy
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
        self._max_undo = 50

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
        # NOTE: do NOT pad to square here — _pad_to_square must run
        # after remove_tm() so TM symbols are still at the image
        # margins when the margin check runs.
        # Reset downstream
        self._labels = None
        self._centers_rgb = None
        self._layers = None
        self._traced = False
        self._undo_stack.clear()
        self._redo_stack.clear()

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
        # Pad to square AFTER margin-based TM detection so symbols
        # that are near the original edges are still caught.
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
            # Erode mask to suppress anti-alias fringe during K-means
            fg_mask_eroded = self._erode_mask(fg_mask)
            labels, centers_rgb = quantize_colors(
                image, fg_mask_eroded, n_colors=n_colors
            )
            # Recover fringe pixels
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

        # OR masks into the keep layer
        for idx in merge_idxs:
            self._layers[keep_idx]["mask"] = np.maximum(
                self._layers[keep_idx]["mask"], self._layers[idx]["mask"]
            )

        # Remove merged layers in reverse order to keep indices stable
        for idx in reversed(merge_idxs):
            del self._layers[idx]

        self._traced = False

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
        """Set the layer separation mode.

        ``'color'`` -- one layer per colour (default).
        ``'object'`` -- one layer per connected component.

        Changing the mode re-builds layers from the existing quantization
        (if available) and resets tracing.
        """
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

    # -- SVG import -------------------------------------------------------

    def import_svg(
        self,
        svg_path: str | Path,
        color_override: str | None = None,
    ) -> int:
        """Import an external SVG file as new colour layer(s).

        Each distinct colour in the SVG becomes a separate layer appended
        to the current layer list.

        Args:
            svg_path: Path to the SVG file.
            color_override: If set, import the entire SVG as one layer
                with this hex colour.

        Returns:
            Number of layers added.
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
        source layer.

        Args:
            index: Source layer index.
            width: Outline thickness in pixels.
            color: Hex colour for the outline.

        Returns:
            Index of the newly created outline layer.
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
        }

        insert_at = index + 1
        self._layers.insert(insert_at, new_layer)
        self._traced = False
        return insert_at

    def add_canvas_border(
        self,
        width: int = 10,
        color: str = "#000000",
    ) -> int:
        """Add a rectangular border around the entire canvas as a new layer.

        Args:
            width: Border thickness in pixels.
            color: Hex colour for the border.

        Returns:
            Index of the newly created border layer.
        """
        if self._image is None:
            raise RuntimeError("No image loaded.")

        if self._layers is None:
            self._layers = []

        self._save_undo("Add canvas border")
        h, w_px = self._image.shape[:2]
        mask = np.zeros((h, w_px), dtype=np.uint8)
        # Top
        mask[:width, :] = 255
        # Bottom
        mask[-width:, :] = 255
        # Left
        mask[:, :width] = 255
        # Right
        mask[:, -width:] = 255

        r, g, b = hex_to_rgb(color)
        new_layer = {
            "rgb": (r, g, b),
            "hex_color": rgb_to_hex(r, g, b),
            "color_name": nearest_color_name(r, g, b),
            "mask": mask,
            "cluster_idx": -1,
            "visible": True,
        }

        self._layers.append(new_layer)
        self._traced = False
        return len(self._layers) - 1

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

        Args:
            text: The text string to render.
            color: Hex colour.
            font_scale: OpenCV font scale factor.
            thickness: Text stroke thickness.
            x: Horizontal position (left edge of text baseline).
            y: Vertical position (text baseline).

        Returns:
            Index of the newly created text layer.
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
        }

        self._layers.append(new_layer)
        self._traced = False
        return len(self._layers) - 1

    # -- undo / redo ------------------------------------------------------

    def _save_undo(self, description: str = "") -> None:
        """Save the current layer state to the undo stack."""
        if self._layers is not None:
            snapshot = _Snapshot(
                layers=copy.deepcopy(self._layers),
                traced=self._traced,
                description=description,
            )
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            # Clear redo stack on new action
            self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo the last layer mutation.  Returns True if undo was performed."""
        if not self._undo_stack:
            return False

        # Save current state to redo
        if self._layers is not None:
            self._redo_stack.append(_Snapshot(
                layers=copy.deepcopy(self._layers),
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

        # Save current to undo
        if self._layers is not None:
            self._undo_stack.append(_Snapshot(
                layers=copy.deepcopy(self._layers),
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

    # -- composite preview ------------------------------------------------

    def get_composite_preview(self) -> np.ndarray | None:
        """Return an (H, W, 4) RGBA uint8 image compositing all *visible* layers.

        Returns ``None`` if no layers exist yet.
        """
        if self._image is None or self._layers is None:
            return None

        h, w = self._image.shape[:2]
        composite = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA, transparent

        for layer in self._layers:
            if not layer.get("visible", True):
                continue
            mask_bool = layer["mask"] > 0
            r, g, b = layer["rgb"]
            composite[mask_bool, 0] = r
            composite[mask_bool, 1] = g
            composite[mask_bool, 2] = b
            composite[mask_bool, 3] = 255

        return composite

    # -- SVG string for preview (no file write) ---------------------------

    def get_composite_svg(self) -> str | None:
        """Return an SVG string compositing all visible, traced layers.

        Returns ``None`` if tracing hasn't been performed yet.
        """
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
    # Internal helpers (ported from pipeline.py so Session is self-contained)
    # =====================================================================

    def _require_layers(self, method: str) -> None:
        if self._layers is None:
            raise RuntimeError(
                f"Cannot call {method}() before quantize()."
            )

    def _build_layers(self) -> None:
        """Build layer dicts from current labels / centres.

        In ``'color'`` mode, uses :func:`separate_layers` (one layer per colour).
        In ``'object'`` mode, further splits each colour into individual
        connected components via :func:`separate_objects`.
        """
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
        # Ensure every layer has a visibility flag
        for layer in self._layers:
            layer.setdefault("visible", True)

    # -- helpers from pipeline.py -----------------------------------------

    def _pad_to_square(self) -> None:
        """Pad image and mask to a square canvas (no cropping, no downscale).

        Uses the detected background colour for the padding area.
        """
        h, w = self._image.shape[:2]
        if h == w:
            return

        size = max(h, w)

        # Determine background colour from non-foreground pixels
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
        fringe_pixels = image[fringe].astype(np.float64)
        centers = centers_rgb.astype(np.float64)
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
            .astype(np.float64)
        )
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
        fg_pixels_lab = image_lab[fg_mask]
        distances = np.linalg.norm(
            fg_pixels_lab[:, np.newaxis, :] - targets_lab[np.newaxis, :, :],
            axis=2,
        )
        nearest = distances.argmin(axis=1).astype(np.int32)
        labels = np.full((h, w), -1, dtype=np.int32)
        labels[fg_mask] = nearest
        return labels, targets_rgb
