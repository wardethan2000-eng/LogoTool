"""Interactive conversion session: holds intermediate state and supports
incremental re-processing.

A :class:`Session` wraps the existing pipeline stages so they can be driven
one-at-a-time from a GUI or any interactive caller, rather than as a single
end-to-end ``process_single`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .color_utils import hex_to_rgb, nearest_color_name, rgb_to_hex
from .image_loader import load_image
from .layer_separator import separate_layers
from .quantizer import quantize_colors
from .svg_writer import write_preview, write_svg_files
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
# Session
# ---------------------------------------------------------------------------

class Session:
    """Holds the intermediate state of a single logo → SVG conversion.

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

        self.min_area = min_area
        self.alphamax = alphamax
        self.opttolerance = opttolerance
        self.turdsize = turdsize
        self.scale = scale
        self.width = width

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

    # -- stage 2: quantize ------------------------------------------------

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
        r, g, b = hex_to_rgb(new_hex)
        self._layers[index]["rgb"] = (r, g, b)
        self._layers[index]["hex_color"] = rgb_to_hex(r, g, b)
        self._layers[index]["color_name"] = nearest_color_name(r, g, b)

    def set_layer_visibility(self, index: int, visible: bool) -> None:
        """Toggle visibility (for preview composite only — does not affect export)."""
        self._require_layers("set_layer_visibility")
        if not 0 <= index < len(self._layers):
            raise IndexError(f"Layer index {index} out of range.")
        self._layers[index]["visible"] = visible

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
        """Build layer dicts from current labels / centers using
        :func:`separate_layers`."""
        assert self._labels is not None
        assert self._centers_rgb is not None
        assert self._fg_mask is not None
        self._layers = separate_layers(
            self._labels,
            self._centers_rgb,
            self._fg_mask,
            self.min_area,
        )
        # Ensure every layer has a visibility flag
        for layer in self._layers:
            layer.setdefault("visible", True)

    # -- helpers from pipeline.py -----------------------------------------

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
