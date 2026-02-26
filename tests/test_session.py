"""Tests for logo2svg.session.Session interactive workflow."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from logo2svg.session import Session, LayerInfo


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _create_two_color_png(path: Path) -> None:
    """Red circle on blue square, transparent background."""
    size = 200
    img = np.zeros((size, size, 4), dtype=np.uint8)
    # Blue square
    img[40:160, 40:160, 0] = 0
    img[40:160, 40:160, 1] = 0
    img[40:160, 40:160, 2] = 200
    img[40:160, 40:160, 3] = 255
    # Red circle
    y, x = np.ogrid[:size, :size]
    circle = (x - 100) ** 2 + (y - 100) ** 2 <= 30 ** 2
    img[circle, 0] = 200
    img[circle, 1] = 0
    img[circle, 2] = 0
    img[circle, 3] = 255
    Image.fromarray(img, "RGBA").save(str(path))


def _create_three_color_png(path: Path) -> None:
    """Green rect, blue rect, red square — white background."""
    size = 200
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[30:90, 30:170, :] = [0, 150, 0]
    img[110:170, 30:170, :] = [0, 0, 150]
    img[80:120, 80:120, :] = [200, 0, 0]
    Image.fromarray(img, "RGB").save(str(path))


# ---------------------------------------------------------------------------
# Test: load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_sets_image(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        assert s.is_loaded
        assert s.image is not None
        assert s.fg_mask is not None
        assert s.image_size == (200, 200)

    def test_load_resets_downstream(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        assert s.is_quantized
        # Re-load should reset quantization
        s.load(png)
        assert not s.is_quantized
        assert not s.is_traced

    def test_load_with_bg_color(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_three_color_png(png)
        s = Session()
        s.load(png, bg_color="#FFFFFF")
        assert s.is_loaded
        fg = np.count_nonzero(s.fg_mask)
        total = s.fg_mask.size
        # White bg removed → less than all pixels are foreground
        assert fg < total


# ---------------------------------------------------------------------------
# Test: quantize
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_quantize_fixed_k(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        assert s.is_quantized
        layers = s.get_layers()
        assert len(layers) == 2
        for l in layers:
            assert isinstance(l, LayerInfo)
            assert l.pixel_count > 0

    def test_quantize_target_colors(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(target_colors=["#C80000", "#0000C8"])
        layers = s.get_layers()
        assert len(layers) == 2

    def test_quantize_resets_trace(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        s.trace()
        assert s.is_traced
        s.quantize(n_colors=2)
        assert not s.is_traced

    def test_quantize_without_load_raises(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.quantize(n_colors=2)


# ---------------------------------------------------------------------------
# Test: layer mutations
# ---------------------------------------------------------------------------

class TestLayerMutations:
    def _session_with_layers(self, tmp_path, n=3):
        png = tmp_path / "logo.png"
        _create_three_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=n)
        return s

    def test_remove_color(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        orig = s.get_layer_count()
        s.remove_color(0)
        assert s.get_layer_count() == orig - 1
        assert not s.is_traced

    def test_remove_invalid_index(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        with pytest.raises(IndexError):
            s.remove_color(99)

    def test_merge_colors(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        orig = s.get_layer_count()
        if orig < 2:
            pytest.skip("Need ≥ 2 layers to test merge")
        s.merge_colors([0, 1])
        assert s.get_layer_count() == orig - 1

    def test_merge_too_few_raises(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        with pytest.raises(ValueError):
            s.merge_colors([0])

    def test_change_color(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        s.change_color(0, "#AABBCC")
        info = s.get_layers()[0]
        assert info.hex_color == "#AABBCC"
        assert info.rgb == (0xAA, 0xBB, 0xCC)

    def test_set_visibility(self, tmp_path):
        s = self._session_with_layers(tmp_path)
        s.set_layer_visibility(0, False)
        assert not s.get_layers()[0].visible

    def test_mutations_before_quantize_raise(self):
        s = Session()
        with pytest.raises(RuntimeError):
            s.remove_color(0)
        with pytest.raises(RuntimeError):
            s.merge_colors([0, 1])
        with pytest.raises(RuntimeError):
            s.change_color(0, "#000000")


# ---------------------------------------------------------------------------
# Test: trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        s.trace()
        assert s.is_traced

    def test_trace_before_quantize_raises(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        with pytest.raises(RuntimeError):
            s.trace()


# ---------------------------------------------------------------------------
# Test: export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_files(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        out = tmp_path / "out"
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        files = s.export(out)
        assert len(files) == 2
        for f in files:
            assert f.exists()
            content = f.read_text()
            assert "<svg" in content

    def test_export_combined(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        out = tmp_path / "out"
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        files = s.export(out, combined=True)
        assert len(files) == 3  # 2 per-color + 1 combined
        combined = [f for f in files if "combined" in f.name]
        assert len(combined) == 1

    def test_export_auto_traces(self, tmp_path):
        """export() should auto-trace if not already traced."""
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        out = tmp_path / "out"
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        assert not s.is_traced
        s.export(out)
        assert s.is_traced


# ---------------------------------------------------------------------------
# Test: composite preview
# ---------------------------------------------------------------------------

class TestCompositePreview:
    def test_returns_rgba(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        comp = s.get_composite_preview()
        assert comp is not None
        assert comp.shape == (200, 200, 4)
        assert comp.dtype == np.uint8

    def test_hidden_layer_excluded(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        # Hide all layers → composite should be fully transparent
        for i in range(s.get_layer_count()):
            s.set_layer_visibility(i, False)
        comp = s.get_composite_preview()
        assert np.all(comp[:, :, 3] == 0)

    def test_no_layers_returns_none(self):
        s = Session()
        assert s.get_composite_preview() is None

    def test_composite_svg(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        s.trace()
        svg = s.get_composite_svg()
        assert svg is not None
        assert "<svg" in svg
        assert "<path" in svg


# ---------------------------------------------------------------------------
# Test: separation mode
# ---------------------------------------------------------------------------

class TestSeparationMode:
    def test_default_mode_is_color(self):
        s = Session()
        assert s.separation_mode == "color"

    def test_set_mode_object(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        color_count = s.get_layer_count()
        s.set_separation_mode("object")
        assert s.separation_mode == "object"
        # Object mode should produce at least as many layers as color mode
        # (each color region may have multiple components)
        assert s.get_layer_count() >= color_count

    def test_set_invalid_mode_raises(self):
        s = Session()
        with pytest.raises(ValueError, match="Invalid separation mode"):
            s.set_separation_mode("invalid")

    def test_mode_change_resets_trace(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        s.trace()
        assert s.is_traced
        s.set_separation_mode("object")
        assert not s.is_traced

    def test_mode_switch_back_to_color(self, tmp_path):
        png = tmp_path / "logo.png"
        _create_two_color_png(png)
        s = Session()
        s.load(png)
        s.quantize(n_colors=2)
        color_count = s.get_layer_count()
        s.set_separation_mode("object")
        s.set_separation_mode("color")
        assert s.get_layer_count() == color_count
