"""Tests for new Session features: undo/redo, text, outline, border, SVG import, preprocessing,
overlap prevention, reorder, duplicate, rename, save/load project, export PNG, RLE encode/decode."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from logo2svg.session import Session, LayerInfo, _rle_encode, _rle_decode


@pytest.fixture
def session_with_layers():
    """Create a session with a synthetic image and non-overlapping layers."""
    s = Session()
    # Inject synthetic data (bypass load/quantize pipeline)
    s._image = np.full((100, 100, 3), 200, dtype=np.uint8)
    s._image[20:60, 20:60] = [255, 0, 0]  # Red square
    s._image[40:80, 40:80] = [0, 0, 255]  # Blue square
    s._fg_mask = np.ones((100, 100), dtype=bool)
    s._layers = [
        {
            "rgb": (255, 0, 0),
            "hex_color": "#FF0000",
            "color_name": "red",
            "mask": np.zeros((100, 100), dtype=np.uint8),
            "cluster_idx": 0,
            "visible": True,
            "name": "Red",
        },
        {
            "rgb": (0, 0, 255),
            "hex_color": "#0000FF",
            "color_name": "blue",
            "mask": np.zeros((100, 100), dtype=np.uint8),
            "cluster_idx": 1,
            "visible": True,
            "name": "Blue",
        },
    ]
    # Fill masks — non-overlapping: red owns top-left, blue owns bottom-right
    s._layers[0]["mask"][20:60, 20:60] = 255
    s._layers[1]["mask"][40:80, 40:80] = 255
    # Resolve overlaps so the fixture is clean (blue at higher index wins overlap)
    s._resolve_overlaps()
    s._traced = False
    return s


@pytest.fixture
def session_loaded():
    """Create a session with an image loaded (no layers yet)."""
    s = Session()
    s._image = np.full((100, 100, 3), 200, dtype=np.uint8)
    s._image[30:70, 30:70] = [128, 64, 32]
    s._fg_mask = np.ones((100, 100), dtype=bool)
    s._path = Path("/tmp/test.png")
    return s


class TestUndoRedo:
    def test_undo_change_color(self, session_with_layers):
        s = session_with_layers
        assert s._layers[0]["hex_color"] == "#FF0000"

        s.change_color(0, "#00FF00")
        assert s._layers[0]["hex_color"] == "#00FF00"
        assert s.can_undo

        s.undo()
        assert s._layers[0]["hex_color"] == "#FF0000"
        assert s.can_redo

    def test_redo_after_undo(self, session_with_layers):
        s = session_with_layers
        s.change_color(0, "#00FF00")
        s.undo()
        assert s._layers[0]["hex_color"] == "#FF0000"

        s.redo()
        assert s._layers[0]["hex_color"] == "#00FF00"

    def test_undo_empty_stack(self, session_with_layers):
        s = session_with_layers
        assert not s.can_undo
        assert not s.undo()

    def test_redo_empty_stack(self, session_with_layers):
        s = session_with_layers
        assert not s.can_redo
        assert not s.redo()

    def test_undo_remove_layer(self, session_with_layers):
        s = session_with_layers
        assert len(s._layers) == 2

        s.remove_color(1)
        assert len(s._layers) == 1

        s.undo()
        assert len(s._layers) == 2

    def test_new_action_clears_redo(self, session_with_layers):
        s = session_with_layers
        s.change_color(0, "#00FF00")
        s.undo()
        assert s.can_redo

        # New action should clear redo
        s.change_color(0, "#FF00FF")
        assert not s.can_redo


class TestAddText:
    def test_basic_text(self, session_loaded):
        s = session_loaded
        s._layers = []
        idx = s.add_text("Hello")
        assert idx == 0
        assert len(s._layers) == 1
        assert s._layers[0]["hex_color"] == "#000000"
        assert np.any(s._layers[0]["mask"] > 0)

    def test_text_with_color(self, session_loaded):
        s = session_loaded
        s._layers = []
        idx = s.add_text("Test", color="#FF0000")
        assert s._layers[idx]["hex_color"] == "#FF0000"
        assert s._layers[idx]["rgb"] == (255, 0, 0)

    def test_text_position(self, session_loaded):
        s = session_loaded
        s._layers = []
        idx = s.add_text("X", x=10, y=50)
        mask = s._layers[idx]["mask"]
        # Some pixels should be lit near the specified position
        assert np.any(mask[40:60, 0:40] > 0)

    def test_text_no_image(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.add_text("Hello")

    def test_text_undo(self, session_with_layers):
        s = session_with_layers
        assert len(s._layers) == 2
        s.add_text("Hello", color="#00FF00")
        assert len(s._layers) == 3
        s.undo()
        assert len(s._layers) == 2


class TestAddOutline:
    def test_basic_outline(self, session_with_layers):
        s = session_with_layers
        idx = s.add_outline(0, width=3, color="#000000")
        assert idx == 1  # inserted after layer 0
        assert len(s._layers) == 3
        # Outline mask should have non-zero pixels
        assert np.any(s._layers[1]["mask"] > 0)

    def test_outline_color(self, session_with_layers):
        s = session_with_layers
        idx = s.add_outline(0, width=2, color="#FF00FF")
        assert s._layers[idx]["hex_color"] == "#FF00FF"

    def test_outline_invalid_index(self, session_with_layers):
        s = session_with_layers
        with pytest.raises(IndexError):
            s.add_outline(99, width=3)

    def test_outline_undo(self, session_with_layers):
        s = session_with_layers
        s.add_outline(0, width=3)
        assert len(s._layers) == 3
        s.undo()
        assert len(s._layers) == 2


class TestAddCanvasBorder:
    def test_basic_border(self, session_loaded):
        s = session_loaded
        s._layers = []
        idx = s.add_canvas_border(width=5, color="#000000")
        assert idx == 0
        assert len(s._layers) == 1
        mask = s._layers[0]["mask"]
        # Top row should be filled
        assert np.all(mask[0, :] == 255)
        # Bottom row should be filled
        assert np.all(mask[-1, :] == 255)
        # Center should be empty
        assert mask[50, 50] == 0

    def test_border_color(self, session_loaded):
        s = session_loaded
        s._layers = []
        s.add_canvas_border(width=5, color="#FF0000")
        assert s._layers[0]["hex_color"] == "#FF0000"

    def test_border_no_image(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.add_canvas_border()

    def test_border_undo(self, session_with_layers):
        s = session_with_layers
        s.add_canvas_border(width=5)
        assert len(s._layers) == 3
        s.undo()
        assert len(s._layers) == 2


class TestImportSvg:
    def test_import_no_image(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.import_svg("/tmp/test.svg")

    def test_import_nonexistent(self, session_loaded):
        s = session_loaded
        with pytest.raises(FileNotFoundError):
            s.import_svg("/nonexistent/path.svg")

    def test_import_creates_layers(self, session_loaded, tmp_path):
        s = session_loaded
        s._layers = []

        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <rect x="10" y="10" width="80" height="80" fill="#FF0000"/>
</svg>"""
        svg_path = tmp_path / "test.svg"
        svg_path.write_text(svg_content)

        count = s.import_svg(svg_path)
        assert count >= 1
        assert len(s._layers) >= 1

    def test_import_with_color_override(self, session_loaded, tmp_path):
        s = session_loaded
        s._layers = []

        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <rect x="10" y="10" width="80" height="80" fill="#FF0000"/>
</svg>"""
        svg_path = tmp_path / "test.svg"
        svg_path.write_text(svg_content)

        count = s.import_svg(svg_path, color_override="#00FF00")
        assert count == 1
        assert s._layers[0]["hex_color"] == "#00FF00"


class TestPreprocessing:
    def test_analyze_image(self, session_loaded):
        s = session_loaded
        report = s.analyze_image()
        assert report is not None
        assert isinstance(report.contrast_low, bool)
        assert isinstance(report.noise_level, float)

    def test_run_preprocessing(self, session_loaded):
        s = session_loaded
        original = s._image.copy()
        report = s.run_preprocessing(contrast=True, sharpen=True)
        assert report is not None
        assert s._image.shape == original.shape

    def test_preprocessing_no_image(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.run_preprocessing()

    def test_analyze_no_image(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.analyze_image()


class TestOverlapPrevention:
    """Test that _resolve_overlaps ensures no pixel belongs to two layers."""

    def test_overlapping_masks_resolved(self):
        s = Session()
        s._image = np.full((50, 50, 3), 200, dtype=np.uint8)
        s._fg_mask = np.ones((50, 50), dtype=bool)
        s._layers = [
            {
                "rgb": (255, 0, 0), "hex_color": "#FF0000", "color_name": "red",
                "mask": np.zeros((50, 50), dtype=np.uint8), "cluster_idx": 0,
                "visible": True, "name": "Red",
            },
            {
                "rgb": (0, 0, 255), "hex_color": "#0000FF", "color_name": "blue",
                "mask": np.zeros((50, 50), dtype=np.uint8), "cluster_idx": 1,
                "visible": True, "name": "Blue",
            },
        ]
        # Both layers claim the same region
        s._layers[0]["mask"][10:30, 10:30] = 255
        s._layers[1]["mask"][10:30, 10:30] = 255

        s._resolve_overlaps()

        # Blue (higher index) wins the overlap
        assert np.all(s._layers[1]["mask"][10:30, 10:30] == 255)
        # Red should have zero pixels in the overlap region
        assert np.all(s._layers[0]["mask"][10:30, 10:30] == 0)

    def test_no_overlap_after_resolve(self, session_with_layers):
        s = session_with_layers
        # After fixture setup (which already resolves), verify no pixel belongs to two layers
        combined = np.zeros(s._layers[0]["mask"].shape, dtype=int)
        for layer in s._layers:
            combined += (layer["mask"] > 0).astype(int)
        assert np.all(combined <= 1)

    def test_resolve_single_layer_noop(self):
        s = Session()
        s._image = np.full((50, 50, 3), 200, dtype=np.uint8)
        s._layers = [
            {
                "rgb": (255, 0, 0), "hex_color": "#FF0000", "color_name": "red",
                "mask": np.zeros((50, 50), dtype=np.uint8), "cluster_idx": 0,
                "visible": True, "name": "Red",
            },
        ]
        s._layers[0]["mask"][10:30, 10:30] = 255
        original_count = np.count_nonzero(s._layers[0]["mask"])
        s._resolve_overlaps()
        assert np.count_nonzero(s._layers[0]["mask"]) == original_count


class TestMoveLayer:
    def test_move_up(self, session_with_layers):
        s = session_with_layers
        original_colors = [l["hex_color"] for l in s._layers]
        result = s.move_layer_up(1)
        assert result is True
        # Layer order should be swapped
        assert s._layers[0]["hex_color"] == original_colors[1]
        assert s._layers[1]["hex_color"] == original_colors[0]

    def test_move_up_at_top(self, session_with_layers):
        s = session_with_layers
        result = s.move_layer_up(0)
        assert result is False

    def test_move_down(self, session_with_layers):
        s = session_with_layers
        original_colors = [l["hex_color"] for l in s._layers]
        result = s.move_layer_down(0)
        assert result is True
        assert s._layers[0]["hex_color"] == original_colors[1]
        assert s._layers[1]["hex_color"] == original_colors[0]

    def test_move_down_at_bottom(self, session_with_layers):
        s = session_with_layers
        result = s.move_layer_down(1)
        assert result is False

    def test_move_resolves_overlaps(self, session_with_layers):
        s = session_with_layers
        s.move_layer_up(1)
        # No pixel should belong to two layers
        combined = np.zeros(s._layers[0]["mask"].shape, dtype=int)
        for layer in s._layers:
            combined += (layer["mask"] > 0).astype(int)
        assert np.all(combined <= 1)

    def test_move_up_undo(self, session_with_layers):
        s = session_with_layers
        original_colors = [l["hex_color"] for l in s._layers]
        s.move_layer_up(1)
        s.undo()
        assert s._layers[0]["hex_color"] == original_colors[0]
        assert s._layers[1]["hex_color"] == original_colors[1]


class TestDuplicateLayer:
    def test_duplicate_creates_copy(self, session_with_layers):
        s = session_with_layers
        assert len(s._layers) == 2
        new_idx = s.duplicate_layer(0)
        assert new_idx == 1
        assert len(s._layers) == 3
        assert s._layers[new_idx]["hex_color"] == s._layers[0]["hex_color"]

    def test_duplicate_name_has_copy_suffix(self, session_with_layers):
        s = session_with_layers
        new_idx = s.duplicate_layer(0)
        assert "(copy)" in s._layers[new_idx]["name"]

    def test_duplicate_resolves_overlaps(self, session_with_layers):
        s = session_with_layers
        s.duplicate_layer(0)
        # After overlap resolution, duplicate should have no pixels
        # (original at lower index owns them, but overlap resolution
        # gives higher-index layers priority — however since the original
        # is at index 0 and dup at index 1, the dup actually keeps them)
        combined = np.zeros(s._layers[0]["mask"].shape, dtype=int)
        for layer in s._layers:
            combined += (layer["mask"] > 0).astype(int)
        assert np.all(combined <= 1)

    def test_duplicate_invalid_index(self, session_with_layers):
        s = session_with_layers
        with pytest.raises(IndexError):
            s.duplicate_layer(99)

    def test_duplicate_undo(self, session_with_layers):
        s = session_with_layers
        assert len(s._layers) == 2
        s.duplicate_layer(0)
        assert len(s._layers) == 3
        s.undo()
        assert len(s._layers) == 2


class TestRenameLayer:
    def test_basic_rename(self, session_with_layers):
        s = session_with_layers
        s.rename_layer(0, "My Red Layer")
        assert s._layers[0]["name"] == "My Red Layer"

    def test_rename_shows_in_get_layers(self, session_with_layers):
        s = session_with_layers
        s.rename_layer(0, "Custom Name")
        layers = s.get_layers()
        assert layers[0].name == "Custom Name"

    def test_rename_invalid_index(self, session_with_layers):
        s = session_with_layers
        with pytest.raises(IndexError):
            s.rename_layer(99, "Name")


class TestRLEEncoding:
    def test_roundtrip_simple(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        encoded = _rle_encode(mask)
        decoded = _rle_decode(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)

    def test_roundtrip_all_zero(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        encoded = _rle_encode(mask)
        decoded = _rle_decode(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)

    def test_roundtrip_all_ones(self):
        mask = np.full((15, 15), 255, dtype=np.uint8)
        encoded = _rle_encode(mask)
        decoded = _rle_decode(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)

    def test_roundtrip_complex_pattern(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:15, 10:20] = 255
        mask[30:40, 25:45] = 255
        mask[0, 0] = 255
        encoded = _rle_encode(mask)
        decoded = _rle_decode(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)

    def test_starts_with_zero_run(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 255  # first pixel is 1
        encoded = _rle_encode(mask)
        # First run should be 0 (zero-length run of 0s)
        assert encoded[0] == 0
        decoded = _rle_decode(encoded, mask.shape)
        np.testing.assert_array_equal(mask, decoded)


class TestSaveLoadProject:
    def test_save_creates_file(self, session_with_layers, tmp_path):
        s = session_with_layers
        project_file = tmp_path / "test.qlp"
        result = s.save_project(project_file)
        assert result.exists()
        assert result == project_file

    def test_save_valid_json(self, session_with_layers, tmp_path):
        s = session_with_layers
        project_file = tmp_path / "test.qlp"
        s.save_project(project_file)
        data = json.loads(project_file.read_text())
        assert "version" in data
        assert "layers" in data
        assert len(data["layers"]) == 2

    def test_save_preserves_layer_info(self, session_with_layers, tmp_path):
        s = session_with_layers
        s.rename_layer(0, "Custom Red")
        project_file = tmp_path / "test.qlp"
        s.save_project(project_file)
        data = json.loads(project_file.read_text())
        assert data["layers"][0]["name"] == "Custom Red"
        assert data["layers"][0]["hex_color"] == "#FF0000"

    def test_save_no_image_raises(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.save_project("/tmp/test.qlp")

    def test_load_nonexistent_raises(self):
        s = Session()
        with pytest.raises(Exception):
            s.load_project("/nonexistent/path.qlp")

    def test_rle_mask_in_project(self, session_with_layers, tmp_path):
        s = session_with_layers
        project_file = tmp_path / "test.qlp"
        s.save_project(project_file)
        data = json.loads(project_file.read_text())
        # Each layer should have mask_rle and mask_shape
        for ld in data["layers"]:
            assert "mask_rle" in ld
            assert "mask_shape" in ld
            assert isinstance(ld["mask_rle"], list)
            assert len(ld["mask_shape"]) == 2


class TestExportPng:
    def test_export_creates_file(self, session_with_layers, tmp_path):
        s = session_with_layers
        out_path = tmp_path / "output.png"
        result = s.export_png(out_path)
        assert result.exists()
        assert result == out_path

    def test_export_no_image_raises(self):
        s = Session()
        with pytest.raises(RuntimeError, match="No image loaded"):
            s.export_png("/tmp/out.png")

    def test_export_no_layers_raises(self, session_loaded):
        s = session_loaded
        s._layers = None
        with pytest.raises(RuntimeError):
            s.export_png("/tmp/out.png")

    def test_export_file_is_valid_png(self, session_with_layers, tmp_path):
        s = session_with_layers
        out_path = tmp_path / "output.png"
        s.export_png(out_path)
        # Check PNG magic bytes
        data = out_path.read_bytes()
        assert data[:4] == b'\x89PNG'
