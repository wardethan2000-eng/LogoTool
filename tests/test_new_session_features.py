"""Tests for new Session features: undo/redo, text, outline, border, SVG import, preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from logo2svg.session import Session, LayerInfo


@pytest.fixture
def session_with_layers():
    """Create a session with a synthetic image and layers."""
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
        },
        {
            "rgb": (0, 0, 255),
            "hex_color": "#0000FF",
            "color_name": "blue",
            "mask": np.zeros((100, 100), dtype=np.uint8),
            "cluster_idx": 1,
            "visible": True,
        },
    ]
    # Fill masks
    s._layers[0]["mask"][20:60, 20:60] = 255
    s._layers[1]["mask"][40:80, 40:80] = 255
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
