"""Tests for cli module: argument parsing, error handling, and batch mode."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from logo2svg.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_simple_logo(path: Path) -> None:
    """Create a small RGBA logo: red circle on transparent background."""
    size = 100
    img = np.zeros((size, size, 4), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    circle = (x - 50) ** 2 + (y - 50) ** 2 <= 30 ** 2
    img[circle] = [200, 0, 0, 255]
    Image.fromarray(img, "RGBA").save(str(path))


def _create_two_color_logo(path: Path) -> None:
    """Red circle on blue square, transparent bg."""
    size = 100
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[20:80, 20:80] = [0, 0, 200, 255]
    y, x = np.ogrid[:size, :size]
    circle = (x - 50) ** 2 + (y - 50) ** 2 <= 20 ** 2
    img[circle] = [200, 0, 0, 255]
    Image.fromarray(img, "RGBA").save(str(path))


def _create_jpeg_logo(path: Path) -> None:
    """Simple RGB image saved as JPEG."""
    size = 100
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[20:80, 20:80] = [0, 0, 0]
    Image.fromarray(img, "RGB").save(str(path), format="JPEG")


# ---------------------------------------------------------------------------
# Basic argument parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Click CliRunner tests for argument validation."""

    def test_no_args_shows_usage(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_nonexistent_file_error(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [str(tmp_path / "nope.png")])
        assert result.exit_code != 0
        assert "Error" in result.output or "does not exist" in result.output

    def test_colors_zero_error(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [str(logo), "--colors", "0"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_colors_negative_error(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [str(logo), "--colors", "-1"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------

class TestSuccessfulRuns:
    """End-to-end CLI runs that should succeed."""

    def test_basic_run(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "1", "--output-dir", str(tmp_path)
        ])
        assert result.exit_code == 0
        svg_files = list(tmp_path.rglob("*.svg"))
        assert len(svg_files) >= 1

    def test_combined_flag(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_two_color_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "2", "--output-dir", str(tmp_path),
            "--combined",
        ])
        assert result.exit_code == 0
        combined = list(tmp_path.rglob("*combined*"))
        assert len(combined) >= 1

    def test_preview_flag(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "1", "--output-dir", str(tmp_path),
            "--preview",
        ])
        assert result.exit_code == 0
        pngs = list(tmp_path.rglob("*preview*"))
        assert len(pngs) >= 1

    def test_bg_color_flag(self, tmp_path):
        logo = tmp_path / "logo.png"
        size = 100
        img = np.full((size, size, 3), [0, 128, 0], dtype=np.uint8)
        img[30:70, 30:70] = [255, 0, 0]
        Image.fromarray(img, "RGB").save(str(logo))

        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "1", "--output-dir", str(tmp_path),
            "--bg-color", "#008000",
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

class TestBatchMode:
    """Tests for --batch directory processing."""

    def test_batch_processes_directory(self, tmp_path):
        for name in ["a.png", "b.png"]:
            _create_simple_logo(tmp_path / name)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(tmp_path), "--batch", "--colors", "1",
        ])
        assert result.exit_code == 0
        svg_files = list(tmp_path.rglob("*.svg"))
        assert len(svg_files) >= 2

    def test_batch_on_file_errors(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [str(logo), "--batch"])
        assert result.exit_code != 0
        assert "not a directory" in result.output


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestTargetColors:
    """Tests for --target-colors validation and usage."""

    def test_invalid_target_color_format(self, tmp_path):
        """Passing an invalid hex color should produce a clean error, not a traceback."""
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--target-colors", "#GGGGGG",
        ])
        assert result.exit_code != 0
        assert "invalid hex color" in result.output.lower() or "Error" in result.output
        # Should NOT contain a Python traceback
        assert "Traceback" not in (result.output or "")

    def test_valid_target_colors_run(self, tmp_path):
        """Passing valid --target-colors with a two-color logo should succeed."""
        logo = tmp_path / "logo.png"
        _create_two_color_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--target-colors", "#FF0000,#0000FF",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        svg_files = list(tmp_path.rglob("*.svg"))
        assert len(svg_files) >= 1


# ---------------------------------------------------------------------------
# --turdsize flag
# ---------------------------------------------------------------------------

class TestTurdsizeFlag:
    """Tests for the --turdsize CLI option."""

    def test_turdsize_flag_accepted(self, tmp_path):
        """The CLI should accept --turdsize without error."""
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "1", "--output-dir", str(tmp_path),
            "--turdsize", "5",
        ])
        assert result.exit_code == 0
        svg_files = list(tmp_path.rglob("*.svg"))
        assert len(svg_files) >= 1

    def test_tm_tuning_flags_accepted(self, tmp_path):
        """The CLI should accept TM tuning flags without error."""
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--colors", "1", "--output-dir", str(tmp_path),
            "--tm-max-area-pct", "2.5", "--tm-margin-pct", "18",
        ])
        assert result.exit_code == 0

    def test_negative_tm_margin_rejected(self, tmp_path):
        """TM margin must be non-negative."""
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--tm-margin-pct", "-1",
        ])
        assert result.exit_code != 0
        assert "tm-margin-pct" in result.output


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestErrorMessages:
    """CLI should give helpful error messages."""

    def test_invalid_bg_color_format(self, tmp_path):
        logo = tmp_path / "logo.png"
        _create_simple_logo(logo)
        runner = CliRunner()
        result = runner.invoke(main, [
            str(logo), "--bg-color", "not-a-color",
        ])
        # Should error (invalid hex) rather than crash with traceback
        assert result.exit_code != 0
