"""Tests for color_utils module."""

from logo2svg.color_utils import hex_to_rgb, nearest_color_name, rgb_to_hex


def test_rgb_to_hex():
    assert rgb_to_hex(255, 0, 0) == "#FF0000"
    assert rgb_to_hex(0, 128, 255) == "#0080FF"
    assert rgb_to_hex(0, 0, 0) == "#000000"
    assert rgb_to_hex(255, 255, 255) == "#FFFFFF"


def test_hex_to_rgb():
    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("0080FF") == (0, 128, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)


def test_hex_to_rgb_roundtrip():
    for r, g, b in [(0, 0, 0), (255, 255, 255), (128, 64, 32), (0, 128, 255)]:
        assert hex_to_rgb(rgb_to_hex(r, g, b)) == (r, g, b)


def test_nearest_color_name():
    # Pure red should be "red"
    assert nearest_color_name(255, 0, 0) == "red"
    # Pure white should be "white"
    assert nearest_color_name(255, 255, 255) == "white"
    # Pure black should be "black"
    assert nearest_color_name(0, 0, 0) == "black"
    # Should return some string (not crash) for any input
    name = nearest_color_name(123, 45, 67)
    assert isinstance(name, str)
    assert len(name) > 0
