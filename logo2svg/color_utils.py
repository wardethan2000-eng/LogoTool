"""Color utility functions: hex/RGB conversion and color name lookup."""

import webcolors


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to a hex color string like '#RRGGBB'."""
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse a hex color string ('#RRGGBB' or 'RRGGBB') to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: '{hex_color}'. Expected 6 hex digits.")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def nearest_color_name(r: int, g: int, b: int) -> str:
    """Return the nearest CSS3 color name for an RGB color.

    Uses Euclidean distance in RGB space to find the closest named color.
    Falls back to the hex string if the distance is very large.
    """
    min_dist = float("inf")
    closest_name = rgb_to_hex(r, g, b)

    for name in webcolors.names("css3"):
        hex_val = webcolors.name_to_hex(name)
        nr, ng, nb = hex_to_rgb(hex_val)
        dist = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name
