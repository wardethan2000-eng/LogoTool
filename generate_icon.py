"""Generate the QuickLayer application icon as an ICO file.

The icon depicts three stacked, slightly offset rounded rectangles
(layers) in the accent palette — a visual metaphor for color layers.
"""

from PIL import Image, ImageDraw

SIZES = [16, 24, 32, 48, 64, 128, 256]

# Palette (indigo accent tones matching the app's look)
COLORS = [
    (99, 102, 241),   # indigo-500 — back layer
    (79, 70, 229),    # indigo-600 — middle layer (accent)
    (67, 56, 202),    # indigo-700 — front layer
]

SHADOW = (0, 0, 0, 40)  # subtle drop-shadow


def _draw_layer(draw: ImageDraw.ImageDraw, bbox, color, radius):
    """Draw a rounded rectangle layer with a thin dark edge."""
    x0, y0, x1, y1 = bbox
    # Shadow / edge (offset 1px down)
    draw.rounded_rectangle(
        (x0, y0 + 1, x1, y1 + 1),
        radius=radius,
        fill=SHADOW,
    )
    # Main shape
    draw.rounded_rectangle(bbox, radius=radius, fill=(*color, 255))
    # Subtle top highlight
    highlight = tuple(min(c + 40, 255) for c in color)
    draw.rounded_rectangle(
        (x0 + 1, y0 + 1, x1 - 1, y0 + max(2, radius // 2)),
        radius=max(1, radius // 2),
        fill=(*highlight, 80),
    )


def generate_icon_image(size: int) -> Image.Image:
    """Return a single-resolution RGBA icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(1, size // 10)
    layer_h = max(4, int((size - 2 * pad) * 0.38))
    layer_w = max(6, size - 2 * pad)
    offset_y = max(2, int(size * 0.14))  # vertical shift per layer
    offset_x = max(1, int(size * 0.04))  # horizontal shift per layer
    radius = max(1, size // 8)

    total_h = layer_h + offset_y * 2
    start_y = (size - total_h) // 2

    for i, color in enumerate(COLORS):
        x0 = pad + (2 - i) * offset_x
        y0 = start_y + i * offset_y
        x1 = x0 + layer_w - (2 - i) * offset_x * 2
        y1 = y0 + layer_h
        _draw_layer(draw, (x0, y0, x1, y1), color, radius)

    return img


def main():
    images = [generate_icon_image(s) for s in SIZES]
    ico_path = "logo2svg/icons/quicklayer.ico"
    # Save ICO with all sizes embedded
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=images[1:],
    )
    print(f"Created {ico_path} with sizes {SIZES}")

    # Also save a 256px PNG for other uses
    png_path = "logo2svg/icons/quicklayer_256.png"
    images[-1].save(png_path)
    print(f"Created {png_path}")


if __name__ == "__main__":
    main()
