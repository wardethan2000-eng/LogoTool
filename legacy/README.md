# Legacy: Catmull-Rom Bézier Fitting

The `bezier_fit.py` module implemented cubic Bézier curve generation via
Catmull-Rom spline interpolation.  It was used by the original OpenCV
contour tracing pipeline to smooth contour points into SVG curves.

This module was **replaced by Potrace** (`tracer.py`), which handles the
entire bitmap → optimised Bézier pipeline in a single step with superior
results.  The code and its tests are preserved here for reference.

Files:
- `bezier_fit.py` — Catmull-Rom spline → cubic Bézier fitting
- `test_bezier_fit.py` — Unit tests for the above

# Legacy: Hue-Based Quantizer

`quantizer_hue.py` implements an alternative color quantization strategy
that groups foreground pixels by hue (HSV) rather than running K-means in
CIELAB space.  It was never wired into the CLI or GUI and has no tests.

The approach is interesting for solid-color logos with very distinct hues,
but the CIELAB K-means quantizer (`logo2svg/quantizer.py`) proved more
robust across a wider range of inputs.  The module is preserved here in
case the hue-based strategy is revisited in the future.

Files:
- `quantizer_hue.py` — Hue-family grouping quantizer
