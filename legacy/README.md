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
