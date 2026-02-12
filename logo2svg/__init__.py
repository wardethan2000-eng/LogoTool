"""logo2svg - Convert PNG logos to color-separated SVGs for 3D printing."""

__version__ = "0.2.0"

from .session import Session, LayerInfo
from .preprocessor import PreprocessReport

__all__ = ["Session", "LayerInfo", "PreprocessReport"]
