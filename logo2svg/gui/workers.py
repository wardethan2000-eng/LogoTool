"""Background worker threads for pipeline operations.

Every heavy operation (load, quantize, trace, export) runs in a
:class:`QThread` so the GUI stay responsive.  Each worker emits
``finished`` on success and ``error`` on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from ..session import Session


class _BaseWorker(QThread):
    """Base class that catches exceptions and emits ``error``."""

    finished = pyqtSignal(object)  # payload depends on subclass
    error = pyqtSignal(str)
    progress = pyqtSignal(str)       # status text

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session

    # Subclasses override _run() instead of run().
    def run(self) -> None:
        try:
            result = self._run()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    def _run(self) -> Any:
        raise NotImplementedError


class LoadWorker(_BaseWorker):
    """Load an image + detect background, optionally remove TM symbols."""

    def __init__(
        self,
        session: Session,
        path: str,
        bg_color: str | None = None,
        remove_tm: bool = True,
        tm_max_area_pct: float = 1.5,
        tm_margin_pct: float = 12.0,
        parent=None,
    ):
        super().__init__(session, parent)
        self._path = path
        self._bg_color = bg_color
        self._remove_tm = remove_tm
        self._tm_max_area_pct = tm_max_area_pct
        self._tm_margin_pct = tm_margin_pct

    def _run(self) -> None:
        self.progress.emit("Loading image\u2026")
        removed = self._session.load_and_prepare(
            self._path,
            bg_color=self._bg_color,
            remove_tm=self._remove_tm,
            tm_max_area_pct=self._tm_max_area_pct,
            tm_margin_pct=self._tm_margin_pct,
        )
        if removed:
            self.progress.emit(f"Removed {removed} TM symbol(s)")
        return None


class PreprocessWorker(_BaseWorker):
    """Run smart image preprocessing (contrast, sharpen, denoise)."""

    def __init__(
        self,
        session: Session,
        *,
        contrast: bool = False,
        sharpen: bool = False,
        denoise: bool = False,
        contrast_strength: float = 2.0,
        sharpen_strength: float = 1.0,
        denoise_strength: int = 10,
        parent=None,
    ):
        super().__init__(session, parent)
        self._contrast = contrast
        self._sharpen = sharpen
        self._denoise = denoise
        self._contrast_strength = contrast_strength
        self._sharpen_strength = sharpen_strength
        self._denoise_strength = denoise_strength

    def _run(self):
        self.progress.emit("Analyzing image\u2026")
        report = self._session.run_preprocessing(
            contrast=self._contrast,
            sharpen=self._sharpen,
            denoise=self._denoise,
            contrast_strength=self._contrast_strength,
            sharpen_strength=self._sharpen_strength,
            denoise_strength=self._denoise_strength,
        )
        if report.recommendations:
            self.progress.emit(report.recommendations[0][:60] + "\u2026")
        return report


class QuantizeWorker(_BaseWorker):
    """Run colour quantization."""

    def __init__(
        self,
        session: Session,
        n_colors: int | None = None,
        target_colors: list[str] | None = None,
        parent=None,
    ):
        super().__init__(session, parent)
        self._n_colors = n_colors
        self._target_colors = target_colors

    def _run(self) -> None:
        self.progress.emit("Quantizing colours\u2026")
        self._session.quantize(
            n_colors=self._n_colors,
            target_colors=self._target_colors,
        )
        return None


class TraceWorker(_BaseWorker):
    """Run Potrace on all layer masks."""

    def _run(self) -> None:
        self.progress.emit("Tracing vector paths\u2026")
        self._session.trace()
        return None


class ExportWorker(_BaseWorker):
    """Write SVG files."""

    def __init__(
        self,
        session: Session,
        output_dir: str,
        combined: bool = False,
        parent=None,
    ):
        super().__init__(session, parent)
        self._output_dir = output_dir
        self._combined = combined

    def _run(self) -> list[Path]:
        self.progress.emit("Exporting SVGs\u2026")
        return self._session.export(self._output_dir, combined=self._combined)


class ImportSvgWorker(_BaseWorker):
    """Import an external SVG as colour layers."""

    def __init__(
        self,
        session: Session,
        svg_path: str,
        color_override: str | None = None,
        parent=None,
    ):
        super().__init__(session, parent)
        self._svg_path = svg_path
        self._color_override = color_override

    def _run(self) -> int:
        self.progress.emit("Importing SVG\u2026")
        count = self._session.import_svg(
            self._svg_path, color_override=self._color_override,
        )
        self.progress.emit(f"Imported {count} layer(s) from SVG")
        return count
