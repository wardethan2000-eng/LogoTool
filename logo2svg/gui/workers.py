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
    """Load an image + detect background."""

    def __init__(self, session: Session, path: str, bg_color: str | None = None, parent=None):
        super().__init__(session, parent)
        self._path = path
        self._bg_color = bg_color

    def _run(self) -> None:
        self.progress.emit("Loading image…")
        self._session.load(self._path, bg_color=self._bg_color)
        return None


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
        self.progress.emit("Quantizing colours…")
        self._session.quantize(
            n_colors=self._n_colors,
            target_colors=self._target_colors,
        )
        return None


class TraceWorker(_BaseWorker):
    """Run Potrace on all layer masks."""

    def _run(self) -> None:
        self.progress.emit("Tracing vector paths…")
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
        self.progress.emit("Exporting SVGs…")
        return self._session.export(self._output_dir, combined=self._combined)
