"""Modern settings dialog with grouped sections.

Exposes the same pipeline tunables as before, but organised into
clear visual groups (Tracing, Filtering, Background) with a cleaner
layout and descriptions.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from .style import TEXT, TEXT_SEC


class SettingsDialog(QDialog):
    """Modal dialog for pipeline settings.

    Callers check ``exec() == QDialog.DialogCode.Accepted`` then read the
    property values.
    """

    def __init__(
        self,
        *,
        min_area: int = 100,
        alphamax: float = 1.0,
        opttolerance: float = 0.2,
        turdsize: int = 2,
        bg_color: str = "",
        remove_tm: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 20)

        # ── Title ──
        title = QLabel("Pipeline Settings")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {TEXT}; "
            "margin-bottom: 4px;"
        )
        layout.addWidget(title)

        subtitle = QLabel("Configure tracing and processing parameters")
        subtitle.setStyleSheet(
            f"font-size: 13px; color: {TEXT_SEC}; margin-bottom: 8px;"
        )
        layout.addWidget(subtitle)

        # ── Tracing section ──
        trace_group = QGroupBox("Tracing")
        trace_form = QFormLayout(trace_group)
        trace_form.setSpacing(12)

        self._alphamax = QDoubleSpinBox()
        self._alphamax.setRange(0.0, 1.334)
        self._alphamax.setDecimals(3)
        self._alphamax.setSingleStep(0.05)
        self._alphamax.setValue(alphamax)
        self._alphamax.setToolTip(
            "Corner detection threshold.\n"
            "Lower = sharper corners, higher = smoother curves."
        )
        trace_form.addRow("Corner threshold:", self._alphamax)

        self._opttolerance = QDoubleSpinBox()
        self._opttolerance.setRange(0.0, 5.0)
        self._opttolerance.setDecimals(3)
        self._opttolerance.setSingleStep(0.05)
        self._opttolerance.setValue(opttolerance)
        self._opttolerance.setToolTip(
            "Curve optimisation tolerance.\n"
            "Lower = more faithful, higher = fewer B\u00e9zier segments."
        )
        trace_form.addRow("Curve tolerance:", self._opttolerance)

        self._turdsize = QSpinBox()
        self._turdsize.setRange(0, 100)
        self._turdsize.setValue(turdsize)
        self._turdsize.setToolTip(
            "Speckle suppression.\n"
            "Components up to this size are discarded during tracing."
        )
        trace_form.addRow("Speckle size (px):", self._turdsize)

        layout.addWidget(trace_group)

        # ── Filtering section ──
        filter_group = QGroupBox("Filtering")
        filter_form = QFormLayout(filter_group)
        filter_form.setSpacing(12)

        self._min_area = QSpinBox()
        self._min_area.setRange(0, 100_000)
        self._min_area.setValue(min_area)
        self._min_area.setToolTip(
            "Minimum connected-component area.\n"
            "Smaller regions are discarded as noise."
        )
        filter_form.addRow("Min area (px):", self._min_area)

        self._remove_tm = QCheckBox("Remove TM / \u00ae symbols from margins")
        self._remove_tm.setChecked(remove_tm)
        self._remove_tm.setToolTip(
            "Auto-detect and remove trademark symbols near edges."
        )
        filter_form.addRow("", self._remove_tm)

        layout.addWidget(filter_group)

        # ── Background section ──
        bg_group = QGroupBox("Background")
        bg_form = QFormLayout(bg_group)
        bg_form.setSpacing(12)

        self._bg_color = QLineEdit(bg_color)
        self._bg_color.setPlaceholderText("#FFFFFF or leave blank for auto")
        self._bg_color.setToolTip(
            "Force a specific background colour, or leave blank for auto-detect."
        )
        bg_form.addRow("Override colour:", self._bg_color)

        layout.addWidget(bg_group)

        layout.addStretch()

        # ── OK / Cancel ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Style the OK button as primary
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setProperty("cssClass", "primary")
            ok_btn.style().unpolish(ok_btn)
            ok_btn.style().polish(ok_btn)

        layout.addWidget(buttons)

    # -- public property accessors ----------------------------------------

    @property
    def min_area(self) -> int:
        return self._min_area.value()

    @property
    def alphamax(self) -> float:
        return self._alphamax.value()

    @property
    def opttolerance(self) -> float:
        return self._opttolerance.value()

    @property
    def turdsize(self) -> int:
        return self._turdsize.value()

    @property
    def bg_color(self) -> str:
        return self._bg_color.text().strip()

    @property
    def remove_tm(self) -> bool:
        return self._remove_tm.isChecked()
