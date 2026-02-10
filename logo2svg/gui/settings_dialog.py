"""Settings dialog: min_area, alphamax, opttolerance, background colour."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    """Modal dialog exposing pipeline tunables.

    Callers check ``exec() == QDialog.DialogCode.Accepted`` then read the
    values through the public properties.
    """

    def __init__(
        self,
        *,
        min_area: int = 100,
        alphamax: float = 1.0,
        opttolerance: float = 0.2,
        bg_color: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # min_area
        self._min_area = QSpinBox()
        self._min_area.setRange(0, 100_000)
        self._min_area.setValue(min_area)
        self._min_area.setToolTip(
            "Minimum connected-component area in pixels. "
            "Smaller regions are filtered as noise."
        )
        form.addRow("Min area (px):", self._min_area)

        # alphamax
        self._alphamax = QDoubleSpinBox()
        self._alphamax.setRange(0.0, 1.334)
        self._alphamax.setDecimals(3)
        self._alphamax.setSingleStep(0.05)
        self._alphamax.setValue(alphamax)
        self._alphamax.setToolTip(
            "Potrace corner-detection threshold. Lower = sharper corners, "
            "higher = smoother curves."
        )
        form.addRow("Alphamax:", self._alphamax)

        # opttolerance
        self._opttolerance = QDoubleSpinBox()
        self._opttolerance.setRange(0.0, 5.0)
        self._opttolerance.setDecimals(3)
        self._opttolerance.setSingleStep(0.05)
        self._opttolerance.setValue(opttolerance)
        self._opttolerance.setToolTip(
            "Potrace curve optimisation tolerance. Lower = more faithful, "
            "higher = fewer Bézier segments."
        )
        form.addRow("Opt tolerance:", self._opttolerance)

        # Background colour override
        self._bg_color = QLineEdit(bg_color)
        self._bg_color.setPlaceholderText("#FFFFFF (leave blank for auto)")
        self._bg_color.setToolTip(
            "Hex colour to force as background, or leave blank for auto-detect."
        )
        form.addRow("Background colour:", self._bg_color)

        layout.addLayout(form)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
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
    def bg_color(self) -> str:
        return self._bg_color.text().strip()
