"""Pop-up showing each issuer's observation span."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.residual_viz import render_coverage
from ycn.gui.styles import APP_STYLE


class CoverageDialog(QDialog):
    """Coverage-by-issuer chart for the currently loaded panel."""

    def __init__(
        self,
        coverage: pl.DataFrame,
        issuer_column: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coverage by issuer")
        self.setStyleSheet(APP_STYLE)
        self.resize(900, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._figure = render_coverage(coverage, issuer_column)
        canvas = FigureCanvas(self._figure)
        canvas.setStyleSheet("background-color: transparent;")

        # A long issuer list makes the figure taller than the dialog; scroll
        # rather than squashing the rows into illegibility.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(canvas)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def closeEvent(self, event) -> None:  # noqa: N802
        plt.close(self._figure)
        super().closeEvent(event)
