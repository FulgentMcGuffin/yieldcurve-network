"""The "NS Residuals" tab: aesthetic pickers over the residual-metric chart.

The chart is redrawn in the GUI thread on every selection change. That is
cheap: the metrics frame has one row per component network (tens, not
thousands), so the whole figure rebuilds in a few milliseconds.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.residual_networks import aesthetic_columns
from ycn.analysis.residual_viz import render_residual_metrics
from ycn.gui.styles import TEXT_MUTED

# Aesthetic defaults, per the tab's specification. Each falls back to the first
# eligible column when the metrics frame does not carry it.
DEFAULT_Y = "modularity"
DEFAULT_SHAPE = "is_connected"
DEFAULT_FILL = "modularity"
DEFAULT_SIZE = "avg_eccentricity"

# Offered in every aesthetic that can be switched off.
NONE_LABEL = "(none)"


class NSResidualsTab(QFrame):
    """Chart plus its four aesthetic pickers and the Coverage button."""

    def __init__(
        self,
        on_coverage: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Canvas")
        self._metrics = pl.DataFrame()
        self._label_order: list[str] = []
        self._label_title = "label"
        self._figure = None
        self._canvas: FigureCanvas | None = None
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.cmb_y = self._picker(controls, "Y axis")
        self.cmb_shape = self._picker(controls, "Shape")
        self.cmb_fill = self._picker(controls, "Fill")
        self.cmb_size = self._picker(controls, "Size")
        controls.addStretch(1)

        self.btn_coverage = QPushButton("Coverage…")
        self.btn_coverage.setObjectName("SecondaryButton")
        self.btn_coverage.setToolTip(
            "Show each issuer's first and last observation in the loaded panel"
        )
        self.btn_coverage.clicked.connect(on_coverage)
        controls.addWidget(self.btn_coverage)
        outer.addLayout(controls)

        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._holder, stretch=1)

        self.set_placeholder(
            "Residual networks will appear here after you build a network."
        )

    def _picker(self, row: QHBoxLayout, title: str) -> QComboBox:
        label = QLabel(f"{title}:")
        label.setStyleSheet(f"color: {TEXT_MUTED};")
        row.addWidget(label)
        combo = QComboBox()
        combo.setMinimumWidth(150)
        combo.currentIndexChanged.connect(self._on_selection_changed)
        row.addWidget(combo)
        return combo

    # ------------------------------------------------------------- content
    def _clear_holder(self) -> None:
        while self._holder_layout.count():
            widget = self._holder_layout.takeAt(0).widget()
            if isinstance(widget, FigureCanvas) and widget.figure:
                plt.close(widget.figure)
            if widget:
                widget.deleteLater()
        self._canvas = None

    def set_placeholder(self, message: str) -> None:
        self._clear_holder()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; background: transparent;"
        )
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._holder_layout.addWidget(label)
        self.btn_coverage.setEnabled(not self._metrics.is_empty())

    def set_result(
        self,
        metrics: pl.DataFrame,
        label_order: list[str],
        label_title: str,
    ) -> None:
        """Adopt a new metrics frame, repopulate the pickers, and draw."""
        self._metrics = metrics
        self._label_order = list(label_order)
        self._label_title = label_title
        self._populate_pickers()
        self.btn_coverage.setEnabled(True)
        self._redraw()

    @property
    def metrics(self) -> pl.DataFrame:
        return self._metrics

    def _populate_pickers(self) -> None:
        options = aesthetic_columns(self._metrics)
        numeric, discrete = options["numeric"], options["discrete"]

        self._syncing = True
        for combo, choices, default, optional in (
            (self.cmb_y, numeric, DEFAULT_Y, False),
            (self.cmb_shape, discrete, DEFAULT_SHAPE, True),
            (self.cmb_fill, numeric, DEFAULT_FILL, True),
            (self.cmb_size, numeric, DEFAULT_SIZE, True),
        ):
            combo.clear()
            if optional:
                combo.addItem(NONE_LABEL, None)
            for name in choices:
                combo.addItem(name, name)
            index = combo.findData(default)
            combo.setCurrentIndex(index if index >= 0 else (1 if optional else 0))
            combo.setEnabled(bool(choices))
        self._syncing = False

    def _on_selection_changed(self, _index: int) -> None:
        if self._syncing:
            return
        self._redraw()

    def selection(self) -> dict[str, str | None]:
        """Current aesthetic choices, for logging and for the data viewer."""
        return {
            "y": self.cmb_y.currentData(),
            "shape": self.cmb_shape.currentData(),
            "fill": self.cmb_fill.currentData(),
            "size": self.cmb_size.currentData(),
        }

    def _redraw(self) -> None:
        if self._metrics.is_empty():
            return
        choice = self.selection()
        y_column = choice["y"]
        if not y_column:
            self.set_placeholder("Pick a numeric column for the y axis.")
            return

        figure = render_residual_metrics(
            self._metrics,
            self._label_order,
            label_column="label",
            label_title=self._label_title,
            y_column=y_column,
            shape_column=choice["shape"],
            fill_column=choice["fill"],
            size_column=choice["size"],
        )
        self._clear_holder()
        self._figure = figure
        self._canvas = FigureCanvas(figure)
        self._canvas.setStyleSheet("background-color: transparent;")
        self._holder_layout.addWidget(self._canvas)
        self._canvas.draw()
