"""The "Evo: Cov(t)" tab: a stress trajectory with a date scrubber.

Two of the four stress series become the axes; the points are joined in time
order, stressed windows are ringed, and a slider walks a cursor along the path.
The slider moves a single artist rather than re-rendering -- rebuilding the
figure on every tick makes a scrub feel broken.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ycn.analysis.mln_evolution import STRESS_SERIES, STRESS_THRESHOLD
from ycn.analysis.mln_evolution_viz import TrajectoryPlot, render_stress_trajectory
from ycn.gui.styles import TEXT, TEXT_MUTED

DEFAULT_X = "avg_abs_corr"
DEFAULT_Y = "corr_variance"


class StressTrajectoryTab(QFrame):
    """Axis pickers, the trajectory canvas, and the date slider."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Canvas")
        self._stress = pl.DataFrame()
        self._dates: list = []
        self._plot: TrajectoryPlot | None = None
        self._canvas: FigureCanvas | None = None
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.cmb_x = self._picker(controls, "X axis")
        self.cmb_y = self._picker(controls, "Y axis")
        controls.addStretch(1)
        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet(f"color: {TEXT_MUTED};")
        controls.addWidget(self.lbl_warning)
        outer.addLayout(controls)

        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._holder, stretch=1)

        scrub = QHBoxLayout()
        scrub.setSpacing(8)
        caption = QLabel("Date:")
        caption.setStyleSheet(f"color: {TEXT_MUTED};")
        scrub.addWidget(caption)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        scrub.addWidget(self.slider, stretch=1)
        self.lbl_date = QLabel("—")
        self.lbl_date.setMinimumWidth(160)
        self.lbl_date.setStyleSheet(f"color: {TEXT}; font-weight: 600;")
        scrub.addWidget(self.lbl_date)
        outer.addLayout(scrub)

        self.set_placeholder(
            "The stress trajectory will appear here after the evolution runs."
        )

    def _picker(self, row: QHBoxLayout, title: str) -> QComboBox:
        label = QLabel(f"{title}:")
        label.setStyleSheet(f"color: {TEXT_MUTED};")
        row.addWidget(label)
        combo = QComboBox()
        combo.setMinimumWidth(230)
        for column, display in STRESS_SERIES.items():
            combo.addItem(display, column)
        combo.currentIndexChanged.connect(self._on_axis_changed)
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
        self._plot = None

    def set_placeholder(self, message: str) -> None:
        self._clear_holder()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; background: transparent;"
        )
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._holder_layout.addWidget(label)
        self.slider.setEnabled(False)
        self.lbl_date.setText("—")

    def set_result(self, stress: pl.DataFrame) -> None:
        """Adopt a new stress frame and draw the default axis pair."""
        self._stress = stress.sort("date_end") if not stress.is_empty() else stress
        self._dates = (
            self._stress.get_column("date_end").to_list()
            if not self._stress.is_empty()
            else []
        )

        self._syncing = True
        x_index = self.cmb_x.findData(DEFAULT_X)
        y_index = self.cmb_y.findData(DEFAULT_Y)
        self.cmb_x.setCurrentIndex(max(x_index, 0))
        self.cmb_y.setCurrentIndex(max(y_index, 0))
        self._syncing = False

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(self._dates) - 1, 0))
        self.slider.setValue(0)
        self.slider.setEnabled(len(self._dates) > 1)
        self.slider.blockSignals(False)
        self._redraw()

    @property
    def stress(self) -> pl.DataFrame:
        return self._stress

    def _on_axis_changed(self, _index: int) -> None:
        if self._syncing:
            return
        self._redraw()

    def _on_slider(self, value: int) -> None:
        if self._plot is None or not self._dates:
            return
        index = max(0, min(value, len(self._dates) - 1))
        self._plot.set_highlight(index)
        self.lbl_date.setText(str(self._dates[index]))

    def _redraw(self) -> None:
        if self._stress.is_empty():
            return
        x_column = self.cmb_x.currentData()
        y_column = self.cmb_y.currentData()
        if x_column == y_column:
            # The spec forbids it and the plot would be a diagonal line.
            self.lbl_warning.setText("Pick two different series.")
            self.set_placeholder("The x and y axes must use different stress series.")
            return
        self.lbl_warning.setText("")

        plot = render_stress_trajectory(self._stress, x_column, y_column)
        self._clear_holder()
        if isinstance(plot, TrajectoryPlot):
            self._plot = plot
            figure = plot.figure
        else:
            figure = plot
        self._canvas = FigureCanvas(figure)
        self._canvas.setStyleSheet("background-color: transparent;")
        self._holder_layout.addWidget(self._canvas)
        self._canvas.draw()

        stressed = (
            self._stress.filter(pl.col("stress_pct") > STRESS_THRESHOLD).height
            if "stress_pct" in self._stress.columns
            else 0
        )
        if stressed == 0:
            self.lbl_warning.setText("No window exceeds the stress threshold.")
        self._on_slider(self.slider.value())
