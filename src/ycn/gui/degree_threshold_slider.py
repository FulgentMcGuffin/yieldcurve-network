"""Live independence-threshold slider under a layer's degree histogram.

Ported from the "Degree histogram" tab in ``tgraphportfolio``. Dragging the
slider re-thresholds that layer's **measure matrix** and redraws the histogram
in place, so you can see how connectivity collapses or fills in as the edge
cutoff moves, without leaving the tab or rebuilding the network.

Two things this needs that a thresholded graph cannot provide, and which is why
``MLNResult`` carries ``layer_measures``: the raw measure matrix (the graph has
already had the cutoff applied, and that is not reversible) and the measure's
own value range, which decides the slider's bounds.

Redrawing reuses the existing ``Figure`` and ``Axes`` rather than building a new
one, so the canvas keeps its size and the tab does not jump on every tick.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from ycn.analysis.degree_hist import (
    BG_APP,
    TEXT,
    TEXT_LOG,
    TEXT_MUTED,
    _annotate_extreme_nodes,
)
from ycn.analysis.network import build_corr_nx

# Measures whose values already sit in [0, 1]; everything else here is a signed
# correlation and needs the slider to reach below zero.
_NORMALISED_MEASURES = {
    "distance_correlation",
    "kendall_tau",
    "dtw_distance",
    "fastdtw_distance",
    "shrinkage_correlation",
    "conditional_correlation",
    "mutual_information",
    "chatterjee_xi",
    "maximal_correlation",
}

_SLIDER_QSS = """
QSlider::groove:horizontal {
    background: #334155;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #7dd3fc;
}
"""


def threshold_bounds(measure: str) -> tuple[float, float]:
    """Slider range for a measure: ``[0, 1]`` normalised, ``[-1, 1]`` signed."""
    return (0.0, 1.0) if measure in _NORMALISED_MEASURES else (-1.0, 1.0)


def build_threshold_slider(
    *,
    measure_df: pl.DataFrame,
    measure: str,
    initial: float,
    figure: Figure,
    canvas: FigureCanvas,
    bins: int,
    title: str,
    on_changed: Callable[[float, dict], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> QWidget:
    """A ``Threshold: [====|====] 0.42`` strip wired to redraw ``figure``.

    Args:
        measure_df: The layer's square, un-thresholded measure matrix.
        measure: Measure id, for the slider's range.
        initial: Threshold the histogram was first drawn at.
        figure/canvas: The histogram to redraw in place.
        bins: Histogram bins, from MLN Settings.
        title: Kept on the axes across redraws.
        on_changed: Notified with ``(threshold, degrees)`` after each redraw,
            so the eye button's table can follow what is on screen.
        on_error: Receives a message if a redraw fails.
    """
    low, high = threshold_bounds(measure)

    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(4, 0, 4, 0)
    row.setSpacing(6)

    caption = QLabel("Threshold:")
    caption.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
    caption.setToolTip(
        "Keep edges whose connection measure is at least this. Redraws this "
        "layer's histogram only — the built network is unchanged."
    )
    row.addWidget(caption)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(int(low * 100), int(high * 100))
    slider.setValue(int(round(initial * 100)))
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setTickInterval(10)
    slider.setStyleSheet(_SLIDER_QSS)
    row.addWidget(slider, 1)

    value_label = QLabel(f"{initial:.2f}")
    value_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; min-width: 36px;")
    row.addWidget(value_label)

    def redraw(raw: int) -> None:
        threshold = raw / 100.0
        value_label.setText(f"{threshold:.2f}")
        try:
            graph = build_corr_nx(measure_df, independent_threshold=threshold)
            degrees = dict(graph.degree())
            deg_list = list(degrees.values())
            if not deg_list:
                return

            axes = figure.axes
            if not axes:
                return
            ax = axes[0]
            ax.clear()

            mean_deg = float(np.mean(deg_list))
            lo, hi = int(min(deg_list)), int(max(deg_list))
            _counts, bin_edges = np.histogram(deg_list, bins=bins)

            sns.histplot(
                deg_list,
                bins=bin_edges,
                kde=True,
                color="#38bdf8",
                edgecolor="#0ea5e9",
                alpha=0.75,
                line_kws={"color": "#7dd3fc", "linewidth": 2},
                ax=ax,
            )
            ax.axvline(
                mean_deg,
                color="#a78bfa",
                linewidth=2.5,
                linestyle="--",
                label=f"Mean = {mean_deg:2.0f}",
            )
            ax.set_xlim(max(0, lo - 1), hi + 1)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, prune=None))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.minorticks_off()
            ax.set_facecolor(BG_APP)
            ax.set_title(title, color=TEXT, fontsize=13, pad=10)
            ax.set_xlabel("Number of Connections", color=TEXT_MUTED, fontsize=10)
            ax.set_ylabel("Density", color=TEXT_MUTED, fontsize=10)
            ax.tick_params(axis="both", which="major", colors=TEXT_MUTED, labelsize=10)
            ax.grid(
                True,
                axis="both",
                which="major",
                color="#334155",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#475569")

            _annotate_extreme_nodes(ax, degrees, lo, hi, bin_edges)

            legend = ax.legend(loc="best", fontsize=11, frameon=True)
            legend.get_frame().set_facecolor("#1a1c24")
            legend.get_frame().set_edgecolor("#7dd3fc")
            for text in legend.get_texts():
                text.set_color(TEXT_LOG)

            # Hover data must follow the redraw, and the cursor must be rebound
            # to the cleared axes or it keeps pointing at dead artists.
            figure._histogram_cursor_data = (ax, degrees, bin_edges)
            from ycn.gui.layer_figure_tabs import attach_histogram_cursor

            attach_histogram_cursor(figure, canvas)

            canvas.draw_idle()
            if on_changed is not None:
                on_changed(threshold, degrees)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- a bad threshold must not kill the tab
            if on_error is not None:
                on_error(f"threshold redraw: {exc}")

    slider.valueChanged.connect(redraw)
    # The slider owns the only reference to `redraw`'s closure otherwise.
    container._ycn_redraw = redraw
    return container
