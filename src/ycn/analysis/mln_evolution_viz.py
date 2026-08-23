"""Figures for the four multi-layer-network evolution tabs.

Built with the ``Figure`` API rather than pyplot so they can be rendered on a
worker thread (see ``plot_theme``). The trajectory plot is the exception: it is
interactive, so it exposes a controller the GUI drives from a slider instead of
re-rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .mln_evolution import STRESS_SERIES, STRESS_THRESHOLD
from .plot_theme import (
    ACCENT,
    CATEGORICAL_10,
    DANGER,
    FG,
    GRID,
    INTER_COLOR,
    INTRA_COLOR,
    MUTED,
    POSITIVE,
    STRESS,
    WARN,
    empty_figure,
    style_axes,
    style_date_axis,
    style_legend,
)
from .plot_theme import BG

FACTOR_ROWS = ("Level", "Slope", "Curvature")
_MEAN_COLUMNS = {
    "Level": "level_mean",
    "Slope": "slope_mean",
    "Curvature": "curvature_mean",
}
_STD_COLUMNS = {
    "Level": "level_std",
    "Slope": "slope_std",
    "Curvature": "curvature_std",
}


def _dates(frame: pl.DataFrame, column: str = "date_end") -> np.ndarray:
    return np.asarray(frame.get_column(column).to_list())


# --------------------------------------------------------------- Evo: Links
def render_edge_evolution(
    edge_types: pl.DataFrame,
    community_k: pl.DataFrame,
    *,
    width: float = 15,
    height: float = 9,
    dpi: int = 100,
) -> Figure:
    """Edge counts + composition over the top half, community k over the bottom."""
    if edge_types.is_empty():
        return empty_figure(
            "No evolution windows produced any multiplex.", width, height, dpi
        )

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    methods = (
        sorted(community_k.get_column("method").unique().to_list())
        if not community_k.is_empty()
        else []
    )
    n_facets = max(len(methods), 1)
    grid = GridSpec(
        2, max(2, n_facets), figure=fig, hspace=0.55, wspace=0.3, height_ratios=[1, 1]
    )

    span = max(2, n_facets) // 2
    _draw_edge_counts(fig.add_subplot(grid[0, :span]), edge_types)
    _draw_edge_composition(fig.add_subplot(grid[0, span:]), edge_types)

    if not methods:
        ax = fig.add_subplot(grid[1, :])
        style_axes(ax)
        ax.text(
            0.5,
            0.5,
            "No community detection results for these windows.",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Sits above the bottom row rather than being a figure suptitle: it titles
    # the community facets only, not the edge panels above them.
    fig.text(
        0.5,
        0.50,
        "#Communities by Method",
        ha="center",
        va="bottom",
        color=FG,
        fontsize=13,
        fontweight="bold",
    )

    shared = None
    for i, method in enumerate(methods):
        ax = fig.add_subplot(grid[1, i], sharey=shared)
        shared = shared or ax
        sub = community_k.filter(pl.col("method") == method).sort("date_end")
        colour = CATEGORICAL_10[i % len(CATEGORICAL_10)]
        ax.plot(
            _dates(sub),
            sub.get_column("n_clusters").to_list(),
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=colour,
        )
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("Window end" if i == len(methods) // 2 else "")
        if i == 0:
            ax.set_ylabel("k")
        else:
            ax.tick_params(labelleft=False)
        style_axes(ax, labelsize=7, grid=True)
        style_date_axis(ax, n_ticks=4)
    return fig


def _draw_edge_counts(ax, edge_types: pl.DataFrame) -> None:
    x = _dates(edge_types)
    for column, label, colour in (
        ("intra_edges", "Intra-layer", INTRA_COLOR),
        ("inter_edges", "Inter-layer", INTER_COLOR),
    ):
        ax.plot(
            x,
            edge_types.get_column(column).to_list(),
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=label,
            color=colour,
        )
    ax.set_title("Edge Counts by Type", fontsize=11, fontweight="bold")
    ax.set_xlabel("Window end")
    ax.set_ylabel("Edge count")
    style_axes(ax, labelsize=8, grid=True)
    style_date_axis(ax)
    style_legend(ax.legend(loc="best", fontsize=8))


def _draw_edge_composition(ax, edge_types: pl.DataFrame) -> None:
    x = _dates(edge_types)
    intra = np.asarray(edge_types.get_column("pct_intra").to_list(), dtype=float)
    inter = np.asarray(edge_types.get_column("pct_inter").to_list(), dtype=float)
    ax.stackplot(
        x,
        intra,
        inter,
        labels=["Intra-layer", "Inter-layer"],
        colors=[INTRA_COLOR, INTER_COLOR],
        alpha=0.75,
    )
    ax.set_ylim(0, 100)
    ax.set_title("Edge Type Composition", fontsize=11, fontweight="bold")
    ax.set_xlabel("Window end")
    ax.set_ylabel("Percentage (%)")
    style_axes(ax, labelsize=8)
    style_date_axis(ax)
    style_legend(ax.legend(loc="lower right", fontsize=8))


# ------------------------------------------------------------------ Evo: NS
def render_factor_evolution(
    factors: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    std: bool = False,
    width: float = 14,
    height: float = 8,
    dpi: int = 100,
) -> Figure:
    """Level / Slope / Curvature over time, shaded by regime.

    ``std=True`` renders the within-window standard deviation (factor
    volatility) instead of the mean, on identical axes and styling so the two
    sub-tabs can be read against each other.
    """
    if factors.is_empty():
        return empty_figure("No factor windows produced.", width, height, dpi)

    columns = _STD_COLUMNS if std else _MEAN_COLUMNS
    title = (
        "Yield Curve Factor Volatility Over Time"
        if std
        else "Yield Curve Factor Evolution Over Time"
    )

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    axes = fig.subplots(len(FACTOR_ROWS), 1, sharex=True)
    x = _dates(factors)

    bands, regime_colours = _regime_bands(factors, regimes)

    for ax, row in zip(np.atleast_1d(axes), FACTOR_ROWS):
        for lo, hi, regime in bands:
            ax.axvspan(lo, hi, color=regime_colours[regime], alpha=0.15, linewidth=0)
        ax.plot(
            x,
            factors.get_column(columns[row]).to_list(),
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=ACCENT,
        )
        ax.set_ylabel(row, color=FG, fontsize=10, fontweight="bold")
        style_axes(ax, labelsize=8, grid=True)

    bottom = np.atleast_1d(axes)[-1]
    bottom.set_xlabel("Window end")
    style_date_axis(bottom)

    if regime_colours:
        handles = [
            Patch(facecolor=colour, alpha=0.5, label=name)
            for name, colour in regime_colours.items()
        ]
        style_legend(
            fig.legend(
                handles=handles,
                loc="upper right",
                fontsize=8,
                title="Regime",
                frameon=True,
            )
        )
    fig.suptitle(title, color=FG, fontsize=14, fontweight="bold")
    fig.subplots_adjust(hspace=0.18, top=0.92)
    return fig


def _regime_bands(
    factors: pl.DataFrame, regimes: pl.DataFrame
) -> tuple[list[tuple], dict[str, str]]:
    """Half-open x spans per window plus a colour per regime name."""
    if regimes.is_empty() or "regime" not in regimes.columns:
        return [], {}
    names = sorted(regimes.get_column("regime").unique().to_list())
    colours = {
        name: CATEGORICAL_10[i % len(CATEGORICAL_10)] for i, name in enumerate(names)
    }
    x = _dates(factors)
    labels = regimes.get_column("regime").to_list()
    if len(x) < 2:
        return [], colours

    # Bands are centred on each window end and meet halfway to their
    # neighbours, so the shading tiles the axis without gaps or overlap.
    half = (x[1] - x[0]) / 2
    bands = [
        (x[i] - half, x[i] + half, labels[i]) for i in range(min(len(x), len(labels)))
    ]
    return bands, colours


# ----------------------------------------------------------------- Evo: Cov
def render_stress_quadrants(
    stress: pl.DataFrame,
    *,
    width: float = 14,
    height: float = 9,
    dpi: int = 100,
) -> Figure:
    """The four stress indicators, one per quadrant."""
    if stress.is_empty():
        return empty_figure("No stress windows produced.", width, height, dpi)

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    axes = fig.subplots(2, 2)
    x = _dates(stress)

    ax = axes[0][0]
    values = stress.get_column("avg_abs_corr").to_list()
    ax.plot(x, values, marker="o", markersize=3, linewidth=1.3, color=INTRA_COLOR)
    ax.axhline(float(np.mean(values)), linestyle="--", color=INTER_COLOR, linewidth=1)
    ax.set_title(STRESS_SERIES["avg_abs_corr"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation")

    ax = axes[0][1]
    ax.plot(
        x,
        stress.get_column("corr_variance").to_list(),
        marker="s",
        markersize=3,
        linewidth=1.3,
        color=POSITIVE,
    )
    ax.set_title("Correlation Variance (Heterogeneity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Variance")

    ax = axes[1][0]
    ax.bar(
        x,
        stress.get_column("n_sig_edges").to_list(),
        width=_bar_width(x),
        color=DANGER,
        alpha=0.85,
    )
    ax.set_title(STRESS_SERIES["n_sig_edges"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")

    ax = axes[1][1]
    pct = np.asarray(stress.get_column("stress_pct").to_list(), dtype=float)
    colours = [STRESS if v > STRESS_THRESHOLD else INTRA_COLOR for v in pct]
    ax.bar(x, pct, width=_bar_width(x), color=colours, alpha=0.9)
    ax.axhline(STRESS_THRESHOLD, linestyle="--", color=STRESS, linewidth=1)
    ax.set_title(STRESS_SERIES["stress_pct"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Stress %")
    style_legend(
        ax.legend(
            handles=[
                Patch(facecolor=INTRA_COLOR, label="Calm"),
                Patch(facecolor=STRESS, label=f"Stressed (>{STRESS_THRESHOLD:g})"),
            ],
            loc="upper left",
            fontsize=8,
        )
    )

    for row in axes:
        for ax in row:
            ax.set_xlabel("Window end")
            style_axes(ax, labelsize=8, grid=True)
            style_date_axis(ax, n_ticks=6)
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.95)
    return fig


def _bar_width(x: np.ndarray) -> float:
    """Bar width in date units, sized to just under one window step."""
    if len(x) < 2:
        return 20.0
    try:
        return float(abs((x[1] - x[0]).days)) * 0.85 or 1.0
    except AttributeError:
        return 0.85


# -------------------------------------------------------------- Evo: Cov(t)
@dataclass
class TrajectoryPlot:
    """A stress trajectory figure plus the handle that moves its cursor.

    The date slider updates this by moving one artist and redrawing, rather
    than rebuilding the figure -- rebuilding on every slider tick is what makes
    a scrub feel broken.
    """

    figure: Figure
    _axes: object
    _cursor: Line2D
    _x: np.ndarray
    _y: np.ndarray
    _annotation: object

    def set_highlight(self, index: int) -> None:
        """Move the cursor to window ``index`` and redraw."""
        if not len(self._x):
            return
        index = max(0, min(index, len(self._x) - 1))
        self._cursor.set_data([self._x[index]], [self._y[index]])
        self._annotation.set_position((self._x[index], self._y[index]))
        self._annotation.xy = (self._x[index], self._y[index])
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw_idle()


def render_stress_trajectory(
    stress: pl.DataFrame,
    x_column: str,
    y_column: str,
    *,
    width: float = 12,
    height: float = 8,
    dpi: int = 100,
) -> TrajectoryPlot | Figure:
    """Dotted time trajectory through two stress series.

    Points are ordered in time and joined by a dotted path; stressed windows
    are ringed, and the first and last windows are labelled so the direction of
    travel is readable.
    """
    if stress.is_empty():
        return empty_figure("No stress windows produced.", width, height, dpi)
    if x_column == y_column:
        return empty_figure(
            "Pick two different series for the x and y axes.", width, height, dpi
        )

    ordered = stress.sort("date_end")
    x = np.asarray(ordered.get_column(x_column).to_list(), dtype=float)
    y = np.asarray(ordered.get_column(y_column).to_list(), dtype=float)
    dates = ordered.get_column("date_end").to_list()
    pct = np.asarray(ordered.get_column("stress_pct").to_list(), dtype=float)

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)

    ax.plot(x, y, linestyle=":", linewidth=1.0, color=MUTED, zorder=1)
    ax.scatter(x, y, s=26, color=ACCENT, alpha=0.8, zorder=2, label="Window")

    stressed = pct > STRESS_THRESHOLD
    if stressed.any():
        ax.scatter(
            x[stressed],
            y[stressed],
            s=110,
            facecolors="none",
            edgecolors=STRESS,
            linewidths=1.8,
            zorder=3,
            label=f"Stressed (>{STRESS_THRESHOLD:g})",
        )

    for idx, prefix in ((0, "Start"), (len(x) - 1, "End")):
        ax.scatter(x[idx], y[idx], s=70, color=WARN, zorder=4)
        ax.annotate(
            f"{prefix}: {dates[idx]}",
            (x[idx], y[idx]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color=FG,
        )

    (cursor,) = ax.plot(
        [x[0]],
        [y[0]],
        marker="o",
        markersize=13,
        markerfacecolor="none",
        markeredgecolor=POSITIVE,
        markeredgewidth=2.4,
        zorder=5,
        linestyle="none",
        label="Selected date",
    )
    annotation = ax.annotate(
        "",
        xy=(x[0], y[0]),
        textcoords="offset points",
        xytext=(10, -14),
        fontsize=8,
        color=POSITIVE,
    )

    ax.set_xlabel(STRESS_SERIES.get(x_column, x_column))
    ax.set_ylabel(STRESS_SERIES.get(y_column, y_column))
    ax.set_title(
        f"{STRESS_SERIES.get(y_column, y_column)} vs "
        f"{STRESS_SERIES.get(x_column, x_column)} (time trajectory)",
        fontsize=12,
        fontweight="bold",
    )
    style_axes(ax, labelsize=9, grid=True)
    style_legend(ax.legend(loc="best", fontsize=8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.12)

    return TrajectoryPlot(
        figure=fig, _axes=ax, _cursor=cursor, _x=x, _y=y, _annotation=annotation
    )
