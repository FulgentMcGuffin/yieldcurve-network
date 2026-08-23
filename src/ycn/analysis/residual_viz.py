"""Figures for the NS Residuals tab: the metric chart and the coverage plot."""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .plot_theme import (
    ACCENT,
    BG,
    FG,
    GRID,
    MUTED,
    empty_figure,
    style_axes,
    style_legend,
)

# Marker cycle for the shape aesthetic, in decreasing visual distinctness.
SHAPE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")

# Point-area bounds for the size aesthetic, in matplotlib's s= units.
SIZE_MIN = 40.0
SIZE_MAX = 420.0


def _scaled_sizes(values: np.ndarray) -> np.ndarray:
    """Map a numeric column onto a readable point-area range.

    A constant or all-NaN column collapses to the midpoint rather than
    producing invisible or absurd markers.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(values.shape, (SIZE_MIN + SIZE_MAX) / 2)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.full(values.shape, (SIZE_MIN + SIZE_MAX) / 2)
    scaled = (values - lo) / (hi - lo)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return SIZE_MIN + scaled * (SIZE_MAX - SIZE_MIN)


def _shape_key(value) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return str(value)


def render_residual_metrics(
    metrics: pl.DataFrame,
    label_order: list[str],
    *,
    label_column: str = "label",
    label_title: str | None = None,
    y_column: str = "modularity",
    shape_column: str | None = "is_connected",
    fill_column: str | None = "modularity",
    size_column: str | None = "avg_eccentricity",
    width: float = 13,
    height: float = 6.5,
    dpi: int = 100,
) -> Figure:
    """Line + point chart of one metric across the component networks.

    Mirrors the notebook's residual-metric chart: a grey line joins the labels
    in order, and each point carries up to three further aesthetics (shape,
    fill, size). Title, axis label and legend all follow the selection.
    """
    if metrics.is_empty() or y_column not in metrics.columns:
        return empty_figure("No residual-network metrics to plot.", width, height, dpi)

    frame = metrics.filter(pl.col(label_column).is_in(label_order))
    order = {label: i for i, label in enumerate(label_order)}
    frame = frame.sort(pl.col(label_column).replace_strict(order, default=len(order)))
    labels = frame.get_column(label_column).to_list()
    x = np.arange(len(labels), dtype=float)
    y = np.asarray(frame.get_column(y_column).to_list(), dtype=float)

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)

    # The connecting line is deliberately neutral: it conveys ordering along
    # the curve, not any of the encoded variables.
    ax.plot(x, y, color=GRID, linewidth=1.6, zorder=1, alpha=0.9)

    sizes = (
        _scaled_sizes(np.asarray(frame.get_column(size_column).to_list(), dtype=float))
        if size_column and size_column in frame.columns
        else np.full(x.shape, 90.0)
    )

    fill_values = None
    if fill_column and fill_column in frame.columns:
        fill_values = np.asarray(frame.get_column(fill_column).to_list(), dtype=float)

    shape_values = (
        [_shape_key(v) for v in frame.get_column(shape_column).to_list()]
        if shape_column and shape_column in frame.columns
        else None
    )

    scatter = None
    if shape_values is None:
        scatter = ax.scatter(
            x,
            y,
            s=sizes,
            c=fill_values if fill_values is not None else ACCENT,
            cmap="viridis" if fill_values is not None else None,
            edgecolors=FG,
            linewidths=0.6,
            zorder=3,
        )
    else:
        groups = sorted(set(shape_values))
        marker_of = {
            key: SHAPE_MARKERS[i % len(SHAPE_MARKERS)] for i, key in enumerate(groups)
        }
        vmin = float(np.nanmin(fill_values)) if fill_values is not None else None
        vmax = float(np.nanmax(fill_values)) if fill_values is not None else None
        for key in groups:
            mask = np.array([v == key for v in shape_values])
            scatter = ax.scatter(
                x[mask],
                y[mask],
                s=sizes[mask],
                c=fill_values[mask] if fill_values is not None else ACCENT,
                cmap="viridis" if fill_values is not None else None,
                vmin=vmin,
                vmax=vmax,
                marker=marker_of[key],
                edgecolors=FG,
                linewidths=0.6,
                zorder=3,
            )
        handles = [
            Line2D(
                [],
                [],
                marker=marker_of[key],
                linestyle="none",
                color=MUTED,
                markerfacecolor=MUTED,
                markeredgecolor=FG,
                markersize=8,
                label=f"{key}",
            )
            for key in groups
        ]
        style_legend(
            ax.legend(handles=handles, title=shape_column, loc="best", fontsize=8)
        )

    if fill_values is not None and scatter is not None:
        bar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.04)
        bar.set_label(fill_column, color=MUTED, fontsize=9)
        bar.ax.tick_params(colors=FG, labelsize=8)
        bar.outline.set_edgecolor(GRID)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    axis_title = label_title or label_column
    ax.set_xlabel(axis_title)
    ax.set_ylabel(y_column)

    encoded = [
        f"fill={fill_column}" if fill_column else "",
        f"shape={shape_column}" if shape_column else "",
        f"size={size_column}" if size_column else "",
    ]
    subtitle = ", ".join(part for part in encoded if part)
    ax.set_title(
        f"{y_column} × {axis_title}" + (f"  ({subtitle})" if subtitle else ""),
        fontsize=12,
        fontweight="bold",
    )
    style_axes(ax, labelsize=9, grid=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.2)
    return fig


def render_coverage(
    coverage: pl.DataFrame,
    issuer_column: str,
    *,
    width: float = 11,
    height: float = 6,
    dpi: int = 100,
) -> Figure:
    """Horizontal span per issuer, from first to last observation.

    Ragged coverage is the norm in curve panels, and this is the quickest way
    to see which issuers only cover part of the window before reading anything
    into a network that excludes them.
    """
    if coverage.is_empty():
        return empty_figure("No coverage to show.", width, height, dpi)

    frame = coverage.sort("coverage_days")
    issuers = frame.get_column(issuer_column).to_list()
    starts = frame.get_column("min_date").to_list()
    ends = frame.get_column("max_date").to_list()
    y = np.arange(len(issuers), dtype=float)

    fig = Figure(figsize=(width, max(height, 0.32 * len(issuers) + 2)), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)

    for row, (lo, hi) in enumerate(zip(starts, ends)):
        ax.plot(
            [lo, hi], [row, row], color=ACCENT, linewidth=2.4, solid_capstyle="butt"
        )
        ax.plot([lo, lo], [row - 0.2, row + 0.2], color=ACCENT, linewidth=2.0)
        ax.plot([hi, hi], [row - 0.2, row + 0.2], color=ACCENT, linewidth=2.0)

    ax.set_yticks(y)
    ax.set_yticklabels(issuers)
    ax.set_ylim(-0.7, len(issuers) - 0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel(issuer_column)
    ax.set_title("Coverage by Issuer", fontsize=12, fontweight="bold")
    style_axes(ax, labelsize=9, grid=True)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.14)
    return fig
