"""Degree-distribution histogram styled for the dark GUI."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .cursor_util import RedrawOnChangeMixin

# Figures here are built directly via the Figure API (not pyplot.subplots) so
# rendering can safely run on a background QThread -- pyplot's state machine is
# tied to the active GUI backend and warns/fails when touched off the main thread.

# Match GUI dark theme (kept local so analysis does not import gui).
BG_APP = "#0f172a"
TEXT = "#e5e7eb"
TEXT_MUTED = "#94a3b8"
TEXT_LOG = "#7dd3fc"
LABEL_COLOR = "#ffffff"
LABEL_SIZE = 8.5
MAX_NODE_LABELS = 5


class _HistogramCursor(RedrawOnChangeMixin):
    """Interactive cursor for histogram hover tooltips."""

    def __init__(self, ax, degrees, bin_edges, canvas):
        """degrees: dict mapping node -> degree"""
        self.ax = ax
        self.degrees = degrees
        self.bin_edges = bin_edges
        self.canvas = canvas
        self.annot = None
        self.cid = canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        """Handle mouse motion events."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            self._redraw_if_changed(None)
            return

        # Find bar at this x position
        x = event.xdata
        best_patch = None
        best_height = 0

        for patch in self.ax.patches:
            left = float(patch.get_x())
            width = float(patch.get_width())
            if left <= x <= left + width:
                height = float(patch.get_height())
                if height > best_height:
                    best_height = height
                    best_patch = patch

        if best_patch is None:
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            return

        # Get bar info
        left = float(best_patch.get_x())
        width = float(best_patch.get_width())
        height = float(best_patch.get_height())
        center_x = left + width / 2

        # Find nodes in this bin
        bin_idx = np.searchsorted(self.bin_edges, center_x) - 1
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= len(self.bin_edges) - 1:
            bin_idx = len(self.bin_edges) - 2

        target_degree = int(center_x)
        nodes_here = [
            str(n) for n, d in self.degrees.items() if int(d) == target_degree
        ]

        # Create annotation only once
        if self.annot is None:
            self.annot = self.ax.annotate(
                "",
                xy=(center_x, height),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.5", fc="#1a1c24", ec="#e2e8f0", alpha=0.9
                ),
                color="#e2e8f0",
                fontsize=9,
                zorder=100,
            )

        # Update annotation
        node_list = ", ".join(nodes_here[:3])
        if len(nodes_here) > 3:
            node_list += f" +{len(nodes_here) - 3}"

        text = f"Degree: {target_degree}\nCount: {height:.1f}\nNodes: {node_list}"
        self.annot.set_text(text)
        self.annot.xy = (center_x, height)
        self.annot.set_visible(True)

        # Moving within the same bar shows the same tooltip -- no repaint needed.
        self._redraw_if_changed((text, center_x, height))


def histogram_title(value_column: str) -> str:
    """Build a title reflecting the series column."""
    return f"Degree Histogram — {value_column}"


def render_degree_histogram(
    graph: nx.Graph,
    title: str,
    current_threshold: float | None = None,
    *,
    bins: int = 15,
) -> Figure:
    """Return a matplotlib Figure of the degree histogram.

    Args:
        graph: NetworkX graph with degree data.
        title: Title for the histogram.
        current_threshold: Current independence threshold (for reference, not used in rendering).
        bins: Number of histogram bins.

    Returns:
        Matplotlib Figure object.
    """
    degrees = dict(graph.degree())
    deg_list = list(degrees.values())
    if not deg_list:
        return _empty_figure("No nodes to plot.")

    mean_deg = float(np.mean(deg_list))
    lo, hi = int(min(deg_list)), int(max(deg_list))
    # Explicit edges so node labels can sit on the same bar centers as histplot.
    _counts, bin_edges = np.histogram(deg_list, bins=bins)

    fig = Figure(figsize=(9, 6.4), dpi=100)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG_APP)
    ax.set_facecolor(BG_APP)

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

    # Spread ticks across the observed degree range (avoids the cramped
    # notebook-style fixed range that piles labels on the left).
    ax.set_xlim(max(0, lo - 1), hi + 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.minorticks_off()

    ax.set_title(title, color=TEXT, fontsize=13, pad=10)
    ax.set_xlabel("Number of Connections", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Density", color=TEXT_MUTED, fontsize=10)
    ax.tick_params(axis="both", which="major", colors=TEXT_MUTED, labelsize=10)
    # Major ticks only — avoids the dense vertical-line clutter from bad xticks.
    ax.grid(True, axis="both", which="major", color="#334155", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#475569")

    _annotate_extreme_nodes(ax, degrees, lo, hi, bin_edges)

    legend = ax.legend(loc="best", fontsize=11, frameon=True)
    legend.get_frame().set_facecolor("#1a1c24")
    legend.get_frame().set_edgecolor("#7dd3fc")
    for text in legend.get_texts():
        text.set_color(TEXT_LOG)

    fig.tight_layout()

    # Store data for cursor attachment later (after FigureCanvas is created)
    fig._histogram_cursor_data = (ax, degrees, bin_edges)

    return fig


def render_degree_histogram_with_dynamic_threshold(
    measure_df,
    title: str,
    current_threshold: float = 0.33,
    *,
    bins: int = 15,
) -> Figure:
    """Return a histogram figure with measure_df attached for dynamic threshold adjustment.

    This is called from the pipeline to create a figure that can be dynamically updated
    by the GUI with different thresholds.
    """
    from ycn.analysis.network import build_corr_nx

    # Build the initial graph with current threshold
    graph = build_corr_nx(measure_df, independent_threshold=current_threshold)

    # Render the histogram
    fig = render_degree_histogram(graph, title, current_threshold, bins=bins)

    # Attach measure_df for dynamic threshold adjustment
    fig._measure_df_stored = measure_df
    fig._current_threshold = current_threshold

    return fig


def _nodes_with_degree(degrees: dict[object, int], target: int) -> list[str]:
    names = sorted(str(n) for n, d in degrees.items() if int(d) == target)
    return names[:MAX_NODE_LABELS]


def _bin_center(value: float, bin_edges: np.ndarray) -> float:
    """Return the center of the hist bin that contains ``value``."""
    edges = np.asarray(bin_edges, dtype=float)
    if value >= edges[-1]:
        idx = len(edges) - 2
    else:
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        idx = int(np.clip(idx, 0, len(edges) - 2))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _bar_height_at(ax: Axes, x_center: float) -> float:
    """Height of the histogram rectangle whose bin contains ``x_center``."""
    best_h = 0.0
    best_dist = float("inf")
    for patch in ax.patches:
        left = float(patch.get_x())
        width = float(patch.get_width())
        mid = left + width / 2.0
        if left <= x_center <= left + width:
            dist = abs(mid - x_center)
            if dist < best_dist:
                best_dist = dist
                best_h = float(patch.get_height())
    return best_h


def _annotate_stack(
    ax: Axes,
    x: float,
    names: list[str],
    *,
    y0: float,
    dy: float,
) -> None:
    from matplotlib import patheffects

    stroke = [patheffects.withStroke(linewidth=2.5, foreground=BG_APP)]
    for i, name in enumerate(names):
        ax.text(
            x,
            y0 + i * dy,
            name,
            color=LABEL_COLOR,
            fontsize=LABEL_SIZE,
            fontweight="normal",
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=6,
            clip_on=False,
            path_effects=stroke,
        )


def _unique_label_degrees(lo: int, hi: int, mode_deg: int) -> list[int]:
    """Degrees to label: low / high / mode, dropping duplicates (1–3 sets)."""
    ordered: list[int] = []
    for deg in (lo, hi, mode_deg):
        if deg not in ordered:
            ordered.append(deg)
    return ordered


def _annotate_extreme_nodes(
    ax: Axes,
    degrees: dict[object, int],
    lo: int,
    hi: int,
    bin_edges: np.ndarray,
) -> None:
    """Label nodes at min / max / mode degree (deduped when those coincide)."""
    counts = Counter(int(d) for d in degrees.values())
    mode_deg, _ = counts.most_common(1)[0]
    label_degrees = _unique_label_degrees(lo, hi, mode_deg)

    ymax = float(ax.get_ylim()[1])
    dy = max(ymax * 0.04, 0.35)

    # Headroom for stacks sitting above each bar.
    max_stack = max(len(_nodes_with_degree(degrees, d)) for d in label_degrees)
    ax.set_ylim(0, ymax + dy * (max_stack + 1.5))

    for deg in label_degrees:
        names = _nodes_with_degree(degrees, deg)
        if not names:
            continue
        x = _bin_center(float(deg), bin_edges)
        bar_h = _bar_height_at(ax, x)
        y0 = bar_h + dy * 0.35
        _annotate_stack(ax, x, names, y0=y0, dy=dy)


def _empty_figure(message: str) -> Figure:
    """Return a placeholder figure with a message."""
    fig = Figure(figsize=(9, 6), dpi=100)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG_APP)
    ax.set_facecolor(BG_APP)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=TEXT_MUTED, fontsize=14)
    return fig
