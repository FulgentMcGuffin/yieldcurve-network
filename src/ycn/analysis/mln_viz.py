"""Matplotlib figures for the multi-layer network tabs.

Raw matplotlib rather than plotnine, deliberately. The Metrics tab is a
composite -- two edge-type charts across the top third, one wide centrality
heatmap underneath -- and plotnine cannot draw into a pre-existing ``Axes``:
``ggplot.draw()`` always returns its own new ``Figure``, with no public API to
target a ``GridSpec`` slot. Rewriting these with plotnine to match the notebooks
would therefore break the composite layout.

As in ``evolution_viz``, figures are built via the ``Figure`` API rather than
``pyplot`` so they can safely be produced on a background QThread.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from .evolution_viz import CATEGORICAL_10
from .yield_curve import sort_terms

BG = "#0f172a"
FG = "#e2e8f0"
MUTED = "#94a3b8"
INTRA_COLOR = "#0ea5e9"
INTER_COLOR = "#a78bfa"
GRID = "#334155"

# Above this many nodes the y-axis labels are thinned -- see _set_node_ticks.
MAX_NODE_TICKS = 60


def _community_palette(n: int) -> list:
    """``n`` visually distinct colours for a categorical community legend.

    ``CATEGORICAL_10`` is curated and covers the common case (it also matches
    the notebook), but ``MLNConfig.max_communities`` allows up to 15, and a
    stale session or an unbounded aligner could hand back more still -- cycling
    ``CATEGORICAL_10`` past 10 communities repeats colours, which reads as
    *fewer* communities than there are. Matplotlib's ``tab20``/``tab20b`` give
    40 combined qualitative colours before this falls back to sampling a
    continuous colormap, which stays distinct-*ish* rather than repeating.
    """
    if n <= len(CATEGORICAL_10):
        return CATEGORICAL_10[:n]
    extended = list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
    if n <= len(extended):
        return extended[:n]
    cmap = colormaps["gist_ncar"]
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _empty(message: str, width: int, height: int, dpi: int) -> Figure:
    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#1e293b")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14,
        color=MUTED,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _style_axes(ax, *, labelsize: int = 8) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=labelsize)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def render_mln_metrics(
    edge_df: pl.DataFrame,
    centrality_df: pl.DataFrame,
    *,
    layer_label: str = "layer",
    node_label: str = "node",
    centrality_name: str = "eigenvector",
    width: int = 13,
    height: int = 9,
    dpi: int = 100,
) -> Figure:
    """Edge-type composition (top ~30%) plus a node x layer centrality heatmap.

    Args:
        edge_df: From ``mln.layer_edge_metrics`` -- one row per layer.
        centrality_df: From ``mln.layer_centrality_matrix`` -- (node, layer, value).
        layer_label: Display name of the layer column (e.g. ``"EqIndex"``).
        node_label: Display name of the node column (e.g. ``"Stock"``).
        centrality_name: Centrality shown, for titling.
    """
    if edge_df.is_empty() and centrality_df.is_empty():
        return _empty("No multi-layer network data available", width, height, dpi)

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    # 30 / 70 split: the two edge-type charts share the top row.
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[3, 7],
        hspace=0.42,
        wspace=0.22,
        left=0.08,
        right=0.93,  # leaves room for the heatmap colorbar and its label
        top=0.92,
        bottom=0.12,
    )

    _draw_edge_counts(fig.add_subplot(gs[0, 0]), edge_df, layer_label)
    _draw_edge_composition(fig.add_subplot(gs[0, 1]), edge_df, layer_label)
    _draw_centrality_heatmap(
        fig.add_subplot(gs[1, :]),
        fig,
        centrality_df,
        layer_label=layer_label,
        node_label=node_label,
        centrality_name=centrality_name,
    )
    return fig


def _draw_edge_counts(ax, edge_df: pl.DataFrame, layer_label: str) -> None:
    """Grouped bars: intra vs inter edge counts per layer.

    Bars rather than the notebooks' lines/points -- with no evolution the x axis
    is a categorical list of layer values, not an ordered window index.
    """
    _style_axes(ax)
    if edge_df.is_empty():
        ax.text(
            0.5,
            0.5,
            "no edges",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    layers = edge_df.get_column("layer").to_list()
    intra = edge_df.get_column("intra_edges").to_list()
    inter = edge_df.get_column("inter_edges").to_list()
    x = np.arange(len(layers))
    w = 0.4
    ax.bar(x - w / 2, intra, w, label="Intra-layer", color=INTRA_COLOR)
    ax.bar(x + w / 2, inter, w, label="Inter-layer", color=INTER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_xlabel(layer_label)
    ax.set_ylabel("Edge count")
    ax.set_title("Edge Counts by Type", fontsize=11, weight="bold")
    ax.grid(True, axis="y", alpha=0.2, color=GRID)
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=8, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(MUTED)


def _draw_edge_composition(ax, edge_df: pl.DataFrame, layer_label: str) -> None:
    """100% stacked bars: intra/inter share per layer."""
    _style_axes(ax)
    if edge_df.is_empty():
        ax.text(
            0.5,
            0.5,
            "no edges",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    layers = edge_df.get_column("layer").to_list()
    pct_intra = edge_df.get_column("pct_intra").to_list()
    pct_inter = edge_df.get_column("pct_inter").to_list()
    x = np.arange(len(layers))
    ax.bar(x, pct_intra, 0.65, label="Intra-layer", color=INTRA_COLOR)
    ax.bar(x, pct_inter, 0.65, bottom=pct_intra, label="Inter-layer", color=INTER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_xlabel(layer_label)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Edge Type Composition", fontsize=11, weight="bold")
    # No legend here: the colours are identical to the counts chart on the left,
    # and a legend inside a full-height stacked bar has nowhere clear to sit.
    # An inter-layer edge touches two layers, so it is counted under both.
    ax.text(
        0.0,
        -0.52,
        "inter-layer edges are counted under both layers they join",
        transform=ax.transAxes,
        fontsize=7,
        color=MUTED,
    )


def _draw_centrality_heatmap(
    ax,
    fig: Figure,
    centrality_df: pl.DataFrame,
    *,
    layer_label: str,
    node_label: str,
    centrality_name: str,
) -> None:
    """node x layer heatmap of the selected centrality."""
    _style_axes(ax, labelsize=7)
    if centrality_df.is_empty():
        ax.text(
            0.5,
            0.5,
            "no centrality data",
            ha="center",
            va="center",
            color=MUTED,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    matrix, nodes, layers = _pivot_matrix(centrality_df, "value")
    # Mask (node, layer) pairs where the node is absent from that layer, so they
    # read as "not present" rather than as a genuine low centrality value.
    cmap = colormaps["viridis"].with_extremes(bad="#1e293b")
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    _set_node_ticks(ax, nodes)
    ax.set_xlabel(layer_label)
    ax.set_ylabel(node_label)
    ax.set_title(
        f"{node_label} x {layer_label} {centrality_name.title()} Centrality",
        fontsize=12,
        weight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(colors=FG, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    cbar.set_label(centrality_name.title(), color=MUTED, fontsize=8)


def _set_node_ticks(ax, nodes: list[str]) -> None:
    """Label the node axis, thinning the labels when there are many nodes.

    Drawing one text object per node dominates ``canvas.draw()`` on the GUI
    thread (measured: ~2.3s for 500 nodes), and at that density the labels are
    too small to read anyway. Showing an evenly-spaced subset keeps the axis
    orientable and the redraw fast.
    """
    n = len(nodes)
    if n <= MAX_NODE_TICKS:
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(nodes, fontsize=max(4, min(7, int(360 / max(n, 1)))))
        return
    step = int(np.ceil(n / MAX_NODE_TICKS))
    idx = np.arange(0, n, step)
    ax.set_yticks(idx)
    ax.set_yticklabels([nodes[i] for i in idx], fontsize=6)


def _pivot_matrix(
    df: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """Long (node, layer, value) -> dense matrix with ordered node/layer axes.

    Both axes use ``sort_terms``: either can carry maturity labels depending on
    the network type, and a plain sort would lay them out 0.5Y, 10Y, 1Y, 20Y,
    which misreads as a yield curve. Non-term labels (issuers) fall back to
    alphabetical, so nothing else changes.
    """
    nodes = sort_terms(df.get_column("node").unique().to_list())
    layers = sort_terms(df.get_column("layer").unique().to_list())
    node_idx = {n: i for i, n in enumerate(nodes)}
    layer_idx = {v: i for i, v in enumerate(layers)}
    matrix = np.full((len(nodes), len(layers)), np.nan)
    for node, layer, value in df.select(["node", "layer", value_column]).iter_rows():
        matrix[node_idx[node], layer_idx[layer]] = value
    return matrix, nodes, layers


def render_mln_communities(
    community_df: pl.DataFrame,
    *,
    layer_label: str = "layer",
    node_label: str = "node",
    width: int = 13,
    height: int = 9,
    dpi: int = 100,
) -> Figure:
    """node x layer tile plot of cross-layer-aligned community ids.

    Community ids have been Jaccard-aligned across layers, so one colour means
    the same community everywhere -- that comparability is the whole point of
    the alignment step, and is why a categorical (not continuous) colour scale
    is used.
    """
    if community_df.is_empty():
        return _empty("No community data available", width, height, dpi)

    matrix, nodes, layers = _pivot_matrix(community_df, "community")

    present = sorted({int(v) for v in matrix[~np.isnan(matrix)]})
    if not present:
        return _empty("No community data available", width, height, dpi)
    colors = _community_palette(len(present))
    remap = {cid: i for i, cid in enumerate(present)}
    indexed = np.full(matrix.shape, np.nan)
    for cid, i in remap.items():
        indexed[matrix == cid] = i

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    _style_axes(ax, labelsize=7)

    cmap = ListedColormap(colors)
    cmap.set_bad(color="#1e293b")  # node absent from that layer
    ax.imshow(
        np.ma.masked_invalid(indexed),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=-0.5,
        vmax=len(present) - 0.5,
    )
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")
    _set_node_ticks(ax, nodes)
    ax.set_xlabel(layer_label)
    ax.set_ylabel(node_label)
    ax.set_title(f"{node_label} x {layer_label} Community", fontsize=13, weight="bold")

    handles = [Patch(facecolor=colors[i], label=f"C{cid}") for cid, i in remap.items()]
    legend = ax.legend(
        handles=handles,
        title="Community",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=8,
        framealpha=0.0,
    )
    legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(MUTED)
    fig.subplots_adjust(left=0.1, right=0.88, top=0.93, bottom=0.12)
    return fig
