"""Visualization for temporal network evolution."""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from .cursor_util import RedrawOnChangeMixin

# Figures here are built directly via the Figure API (not pyplot.subplots) so
# rendering can safely run on a background QThread -- pyplot's state machine is
# tied to the active GUI backend and warns/fails when touched off the main thread.

# Shared qualitative palette (matches the notebook's CATEGORICAL_6, extended to 10)
# for the centrality trajectories lines and the community heatmap categories.
CATEGORICAL_10 = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]


class _HeatmapCursor(RedrawOnChangeMixin):
    """Interactive cursor for heatmap hover tooltips."""

    def __init__(self, ax, matrix, nodes, window_ends_sorted, canvas):
        self.ax = ax
        self.matrix = matrix
        self.nodes = nodes
        self.window_ends_sorted = window_ends_sorted
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

        # Get nearest grid cell
        x_idx = int(np.round(event.xdata))
        y_idx = int(np.round(event.ydata))

        # Bounds check
        if not (
            0 <= x_idx < len(self.window_ends_sorted) and 0 <= y_idx < len(self.nodes)
        ):
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            return

        value = self.matrix[y_idx, x_idx]
        node = self.nodes[y_idx]
        date = self.window_ends_sorted[x_idx]

        # Create annotation only once
        if self.annot is None:
            self.annot = self.ax.annotate(
                "",
                xy=(0, 0),
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
        text = f"{node}\n{date}\nDegree: {value:.2f}"
        self.annot.set_text(text)
        self.annot.xy = (event.xdata, event.ydata)
        self.annot.set_visible(True)

        # Moving within the same cell shows the same tooltip -- skip the repaint.
        self._redraw_if_changed(text)


class _LineCursor(RedrawOnChangeMixin):
    """Interactive cursor for line-plot hover tooltips and click-to-highlight."""

    def __init__(self, ax, line_data, canvas, line_objects=None, original_styles=None):
        """
        Args:
            ax: The Axes containing the lines.
            line_data: dict mapping node -> (x_numeric, y_vals) arrays.
            canvas: The Qt FigureCanvas to bind events on and redraw.
            line_objects: dict mapping node -> Line2D (enables click-to-highlight).
            original_styles: dict mapping node -> dict(linewidth, linestyle, alpha, zorder),
                the style to restore when a highlighted series is deselected.
        """
        self.ax = ax
        self.line_data = line_data
        self.canvas = canvas
        self.line_objects = line_objects or {}
        self.original_styles = original_styles or {}
        self.selected_node = None
        self.annot = None
        self.cid = canvas.mpl_connect("motion_notify_event", self.on_move)
        self.cid_click = canvas.mpl_connect("button_press_event", self.on_click)

    def _nearest_node(self, xdata, ydata):
        """Find the node whose line has the nearest point to (xdata, ydata).

        Returns (node, normalized_distance, x_numeric_at_nearest) or (None, inf, None).
        """
        min_dist = float("inf")
        best_node = None
        best_x_numeric = None

        for node, xy_pairs in self.line_data.items():
            if len(xy_pairs) == 0:
                continue
            x_vals, y_vals = xy_pairs
            # Calculate normalized distance (scale x and y to same units)
            x_range = np.max(x_vals) - np.min(x_vals)
            y_range = np.max(y_vals) - np.min(y_vals)
            if x_range == 0 or y_range == 0:
                continue

            x_norm = (x_vals - np.min(x_vals)) / x_range if x_range > 0 else x_vals
            y_norm = (y_vals - np.min(y_vals)) / y_range if y_range > 0 else y_vals
            event_x_norm = (xdata - np.min(x_vals)) / x_range if x_range > 0 else 0
            event_y_norm = (ydata - np.min(y_vals)) / y_range if y_range > 0 else 0

            dist = np.sqrt((x_norm - event_x_norm) ** 2 + (y_norm - event_y_norm) ** 2)
            nearest_idx = np.argmin(dist)
            if dist[nearest_idx] < min_dist:
                min_dist = dist[nearest_idx]
                best_node = node
                best_x_numeric = x_vals[nearest_idx]

        return best_node, min_dist, best_x_numeric

    def _highlight(self, node) -> None:
        """Make a series prominent: thicker, solid, fully opaque, drawn on top."""
        line = self.line_objects.get(node)
        if line is None:
            return
        line.set_linewidth(3.0)
        line.set_linestyle("-")
        line.set_alpha(1.0)
        line.set_zorder(10)

    def _unhighlight(self, node) -> None:
        """Restore a series to its original plotted style."""
        line = self.line_objects.get(node)
        style = self.original_styles.get(node)
        if line is None or style is None:
            return
        line.set_linewidth(style["linewidth"])
        line.set_linestyle(style["linestyle"])
        line.set_alpha(style["alpha"])
        line.set_zorder(style["zorder"])

    def on_click(self, event):
        """Toggle highlight of the series nearest the clicked point."""
        if not self.line_objects:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        node, min_dist, _ = self._nearest_node(event.xdata, event.ydata)
        if node is None or min_dist > 0.15:
            return

        if self.selected_node == node:
            # Same series clicked again -> deselect, back to original state.
            self._unhighlight(node)
            self.selected_node = None
        else:
            # Different series (or none) selected -> switch highlight.
            if self.selected_node is not None:
                self._unhighlight(self.selected_node)
            self._highlight(node)
            self.selected_node = node

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def on_move(self, event):
        """Handle mouse motion events."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            self._redraw_if_changed(None)
            return

        best_node, min_dist, best_x_numeric = self._nearest_node(
            event.xdata, event.ydata
        )
        best_point = (event.xdata, event.ydata) if best_node is not None else None

        # Only show if close enough
        if min_dist > 0.15 or best_point is None:
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            return

        # Create annotation only once
        if self.annot is None:
            self.annot = self.ax.annotate(
                "",
                xy=best_point,
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
        x_val, y_val = best_point
        # Convert numeric date back to string for display
        try:
            from matplotlib.dates import num2date

            date_str = num2date(best_x_numeric).strftime("%Y-%m-%d")
        except Exception:
            date_str = str(x_val)

        text = f"{best_node}\nDate: {date_str}\nCentrality: {y_val:.4f}"
        self.annot.set_text(text)
        self.annot.xy = best_point
        self.annot.set_visible(True)

        # Hovering near the same point shows the same tooltip -- skip the repaint.
        self._redraw_if_changed((text, best_point))


class _MultiPanelCursor(RedrawOnChangeMixin):
    """Interactive cursor for a grid of line-plot subplots (nearest-point per panel)."""

    def __init__(self, panels: list[tuple], canvas):
        """panels: list of (ax, x_numeric, y_vals, metric_name) tuples, one per subplot."""
        self.panels = panels
        self.canvas = canvas
        self.annot_by_ax = {}
        self.cid = canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        """Handle mouse motion events."""
        # Collected across all panels so the whole row repaints at most once.
        shown: list[tuple] = []
        for ax, x_numeric, y_vals, metric_name in self.panels:
            annot = self.annot_by_ax.get(ax)
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                if annot and annot.get_visible():
                    annot.set_visible(False)
                continue

            if len(x_numeric) == 0:
                continue

            dist = np.abs(x_numeric - event.xdata)
            nearest_idx = int(np.argmin(dist))
            x_range = float(np.max(x_numeric) - np.min(x_numeric)) or 1.0
            if dist[nearest_idx] / x_range > 0.03:
                if annot and annot.get_visible():
                    annot.set_visible(False)
                continue

            x_val = x_numeric[nearest_idx]
            y_val = y_vals[nearest_idx]

            if annot is None:
                annot = ax.annotate(
                    "",
                    xy=(x_val, y_val),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.5", fc="#1a1c24", ec="#e2e8f0", alpha=0.9
                    ),
                    color="#e2e8f0",
                    fontsize=8,
                    zorder=100,
                )
                self.annot_by_ax[ax] = annot

            try:
                from matplotlib.dates import num2date

                date_str = num2date(x_val).strftime("%Y-%m-%d")
            except Exception:
                date_str = f"{x_val:.2f}"

            annot.set_text(f"{metric_name}\n{date_str}\nValue: {y_val:.3f}")
            annot.xy = (x_val, y_val)
            annot.set_visible(True)
            shown.append((metric_name, x_val, y_val))

        # One repaint for the whole panel row, and only when something changed.
        self._redraw_if_changed(tuple(shown))


class _CommunityHeatmapCursor(RedrawOnChangeMixin):
    """Interactive cursor for the node x window community-membership heatmap."""

    def __init__(self, ax, matrix, nodes, window_ends, canvas):
        self.ax = ax
        self.matrix = matrix
        self.nodes = nodes
        self.window_ends = window_ends
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

        x_idx = int(np.round(event.xdata))
        y_idx = int(np.round(event.ydata))

        if not (0 <= x_idx < len(self.window_ends) and 0 <= y_idx < len(self.nodes)):
            if self.annot and self.annot.get_visible():
                self.annot.set_visible(False)
            return

        community = int(self.matrix[y_idx, x_idx])
        node = self.nodes[y_idx]
        window_end = self.window_ends[x_idx]

        if self.annot is None:
            self.annot = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.5", fc="#1a1c24", ec="#e2e8f0", alpha=0.9
                ),
                color="#e2e8f0",
                fontsize=9,
                zorder=100,
            )

        text = f"{node}\nWindow: {window_end}\nCommunity: {community}"
        self.annot.set_text(text)
        self.annot.xy = (event.xdata, event.ydata)
        self.annot.set_visible(True)

        # Moving within the same cell shows the same tooltip -- skip the repaint.
        self._redraw_if_changed(text)


def render_weighted_degree_heatmap(
    node_metrics: pl.DataFrame,
    *,
    width: int = 12,
    height: int = 8,
    dpi: int = 100,
) -> Figure:
    """Render weighted-degree heatmap (Plot b) as matplotlib Figure.

    Args:
        node_metrics: Long-format DataFrame with columns
            (window_end, node, metric, value).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        Matplotlib Figure object.
    """
    # Filter to weighted_degree metric
    heatmap_df = node_metrics.filter(pl.col("metric") == "weighted_degree")

    if heatmap_df.is_empty():
        return _empty_plot(
            "No weighted-degree data available", width=width, height=height, dpi=dpi
        )

    # Pivot to matrix form
    pivot = heatmap_df.pivot(
        on="window_end", index="node", values="value", aggregate_function="first"
    )
    pivot = pivot.sort("node")  # Sort nodes for consistent axis

    nodes = pivot.get_column("node").to_list()
    window_ends = [col for col in pivot.columns if col != "node"]
    window_ends_sorted = sorted(window_ends)

    # Build matrix
    matrix = np.array(
        [
            pivot.filter(pl.col("node") == n).select(window_ends_sorted).to_numpy()[0]
            for n in nodes
        ]
    )

    # Render
    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Heatmap with dark-friendly colormap (grayscale works better on dark)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")

    # Show x-axis tick labels for legibility (show ~16 labels)
    tick_interval = max(1, len(window_ends_sorted) // 16)  # Show ~16 labels
    tick_indices = np.arange(0, len(window_ends_sorted), tick_interval)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [str(window_ends_sorted[i]) for i in tick_indices],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=6)

    ax.set_xlabel("Window end date", color="#e2e8f0", fontsize=9)
    ax.set_ylabel("Stock", color="#e2e8f0", fontsize=9)
    ax.set_title("Node weighted-degree evolution", color="#e2e8f0", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Weighted degree", rotation=270, labelpad=20, color="#e2e8f0", fontsize=8
    )
    cbar.ax.tick_params(colors="#e2e8f0", labelsize=7)

    # Tick colors
    ax.tick_params(colors="#e2e8f0", labelsize=7)

    # Dark spine colors for better blend
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()

    # Store data for cursor attachment later (after FigureCanvas is created)
    fig._heatmap_cursor_data = (ax, matrix, nodes, window_ends_sorted)

    return fig


def render_centrality_trajectories(
    node_metrics: pl.DataFrame,
    centrality_metric: str = "eigenvector",
    n_nodes: int | None = None,
    *,
    width: int = 12,
    height: int = 10,
    dpi: int = 100,
) -> Figure:
    """Render centrality trajectories as two vertically stacked plots (matplotlib Figure).

    Top plot: nodes with highest centrality variability (by std).
    Bottom plot: nodes with lowest centrality variability (by std).

    Args:
        node_metrics: Long-format DataFrame with columns
            (window_end, node, metric, value).
        centrality_metric: Which centrality to plot (default "eigenvector").
        n_nodes: Number of nodes per plot (default 10, capped at total_unique_nodes / 2).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        Matplotlib Figure object.
    """
    # Default to 10 nodes if not specified
    if n_nodes is None:
        n_nodes = 10

    # Filter to centrality metric
    cent_df = node_metrics.filter(pl.col("metric") == centrality_metric)

    if cent_df.is_empty():
        return _empty_plot(
            f"No {centrality_metric} centrality data available",
            width=width,
            height=height,
            dpi=dpi,
        )

    # Get all unique nodes and cap n_nodes at total_nodes / 2
    all_nodes = cent_df.get_column("node").unique().to_list()
    max_n_nodes = max(1, len(all_nodes) // 2)
    plot_n_nodes = min(n_nodes, max_n_nodes, 20)

    # Find top and bottom variable nodes
    node_stats = (
        cent_df.group_by("node")
        .agg(pl.col("value").std().alias("std"))
        .sort("std", descending=True)
    )
    top_nodes = node_stats.head(plot_n_nodes).get_column("node").to_list()
    bottom_nodes = node_stats.tail(plot_n_nodes).get_column("node").to_list()

    # Render
    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")

    # Create two subplots vertically stacked
    ax_top = fig.add_subplot(211)
    ax_bottom = fig.add_subplot(212)

    for ax in [ax_top, ax_bottom]:
        ax.set_facecolor("#0f172a")

    # Color palette (slightly muted for less prominence)
    colors = CATEGORICAL_10

    # Line styles and markers for visual distinction
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "+", "x"]

    # Sort by window_end for consistent plotting
    pandas_df = cent_df.to_pandas().sort_values("window_end")

    import matplotlib.dates as mdates

    # Helper function to plot nodes on an axis
    def plot_nodes_on_axis(ax, nodes_to_plot, plot_title):
        line_data = {}
        line_objects = {}
        original_styles = {}

        for i, node in enumerate(nodes_to_plot):
            node_data = pandas_df[pandas_df["node"] == node]
            if node_data.empty:
                continue

            # Convert dates to numeric matplotlib dates for cursor distance calculation
            x_numeric = mdates.date2num(node_data["window_end"].values)
            y_vals = node_data["value"].values
            line_data[node] = (x_numeric, y_vals)

            node_linewidth = 1.2  # Thinner lines for less prominence
            node_linestyle = line_styles[i % len(line_styles)]  # Vary line style
            node_alpha = 0.8  # Slightly transparent
            (line,) = ax.plot(
                node_data["window_end"],
                node_data["value"],
                label=node,
                color=colors[i % len(colors)],
                linewidth=node_linewidth,
                linestyle=node_linestyle,
                marker=markers[i % len(markers)],  # Vary marker shape
                markersize=3,  # Smaller markers
                alpha=node_alpha,
            )
            line_objects[node] = line
            original_styles[node] = {
                "linewidth": node_linewidth,
                "linestyle": node_linestyle,
                "alpha": node_alpha,
                "zorder": line.get_zorder(),
            }

        ax.set_ylabel(
            f"{centrality_metric.title()} centrality", color="#e2e8f0", fontsize=9
        )
        ax.set_title(plot_title, color="#e2e8f0", fontsize=11)

        ax.legend(
            loc="best",
            fontsize=7,
            facecolor="#0f172a",
            edgecolor="#e2e8f0",
            labelcolor="#e2e8f0",
            framealpha=0.9,
        )
        ax.grid(True, alpha=0.2, color="#e2e8f0")
        ax.tick_params(colors="#e2e8f0", labelsize=7)

        # Axis spines
        for spine in ax.spines.values():
            spine.set_edgecolor("#e2e8f0")

        return line_data, line_objects, original_styles

    # Plot top nodes
    top_line_data, top_line_objects, top_original_styles = plot_nodes_on_axis(
        ax_top, top_nodes, f"Highest variability nodes ({centrality_metric})"
    )

    # Plot bottom nodes
    bottom_line_data, bottom_line_objects, bottom_original_styles = plot_nodes_on_axis(
        ax_bottom, bottom_nodes, f"Lowest variability nodes ({centrality_metric})"
    )

    # Set xlabel only on bottom plot
    ax_bottom.set_xlabel("Window end date", color="#e2e8f0", fontsize=9)

    fig.tight_layout()

    # Store data for cursor attachment later (after FigureCanvas is created)
    # Format: {ax: (ax_obj, line_data, line_objects, original_styles)}
    fig._line_cursor_data = {
        ax_top: (ax_top, top_line_data, top_line_objects, top_original_styles),
        ax_bottom: (
            ax_bottom,
            bottom_line_data,
            bottom_line_objects,
            bottom_original_styles,
        ),
    }

    return fig


# Metrics shown in render_extended_metrics, in facet order (2 rows x 4 cols).
_EXTENDED_METRICS = [
    "avg_clustering",
    "density",
    "giant_component_avg_shortest_path",
    "largest_component_size",
    "n_components",
    "n_edges",
    "n_nodes",
    "transitivity",
]


def render_extended_metrics(
    network_metrics: pl.DataFrame,
    *,
    width: int = 12,
    height: int = 7,
    dpi: int = 100,
) -> Figure:
    """Render extended rolling network metrics as a faceted grid (matplotlib Figure).

    Args:
        network_metrics: One row per window with columns including window_end and
            the metrics in _EXTENDED_METRICS (avg_clustering, density,
            giant_component_avg_shortest_path, largest_component_size, n_components,
            n_edges, n_nodes, transitivity).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        Matplotlib Figure object with one subplot per metric.
    """
    if network_metrics.is_empty():
        return _empty_plot(
            "No network metrics available", width=width, height=height, dpi=dpi
        )

    metrics = [m for m in _EXTENDED_METRICS if m in network_metrics.columns]
    if not metrics:
        return _empty_plot(
            "No extended metrics available", width=width, height=height, dpi=dpi
        )

    import matplotlib.dates as mdates

    pandas_df = network_metrics.to_pandas().sort_values("window_end")
    x_numeric = mdates.date2num(pandas_df["window_end"].values)

    n_cols = 4
    n_rows = int(np.ceil(len(metrics) / n_cols))

    fig = Figure(figsize=(width, height), dpi=dpi)
    axes = fig.subplots(n_rows, n_cols)
    fig.patch.set_facecolor("#0f172a")
    axes_flat = np.atleast_1d(axes).flatten()

    panels = []
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        ax.set_facecolor("#0f172a")
        y_vals = pandas_df[metric].to_numpy(dtype=float)

        ax.plot(
            pandas_df["window_end"],
            y_vals,
            color="#38bdf8",
            linewidth=1.2,
            marker="o",
            markersize=2,
            alpha=0.9,
        )
        ax.set_title(metric, color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=6)
        ax.grid(True, alpha=0.2, color="#334155")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        # Sparse x tick labels to avoid overlap in the small facet panels.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

        panels.append((ax, x_numeric, y_vals, metric))

    # Hide unused trailing axes if metric count doesn't fill the grid.
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Extended Rolling Metrics", color="#e2e8f0", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Store data for cursor attachment later (after FigureCanvas is created)
    fig._extended_cursor_data = panels

    return fig


def render_community_heatmap(
    community_metrics: pl.DataFrame,
    *,
    width: int = 12,
    height: int = 8,
    dpi: int = 100,
) -> Figure:
    """Render node x window community-membership heatmap (matplotlib Figure).

    Args:
        community_metrics: Long-format DataFrame with columns
            (window_idx, window_end, node, community).
        width, height: Figure dimensions in inches.
        dpi: Dots per inch.

    Returns:
        Matplotlib Figure object.
    """
    if community_metrics.is_empty():
        return _empty_plot(
            "No community data available", width=width, height=height, dpi=dpi
        )

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    pivot = community_metrics.pivot(
        on="window_idx", index="node", values="community", aggregate_function="first"
    )
    pivot = pivot.sort("node")

    nodes = pivot.get_column("node").to_list()
    window_idx_cols = sorted(
        (col for col in pivot.columns if col != "node"), key=lambda c: int(c)
    )
    matrix = np.array(
        [
            pivot.filter(pl.col("node") == n).select(window_idx_cols).to_numpy()[0]
            for n in nodes
        ]
    )
    matrix = matrix.astype(int)

    # Map window_idx -> window_end date for tooltips/x-tick labels.
    idx_to_date = dict(
        community_metrics.select("window_idx", "window_end")
        .unique()
        .sort("window_idx")
        .iter_rows()
    )
    window_ends = [idx_to_date[int(c)] for c in window_idx_cols]

    n_communities = int(matrix.max()) + 1 if matrix.size else 1
    palette = [CATEGORICAL_10[i % len(CATEGORICAL_10)] for i in range(n_communities)]
    cmap = ListedColormap(palette)

    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=-0.5,
        vmax=n_communities - 0.5,
    )

    tick_interval = max(1, len(window_ends) // 16)
    tick_indices = np.arange(0, len(window_ends), tick_interval)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [str(window_ends[i]) for i in tick_indices], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=6)

    ax.set_xlabel("Window end date", color="#e2e8f0", fontsize=9)
    ax.set_ylabel("Stock", color="#e2e8f0", fontsize=9)
    ax.set_title("Node community membership evolution", color="#e2e8f0", fontsize=12)
    ax.tick_params(colors="#e2e8f0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    legend_handles = [
        Patch(facecolor=palette[i], edgecolor="#334155", label=str(i))
        for i in range(n_communities)
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Community",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        facecolor="#0f172a",
        edgecolor="#e2e8f0",
        labelcolor="#e2e8f0",
    )
    legend.get_title().set_color("#e2e8f0")

    fig.tight_layout()

    # Store data for cursor attachment later (after FigureCanvas is created)
    fig._community_cursor_data = (ax, matrix, nodes, window_ends)

    return fig


def _empty_plot(
    message: str, width: int = 12, height: int = 8, dpi: int = 100
) -> Figure:
    """Render a placeholder plot with a message."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14,
        color="#94a3b8",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
