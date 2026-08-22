"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date

from matplotlib.figure import Figure
import networkx as nx
from PySide6.QtCore import QObject, Signal, Slot
import polars as pl

from ycn.analysis.config import PipelineConfig
from ycn.analysis.evolution import (
    EvolutionConfig,
    compute_community_metrics,
    compute_evolution_metrics,
)
from ycn.analysis.evolution_viz import (
    render_weighted_degree_heatmap,
    render_centrality_trajectories,
    render_extended_metrics,
    render_community_heatmap,
)
from ycn.analysis.gui_cache import GuiDataCache
from ycn.analysis.mln import (
    MLNConfig,
    build_layer_graphs,
    build_multilayer_network,
    count_edge_types,
    layer_centrality_matrix,
    layer_community_matrix,
    layer_edge_metrics,
    layer_values_of,
    prepare_mln_frame,
)
from ycn.analysis.mln_viz import (
    render_mln_communities,
    render_mln_metrics,
)
from ycn.analysis.multiplex_data import multiplex_tables
from ycn.analysis.pipeline import PipelineResult, run_pipeline
from ycn.analysis.yield_curve import CurvePanel, load_long_panel

# Progress callbacks fire once per node pair -- tens of thousands of times for a
# large run. Every emit is a queued cross-thread signal that makes the GUI thread
# repaint the progress bar and status label, which is what made the window
# sluggish while a build was running. Emitting at most this often keeps the UI
# informative (well above human perception) at a fraction of the cost. The
# cancellation check still runs on *every* call, so responsiveness to Cancel
# Render is unaffected.
_PROGRESS_EMIT_INTERVAL_S = 0.08


class WorkerCancelled(Exception):
    """Raised from inside a progress/status callback to unwind a cancelled worker.

    ``run_pipeline``/``compute_evolution_metrics`` invoke their ``progress``
    callback on every pair/window with no cancellation hook of their own, so
    this is the cheapest place to interrupt a long-running computation:
    raising here propagates as a normal exception up through the (otherwise
    unmodified) computation and back into ``run()``, where it is caught
    separately from real failures.
    """


class _ThrottledProgressMixin:
    """Cancellation check on every tick; signal emission rate-limited."""

    def _init_throttle(self, cancel_event: threading.Event | None) -> None:
        self._cancel_event = cancel_event or threading.Event()
        self._last_emit = 0.0

    def _progress_wrapper(self, done: int, total: int, desc: str) -> None:
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        now = time.monotonic()
        # Always emit the terminal tick so the bar reliably reaches 100%.
        if done >= total or (now - self._last_emit) >= _PROGRESS_EMIT_INTERVAL_S:
            self._last_emit = now
            self.progress.emit(done, total, desc)

    def _status_wrapper(self, msg: str) -> None:
        if self._cancel_event.is_set():
            raise WorkerCancelled()
        self.status.emit(msg)


@dataclass
class EvolutionResult:
    """Artifacts produced by evolution analysis."""

    node_metrics: pl.DataFrame
    network_metrics: pl.DataFrame
    community_metrics: pl.DataFrame
    heatmap_fig: Figure
    centrality_fig: Figure
    extended_metrics_fig: Figure
    community_fig: Figure


@dataclass
class MLNResult:
    """Artifacts produced by multi-layer network analysis.

    The multiplex tables are carried through so the GUI can re-render the 3D
    view for a different layer selection without recomputing anything.
    """

    graph: nx.Graph
    nodes: pl.DataFrame
    intra: pl.DataFrame
    inter: pl.DataFrame
    layer_values: list[str]
    layer_column: str
    node_column: str
    centrality: str
    n_intra_edges: int
    n_inter_edges: int
    edge_df: pl.DataFrame
    centrality_df: pl.DataFrame
    community_df: pl.DataFrame
    metrics_fig: Figure
    community_fig: Figure


class PipelineWorker(_ThrottledProgressMixin, QObject):
    """Runs ``run_pipeline`` off the UI thread."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # PipelineResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        data_cache: GuiDataCache | None = None,
        edge_settings: dict | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._data_cache = data_cache
        self._edge_settings = edge_settings or {}
        self._init_throttle(cancel_event)
        # Will be populated during run
        self.df_returns: pl.DataFrame | None = None
        self.dates: list[date] | None = None
        # Store original date column name for reference
        self.date_column_original: str = config.date_column

    # Aliases kept so run() reads the same as the other workers.
    _progress = _ThrottledProgressMixin._progress_wrapper
    _status = _ThrottledProgressMixin._status_wrapper

    @Slot()
    def run(self) -> None:
        try:
            result: PipelineResult = run_pipeline(
                self._config,
                progress=self._progress,
                status=self._status,
                data_cache=self._data_cache,
                edge_settings=self._edge_settings,
            )
            if self._cancel_event.is_set():
                raise WorkerCancelled()

            # Store prepared data for evolution analysis
            # Get from cache if available
            from ycn.analysis.data_access import load_table
            from ycn.analysis.transforms import apply_transforms

            try:
                needed_cols = [
                    self._config.date_column,
                    self._config.name_column,
                    self._config.value_column,
                ]

                df = load_table(
                    self._config.db_path,
                    self._config.table,
                    columns=needed_cols,
                    where_clause=self._config.where_clause,
                )
                df = df.with_columns(
                    pl.col(self._config.date_column).cast(pl.Date, strict=False)
                )
                df = df.sort(self._config.date_column, self._config.name_column)

                if self._config.date_start is not None:
                    df = df.filter(
                        pl.col(self._config.date_column) >= self._config.date_start
                    )
                if self._config.date_end is not None:
                    df = df.filter(
                        pl.col(self._config.date_column) <= self._config.date_end
                    )

                df = df.drop_nulls(
                    subset=[
                        self._config.date_column,
                        self._config.name_column,
                        self._config.value_column,
                    ]
                )

                # Apply transforms
                df = apply_transforms(
                    df,
                    self._config.transforms,
                    self._config.date_column,
                    self._config.name_column,
                    [self._config.value_column],
                )

                # Normalize column names for evolution analysis
                # Rename to standard internal names to avoid column name issues
                df = df.rename(
                    {
                        self._config.date_column: "Date",
                        self._config.name_column: "Name",
                        self._config.value_column: "Close",
                    }
                )

                self.df_returns = df
                self.dates = [d for d in df.get_column("Date").unique().sort()]
            except Exception:
                pass  # Evolution worker will handle missing data gracefully

            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self.failed.emit(str(exc))


class EvolutionWorker(_ThrottledProgressMixin, QObject):
    """Background worker for temporal network evolution analysis."""

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # EvolutionResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        df_returns: pl.DataFrame,
        dates: list[date],
        evolution_config: EvolutionConfig,
        data_cache: GuiDataCache | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.df_returns = df_returns
        self.dates = dates
        self.evolution_config = evolution_config
        self.data_cache = data_cache
        self._init_throttle(cancel_event)

    @Slot()
    def run(self) -> None:
        try:
            self._status_wrapper("Computing evolution metrics...")

            # Compute metrics (node-level, network-level, and per-window graphs)
            # Use normalized column names (Date, Name, Close)
            metrics_result = compute_evolution_metrics(
                self.df_returns,
                self.dates,
                self.evolution_config,
                date_column="Date",
                name_column="Name",
                value_column="Close",
                progress=self._progress_wrapper,
                status=self._status_wrapper,
            )
            node_metrics = metrics_result.node_metrics
            network_metrics = metrics_result.network_metrics
            graphs = metrics_result.graphs

            if node_metrics.is_empty():
                self.failed.emit("No windows produced with minimum node count")
                return

            self._status_wrapper("Rendering weighted-degree heatmap...")
            heatmap_fig = render_weighted_degree_heatmap(node_metrics)

            self._status_wrapper("Rendering centrality trajectories...")
            n_unique_nodes = len(node_metrics["node"].unique())
            centrality_fig = render_centrality_trajectories(
                node_metrics,
                centrality_metric=self.evolution_config.centrality,
                n_nodes=min(self.evolution_config.n_top_nodes, n_unique_nodes),
            )

            self._status_wrapper("Rendering extended rolling metrics...")
            extended_metrics_fig = render_extended_metrics(network_metrics)

            self._status_wrapper("Detecting communities per window...")
            community_metrics = pl.DataFrame()
            try:
                community_metrics = compute_community_metrics(
                    graphs,
                    max_clusters=self.evolution_config.max_communities,
                    min_nodes=self.evolution_config.min_nodes,
                    community_method=self.evolution_config.community_method,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                )
                community_fig = render_community_heatmap(community_metrics)
            except ValueError as exc:
                self._status_wrapper(f"Community detection skipped: {exc}")
                community_fig = render_community_heatmap(pl.DataFrame())

            self._status_wrapper("Evolution analysis complete.")
            self.progress.emit(1, 1, "done")

            result = EvolutionResult(
                node_metrics=node_metrics,
                network_metrics=network_metrics,
                community_metrics=community_metrics,
                heatmap_fig=heatmap_fig,
                centrality_fig=centrality_fig,
                extended_metrics_fig=extended_metrics_fig,
                community_fig=community_fig,
            )
            self.finished.emit(result)

        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Evolution analysis failed: {str(exc)}")


class MLNWorker(_ThrottledProgressMixin, QObject):
    """Background worker for multi-layer network analysis.

    Runs between the pipeline and evolution stages. Every status line is
    prefixed ``"MLN: "`` so its log output stays distinguishable from the
    evolution worker's.
    """

    progress = Signal(int, int, str)  # done, total, description
    status = Signal(str)
    finished = Signal(object)  # MLNResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        mln_config: MLNConfig,
        edge_settings: dict | None = None,
        cancel_event: threading.Event | None = None,
        panel: CurvePanel | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._mln_config = mln_config
        self._edge_settings = edge_settings or {}
        self._panel = panel
        self._init_throttle(cancel_event)

    def _load_frame(self) -> pl.DataFrame:
        """Long frame for the MLN, from a wide curve panel or a long table."""
        if self._panel is not None:
            return load_long_panel(self._config, self._panel)
        return prepare_mln_frame(
            self._config, self._mln_config.layer_column, status=self._status_wrapper
        )

    @Slot()
    def run(self) -> None:
        try:
            cfg = self._config
            mcfg = self._mln_config
            layer_column = mcfg.layer_column

            self._status_wrapper(f"MLN: loading data for layer column {layer_column!r}")
            df = self._load_frame()
            if df.is_empty():
                self.failed.emit(
                    "No rows remain after filtering / date range for the MLN. "
                    "Check the Optional Filter, the date range, and the User Filter "
                    "cell selection."
                )
                return

            values = layer_values_of(df, layer_column)
            if len(values) < 2:
                self.failed.emit(
                    f"Column {layer_column!r} has {len(values)} distinct value(s) "
                    "after filtering -- at least 2 are needed for a multi-layer network."
                )
                return
            self._status_wrapper(f"MLN: {len(values)} layers -- {', '.join(values)}")
            if len(values) > 12:
                self._status_wrapper(
                    f"MLN: {len(values)} layers is a lot; each is an O(n^2) "
                    f"{cfg.measure} computation, so this may take a while."
                )

            layer_graphs = build_layer_graphs(
                df,
                cfg,
                layer_column,
                layer_values=values,
                progress=self._progress_wrapper,
                status=self._status_wrapper,
                edge_settings=self._edge_settings,
            )

            self._status_wrapper("MLN: assembling multiplex network")
            graph = build_multilayer_network(layer_graphs, values)
            n_intra, n_inter = count_edge_types(graph)
            self._status_wrapper(
                f"MLN: multiplex has {graph.number_of_nodes()} nodes, "
                f"{n_intra} intra-layer and {n_inter} inter-layer edges."
            )
            if n_inter == 0:
                # A data property, not a failure: no node appears in >1 layer.
                self._status_wrapper(
                    "MLN: no node appears in more than one layer -- the multiplex "
                    "has no inter-layer edges."
                )

            nodes, intra, inter = multiplex_tables(graph)

            self._status_wrapper("MLN: computing per-layer edge metrics")
            edge_df = layer_edge_metrics(graph, values)

            self._status_wrapper(f"MLN: computing {mcfg.centrality} centrality")
            centrality_df = layer_centrality_matrix(
                layer_graphs, values, mcfg.centrality
            )

            self._status_wrapper(
                f"MLN: detecting communities ({mcfg.community_method.value}, "
                f"Jaccard >= {mcfg.jaccard_threshold:.2f})"
            )
            community_df = layer_community_matrix(layer_graphs, values, mcfg)

            self._status_wrapper("MLN: rendering metrics")
            metrics_fig = render_mln_metrics(
                edge_df,
                centrality_df,
                layer_label=layer_column,
                node_label=cfg.name_column,
                centrality_name=mcfg.centrality,
            )

            self._status_wrapper("MLN: rendering communities")
            community_fig = render_mln_communities(
                community_df,
                layer_label=layer_column,
                node_label=cfg.name_column,
            )

            self._status_wrapper("MLN: complete.")
            self.finished.emit(
                MLNResult(
                    graph=graph,
                    nodes=nodes,
                    intra=intra,
                    inter=inter,
                    layer_values=values,
                    layer_column=layer_column,
                    node_column=cfg.name_column,
                    centrality=mcfg.centrality,
                    n_intra_edges=n_intra,
                    n_inter_edges=n_inter,
                    edge_df=edge_df,
                    centrality_df=centrality_df,
                    community_df=community_df,
                    metrics_fig=metrics_fig,
                    community_fig=community_fig,
                )
            )
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"MLN analysis failed: {str(exc)}")
