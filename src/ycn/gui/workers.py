"""Background workers so the UI stays responsive during long dcor runs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date

from matplotlib.figure import Figure
import networkx as nx
from PySide6.QtCore import QObject, Signal, Slot
import polars as pl

from ycn.analysis.af_models import ModelSpec, ResidualModel
from ycn.analysis.cancellation import ComputationCancelled
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
from ycn.analysis.mln_evolution import (
    compute_curve_factors,
    compute_curve_factors_by_issuer,
    compute_multiplex_evolution,
    compute_stress_metrics,
    compute_stress_metrics_by_issuer,
)
from ycn.analysis.mln_evolution_viz import (
    render_edge_evolution,
    render_factor_evolution,
    render_stress_quadrants,
)
from ycn.analysis.multiplex_data import multiplex_tables
from ycn.analysis.pipeline import PipelineResult, run_pipeline
from ycn.analysis.residual_networks import compute_residual_networks, residual_cube
from ycn.analysis.yield_curve import CurvePanel, NetworkKind, load_long_panel

# Progress callbacks fire once per node pair -- tens of thousands of times for a
# large run. Every emit is a queued cross-thread signal that makes the GUI thread
# repaint the progress bar and status label, which is what made the window
# sluggish while a build was running. Emitting at most this often keeps the UI
# informative (well above human perception) at a fraction of the cost. The
# cancellation check still runs on *every* call, so responsiveness to Cancel
# Render is unaffected.
_PROGRESS_EMIT_INTERVAL_S = 0.08


# Raised from inside a progress/status callback to unwind a cancelled worker.
# Computations invoke their ``progress`` callback at checkpoints with no
# cancellation hook of their own, so this is the cheapest place to interrupt
# one: raising here propagates as a normal exception up through the (otherwise
# unmodified) computation and back into ``run()``, where it is caught
# separately from real failures. Defined in the analysis layer so code there
# can re-raise it past its own per-item ``except Exception`` handlers.
WorkerCancelled = ComputationCancelled


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
    # Per-layer un-thresholded measure matrices, keyed by layer value. Feeds
    # the MLN: Degree tab's threshold slider, which re-thresholds a layer
    # without recomputing its (expensive) measure. Empty on a restored
    # session -- the slider is then simply unavailable, since the archive
    # stores thresholded edges, not measures.
    layer_measures: dict[str, pl.DataFrame] = field(default_factory=dict)
    independent_threshold: float = 0.33


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

            layer_measures: dict[str, pl.DataFrame] = {}
            layer_graphs = build_layer_graphs(
                df,
                cfg,
                layer_column,
                layer_values=values,
                progress=self._progress_wrapper,
                status=self._status_wrapper,
                edge_settings=self._edge_settings,
                measures_out=layer_measures,
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
                    layer_measures=layer_measures,
                    independent_threshold=cfg.independent_threshold,
                )
            )
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"MLN analysis failed: {str(exc)}")


@dataclass
class ResidualResult:
    """Artifacts from the Nelson-Siegel residual-network pass."""

    metrics: pl.DataFrame
    coverage: pl.DataFrame
    label_column: str
    label_order: list[str]
    node_label: str
    issuer_column: str
    skipped: list[str]


@dataclass
class MLNEvolutionResult:
    """Artifacts from the multi-layer-network evolution pass.

    The three static figures are rendered here on the worker thread; the Cov(t)
    trajectory is not, because it depends on axis choices the user makes after
    the run and is cheap to draw on demand.
    """

    edge_types: pl.DataFrame
    community_k: pl.DataFrame
    communities: pl.DataFrame
    layer_metrics: pl.DataFrame
    factors: pl.DataFrame
    regimes: pl.DataFrame
    stress: pl.DataFrame
    links_fig: Figure
    factor_fig: Figure
    factor_std_fig: Figure
    stress_fig: Figure
    # Per-issuer analogues of factors/regimes/stress, keyed by issuer -- feed
    # the "Evo: Resids"/"Evo: Cov"/"Evo: Cov(t)" issuer picker. Rendered on
    # demand in MainWindow, not here: pre-rendering a figure per issuer up
    # front would pay for every issuer even though a run usually only looks
    # at a few.
    issuer_factors: dict[str, pl.DataFrame] = field(default_factory=dict)
    issuer_regimes: dict[str, pl.DataFrame] = field(default_factory=dict)
    issuer_stress: dict[str, pl.DataFrame] = field(default_factory=dict)
    skipped_issuers: list[str] = field(default_factory=list)


@dataclass
class NeuralEvolutionResult:
    """Artifacts from the optional Neural-HJM evolution pass.

    Deliberately shaped like the factor/regime/stress half of
    :class:`MLNEvolutionResult` -- same field names, same meaning -- so the GUI
    can swap between the two by which object it reads from, not by branching on
    shape. There is no ``edge_types``/``community_k``/``links_fig`` here: the
    multiplex structure itself does not depend on which curve model produced
    the residuals, only the factor trajectory and the stress indicators do.
    """

    factors: pl.DataFrame
    regimes: pl.DataFrame
    stress: pl.DataFrame
    factor_fig: Figure
    factor_std_fig: Figure
    stress_fig: Figure
    issuer_factors: dict[str, pl.DataFrame] = field(default_factory=dict)
    issuer_regimes: dict[str, pl.DataFrame] = field(default_factory=dict)
    issuer_stress: dict[str, pl.DataFrame] = field(default_factory=dict)
    skipped_issuers: list[str] = field(default_factory=list)


class ResidualWorker(_ThrottledProgressMixin, QObject):
    """Builds the NS residual networks off the UI thread.

    Independent of both the multiplex build and the evolution pass -- residual
    networks are a single snapshot over the configured date range -- so this
    runs on its own thread and its results land as soon as they are ready.
    """

    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(object)  # ResidualResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        panel: CurvePanel,
        kind: NetworkKind,
        *,
        threshold: float = 0.3,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._panel = panel
        self._kind = kind
        self._threshold = threshold
        self._init_throttle(cancel_event)

    @Slot()
    def run(self) -> None:
        try:
            self._status_wrapper("NS: loading panel for residual networks...")
            long = load_long_panel(self._config, self._panel)
            if long.is_empty():
                self.failed.emit("No rows remain for the residual networks.")
                return

            result = compute_residual_networks(
                long,
                self._panel,
                self._kind,
                self._config.date_column,
                threshold=self._threshold,
                progress=self._progress_wrapper,
                status=self._status_wrapper,
            )
            if self._cancel_event.is_set():
                raise WorkerCancelled()

            if result.skipped:
                names = ", ".join(result.skipped[:6])
                self._status_wrapper(
                    f"NS: {len(result.skipped)} issuer(s) skipped ({names})"
                )
            self._status_wrapper("NS: residual networks complete.")
            self.finished.emit(
                ResidualResult(
                    metrics=result.metrics,
                    coverage=result.coverage,
                    label_column=result.label_column,
                    label_order=result.label_order,
                    node_label=result.node_label,
                    issuer_column=self._panel.issuer_column,
                    skipped=result.skipped,
                )
            )
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 -- surface to UI
            self.failed.emit(f"Residual networks failed: {exc}")


class MLNEvolutionWorker(_ThrottledProgressMixin, QObject):
    """Rolling-window evolution of the multiplex, the curve factors and stress.

    Every status line is prefixed ``"Evolution: "`` so its log output stays
    distinguishable from the residual worker's, which runs concurrently.
    """

    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(object)  # MLNEvolutionResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        panel: CurvePanel,
        mln_config: MLNConfig,
        evolution_config: EvolutionConfig,
        *,
        edge_settings: dict | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._panel = panel
        self._mln_config = mln_config
        self._evolution_config = evolution_config
        self._edge_settings = edge_settings or {}
        self._init_throttle(cancel_event)

    @Slot()
    def run(self) -> None:
        try:
            cfg, panel = self._config, self._panel
            self._status_wrapper("Evolution: loading panel...")
            long = load_long_panel(cfg, panel)
            if long.is_empty():
                self.failed.emit("No rows remain for the evolution analysis.")
                return

            edge_types, community_k, communities, layer_metrics, _windows = (
                compute_multiplex_evolution(
                    long,
                    cfg,
                    self._mln_config,
                    self._evolution_config,
                    edge_settings=self._edge_settings,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                )
            )
            if self._cancel_event.is_set():
                raise WorkerCancelled()

            # Factors and stress are curve-level, not multiplex-level, so a
            # failure in either must not discard the multiplex evolution that
            # already succeeded -- the Links tab still has something to show.
            # Same reasoning splits each of those two into a market half and a
            # per-issuer half: a per-issuer failure must not lose the market
            # result the "Average" selection needs.
            factors = regimes = stress = pl.DataFrame()
            issuer_factors: dict[str, pl.DataFrame] = {}
            issuer_regimes: dict[str, pl.DataFrame] = {}
            issuer_stress: dict[str, pl.DataFrame] = {}
            skipped_issuers: list[str] = []
            try:
                factors, regimes = compute_curve_factors(
                    long,
                    panel,
                    cfg.date_column,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                    status=self._status_wrapper,
                )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Evolution: factor evolution skipped ({exc})")

            try:
                self._status_wrapper("Evolution: computing per-issuer factor evolution...")
                issuer_factors, issuer_regimes, skipped_issuers = (
                    compute_curve_factors_by_issuer(
                        long,
                        panel,
                        cfg.date_column,
                        window_size=self._evolution_config.window_size,
                        step_size=self._evolution_config.step,
                        progress=self._progress_wrapper,
                        status=self._status_wrapper,
                    )
                )
                if skipped_issuers:
                    names = ", ".join(skipped_issuers[:6])
                    self._status_wrapper(
                        f"Evolution: {len(skipped_issuers)} issuer(s) skipped for "
                        f"per-issuer factors ({names})"
                    )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Evolution: per-issuer factor evolution skipped ({exc})")

            try:
                self._status_wrapper("Evolution: computing correlation stress...")
                issuers, dates, cube, _terms, _skipped = residual_cube(
                    long,
                    panel,
                    cfg.date_column,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                )
                stress = compute_stress_metrics(
                    cube,
                    dates,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                )

                self._status_wrapper("Evolution: computing per-issuer correlation stress...")
                issuer_stress = compute_stress_metrics_by_issuer(
                    issuers,
                    dates,
                    cube,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Evolution: stress analysis skipped ({exc})")

            if self._cancel_event.is_set():
                raise WorkerCancelled()

            self._status_wrapper("Evolution: rendering figures...")
            result = MLNEvolutionResult(
                edge_types=edge_types,
                community_k=community_k,
                communities=communities,
                layer_metrics=layer_metrics,
                factors=factors,
                regimes=regimes,
                stress=stress,
                links_fig=render_edge_evolution(edge_types, community_k),
                factor_fig=render_factor_evolution(factors, regimes, std=False),
                factor_std_fig=render_factor_evolution(factors, regimes, std=True),
                stress_fig=render_stress_quadrants(stress),
                issuer_factors=issuer_factors,
                issuer_regimes=issuer_regimes,
                issuer_stress=issuer_stress,
                skipped_issuers=skipped_issuers,
            )
            self._status_wrapper("Evolution: complete.")
            self.progress.emit(1, 1, "done")
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Evolution analysis failed: {exc}")


class NeuralEvolutionWorker(_ThrottledProgressMixin, QObject):
    """Rolling-window factor and stress evolution under the Neural-HJM model.

    Opt-in and gated on the NS evolution pass already being requested; runs
    **after** :class:`MLNEvolutionWorker` finishes, not alongside it -- the
    same reasoning as every stage in this chain: compute-bound Python gains
    nothing from a second thread competing with the GUI for the interpreter,
    and training one small network per issuer is the most CPU-heavy stage of
    the four. Every status line is prefixed ``"Neural: "``.

    Does not touch multiplex structure (edge composition, community *k*) --
    that depends only on the connection measure, not on which curve model
    produced the residuals, so it is computed once by ``MLNEvolutionWorker``
    regardless of this stage running.
    """

    progress = Signal(int, int, str)
    status = Signal(str)
    finished = Signal(object)  # NeuralEvolutionResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: PipelineConfig,
        panel: CurvePanel,
        evolution_config: EvolutionConfig,
        *,
        seed: int = 0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._panel = panel
        self._evolution_config = evolution_config
        self._seed = seed
        self._init_throttle(cancel_event)

    @Slot()
    def run(self) -> None:
        try:
            cfg, panel = self._config, self._panel
            self._status_wrapper("Neural: loading panel...")
            long = load_long_panel(cfg, panel)
            if long.is_empty():
                self.failed.emit("No rows remain for the Neural-HJM evolution.")
                return

            model = ModelSpec(ResidualModel.NEURAL_HJM, seed=self._seed)

            # As in MLNEvolutionWorker: factors and stress are independent, so
            # a failure in either must not discard the other -- and each
            # splits into a market half and a per-issuer half for the same
            # reason.
            factors = regimes = stress = pl.DataFrame()
            issuer_factors: dict[str, pl.DataFrame] = {}
            issuer_regimes: dict[str, pl.DataFrame] = {}
            issuer_stress: dict[str, pl.DataFrame] = {}
            skipped_issuers: list[str] = []
            try:
                factors, regimes = compute_curve_factors(
                    long,
                    panel,
                    cfg.date_column,
                    model=model,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                    status=self._status_wrapper,
                    log_prefix="Neural",
                )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Neural: factor evolution skipped ({exc})")

            try:
                self._status_wrapper("Neural: computing per-issuer factor evolution...")
                issuer_factors, issuer_regimes, skipped_issuers = (
                    compute_curve_factors_by_issuer(
                        long,
                        panel,
                        cfg.date_column,
                        model=model,
                        window_size=self._evolution_config.window_size,
                        step_size=self._evolution_config.step,
                        progress=self._progress_wrapper,
                        status=self._status_wrapper,
                        log_prefix="Neural",
                    )
                )
                if skipped_issuers:
                    names = ", ".join(skipped_issuers[:6])
                    self._status_wrapper(
                        f"Neural: {len(skipped_issuers)} issuer(s) skipped for "
                        f"per-issuer factors ({names})"
                    )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Neural: per-issuer factor evolution skipped ({exc})")

            try:
                self._status_wrapper("Neural: computing correlation stress...")
                issuers, dates, cube, _terms, _skipped = residual_cube(
                    long,
                    panel,
                    cfg.date_column,
                    model=model,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                    log_prefix="Neural",
                )
                stress = compute_stress_metrics(
                    cube,
                    dates,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                )

                self._status_wrapper("Neural: computing per-issuer correlation stress...")
                issuer_stress = compute_stress_metrics_by_issuer(
                    issuers,
                    dates,
                    cube,
                    window_size=self._evolution_config.window_size,
                    step_size=self._evolution_config.step,
                    progress=self._progress_wrapper,
                    status=self._status_wrapper,
                    log_prefix="Neural",
                )
            except WorkerCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                self._status_wrapper(f"Neural: stress analysis skipped ({exc})")

            if self._cancel_event.is_set():
                raise WorkerCancelled()

            self._status_wrapper("Neural: rendering figures...")
            result = NeuralEvolutionResult(
                factors=factors,
                regimes=regimes,
                stress=stress,
                factor_fig=render_factor_evolution(factors, regimes, std=False),
                factor_std_fig=render_factor_evolution(factors, regimes, std=True),
                stress_fig=render_stress_quadrants(stress),
                issuer_factors=issuer_factors,
                issuer_regimes=issuer_regimes,
                issuer_stress=issuer_stress,
                skipped_issuers=skipped_issuers,
            )
            self._status_wrapper("Neural: complete.")
            self.progress.emit(1, 1, "done")
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Neural-HJM evolution failed: {exc}")
