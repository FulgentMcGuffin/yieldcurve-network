"""Temporal evolution of network metrics via rolling windows."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from enum import Enum

from graspologic.embed import AdjacencySpectralEmbed
import networkx as nx
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from . import measures, network, transforms

# Suppress sklearn ConvergenceWarning from KMeans when n_distinct_clusters < n_clusters
# This occurs when analyzing sparse networks with highly clustered node embeddings
warnings.filterwarnings("ignore", message="Number of distinct clusters.*found smaller")

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]

# Sub-steps reported per window, so progress (and therefore the cancellation
# check that rides on it) advances during a window rather than only between them.
PROGRESS_SCALE = 100


class CommunityMethod(str, Enum):
    """Community detection optimization method for rolling-window networks.

    Each method determines the number of communities k independently per window
    (no lookahead bias): only data from the current window informs that window's k.
    """

    FIXED = "fixed"
    """Fixed k (max_communities parameter) for every window."""

    SILHOUETTE = "silhouette"
    """Maximize average silhouette coefficient in latent space (graspologic default)."""

    MODULARITY = "modularity"
    """Maximize modularity in the original network (graph-aware)."""

    DAVIES_BOULDIN = "davies_bouldin"
    """Minimize Davies-Bouldin index in latent space (cluster separation/compactness)."""

    CALINSKI_HARABASZ = "calinski_harabasz"
    """Maximize Calinski-Harabasz index in latent space (between/within cluster variance)."""


@dataclass
class EvolutionConfig:
    """Configuration for rolling/expanding-window network evolution analysis.

    Attributes:
        window_size: Number of observations per window (trading days, default 252 ~1 year).
        step: Observations to advance between windows (default 21 ~1 month).
        expanding: If True, anchor window start at first date (expanding window mode).
        min_nodes: Minimum nodes required per window to include it (default 5).
        independent_threshold: Correlation threshold for network edges (default 0.33).
        centrality: Per-node centrality measure: "eigenvector", "betweenness", or "degree".
        n_top_nodes: Number of top variable nodes to show in centrality plot (default 10).
        max_communities: For FIXED method, exact k per window; for other methods,
            upper bound when searching for optimal k per window.
        community_method: Strategy for determining optimal k per window; see CommunityMethod enum.
        measure: Connection measure used to build each window's network. The GUI
            sets this from the sidebar's selection so evolution matches the
            Network and MLN tabs; the default preserves the historical behaviour
            for callers that do not set it.
        edge_settings: Optional measure-specific parameters (e.g.
            ``{"conditional_quantile": 0.9}``), passed through to
            ``measures.compute_measure``.
    """

    window_size: int = 252
    step: int = 21
    expanding: bool = False
    min_nodes: int = 5
    independent_threshold: float = 0.33
    centrality: str = "eigenvector"
    n_top_nodes: int = 10
    max_communities: int = 10
    community_method: CommunityMethod = CommunityMethod.FIXED
    measure: str = "distance_correlation"
    edge_settings: dict | None = None

    def __post_init__(self) -> None:
        if isinstance(self.community_method, str):
            self.community_method = CommunityMethod(self.community_method)


def generate_windows(
    dates: list[date],
    window_size: int,
    step: int,
    *,
    expanding: bool = False,
) -> Iterator[tuple[date, date, list[date]]]:
    """Yield rolling or expanding windows over sorted unique trading dates.

    Args:
        dates: Sorted, unique trading dates (ascending).
        window_size: Minimum/initial number of observations per window.
        step: Number of observations to advance between windows.
        expanding: Anchor window start at dates[0] instead of sliding it.

    Yields:
        (window_start, window_end, window_dates) tuples.

    Raises:
        ValueError: If window_size < 3.
    """
    if window_size < 3:
        raise ValueError("window_size must be >= 3")
    n = len(dates)
    end_idx = window_size - 1
    while end_idx < n:
        start_idx = 0 if expanding else end_idx - window_size + 1
        window_dates = dates[start_idx : end_idx + 1]
        yield window_dates[0], window_dates[-1], window_dates
        end_idx += step


def _drop_nan_edges(graph: nx.Graph) -> nx.Graph:
    """Remove NaN-weight edges."""
    graph = graph.copy()
    bad_edges = [(u, v) for u, v, w in graph.edges(data="weight") if w != w]
    graph.remove_edges_from(bad_edges)
    return graph


def _add_strength_attr(graph: nx.Graph) -> None:
    """Attach strength attribute (1 - weight) in place."""
    for _, _, d in graph.edges(data=True):
        d["strength"] = 1.0 - d["weight"]


def _node_centrality(graph: nx.Graph, centrality: str) -> dict[str, float]:
    """Compute per-node centrality, robust to disconnected graphs."""
    if graph.number_of_edges() == 0:
        return dict.fromkeys(graph.nodes(), 0.0)
    if centrality == "eigenvector":
        try:
            return nx.eigenvector_centrality(graph, weight="strength", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            return nx.betweenness_centrality(graph, weight="weight")
    if centrality == "betweenness":
        return nx.betweenness_centrality(graph, weight="weight")
    if centrality == "degree":
        return nx.degree_centrality(graph)
    raise ValueError(f"Unknown centrality: {centrality!r}")


def _network_summary(graph: nx.Graph, window_start: date, window_end: date) -> dict:
    """One row of network-level summary metrics for a window's graph."""
    n_nodes = graph.number_of_nodes()
    degrees = [d for _, d in graph.degree()]
    largest_cc = max((len(c) for c in nx.connected_components(graph)), default=0)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": (sum(degrees) / n_nodes) if n_nodes else float("nan"),
        "n_components": nx.number_connected_components(graph) if n_nodes else 0,
        "largest_component_size": largest_cc,
        "avg_clustering": nx.average_clustering(graph) if n_nodes else float("nan"),
    }


def _extended_network_summary(graph: nx.Graph) -> dict:
    """Extra global structural metrics: transitivity and giant-component ASP.

    Returns:
        Dict with keys: transitivity, giant_component_avg_shortest_path.
        NaN for giant_component_avg_shortest_path if the giant component has <2 nodes.
    """
    result: dict = {"transitivity": nx.transitivity(graph) or float("nan")}
    largest_cc = max(nx.connected_components(graph), key=len, default=set())
    if len(largest_cc) >= 2:
        subgraph = graph.subgraph(largest_cc)
        result["giant_component_avg_shortest_path"] = nx.average_shortest_path_length(
            subgraph
        )
    else:
        result["giant_component_avg_shortest_path"] = float("nan")
    return result


def _node_summary(graph: nx.Graph, window_end: date, *, centrality: str) -> list[dict]:
    """Long-format per-node metric rows."""
    cent = _node_centrality(graph, centrality)
    weighted_degree = dict(graph.degree(weight="strength"))
    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "window_end": window_end,
                "node": node,
                "metric": "weighted_degree",
                "value": float(weighted_degree[node]),
            }
        )
        rows.append(
            {
                "window_end": window_end,
                "node": node,
                "metric": centrality,
                "value": float(cent.get(node, float("nan"))),
            }
        )
    return rows


@dataclass
class EvolutionMetricsResult:
    """Aggregated outputs of :func:`compute_evolution_metrics`.

    Attributes:
        node_metrics: Long-format per-node metrics (window_end, node, metric, value).
        network_metrics: One row per window of network-level + extended structural metrics.
        graphs: window_end -> pruned, strength-annotated graph (for downstream chapters,
            e.g. community detection, that need the graph itself rather than a summary row).
    """

    node_metrics: pl.DataFrame
    network_metrics: pl.DataFrame
    graphs: dict[date, nx.Graph]


def compute_evolution_metrics(
    df_returns: pl.DataFrame,
    dates: list[date],
    cfg: EvolutionConfig,
    *,
    date_column: str = "Date",
    name_column: str = "Name",
    value_column: str = "Close",
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> EvolutionMetricsResult:
    """Compute node- and network-level metrics for every window.

    Args:
        df_returns: Long-format daily-returns dataframe.
        dates: Sorted unique trading dates.
        cfg: Evolution configuration.
        date_column, name_column, value_column: Column names.
        progress: Progress callback (done, total, desc).
        status: Status message callback.

    Returns:
        EvolutionMetricsResult with node-level metrics (long format), network-level
        metrics (one row per window), and the per-window graphs themselves.
    """
    if status is not None:
        status(
            f"Generating {cfg.window_size}-obs windows (step={cfg.step}) "
            f"using {cfg.measure}..."
        )

    windows = list(
        generate_windows(dates, cfg.window_size, cfg.step, expanding=cfg.expanding)
    )
    node_rows = []
    network_rows = []
    graphs: dict[date, nx.Graph] = {}
    n_windows = len(windows)

    # Progress is reported on a 0..PROGRESS_SCALE sub-range per window so the
    # bar advances *within* a window too, not only between windows.
    total_units = max(n_windows, 1) * PROGRESS_SCALE

    for w_idx, (window_start, window_end, window_dates) in enumerate(windows):
        if progress is not None:
            progress(w_idx * PROGRESS_SCALE, total_units, f"{window_end}")

        window_df = df_returns.filter(
            pl.col(date_column).is_between(window_start, window_end)
        )
        wide = network.pivot_to_wide(window_df, date_column, name_column, value_column)
        nodes = [c for c in wide.columns if c != date_column]
        nodes = [n for n in nodes if wide.get_column(n).drop_nulls().len() >= 3]

        if len(nodes) < cfg.min_nodes:
            continue

        # Forwarding a per-pair progress callback is what makes a window
        # interruptible: callers detect cancellation inside this callback, and
        # a single window's measure can run for tens of seconds on a large
        # network. Without it the UI froze until the whole window finished.
        def _pair_progress(done: int, total: int, desc: str, _w: int = w_idx) -> None:
            if progress is None:
                return
            frac = (done / total) if total else 1.0
            progress(
                int(_w * PROGRESS_SCALE + frac * PROGRESS_SCALE),
                total_units,
                f"{window_end}: {desc}",
            )

        measure_df = measures.compute_measure(
            cfg.measure,
            wide.select(nodes),
            nodes,
            progress=_pair_progress,
            edge_settings=cfg.edge_settings,
        )
        graph = network.build_corr_nx(
            measure_df, independent_threshold=cfg.independent_threshold
        )
        graph = _drop_nan_edges(graph)
        _add_strength_attr(graph)
        node_rows.extend(_node_summary(graph, window_end, centrality=cfg.centrality))
        network_row = _network_summary(graph, window_start, window_end)
        network_row.update(_extended_network_summary(graph))
        network_rows.append(network_row)
        graphs[window_end] = graph

    if progress is not None:
        progress(total_units, total_units, "done")

    return EvolutionMetricsResult(
        node_metrics=pl.DataFrame(node_rows),
        network_metrics=pl.DataFrame(network_rows),
        graphs=graphs,
    )


def build_adjacency_tensor(
    graphs: dict[date, nx.Graph],
    *,
    min_nodes: int = 2,
    progress: ProgressCallback | None = None,
) -> tuple[list[date], list[str], np.ndarray]:
    """Stack per-window graphs into a binary (T, n, n) adjacency tensor.

    Restricts to the intersection of node sets across all collected windows
    (computed, not assumed) and uses binary edges rather than the continuous
    dissimilarity 'weight' attribute (matches graspologic's RDPG/ASE assumptions).

    Args:
        graphs: window_end -> nx.Graph, as returned in EvolutionMetricsResult.graphs.
        min_nodes: Minimum common-node count required (raises otherwise).

    Returns:
        (window_ends, common_nodes, tensor): window_ends is the sorted list of
        window-end dates (tensor axis-0 order); common_nodes is the sorted node
        names shared by every graph (tensor axis-1/2 order); tensor has shape
        (T, n, n), dtype float64, values in {0.0, 1.0}.

    Raises:
        ValueError: If fewer than min_nodes nodes are common to every window,
            or fewer than 2 windows are available.
    """
    window_ends = sorted(graphs.keys())
    if len(window_ends) < 2:
        raise ValueError("Need at least 2 windows to build an adjacency tensor.")
    node_sets = [set(graphs[we].nodes()) for we in window_ends]
    common_nodes = sorted(set.intersection(*node_sets))
    if len(common_nodes) < min_nodes:
        raise ValueError(
            f"Only {len(common_nodes)} common nodes across all windows -- "
            f"too few for community detection."
        )
    # Built as a loop rather than a comprehension so ``progress`` fires per
    # window: callers hang their cancellation check on it, and stacking many
    # windows is otherwise a multi-second stretch with no way to interrupt.
    mats = []
    for idx, we in enumerate(window_ends):
        if progress is not None:
            progress(idx, len(window_ends), f"adjacency {we}")
        mats.append(nx.to_numpy_array(graphs[we], nodelist=common_nodes, weight=None))
    tensor = np.stack(mats)
    return window_ends, common_nodes, tensor


def _kmeans_labels(
    embedding: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster ASE coordinates with a hard k (sklearn KMeans).

    Args:
        embedding: (n, d) latent positions.
        n_clusters: Requested community count; clipped to ``[2, n]``.
        random_state: Seed for reproducibility.

    Returns:
        Integer label per row of ``embedding``.
    """
    k = max(2, min(int(n_clusters), len(embedding)))
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(
        embedding
    )


def _optimal_k_silhouette(
    embedding: np.ndarray,
    max_clusters: int,
    random_state: int = 0,
) -> tuple[int, float]:
    """Find optimal k by maximizing average silhouette coefficient in latent space.

    Args:
        embedding: (n, d) ASE-embedded coordinates.
        max_clusters: Upper bound for k search.
        random_state: Seed for reproducibility.

    Returns:
        (optimal_k, best_score).
    """
    best_k = 2
    best_score = -1.0
    for k in range(2, min(max_clusters + 1, len(embedding))):
        labels = _kmeans_labels(embedding, k, random_state)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(embedding, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def _optimal_k_modularity(
    adjacency: np.ndarray,
    embedding: np.ndarray,
    max_clusters: int,
    random_state: int = 0,
) -> tuple[int, float]:
    """Find optimal k by maximizing modularity in the original network.

    Args:
        adjacency: (n, n) binary adjacency matrix.
        embedding: (n, d) ASE-embedded coordinates (used for kmeans clustering).
        max_clusters: Upper bound for k search.
        random_state: Seed for reproducibility.

    Returns:
        (optimal_k, best_modularity).
    """
    G = nx.Graph(adjacency)
    best_k = 2
    best_mod = -1.0
    for k in range(2, min(max_clusters + 1, len(embedding))):
        labels = _kmeans_labels(embedding, k, random_state)
        if len(np.unique(labels)) < 2:
            continue
        # Convert labels to partition of node lists
        partition = [[] for _ in range(k)]
        for node_idx, label in enumerate(labels):
            partition[label].append(node_idx)
        partition = [p for p in partition if p]  # Remove empty communities
        if len(partition) < 2:
            continue
        try:
            mod = nx.algorithms.community.modularity(G, partition)
            if mod > best_mod:
                best_mod = mod
                best_k = len(partition)
        except Exception:
            pass
    return best_k, best_mod


def _optimal_k_davies_bouldin(
    embedding: np.ndarray,
    max_clusters: int,
    random_state: int = 0,
) -> tuple[int, float]:
    """Find optimal k by minimizing Davies-Bouldin index in latent space.

    Lower DB index is better (indicates better-separated, more compact clusters).

    Args:
        embedding: (n, d) ASE-embedded coordinates.
        max_clusters: Upper bound for k search.
        random_state: Seed for reproducibility.

    Returns:
        (optimal_k, best_score) where best_score is Davies-Bouldin index (lower is better).
    """
    best_k = 2
    best_score = float("inf")
    for k in range(2, min(max_clusters + 1, len(embedding))):
        labels = _kmeans_labels(embedding, k, random_state)
        if len(np.unique(labels)) < 2:
            continue
        score = davies_bouldin_score(embedding, labels)
        if score < best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def _optimal_k_calinski_harabasz(
    embedding: np.ndarray,
    max_clusters: int,
    random_state: int = 0,
) -> tuple[int, float]:
    """Find optimal k by maximizing Calinski-Harabasz index in latent space.

    Higher CH index is better (indicates better between-to-within-cluster variance ratio).

    Args:
        embedding: (n, d) ASE-embedded coordinates.
        max_clusters: Upper bound for k search.
        random_state: Seed for reproducibility.

    Returns:
        (optimal_k, best_score).
    """
    best_k = 2
    best_score = -1.0
    for k in range(2, min(max_clusters + 1, len(embedding))):
        labels = _kmeans_labels(embedding, k, random_state)
        if len(np.unique(labels)) < 2:
            continue
        score = calinski_harabasz_score(embedding, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def compute_window_communities(
    adjacency: np.ndarray,
    *,
    max_clusters: int = 10,
    ase_n_components: int | None = None,
    random_state: int = 0,
    method: CommunityMethod = CommunityMethod.FIXED,
) -> tuple[np.ndarray, int, float]:
    """ASE + KMeans community detection with configurable k-selection strategy.

    Each window independently determines its optimal k based only on that window's
    data (no lookahead bias). Method determines how k is selected.

    Args:
        adjacency: Binary (n, n) adjacency matrix.
        max_clusters: For FIXED method, exact k; for other methods, upper bound for k search.
        ase_n_components: Latent dimension for ASE (if None, uses sqrt(n)).
        random_state: Seed for reproducibility.
        method: Community detection optimization method (CommunityMethod enum).

    Returns:
        (labels, selected_k, score): cluster assignments, number of communities
        detected, and optimization score for the method (higher/lower depending on method).
    """
    if ase_n_components is None:
        ase_n_components = max(2, int(np.sqrt(adjacency.shape[0])))
    # check_lcc=False: pruned correlation-network windows are routinely
    # disconnected by design (independent_threshold removes weak edges), so the
    # "not fully connected" warning is expected noise here, not a data problem.
    # It only gates the warning -- ASE still embeds the full adjacency either way.
    embedding = AdjacencySpectralEmbed(
        n_components=ase_n_components, check_lcc=False
    ).fit_transform(adjacency)
    if isinstance(embedding, tuple):
        embedding = np.hstack(embedding)

    if method == CommunityMethod.FIXED:
        labels = _kmeans_labels(embedding, max_clusters, random_state)
        selected_k = int(len(np.unique(labels)))
        score = float(silhouette_score(embedding, labels)) if selected_k >= 2 else -1.0
    elif method == CommunityMethod.SILHOUETTE:
        selected_k, score = _optimal_k_silhouette(embedding, max_clusters, random_state)
        labels = _kmeans_labels(embedding, selected_k, random_state)
    elif method == CommunityMethod.MODULARITY:
        selected_k, score = _optimal_k_modularity(
            adjacency, embedding, max_clusters, random_state
        )
        labels = _kmeans_labels(embedding, selected_k, random_state)
    elif method == CommunityMethod.DAVIES_BOULDIN:
        selected_k, score = _optimal_k_davies_bouldin(
            embedding, max_clusters, random_state
        )
        labels = _kmeans_labels(embedding, selected_k, random_state)
    elif method == CommunityMethod.CALINSKI_HARABASZ:
        selected_k, score = _optimal_k_calinski_harabasz(
            embedding, max_clusters, random_state
        )
        labels = _kmeans_labels(embedding, selected_k, random_state)
    else:
        raise ValueError(f"Unknown community method: {method}")

    return labels, selected_k, score


def compute_community_metrics(
    graphs: dict[date, nx.Graph],
    *,
    max_clusters: int = 10,
    min_nodes: int = 2,
    random_state: int = 0,
    community_method: CommunityMethod = CommunityMethod.FIXED,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> pl.DataFrame:
    """Detect per-window communities (ASE+KMeans) across a common node set.

    Each window independently selects its community count based on the method
    specified, ensuring no lookahead bias: k for window t depends only on data
    from window t, not future windows.

    Args:
        graphs: window_end -> nx.Graph, as returned in EvolutionMetricsResult.graphs.
        max_clusters: For FIXED method, exact k per window; for other methods,
            upper bound when searching for optimal k per window.
        min_nodes: Minimum common-node count required.
        random_state: Seed for reproducibility.
        community_method: Strategy for selecting k per window (CommunityMethod enum).
        progress: Progress callback (done, total, desc).
        status: Status message callback.

    Returns:
        Long-format DataFrame: window_idx, window_end, node, community (int).

    Raises:
        ValueError: If too few common nodes or windows are available (propagated
            from build_adjacency_tensor).
    """
    if status is not None:
        status("Building common-node adjacency tensor...")
    window_ends, common_nodes, tensor = build_adjacency_tensor(
        graphs, min_nodes=min_nodes, progress=progress
    )

    n_windows = len(window_ends)
    rows = []
    for t in range(n_windows):
        if progress is not None:
            progress(t, n_windows, f"{window_ends[t]}")
        labels, selected_k, score = compute_window_communities(
            tensor[t],
            max_clusters=max_clusters,
            random_state=random_state,
            method=community_method,
        )
        for node_idx, node_name in enumerate(common_nodes):
            rows.append(
                {
                    "window_idx": t,
                    "window_end": window_ends[t],
                    "node": node_name,
                    "community": int(labels[node_idx]),
                }
            )

    if progress is not None:
        progress(n_windows, n_windows, "done")

    return pl.DataFrame(rows)


def get_top_variable_nodes(
    node_metrics: pl.DataFrame, metric: str, k: int = 10
) -> list[str]:
    """Return k node names with highest across-window std for metric."""
    return (
        node_metrics.filter(pl.col("metric") == metric)
        .group_by("node")
        .agg(pl.col("value").std().alias("std"))
        .sort("std", descending=True)
        .head(k)
        .get_column("node")
        .to_list()
    )
