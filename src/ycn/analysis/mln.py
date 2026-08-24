"""Multi-layer (multiplex) network construction and metrics.

One layer per distinct value of ``layer_column``; nodes come from
``name_column``. Every layer's network is built with the same measure, date
range, transforms, filter and threshold, so layers are directly comparable to
each other rather than each being its own analysis.

No temporal evolution here -- this is a single snapshot over the configured
date range.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import networkx as nx
import polars as pl

from .config import PipelineConfig
from .data_access import load_table
from .evolution import CommunityMethod, _add_strength_attr, _node_centrality
from .measures import compute_measure
from .multilayer_communities import align_community_labels, detect_layer_communities
from .network import build_corr_nx, pivot_to_wide
from .transforms import apply_transforms
from .yield_curve import sort_terms

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]

# Per-layer progress is reported on a 0..PROGRESS_SCALE sub-range so the global
# bar advances smoothly across layers instead of restarting at each one.
PROGRESS_SCALE = 100


@dataclass
class MLNConfig:
    """User choices for the multi-layer network workstream.

    Args:
        layer_column: Column whose distinct values become layers.
        centrality: Per-node centrality for the Metrics tab heatmap.
        jaccard_threshold: Minimum Jaccard overlap for two per-layer
            communities to be considered the same community across layers.
        community_method: k-selection strategy, per layer.
        max_communities: Exact k for FIXED; search upper bound otherwise. Also
            the ceiling on the total distinct ids the MLN: Community tab shows
            after cross-layer alignment (see ``layer_community_matrix``).
        min_nodes: Layers with fewer nodes are skipped for community detection.
    """

    layer_column: str
    centrality: str = "eigenvector"
    jaccard_threshold: float = 0.6
    community_method: CommunityMethod = CommunityMethod.FIXED
    max_communities: int = 10
    min_nodes: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.community_method, str):
            self.community_method = CommunityMethod(self.community_method)


def prepare_mln_frame(
    cfg: PipelineConfig,
    layer_column: str,
    *,
    status: StatusCallback | None = None,
) -> pl.DataFrame:
    """Load and filter the long frame for MLN work, WITHOUT applying transforms.

    Mirrors ``pipeline._prepare_frame`` except that transforms are deliberately
    left for :func:`build_layer_graphs` to apply *per layer*. Applying them here
    would be wrong: ``daily_returns`` does ``pct_change().over(name_column)``,
    so for a node present in two layers a globally-transformed frame would
    compute returns across interleaved rows from different layers.
    """
    cols = [cfg.date_column, cfg.name_column, cfg.value_column, layer_column]
    cols = list(dict.fromkeys(c for c in cols if c))

    if status is not None and cfg.where_clause:
        status(f"MLN: loading with SQL filter: WHERE {cfg.where_clause}")
    df = load_table(cfg.db_path, cfg.table, columns=cols, where_clause=cfg.where_clause)

    df = df.with_columns(pl.col(cfg.date_column).cast(pl.Date, strict=False))
    df = df.sort(cfg.date_column, cfg.name_column)

    if cfg.date_start is not None:
        df = df.filter(pl.col(cfg.date_column) >= cfg.date_start)
    if cfg.date_end is not None:
        df = df.filter(pl.col(cfg.date_column) <= cfg.date_end)

    return df.drop_nulls(
        subset=[cfg.date_column, cfg.name_column, cfg.value_column, layer_column]
    )


def layer_values_of(df: pl.DataFrame, layer_column: str) -> list[str]:
    """Distinct layer values, as strings, in display order.

    Maturity terms are ordered by maturity rather than alphabetically -- with
    a plain sort a term-layered multiplex would stack as 0.5Y, 10Y, 1Y, 20Y,
    which misreads as a yield curve in every layered plot. ``sort_terms``
    falls back to alphabetical for non-term labels, so issuer layers are
    unaffected.
    """
    if df.is_empty() or layer_column not in df.columns:
        return []
    values = df.get_column(layer_column).cast(pl.Utf8).unique().to_list()
    return sort_terms(v for v in values if v is not None)


def build_layer_graphs(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    layer_column: str,
    *,
    layer_values: list[str] | None = None,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
    edge_settings: dict | None = None,
) -> dict[str, nx.Graph]:
    """One correlation network per layer, using the single network's settings.

    Per layer: subset -> transform -> pivot wide -> measure -> threshold. The
    subset-then-transform ordering is required for correctness (see
    :func:`prepare_mln_frame`) and also because ``pivot_to_wide`` needs
    ``(date, name)`` unique, which in a layered table only holds within a layer.
    """
    values = (
        layer_values if layer_values is not None else layer_values_of(df, layer_column)
    )
    graphs: dict[str, nx.Graph] = {}
    n_layers = max(len(values), 1)

    for idx, value in enumerate(values):
        if status is not None:
            status(f"MLN: building layer {idx + 1}/{len(values)} ({value})")

        sub = df.filter(pl.col(layer_column).cast(pl.Utf8) == value)
        if cfg.transforms:
            sub = apply_transforms(
                sub,
                cfg.transforms,
                cfg.date_column,
                cfg.name_column,
                [cfg.value_column],
            )
        if sub.height < 3:
            graphs[value] = nx.Graph()
            continue

        wide = pivot_to_wide(
            sub,
            date_column=cfg.date_column,
            name_column=cfg.name_column,
            value_column=cfg.value_column,
        )
        nodes = [c for c in wide.columns if c != cfg.date_column]
        if len(nodes) < 2:
            graphs[value] = nx.Graph()
            continue

        # Map this layer's per-pair progress onto its slice of the global bar,
        # and forward every tick so cancellation latency stays at one pair.
        def _layer_progress(done: int, total: int, desc: str, _i: int = idx) -> None:
            if progress is None:
                return
            frac = (done / total) if total else 1.0
            scaled = int(_i * PROGRESS_SCALE + frac * PROGRESS_SCALE)
            progress(scaled, n_layers * PROGRESS_SCALE, f"{value}: {desc}")

        measure_df = compute_measure(
            cfg.measure,
            wide.select(nodes),
            nodes,
            progress=_layer_progress,
            edge_settings=edge_settings,
        )
        graphs[value] = build_corr_nx(
            measure_df, independent_threshold=cfg.independent_threshold
        )

    if progress is not None:
        progress(n_layers * PROGRESS_SCALE, n_layers * PROGRESS_SCALE, "layers done")
    return graphs


def build_multilayer_network(
    layer_graphs: Mapping[str, nx.Graph],
    layer_values: list[str],
    *,
    inter_layer_weight: float = 1.0,
) -> nx.Graph:
    """Assemble per-layer graphs into one multiplex graph.

    Node ids are ``(node, layer)`` tuples. Intra-layer edges carry the layer's
    correlation strength (``1 - distance``) and ``layer="intra"``; the layer
    value is stored under the key ``term`` because the vendored
    ``multiplex_data``/``multiplex_plotly`` modules read that key. Inter-layer
    edges connect a node across every pair of layers it appears in -- unlike
    maturities, arbitrary layer values have no natural adjacency order.
    """
    M = nx.Graph()
    for layer in layer_values:
        graph = layer_graphs.get(layer)
        if graph is None:
            continue
        for node in graph.nodes():
            M.add_node((node, layer))
        for u, v, data in graph.edges(data=True):
            M.add_edge(
                (u, layer),
                (v, layer),
                weight=1.0 - float(data.get("weight", 0.5)),
                layer="intra",
                term=layer,
            )

    node_layers: dict[str, set[str]] = {}
    for layer in layer_values:
        graph = layer_graphs.get(layer)
        if graph is None:
            continue
        for node in graph.nodes():
            node_layers.setdefault(node, set()).add(layer)

    for node, present in node_layers.items():
        for a, b in itertools.combinations(sorted(present), 2):
            M.add_edge(
                (node, a),
                (node, b),
                weight=inter_layer_weight,
                layer="inter",
                node=node,
            )
    return M


def count_edge_types(M: nx.Graph) -> tuple[int, int]:
    """Total (intra, inter) edge counts for the whole multiplex."""
    intra = sum(1 for _, _, d in M.edges(data=True) if d.get("layer") == "intra")
    inter = sum(1 for _, _, d in M.edges(data=True) if d.get("layer") == "inter")
    return intra, inter


def layer_edge_metrics(M: nx.Graph, layer_values: list[str]) -> pl.DataFrame:
    """Per-layer intra/inter edge counts and composition -- single snapshot.

    An inter-layer edge touches two layers and is counted under both, so the
    inter column sums to twice the multiplex inter-edge total. The charts say so
    in their subtitle.
    """
    intra_counts = dict.fromkeys(layer_values, 0)
    inter_counts = dict.fromkeys(layer_values, 0)

    for u, v, data in M.edges(data=True):
        if data.get("layer") == "inter":
            for layer in {u[1], v[1]}:
                if layer in inter_counts:
                    inter_counts[layer] += 1
        else:
            layer = data.get("term", u[1])
            if layer in intra_counts:
                intra_counts[layer] += 1

    rows = []
    for layer in layer_values:
        intra = intra_counts.get(layer, 0)
        inter = inter_counts.get(layer, 0)
        total = intra + inter
        rows.append(
            {
                "layer": layer,
                "intra_edges": intra,
                "inter_edges": inter,
                "pct_intra": (100.0 * intra / total) if total else 0.0,
                "pct_inter": (100.0 * inter / total) if total else 0.0,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "layer": pl.Utf8,
            "intra_edges": pl.Int64,
            "inter_edges": pl.Int64,
            "pct_intra": pl.Float64,
            "pct_inter": pl.Float64,
        },
    )


def layer_centrality_matrix(
    layer_graphs: Mapping[str, nx.Graph],
    layer_values: list[str],
    centrality: str,
) -> pl.DataFrame:
    """Long frame ``(node, layer, value)`` of per-layer node centrality.

    Computed on the per-layer correlation graphs rather than multiplex
    subgraphs, because ``evolution._node_centrality`` expects that graph's
    convention (``weight`` = distance, ``strength`` = 1 - weight).
    """
    rows = []
    for layer in layer_values:
        graph = layer_graphs.get(layer)
        if graph is None or graph.number_of_nodes() == 0:
            continue
        _add_strength_attr(graph)
        scores = _node_centrality(graph, centrality)
        for node, score in scores.items():
            rows.append({"node": str(node), "layer": layer, "value": float(score)})
    return pl.DataFrame(
        rows, schema={"node": pl.Utf8, "layer": pl.Utf8, "value": pl.Float64}
    )


def layer_community_matrix(
    layer_graphs: Mapping[str, nx.Graph],
    layer_values: list[str],
    cfg: MLNConfig,
) -> pl.DataFrame:
    """Long frame ``(node, layer, community)`` with cross-layer-stable ids.

    Per-layer ASE + KMeans (k chosen by ``cfg.community_method``, capped by
    ``cfg.max_communities``), then Jaccard alignment so a community id means the
    same group of nodes in every layer -- that is what makes the heatmap's
    colours comparable across layers. ``cfg.max_communities`` also bounds the
    *total* distinct ids the alignment step hands out, not just each layer's own
    cluster count -- see ``align_community_labels``.
    """
    ordered = {
        layer: layer_graphs[layer] for layer in layer_values if layer in layer_graphs
    }
    raw, _diagnostics = detect_layer_communities(
        ordered,
        cfg.community_method,
        max_clusters=cfg.max_communities,
        min_nodes=cfg.min_nodes,
    )
    aligned = align_community_labels(
        raw,
        list(ordered.keys()),
        min_jaccard=cfg.jaccard_threshold,
        max_ids=cfg.max_communities,
    )
    rows = [
        {"node": str(node), "layer": layer, "community": int(label)}
        for layer, labels in aligned.items()
        for node, label in labels.items()
    ]
    return pl.DataFrame(
        rows, schema={"node": pl.Utf8, "layer": pl.Utf8, "community": pl.Int64}
    )
