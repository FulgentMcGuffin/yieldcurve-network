"""Per-layer views of the multiplex: degree distributions and centrality paths.

The MLN tabs describe the multiplex *as a whole* -- edge composition across
layers, one centrality heatmap, one community map. These two helpers go the
other way and describe each **component network** on its own terms, which is
what the "MLN: Degree" and "MLN: Centrality" tabs chart.

Degree histograms are rebuilt from the multiplex edge tables rather than from
the live ``layer_graphs`` dict. Those tables are a faithful record of the graph
and are the only form that survives a session save/load, so a restored analysis
gets the same histograms as a fresh one without storing anything extra.

Centrality *trajectories* need windows, so they cannot come from the single
whole-range multiplex -- they are produced by the rolling-window evolution pass
(``mln_evolution.compute_multiplex_evolution``), which already has each
window's layer graphs in hand.
"""

from __future__ import annotations

from datetime import date

import networkx as nx
import polars as pl

from .evolution import _add_strength_attr, _node_centrality

# Long-format schema of the per-layer, per-window metric frame. Matches what
# ``evolution_viz.render_centrality_trajectories`` consumes, plus ``layer``.
LAYER_METRICS_SCHEMA = {
    "layer": pl.Utf8,
    "window_end": pl.Date,
    "node": pl.Utf8,
    "metric": pl.Utf8,
    "value": pl.Float64,
}


def layer_subgraph(nodes: pl.DataFrame, intra: pl.DataFrame, layer: str) -> nx.Graph:
    """The component network for one layer, rebuilt from the multiplex tables.

    Node identity is the plain node name (not the ``(node, layer)`` pair the
    multiplex uses), because a degree histogram of one layer is a statement
    about that layer's own nodes.

    Isolated nodes are kept: a node that is present in the layer but connects
    to nothing has degree 0, and dropping it would quietly bias the
    distribution away from exactly the nodes worth noticing.
    """
    graph = nx.Graph()
    if not nodes.is_empty() and {"issuer", "term"} <= set(nodes.columns):
        present = nodes.filter(pl.col("term").cast(pl.Utf8) == layer)
        for issuer in present.get_column("issuer").to_list():
            graph.add_node(str(issuer))
    if intra.is_empty() or "term" not in intra.columns:
        return graph
    edges = intra.filter(pl.col("term").cast(pl.Utf8) == layer)
    for row in edges.iter_rows(named=True):
        graph.add_edge(
            str(row["source_issuer"]),
            str(row["target_issuer"]),
            weight=float(row.get("weight", 0.5)),
        )
    return graph


def layer_degree_graphs(
    nodes: pl.DataFrame, intra: pl.DataFrame, layer_values: list[str]
) -> dict[str, nx.Graph]:
    """``layer -> component network``, in the caller's layer order."""
    return {layer: layer_subgraph(nodes, intra, layer) for layer in layer_values}


def layer_node_metric_rows(
    layer_graphs: dict[str, nx.Graph],
    window_end: date,
    *,
    centrality: str,
) -> list[dict]:
    """Long-format centrality/weighted-degree rows for one window's layers.

    Mirrors ``evolution._node_summary`` -- same two metrics, same column names
    -- so the per-layer frame can be handed straight to the same trajectory
    renderer the single-network evolution uses, just filtered by layer first.
    """
    rows: list[dict] = []
    for layer, graph in layer_graphs.items():
        if graph.number_of_nodes() == 0:
            continue
        _add_strength_attr(graph)
        try:
            cent = _node_centrality(graph, centrality)
        except Exception:  # noqa: BLE001 -- one bad layer must not stop the pass
            continue
        weighted_degree = dict(graph.degree(weight="strength"))
        for node in graph.nodes():
            rows.append(
                {
                    "layer": str(layer),
                    "window_end": window_end,
                    "node": str(node),
                    "metric": "weighted_degree",
                    "value": float(weighted_degree.get(node, 0.0)),
                }
            )
            rows.append(
                {
                    "layer": str(layer),
                    "window_end": window_end,
                    "node": str(node),
                    "metric": centrality,
                    "value": float(cent.get(node, float("nan"))),
                }
            )
    return rows


def layer_metrics_frame(rows: list[dict]) -> pl.DataFrame:
    """Build the per-layer metric frame, typed even when empty."""
    if not rows:
        return pl.DataFrame(schema=LAYER_METRICS_SCHEMA)
    return pl.DataFrame(rows, schema=LAYER_METRICS_SCHEMA)


def metrics_for_layer(layer_metrics: pl.DataFrame, layer: str) -> pl.DataFrame:
    """One layer's rows, shaped exactly as the trajectory renderer expects.

    Drops the ``layer`` column: ``render_centrality_trajectories`` takes a
    single network's ``(window_end, node, metric, value)`` frame and would
    otherwise treat the extra column as an unknown metric dimension.
    """
    if layer_metrics.is_empty() or "layer" not in layer_metrics.columns:
        return pl.DataFrame(
            schema={k: v for k, v in LAYER_METRICS_SCHEMA.items() if k != "layer"}
        )
    return layer_metrics.filter(pl.col("layer") == layer).drop("layer")
