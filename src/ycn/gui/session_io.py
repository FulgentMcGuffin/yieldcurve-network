"""Convert between the GUI's result objects and a saved :mod:`session` archive.

Kept out of ``main_window`` so the round-trip is testable on its own, and out
of the analysis layer because the result dataclasses are GUI-side.

Figures are rebuilt here on load rather than stored: they are pure functions of
the frames plus the settings, so re-rendering is exact and the archive stays
free of pickled objects.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import polars as pl

from ycn.analysis.mln_evolution_viz import (
    render_edge_evolution,
    render_factor_evolution,
    render_stress_quadrants,
)
from ycn.analysis.mln_viz import render_mln_communities, render_mln_metrics
from ycn.analysis.session import Session
from ycn.gui.workers import (
    MLNEvolutionResult,
    MLNResult,
    NeuralEvolutionResult,
    ResidualResult,
)

MLN = "mln"
RESIDUAL = "residual"
EVOLUTION = "evolution"
NEURAL_EVOLUTION = "neural_evolution"


# ------------------------------------------------------------------- capture
def capture_mln(session: Session, result: MLNResult) -> None:
    """Store an :class:`MLNResult` into ``session``."""
    session.scalars[MLN] = {
        "layer_values": list(result.layer_values),
        "layer_column": result.layer_column,
        "node_column": result.node_column,
        "centrality": result.centrality,
        "n_intra_edges": int(result.n_intra_edges),
        "n_inter_edges": int(result.n_inter_edges),
    }
    session.frames.update(
        {
            f"{MLN}.nodes": result.nodes,
            f"{MLN}.intra": result.intra,
            f"{MLN}.inter": result.inter,
            f"{MLN}.edge": result.edge_df,
            f"{MLN}.centrality": result.centrality_df,
            f"{MLN}.community": result.community_df,
        }
    )


def capture_residual(session: Session, result: ResidualResult) -> None:
    """Store a :class:`ResidualResult` into ``session``."""
    session.scalars[RESIDUAL] = {
        "label_column": result.label_column,
        "label_order": list(result.label_order),
        "node_label": result.node_label,
        "issuer_column": result.issuer_column,
        "skipped": list(result.skipped),
    }
    session.frames.update(
        {
            f"{RESIDUAL}.metrics": result.metrics,
            f"{RESIDUAL}.coverage": result.coverage,
        }
    )


def capture_evolution(session: Session, result: MLNEvolutionResult) -> None:
    """Store an :class:`MLNEvolutionResult` into ``session``."""
    session.scalars[EVOLUTION] = {}
    session.frames.update(
        {
            f"{EVOLUTION}.edge_types": result.edge_types,
            f"{EVOLUTION}.community_k": result.community_k,
            f"{EVOLUTION}.communities": result.communities,
            f"{EVOLUTION}.factors": result.factors,
            f"{EVOLUTION}.regimes": result.regimes,
            f"{EVOLUTION}.stress": result.stress,
        }
    )


def capture_neural_evolution(session: Session, result: NeuralEvolutionResult) -> None:
    """Store a :class:`NeuralEvolutionResult` into ``session``."""
    session.scalars[NEURAL_EVOLUTION] = {}
    session.frames.update(
        {
            f"{NEURAL_EVOLUTION}.factors": result.factors,
            f"{NEURAL_EVOLUTION}.regimes": result.regimes,
            f"{NEURAL_EVOLUTION}.stress": result.stress,
        }
    )


# ------------------------------------------------------------------ restore
def _rebuild_multiplex(
    nodes: pl.DataFrame, intra: pl.DataFrame, inter: pl.DataFrame
) -> nx.Graph:
    """Reconstruct the multiplex from its edge tables.

    The tables are a faithful record of the graph, so this is exact rather than
    an approximation. Nothing in the GUI reads ``MLNResult.graph`` today, but
    leaving the field empty would make a restored result quietly different from
    a freshly computed one.
    """
    graph = nx.Graph()
    if not nodes.is_empty():
        for issuer, term in nodes.select(["issuer", "term"]).iter_rows():
            graph.add_node((str(issuer), str(term)))
    for row in intra.iter_rows(named=True):
        graph.add_edge(
            (str(row["source_issuer"]), str(row["term"])),
            (str(row["target_issuer"]), str(row["term"])),
            weight=float(row.get("weight", 0.5)),
            layer="intra",
            term=str(row["term"]),
        )
    for row in inter.iter_rows(named=True):
        graph.add_edge(
            (str(row["source_issuer"]), str(row["term_from"])),
            (str(row["target_issuer"]), str(row["term_to"])),
            weight=float(row.get("weight", 1.0)),
            layer="inter",
            node=str(row["source_issuer"]),
        )
    return graph


def restore_mln(session: Session) -> MLNResult | None:
    """Rebuild the MLN result, re-rendering its two figures."""
    scalars = session.scalars.get(MLN)
    if scalars is None:
        return None
    nodes = session.frame(f"{MLN}.nodes")
    intra = session.frame(f"{MLN}.intra")
    inter = session.frame(f"{MLN}.inter")
    edge_df = session.frame(f"{MLN}.edge")
    centrality_df = session.frame(f"{MLN}.centrality")
    community_df = session.frame(f"{MLN}.community")
    layer_column = scalars.get("layer_column", "layer")
    node_column = scalars.get("node_column", "node")
    centrality = scalars.get("centrality", "eigenvector")

    return MLNResult(
        graph=_rebuild_multiplex(nodes, intra, inter),
        nodes=nodes,
        intra=intra,
        inter=inter,
        layer_values=list(scalars.get("layer_values", [])),
        layer_column=layer_column,
        node_column=node_column,
        centrality=centrality,
        n_intra_edges=int(scalars.get("n_intra_edges", 0)),
        n_inter_edges=int(scalars.get("n_inter_edges", 0)),
        edge_df=edge_df,
        centrality_df=centrality_df,
        community_df=community_df,
        metrics_fig=render_mln_metrics(
            edge_df,
            centrality_df,
            layer_label=layer_column,
            node_label=node_column,
            centrality_name=centrality,
        ),
        community_fig=render_mln_communities(
            community_df, layer_label=layer_column, node_label=node_column
        ),
    )


def restore_residual(session: Session) -> ResidualResult | None:
    """Rebuild the residual result. Its chart is drawn by the tab on demand."""
    scalars = session.scalars.get(RESIDUAL)
    if scalars is None:
        return None
    return ResidualResult(
        metrics=session.frame(f"{RESIDUAL}.metrics"),
        coverage=session.frame(f"{RESIDUAL}.coverage"),
        label_column=scalars.get("label_column", "label"),
        label_order=list(scalars.get("label_order", [])),
        node_label=scalars.get("node_label", "node"),
        issuer_column=scalars.get("issuer_column", "issuer"),
        skipped=list(scalars.get("skipped", [])),
    )


def restore_evolution(session: Session) -> MLNEvolutionResult | None:
    """Rebuild the evolution result, re-rendering its four figures."""
    if EVOLUTION not in session.scalars:
        return None
    edge_types = session.frame(f"{EVOLUTION}.edge_types")
    community_k = session.frame(f"{EVOLUTION}.community_k")
    factors = session.frame(f"{EVOLUTION}.factors")
    regimes = session.frame(f"{EVOLUTION}.regimes")
    stress = session.frame(f"{EVOLUTION}.stress")

    return MLNEvolutionResult(
        edge_types=edge_types,
        community_k=community_k,
        communities=session.frame(f"{EVOLUTION}.communities"),
        factors=factors,
        regimes=regimes,
        stress=stress,
        links_fig=render_edge_evolution(edge_types, community_k),
        factor_fig=render_factor_evolution(factors, regimes, std=False),
        factor_std_fig=render_factor_evolution(factors, regimes, std=True),
        stress_fig=render_stress_quadrants(stress),
    )


def restore_neural_evolution(session: Session) -> NeuralEvolutionResult | None:
    """Rebuild the Neural-HJM evolution result, re-rendering its three figures."""
    if NEURAL_EVOLUTION not in session.scalars:
        return None
    factors = session.frame(f"{NEURAL_EVOLUTION}.factors")
    regimes = session.frame(f"{NEURAL_EVOLUTION}.regimes")
    stress = session.frame(f"{NEURAL_EVOLUTION}.stress")

    return NeuralEvolutionResult(
        factors=factors,
        regimes=regimes,
        stress=stress,
        factor_fig=render_factor_evolution(factors, regimes, std=False),
        factor_std_fig=render_factor_evolution(factors, regimes, std=True),
        stress_fig=render_stress_quadrants(stress),
    )


def settings_summary(settings: dict[str, Any]) -> str:
    """Short human description of what produced a session."""
    bits = [
        settings.get("table") or "?",
        settings.get("network_kind_label") or settings.get("network_kind") or "?",
        settings.get("measure") or "?",
    ]
    start, end = settings.get("date_start"), settings.get("date_end")
    if start and end:
        bits.append(f"{start} → {end}")
    return " · ".join(str(b) for b in bits)
