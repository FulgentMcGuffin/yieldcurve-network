"""Polars tables for a ``(node, layer)`` multiplex NetworkX graph.

Vendored from https://github.com/FulgentMcGuffin/MultiLayerNetViz
(``multiplex_data.py``), unmodified apart from this docstring.

Vendored rather than imported because the published ``multilayer-netviz``
package ships only ``multi_layer_NetViz_fcts`` -- this module is not part of
its ``py-modules`` list and is therefore not importable from the installed
distribution.

The column names ``issuer``/``term`` are historical (the original use case was
a yield-curve multiplex). They are generic: ``issuer`` is whatever the GUI's
node-name column is, ``term`` is whatever layer column the user selected.
"""

from __future__ import annotations

import networkx as nx
import polars as pl


def _scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _node_pair(node) -> tuple[str, str] | None:
    if not (isinstance(node, tuple) and len(node) >= 2):
        return None
    return str(_scalar(node[0])), str(_scalar(node[1]))


def multiplex_tables(
    G: nx.Graph,
    terms: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (nodes, intra_edges, inter_edges) as Polars frames.

    Nodes are ``(issuer, term)`` tuples. Intra edges share a term
    (``layer != "inter"``). Inter edges use ``layer == "inter"``.
    """
    term_filter = set(terms) if terms is not None else None

    node_rows: list[dict] = []
    for node, attrs in G.nodes(data=True):
        pair = _node_pair(node)
        if pair is None:
            continue
        issuer, term = pair
        if term_filter is not None and term not in term_filter:
            continue
        row = {"issuer": issuer, "term": term}
        for key, val in attrs.items():
            row[str(key)] = _scalar(val)
        node_rows.append(row)

    intra_rows: list[dict] = []
    inter_rows: list[dict] = []
    for u, v, data in G.edges(data=True):
        u_pair, v_pair = _node_pair(u), _node_pair(v)
        if u_pair is None or v_pair is None:
            continue
        u_iss, u_term = u_pair
        v_iss, v_term = v_pair
        weight = float(data.get("weight", 0.5))
        extra = {
            str(k): _scalar(val)
            for k, val in data.items()
            if k not in ("weight", "layer")
        }
        layer = data.get("layer")
        if layer == "inter":
            if term_filter is not None and (
                u_term not in term_filter or v_term not in term_filter
            ):
                continue
            inter_rows.append(
                {
                    "source_issuer": u_iss,
                    "target_issuer": v_iss,
                    "term_from": u_term,
                    "term_to": v_term,
                    "weight": weight,
                    **extra,
                }
            )
        else:
            term = str(_scalar(data.get("term", u_term)))
            if term_filter is not None and term not in term_filter:
                continue
            intra_rows.append(
                {
                    "source_issuer": u_iss,
                    "target_issuer": v_iss,
                    "term": term,
                    "weight": weight,
                    **extra,
                }
            )

    nodes = (
        pl.DataFrame(node_rows)
        if node_rows
        else pl.DataFrame(schema={"issuer": pl.Utf8, "term": pl.Utf8})
    )
    intra = (
        pl.DataFrame(intra_rows)
        if intra_rows
        else pl.DataFrame(
            schema={
                "source_issuer": pl.Utf8,
                "target_issuer": pl.Utf8,
                "term": pl.Utf8,
                "weight": pl.Float64,
            }
        )
    )
    inter = (
        pl.DataFrame(inter_rows)
        if inter_rows
        else pl.DataFrame(
            schema={
                "source_issuer": pl.Utf8,
                "target_issuer": pl.Utf8,
                "term_from": pl.Utf8,
                "term_to": pl.Utf8,
                "weight": pl.Float64,
            }
        )
    )
    return nodes, intra, inter


def available_terms(nodes: pl.DataFrame) -> list[str]:
    if nodes.is_empty() or "term" not in nodes.columns:
        return []
    return sorted(nodes.get_column("term").unique().to_list())


def filter_tables(
    nodes: pl.DataFrame,
    intra: pl.DataFrame,
    inter: pl.DataFrame,
    terms: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    term_list = list(terms)
    nodes_f = (
        nodes.filter(pl.col("term").is_in(term_list)) if not nodes.is_empty() else nodes
    )
    intra_f = (
        intra.filter(pl.col("term").is_in(term_list)) if not intra.is_empty() else intra
    )
    if inter.is_empty():
        inter_f = inter
    else:
        inter_f = inter.filter(
            pl.col("term_from").is_in(term_list) & pl.col("term_to").is_in(term_list)
        )
    return nodes_f, intra_f, inter_f
