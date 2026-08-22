"""Wide-format pivoting and NetworkX construction from a connection matrix."""

from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl


def pivot_to_wide(
    df: pl.DataFrame,
    date_column: str,
    name_column: str,
    value_column: str,
) -> pl.DataFrame:
    """Pivot long data to wide format: dates as rows, nodes as columns."""
    pivoted = df.pivot(on=name_column, index=date_column, values=value_column).sort(
        date_column
    )
    node_cols = sorted(col for col in pivoted.columns if col != date_column)
    return pivoted.select([date_column, *node_cols])


def build_corr_nx(
    measure_df: pl.DataFrame,
    independent_threshold: float = 0.33,
) -> nx.Graph:
    """Build a similarity network from a square connection-measure matrix.

    Measure values near 1 mean strong dependence. Edges are stored as
    ``1 - measure`` so strongly related nodes sit closer in force layouts.
    Weak edges (measure < independent_threshold), self-loops, and NaN edges are removed.

    Args:
        measure_df: Square (n, n) measure matrix with rows/cols as node names.
        independent_threshold: Keep edges where measure >= threshold
            (e.g., 0.4 means keep correlation >= 0.4 or distance-correlation >= 0.4).

    Returns:
        Undirected NetworkX Graph with weak edges pruned.
    """
    cor_matrix = measure_df.to_numpy().astype(float)

    # Replace NaN values with 0 (no correlation) to avoid issues in rendering
    cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)

    sim_matrix = 1.0 - cor_matrix
    G = nx.from_numpy_array(sim_matrix)
    node_names = np.array(measure_df.columns)
    G = nx.relabel_nodes(G, lambda x: node_names[x])
    H = G.copy()
    for u, v, wt in G.edges.data("weight"):
        # Remove weak edges and self-loops
        # wt = 1 - measure, so to keep edges where measure >= threshold,
        # we remove edges where wt > 1 - threshold (i.e., measure < threshold)
        # NaN handling: wt should now be finite (converted above)
        if not np.isfinite(wt) or wt > 1.0 - independent_threshold or u == v:
            H.remove_edge(u, v)
    return H
