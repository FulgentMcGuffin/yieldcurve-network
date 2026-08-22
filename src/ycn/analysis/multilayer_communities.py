"""Per-layer ASE community detection with stable IDs across layers.

Each layer is treated like a single-network window: k is chosen from that
layer's adjacency only (no lookahead across terms or indices).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
from graspologic.embed import AdjacencySpectralEmbed
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .evolution import CommunityMethod, compute_window_communities

METHOD_LABELS: dict[CommunityMethod, str] = {
    CommunityMethod.FIXED: "FIXED",
    CommunityMethod.SILHOUETTE: "SILHOUETTE",
    CommunityMethod.MODULARITY: "MODULARITY",
    CommunityMethod.DAVIES_BOULDIN: "DAVIES-BOULDIN",
    CommunityMethod.CALINSKI_HARABASZ: "CALINSKI-HARABASZ",
}


def detect_multilayer_communities(
    graph: nx.Graph,
    method: CommunityMethod,
    *,
    max_clusters: int,
    ase_n_components: int | None = None,
    random_state: int = 0,
    min_nodes: int = 3,
) -> dict[str, object]:
    """ASE + KMeans on one multiplex snapshot (supra-adjacency).

    Uses the same k-selection strategies as single-window evolution analysis.

    Args:
        graph: Multiplex graph; nodes are typically ``(issuer, term)`` tuples.
        method: k-selection strategy (see ``CommunityMethod``).
        max_clusters: Exact k for FIXED; upper bound for the other methods.
        ase_n_components: ASE latent dimension (default ``sqrt(n)``).
        random_state: Seed for reproducibility.
        min_nodes: Skip graphs with fewer nodes than this.

    Returns:
        Dict with ``communities``, ``n_clusters``, ``score``, ``inertia``,
        and ``method`` (human-readable label).
    """
    empty: dict[str, object] = {
        "communities": {},
        "n_clusters": 0,
        "score": float("nan"),
        "inertia": float("nan"),
        "method": METHOD_LABELS[method],
    }
    nodes = sorted(graph.nodes())
    n_nodes = len(nodes)
    if n_nodes < min_nodes:
        return empty

    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    k_arg = max_clusters
    if method != CommunityMethod.FIXED:
        k_arg = max(2, min(max_clusters, n_nodes - 1))
    try:
        labels, selected_k, score = compute_window_communities(
            adjacency,
            max_clusters=k_arg,
            ase_n_components=ase_n_components,
            random_state=random_state,
            method=method,
        )
    except Exception as exc:
        print(f"multilayer ({method.value}): {exc}")
        return empty

    n_comp = ase_n_components
    if n_comp is None:
        n_comp = max(2, int(np.sqrt(n_nodes)))
    embedding = AdjacencySpectralEmbed(
        n_components=n_comp, check_lcc=False
    ).fit_transform(adjacency)
    if isinstance(embedding, tuple):
        embedding = np.hstack(embedding)
    k = max(2, min(int(selected_k), len(embedding)))
    inertia = float(
        KMeans(n_clusters=k, n_init=10, random_state=random_state)
        .fit(embedding)
        .inertia_
    )

    return {
        "communities": {node: int(label) for node, label in zip(nodes, labels)},
        "n_clusters": int(selected_k),
        "score": float(score),
        "inertia": inertia,
        "method": METHOD_LABELS[method],
    }


def detect_layer_communities(
    layer_graphs: Mapping[str, nx.Graph],
    method: CommunityMethod,
    *,
    max_clusters: int,
    random_state: int = 0,
    min_nodes: int = 3,
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[int, float]]]:
    """Run ASE + KMeans independently on each layer.

    Args:
        layer_graphs: Layer name -> residual graph.
        method: k-selection strategy (see ``CommunityMethod``).
        max_clusters: Exact k for FIXED; upper bound for the other methods.
        random_state: Seed for reproducibility.
        min_nodes: Skip layers with fewer nodes than this.

    Returns:
        (communities, diagnostics) where communities is layer -> node -> label
        and diagnostics is layer -> (selected_k, method_score).
    """
    communities: dict[str, dict[str, int]] = {}
    diagnostics: dict[str, tuple[int, float]] = {}
    for layer, graph in layer_graphs.items():
        nodes = sorted(graph.nodes())
        n_nodes = len(nodes)
        if n_nodes < min_nodes:
            communities[layer] = {}
            diagnostics[layer] = (0, float("nan"))
            continue
        adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
        k_cap = max(2, min(max_clusters, n_nodes - 1))
        try:
            labels, selected_k, score = compute_window_communities(
                adjacency,
                max_clusters=k_cap,
                method=method,
                random_state=random_state,
            )
        except Exception as exc:
            print(f"{layer} ({method.value}): {exc}")
            communities[layer] = {}
            diagnostics[layer] = (0, float("nan"))
            continue
        communities[layer] = {node: int(label) for node, label in zip(nodes, labels)}
        diagnostics[layer] = (int(selected_k), float(score))
    return communities, diagnostics


def _groups(labels: Mapping[str, int]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for node, cid in labels.items():
        out.setdefault(cid, set()).add(node)
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return (len(left & right) / union) if union else 0.0


def align_community_labels(
    communities: Mapping[str, Mapping[str, int]],
    layers: Sequence[str],
    *,
    min_jaccard: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Relabel per-layer clusters so overlapping groups keep a stable id.

    Each layer is matched to already-seen canonical groups by Jaccard
    similarity (Hungarian assignment). Matches below ``min_jaccard`` and
    leftover clusters receive a new id. Canonical prototypes are the
    founding member sets, so a group that splits and later reforms keeps
    its original label.
    """
    aligned: dict[str, dict[str, int]] = {}
    canonical: dict[int, set[str]] = {}
    next_id = 0

    for layer in layers:
        raw = dict(communities.get(layer) or {})
        groups = _groups(raw)
        if not groups:
            aligned[layer] = {}
            continue

        if not canonical:
            for cid, members in groups.items():
                canonical[cid] = set(members)
                next_id = max(next_id, cid + 1)
            aligned[layer] = raw
            continue

        old_ids = sorted(groups)
        can_ids = sorted(canonical)
        cost = np.ones((len(old_ids), len(can_ids)), dtype=float)
        for i, oid in enumerate(old_ids):
            for j, cid in enumerate(can_ids):
                cost[i, j] = 1.0 - _jaccard(groups[oid], canonical[cid])

        mapping: dict[int, int] = {}
        row_ind, col_ind = linear_sum_assignment(cost)
        for row, col in zip(row_ind, col_ind):
            if (1.0 - cost[row, col]) >= min_jaccard:
                mapping[old_ids[row]] = can_ids[col]

        for oid in old_ids:
            if oid in mapping:
                continue
            mapping[oid] = next_id
            canonical[next_id] = set(groups[oid])
            next_id += 1

        aligned[layer] = {node: mapping[cid] for node, cid in raw.items()}

    return aligned


def detect_all_community_methods(
    layer_graphs: Mapping[str, nx.Graph],
    *,
    max_clusters: int,
    min_jaccard: float = 0.5,
    random_state: int = 0,
) -> tuple[
    dict[str, dict[str, dict[str, int]]],
    dict[str, dict[str, tuple[int, float]]],
]:
    """Run every ``CommunityMethod`` and align labels within each method.

    Args:
        layer_graphs: Layer name -> residual graph.
        max_clusters: Exact k for FIXED; search cap for the other methods.
        min_jaccard: Alignment threshold across layers (within a method).
        random_state: Seed for reproducibility.

    Returns:
        (communities_by_method, diagnostics_by_method) keyed by method label
        (e.g. ``"SILHOUETTE"``), then layer.
    """
    layers = list(layer_graphs.keys())
    communities_by_method: dict[str, dict[str, dict[str, int]]] = {}
    diagnostics_by_method: dict[str, dict[str, tuple[int, float]]] = {}
    for method in CommunityMethod:
        raw, diagnostics = detect_layer_communities(
            layer_graphs,
            method,
            max_clusters=max_clusters,
            random_state=random_state,
        )
        label = METHOD_LABELS[method]
        communities_by_method[label] = align_community_labels(
            raw, layers, min_jaccard=min_jaccard
        )
        diagnostics_by_method[label] = diagnostics
    return communities_by_method, diagnostics_by_method
