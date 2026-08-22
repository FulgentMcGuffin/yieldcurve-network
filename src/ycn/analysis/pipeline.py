"""End-to-end notebook-equivalent pipeline for a single network view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from matplotlib.figure import Figure
import networkx as nx
import polars as pl

from .config import PipelineConfig
from .data_access import load_table
from .degree_hist import (
    histogram_title,
    render_degree_histogram,
    render_degree_histogram_with_dynamic_threshold,
)
from .gui_cache import GuiDataCache
from .measures import compute_measure
from .network import build_corr_nx, pivot_to_wide
from .pyvis_plot import graph_to_html
from .transforms import apply_transforms

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]


@dataclass
class PipelineResult:
    """Artifacts produced by a successful pipeline run."""

    network_html: str
    degree_hist_fig: Figure
    graph: nx.Graph


def _needed_columns(cfg: PipelineConfig) -> list[str]:
    cols = [cfg.date_column, cfg.name_column, cfg.value_column]
    return list(dict.fromkeys(cols))


def _prepare_frame(
    cfg: PipelineConfig, status: StatusCallback | None = None
) -> pl.DataFrame:
    needed = _needed_columns(cfg)
    if cfg.where_clause and status is not None:
        status(f"Loading with SQL filter: WHERE {cfg.where_clause}…")
    df = load_table(
        cfg.db_path, cfg.table, columns=needed, where_clause=cfg.where_clause
    )

    df = df.with_columns(pl.col(cfg.date_column).cast(pl.Date, strict=False))
    df = df.sort(cfg.date_column, cfg.name_column)

    if cfg.date_start is not None:
        df = df.filter(pl.col(cfg.date_column) >= cfg.date_start)
    if cfg.date_end is not None:
        df = df.filter(pl.col(cfg.date_column) <= cfg.date_end)

    df = df.drop_nulls(subset=[cfg.date_column, cfg.name_column, cfg.value_column])

    transform_cols = [cfg.value_column]
    if cfg.transforms:
        if status is not None:
            status(f"Applying transforms: {', '.join(cfg.transforms)}…")
        df = apply_transforms(
            df,
            cfg.transforms,
            cfg.date_column,
            cfg.name_column,
            transform_cols,
        )

    return df


def run_pipeline(
    cfg: PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
    data_cache: GuiDataCache | None = None,
    edge_settings: dict | None = None,
) -> PipelineResult:
    """Run the full pipeline and return network HTML + degree histogram PNG.

    When ``data_cache`` is provided (GUI session), Polars frames for load /
    prepare / measure are memoized via framecache in an in-memory SQLite DB.

    Args:
        cfg: Pipeline configuration.
        progress: Optional progress callback.
        status: Optional status callback.
        data_cache: Optional memoization cache.
        edge_settings: Optional dict with measure-specific parameters
            (e.g., {'conditional_quantile': 0.90}).
    """

    def _status(msg: str) -> None:
        if status is not None:
            status(msg)

    if data_cache is not None:
        df = data_cache.get_prepared(cfg, status=status)
    else:
        _status("Loading data…")
        df = _prepare_frame(cfg, status=status)

    if df.is_empty():
        raise ValueError("No rows remain after filtering / date range / transforms.")
    _status(f"Prepared frame: {df.height:,} rows.")

    if data_cache is not None:
        measure_hit = data_cache.has_measure(cfg)
        measure_df = data_cache.get_measure_matrix(
            cfg, progress=progress, status=status
        )
        if measure_hit and progress is not None:
            progress(1, 1, "cached")
        nodes = measure_df.columns
        _status(f"Measure matrix: {len(nodes)} × {len(nodes)}.")
    else:
        _status(
            f"Pivoting wide on {cfg.value_column!r} "
            f"(date={cfg.date_column!r}, node={cfg.name_column!r})…"
        )
        wide = pivot_to_wide(
            df,
            date_column=cfg.date_column,
            name_column=cfg.name_column,
            value_column=cfg.value_column,
        )
        nodes = [c for c in wide.columns if c != cfg.date_column]
        if len(nodes) < 2:
            raise ValueError("Need at least two nodes to build a network.")
        _status(f"Wide matrix: {wide.height:,} dates × {len(nodes)} nodes.")

        n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
        _status(f"Computing {cfg.measure} ({n_pairs:,} pairs)…")
        measure_df = compute_measure(
            cfg.measure,
            wide.select(nodes),
            nodes,
            progress=progress,
            edge_settings=edge_settings,
        )

    _status(
        f"Building network (independence threshold={cfg.independent_threshold:.2f})…"
    )
    graph = build_corr_nx(measure_df, independent_threshold=cfg.independent_threshold)
    # Store measure_df on graph for dynamic threshold adjustment in GUI
    graph._measure_df = measure_df
    graph._measure_name = cfg.measure
    _status(
        f"Network built: |N|={graph.number_of_nodes()}, "
        f"|E|={graph.number_of_edges():,}."
    )

    _status("Rendering interactive network…")
    html = graph_to_html(graph, title=cfg.title)

    hist_title = histogram_title(cfg.value_column)
    _status("Rendering degree histogram…")
    hist_fig = render_degree_histogram_with_dynamic_threshold(
        measure_df, hist_title, current_threshold=cfg.independent_threshold
    )
    # Also store measure info on the figure for GUI slider
    hist_fig._measure_name = cfg.measure

    _status("Done.")
    return PipelineResult(
        network_html=html,
        degree_hist_fig=hist_fig,
        graph=graph,
    )
