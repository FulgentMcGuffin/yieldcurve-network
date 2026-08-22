"""Session-scoped [framecache](https://github.com/FulgentMcGuffin/framecache) for the GUI.

Caches Polars frames in an in-memory SQLite DB so repeated builds with the same
inputs skip DuckDB loads, transforms, and expensive measure recomputation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from framecache import CacheConfig, FrameCache

from ycn.analysis.config import PipelineConfig
from ycn.analysis.data_access import load_table
from ycn.analysis.measures import compute_measure
from ycn.analysis.network import pivot_to_wide
from ycn.analysis.transforms import apply_transforms

ProgressCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[str], None]

# Progress is not part of the cache key; the worker sets it for the current run.
_progress: threading.local = threading.local()


def create_gui_frame_cache() -> FrameCache:
    """In-memory SQLite FrameCache for the lifetime of the GUI process."""
    config = CacheConfig(
        backend_type="sqlite",
        db_path=":memory:",
        framecache_key="YCNGUI",
        use_hash_keys=True,
        # Session cache; long TTL (framecache treats None as the 1h class default).
        default_ttl_hours=24.0 * 365.0,
    )
    return FrameCache.from_config(config)


def _iso(d: date | None) -> str:
    return "" if d is None else d.isoformat()


def _none_str(value: str | None) -> str:
    return "" if value is None else value


def _transforms_key(transforms: list[str] | tuple[str, ...]) -> str:
    return ",".join(transforms)


def _columns_key(columns: list[str]) -> str:
    return ",".join(columns)


def cache_key_parts(cfg: PipelineConfig) -> dict[str, str]:
    """Primitive, stable args used as framecache keys for a pipeline config."""
    cols = list(dict.fromkeys([cfg.date_column, cfg.name_column, cfg.value_column]))
    return {
        "db_path": str(Path(cfg.db_path).resolve()),
        "table": cfg.table,
        "columns": _columns_key(cols),
        "date_column": cfg.date_column,
        "name_column": cfg.name_column,
        "value_column": cfg.value_column,
        "where_clause": _none_str(cfg.where_clause),
        "transforms": _transforms_key(cfg.transforms),
        "date_start": _iso(cfg.date_start),
        "date_end": _iso(cfg.date_end),
        "measure": cfg.measure,
    }


@dataclass
class GuiDataCache:
    """Bound framecache wrappers for load → prepare → measure DataFrames."""

    frame_cache: FrameCache
    load_columns: Callable[..., pl.DataFrame]
    prepare_frame: Callable[..., pl.DataFrame]
    measure_matrix: Callable[..., pl.DataFrame]

    @classmethod
    def create(cls, frame_cache: FrameCache | None = None) -> GuiDataCache:
        fc = frame_cache or create_gui_frame_cache()

        @fc.cache(method="pyarrow")
        def load_columns(
            db_path: str, table: str, columns: str, where_clause: str
        ) -> pl.DataFrame:
            cols = [c for c in columns.split(",") if c]
            return load_table(
                db_path, table, columns=cols or None, where_clause=where_clause or None
            )

        @fc.cache(method="pyarrow")
        def prepare_frame(
            db_path: str,
            table: str,
            columns: str,
            date_column: str,
            name_column: str,
            value_column: str,
            where_clause: str,
            transforms: str,
            date_start: str,
            date_end: str,
        ) -> pl.DataFrame:
            df = load_columns(db_path, table, columns, where_clause)
            df = df.with_columns(pl.col(date_column).cast(pl.Date, strict=False))
            df = df.sort(date_column, name_column)

            if date_start:
                df = df.filter(pl.col(date_column) >= date.fromisoformat(date_start))
            if date_end:
                df = df.filter(pl.col(date_column) <= date.fromisoformat(date_end))

            df = df.drop_nulls(subset=[date_column, name_column, value_column])

            transform_ids = [t for t in transforms.split(",") if t]
            if transform_ids:
                df = apply_transforms(
                    df,
                    transform_ids,
                    date_column,
                    name_column,
                    [value_column],
                )
            return df

        @fc.cache(method="pyarrow")
        def measure_matrix(
            db_path: str,
            table: str,
            columns: str,
            date_column: str,
            name_column: str,
            value_column: str,
            where_clause: str,
            transforms: str,
            date_start: str,
            date_end: str,
            measure: str,
        ) -> pl.DataFrame:
            df = prepare_frame(
                db_path,
                table,
                columns,
                date_column,
                name_column,
                value_column,
                where_clause,
                transforms,
                date_start,
                date_end,
            )
            if df.is_empty():
                raise ValueError(
                    "No rows remain after filtering / date range / transforms."
                )

            wide = pivot_to_wide(
                df,
                date_column=date_column,
                name_column=name_column,
                value_column=value_column,
            )
            nodes = [c for c in wide.columns if c != date_column]
            if len(nodes) < 2:
                raise ValueError("Need at least two nodes to build a network.")

            progress = getattr(_progress, "callback", None)
            return compute_measure(
                measure, wide.select(nodes), nodes, progress=progress
            )

        return cls(fc, load_columns, prepare_frame, measure_matrix)

    def has_prepare(self, cfg: PipelineConfig) -> bool:
        k = cache_key_parts(cfg)
        return (
            self.frame_cache.latest_cache_instance_id(
                self.prepare_frame,
                "pyarrow",
                k["db_path"],
                k["table"],
                k["columns"],
                k["date_column"],
                k["name_column"],
                k["value_column"],
                k["where_clause"],
                k["transforms"],
                k["date_start"],
                k["date_end"],
            )
            is not None
        )

    def has_measure(self, cfg: PipelineConfig) -> bool:
        k = cache_key_parts(cfg)
        return (
            self.frame_cache.latest_cache_instance_id(
                self.measure_matrix,
                "pyarrow",
                k["db_path"],
                k["table"],
                k["columns"],
                k["date_column"],
                k["name_column"],
                k["value_column"],
                k["where_clause"],
                k["transforms"],
                k["date_start"],
                k["date_end"],
                k["measure"],
            )
            is not None
        )

    def get_prepared(
        self,
        cfg: PipelineConfig,
        *,
        status: StatusCallback | None = None,
    ) -> pl.DataFrame:
        k = cache_key_parts(cfg)
        hit = self.has_prepare(cfg)
        if status is not None:
            if hit:
                status("Cache hit: prepared frame (skipping load/filters/transforms).")
            else:
                status("Loading data and applying filters/transforms…")
        return self.prepare_frame(
            k["db_path"],
            k["table"],
            k["columns"],
            k["date_column"],
            k["name_column"],
            k["value_column"],
            k["where_clause"],
            k["transforms"],
            k["date_start"],
            k["date_end"],
        )

    def get_measure_matrix(
        self,
        cfg: PipelineConfig,
        *,
        progress: ProgressCallback | None = None,
        status: StatusCallback | None = None,
    ) -> pl.DataFrame:
        k = cache_key_parts(cfg)
        hit = self.has_measure(cfg)
        if status is not None:
            if hit:
                status(
                    f"Cache hit: {cfg.measure} matrix "
                    "(skipping pairwise recomputation)."
                )
            else:
                status(f"Computing {cfg.measure}…")
        _progress.callback = None if hit else progress
        try:
            return self.measure_matrix(
                k["db_path"],
                k["table"],
                k["columns"],
                k["date_column"],
                k["name_column"],
                k["value_column"],
                k["where_clause"],
                k["transforms"],
                k["date_start"],
                k["date_end"],
                k["measure"],
            )
        finally:
            _progress.callback = None
