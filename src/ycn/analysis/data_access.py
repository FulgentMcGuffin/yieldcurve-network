"""Thin helpers for listing/loading tables from DuckDB or SQLite files."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from ycn.backends.base import _normalize_sql_type
from ycn.backends.duckdb_backend import DuckDBSource
from ycn.backends.sqlite_backend import SQLiteSource

# A multi-layer network needs a column with a small, discrete set of values:
# one layer per distinct value. These bounds keep the layer count sane (and the
# per-layer O(n^2) measure cost bounded).
MAX_LAYER_VALUES = 50
MAX_INTEGER_LAYER_VALUES = 20


def open_backend(db_path: Path | str, *, read_only: bool = True):
    """Open the appropriate backend based on file extension."""
    path = Path(db_path)
    suffix = path.suffix.lower()
    if suffix == ".duckdb":
        return DuckDBSource(path, read_only=read_only)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return SQLiteSource(path, read_only=read_only)
    raise ValueError(
        f"Unsupported database type {suffix!r}. Use .duckdb or .sqlite/.db."
    )


def list_tables(db_path: Path | str) -> list[str]:
    with open_backend(db_path) as db:
        return db.list_tables()


def list_columns(db_path: Path | str, table: str) -> list[str]:
    with open_backend(db_path) as db:
        schema = db.get_schema(table)
        return [col.name for col in schema.columns]


def eligible_layer_columns(
    db_path: Path | str,
    table: str,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Columns usable as a multi-layer-network layer key.

    A layer column must be a discrete, low-cardinality label. Text columns
    qualify with at most ``MAX_LAYER_VALUES`` distinct values; integer columns
    are allowed only as an exception, and only below
    ``MAX_INTEGER_LAYER_VALUES``. Floats, dates and timestamps are continuous
    and never qualify. Columns with fewer than two distinct values are dropped
    too -- they would collapse the multiplex to a single layer.

    Args:
        db_path: Database file.
        table: Table to inspect.
        exclude: Columns already claimed by other roles (date/name/value).

    Returns:
        Eligible column names, sorted. Columns whose cardinality cannot be
        determined are skipped rather than raising.
    """
    skip = {c.lower() for c in (exclude or set()) if c}
    eligible: list[str] = []
    with open_backend(db_path) as db:
        for col in db.get_schema(table).columns:
            if col.name.lower() in skip:
                continue
            kind = _normalize_sql_type(col.type)
            if kind not in {"TEXT", "INTEGER"}:
                continue  # REAL / DATE / TIMESTAMP are continuous
            try:
                frame = db.run_query(
                    f'SELECT COUNT(DISTINCT "{col.name}") AS n FROM "{table}"'
                )
            except Exception:  # noqa: BLE001 -- an odd column must not break the GUI
                continue
            if frame.is_empty():
                continue
            n_distinct = int(frame.row(0)[0] or 0)
            if n_distinct < 2:
                continue
            if kind == "INTEGER":
                # Integers are the documented exception: strictly under 20.
                ok = n_distinct < MAX_INTEGER_LAYER_VALUES
            else:
                ok = n_distinct <= MAX_LAYER_VALUES
            if ok:
                eligible.append(col.name)
    return sorted(eligible)


def load_table(
    db_path: Path | str,
    table: str,
    columns: list[str] | None = None,
    where_clause: str | None = None,
) -> pl.DataFrame:
    """Load selected columns from ``table`` (all columns if ``columns`` is None).

    ``where_clause`` is raw user-authored SQL inserted verbatim after ``WHERE``
    (e.g. from the GUI's "WHERE clause" filter mode) — not parameterized, same
    trust model as the rest of this module's f-string SQL for this local tool.
    """
    with open_backend(db_path) as db:
        if columns:
            quoted = ", ".join(f'"{c}"' for c in columns)
            sql = f'SELECT {quoted} FROM "{table}"'
        else:
            sql = f'SELECT * FROM "{table}"'
        if where_clause:
            sql += f" WHERE {where_clause}"
        return db.run_query(sql)


def distinct_values(
    db_path: Path | str,
    table: str,
    column: str,
    limit: int = 500,
) -> list[str]:
    """Return distinct stringified values for a filter dropdown."""
    with open_backend(db_path) as db:
        sql = (
            f'SELECT DISTINCT "{column}" AS v FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f"ORDER BY v LIMIT {int(limit)}"
        )
        frame = db.run_query(sql)
        if frame.is_empty():
            return []
        return [str(v) for v in frame.get_column("v").to_list()]


def column_date_bounds(
    db_path: Path | str,
    table: str,
    column: str,
) -> tuple[date | None, date | None]:
    """Return ``(min_date, max_date)`` for a date/datetime column."""
    with open_backend(db_path) as db:
        sql = f'SELECT MIN("{column}") AS lo, MAX("{column}") AS hi ' f'FROM "{table}"'
        frame = db.run_query(sql)
        if frame.is_empty():
            return None, None
        lo, hi = frame.row(0)
        return _as_date(lo), _as_date(hi)


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
