"""Database backends.

``base`` defines the generic :class:`DataBackend` protocol; concrete backends
(SQLite, DuckDB) implement it so notebooks and query code never depend on a
specific storage engine.
"""

from .base import (
    ColumnInfo,
    DataBackend,
    DataSink,
    QueryError,
    TableSchema,
    is_read_only_sql,
)
from .duckdb_backend import DuckDBSource
from .sqlite_backend import SQLiteSource

__all__ = [
    "ColumnInfo",
    "DataBackend",
    "DataSink",
    "DuckDBSource",
    "QueryError",
    "SQLiteSource",
    "TableSchema",
    "create_backend",
    "is_read_only_sql",
]


def create_backend(
    db_type: str = "duckdb",
    db_path: str | None = None,
) -> DataBackend:
    """Construct a read-only backend for the given engine and database path."""

    if db_type == "duckdb":
        return DuckDBSource(db_path, read_only=True)
    return SQLiteSource(db_path, read_only=True)
