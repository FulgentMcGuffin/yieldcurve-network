"""DuckDB implementation of the read (:class:`DataBackend`) and write
(:class:`DataSink`) contracts.

A single :class:`DuckDBSource` class serves both roles so the rest of the
project depends on one DuckDB type. The serving path is still protected: with
``read_only=True`` the connection is
opened in read-only mode and the write methods are rejected, so it physically
cannot mutate the database. Ingestion code opts in with ``read_only=False``.
"""

from __future__ import annotations

import os
import duckdb
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from .base import (
    ColumnInfo,
    DataSink,
    QueryError,
    TableSchema,
    _polars_dtype_to_sql,
    is_read_only_sql,
)

# Load the project's .env and .secrets so DUCKDB_PATH is available even when
# not exported in the shell. Existing OS env vars take precedence.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[3] / ".secrets", override=False)


def _resolve_default_db_path() -> tuple[str, str]:
    """Resolve ``(directory, filename)`` for the default DB from ``DUCKDB_PATH``.

    Evaluated lazily (never at import time) so importing this module never
    requires ``DUCKDB_PATH`` to be configured.

    Raises:
        ValueError: if ``DUCKDB_PATH`` is unset or does not end in ``.duckdb``.
    """
    raw = os.getenv("DUCKDB_PATH")
    if raw is None:
        raise ValueError("DUCKDB_PATH environment variable is not set")
    name = os.path.basename(raw)
    if not name.endswith(".duckdb"):
        raise ValueError(
            f"DUCKDB_PATH environment variable must end with .duckdb but found {name}"
        )
    return os.path.dirname(raw), name


class DuckDBSource(DataSink):
    """Read/write DuckDB data source.

    Implements the read-only :class:`DataBackend` protocol (consumed by the
    server and query pipeline) *and* the writable :class:`DataSink` interface
    (consumed by ingestion scripts). Usable as a context manager::

        with DuckDBSource("company.duckdb", read_only=False) as db:
            db.create_table("employees", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
            db.insert("employees", {"id": 1, "name": "Alice"})
            rows = db.select("employees", where={"id": 1})

    Args:
        db_path: Path to the DuckDB file, ``:memory:`` for an in-memory db, or
            ``None`` to derive the location from the ``DUCKDB_PATH`` env var.
        read_only: When ``True`` (default) the connection is opened read-only and
            mutating methods raise :class:`QueryError`.
    """

    name = "duckdb"

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        read_only: bool = True,
    ) -> None:
        if db_path is None:
            default_dir, default_name = _resolve_default_db_path()
            db_path = os.path.join(default_dir, default_name)
        self._db_path = str(db_path)
        self._read_only = read_only
        self._connection: duckdb.DuckDBPyConnection | None = None
        self.connect()

    # ------------------------------------------------------------------
    # Default-path helpers (derived from DUCKDB_PATH, resolved lazily)
    # ------------------------------------------------------------------

    @classmethod
    def default_db_dir(cls) -> str:
        """Directory portion of ``DUCKDB_PATH``."""
        return _resolve_default_db_path()[0]

    @classmethod
    def default_db_name(cls) -> str:
        """Filename portion of ``DUCKDB_PATH``."""
        return _resolve_default_db_path()[1]

    @classmethod
    def get_full_db_path(cls, db_name: str | None = None) -> str:
        default_dir, default_name = _resolve_default_db_path()
        if db_name is None:
            db_name = default_name
        return os.path.join(
            default_dir,
            f"{db_name}.duckdb" if not db_name.endswith(".duckdb") else db_name,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the connection if not already open.

        In read-only mode an existing on-disk file is opened in read-only mode.
        In read-write mode the file (and any missing parent directories) is
        created on demand.
        """
        if self._connection is not None:
            return
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        if self._read_only and self._db_path != ":memory:":
            path = Path(self._db_path)
            if not path.exists():
                raise QueryError(
                    f"DuckDB database not found at {path}. " "Run the seeder first."
                )
            self._connection = duckdb.connect(self._db_path, read_only=True)
        else:
            self._connection = duckdb.connect(self._db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "DuckDBSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _conn(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise RuntimeError(
                "No active connection. Call connect() or use DuckDBSource as a context manager."
            )
        return self._connection

    def _require_writable(self) -> None:
        if self._read_only:
            raise QueryError(
                "This DuckDBSource is read-only; construct it with read_only=False to modify data."
            )

    # ------------------------------------------------------------------
    # DataBackend interface (read side)
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        cursor = self._conn().execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' "
            "ORDER BY table_name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table: str) -> TableSchema:
        if table not in self.list_tables():
            raise QueryError(f"Unknown table: {table!r}")
        cursor = self._conn().execute(f"PRAGMA table_info({table})")
        columns = [
            ColumnInfo(
                name=row[1],
                type=row[2] or "UNKNOWN",
                nullable=not bool(row[3]),
                primary_key=bool(row[5]),
            )
            for row in cursor.fetchall()
        ]
        return TableSchema(table=table, columns=columns)

    def run_query(self, sql: str) -> pl.DataFrame:
        if not is_read_only_sql(sql):
            raise QueryError(
                "Only single read-only statements are allowed "
                "(SELECT / WITH / PRAGMA / EXPLAIN)."
            )
        try:
            cursor = self._conn().execute(sql)
        except duckdb.Error as exc:
            raise QueryError(f"DuckDB error: {exc}") from exc

        result = cursor.fetchall()
        if not result:
            # Return empty DataFrame with column names from cursor description
            if cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                return pl.DataFrame(schema=column_names)
            return pl.DataFrame()

        column_names = [desc[0] for desc in cursor.description]
        return pl.DataFrame(result, schema=column_names, orient="row")

    # ------------------------------------------------------------------
    # DataSink interface (write side)
    # ------------------------------------------------------------------

    def create_table(
        self,
        table_name: str,
        schema: dict[str, str],
        overwrite_if_exists: bool = False,
    ) -> bool:
        self._require_writable()
        columns_sql = ",\n    ".join(
            f"{col} {definition}" for col, definition in schema.items()
        )
        conn = self._conn()
        already_exists = (
            conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name = ?",
                [table_name],
            ).fetchone()
            is not None
        )

        if already_exists and not overwrite_if_exists:
            return False

        if overwrite_if_exists:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.execute(f"CREATE TABLE {table_name} (\n    {columns_sql}\n);")
        return True

    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        cols_sql = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols_sql} FROM {table_name}"
        params: list = []
        if where:
            conditions = " AND ".join(f"{col} = ?" for col in where)
            query += f" WHERE {conditions}"
            params = list(where.values())
        cursor = self._conn().execute(query, params)
        rows = cursor.fetchall()
        if not rows:
            return []
        # Convert tuples to dicts using column names
        column_names = [desc[0] for desc in cursor.description]
        return [dict(zip(column_names, row)) for row in rows]

    def _insert_dataframe(self, table_name: str, df: pl.DataFrame) -> None:
        """Bulk-insert a Polars frame using DuckDB's native columnar path."""
        conn = self._conn()
        conn.register("_bulk_insert_src", df)
        try:
            conn.execute(
                f"INSERT INTO {table_name} BY NAME SELECT * FROM _bulk_insert_src"
            )
        finally:
            conn.unregister("_bulk_insert_src")

    def _insert_dataframe_skip_duplicates(
        self,
        table_name: str,
        df: pl.DataFrame,
        dup_check_cols_lower: list[str],
    ) -> int:
        """Bulk-insert rows whose dup-check key is not already in the table."""
        conn = self._conn()
        table_col_map = {
            col.name.lower(): col.name for col in self.get_schema(table_name).columns
        }
        df_col_map = {col.lower(): col for col in df.columns}
        conditions = " AND ".join(
            f'existing."{table_col_map[col]}" = new."{df_col_map[col]}"'
            for col in dup_check_cols_lower
        )
        before = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        conn.register("_append_src", df)
        try:
            conn.execute(
                f"INSERT INTO {table_name} BY NAME "
                f"SELECT new.* FROM _append_src AS new "
                f"WHERE NOT EXISTS ("
                f"SELECT 1 FROM {table_name} AS existing WHERE {conditions}"
                f")"
            )
        finally:
            conn.unregister("_append_src")
        after = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return after - before

    def append_to_table(
        self,
        table_name: str,
        df: pl.DataFrame,
        duplicate_check_columns: list[str] | None = None,
    ) -> int:
        """Append via columnar bulk insert, with SQL-based duplicate filtering."""
        self._require_writable()
        if len(df) == 0:
            return 0

        dup_check_cols_lower = self._validate_append_schema(
            table_name, df, duplicate_check_columns
        )
        if dup_check_cols_lower is None:
            self._insert_dataframe(table_name, df)
            return df.height
        return self._insert_dataframe_skip_duplicates(
            table_name, df, dup_check_cols_lower
        )

    def create_table_from_polars(
        self,
        table_name: str,
        df: pl.DataFrame,
        overwrite_if_exists: bool = False,
        num_splits: int = 20,
    ) -> bool:
        """Create a table and bulk-load rows without dict round-trips."""
        schema = {col: _polars_dtype_to_sql(dtype) for col, dtype in df.schema.items()}
        created = self.create_table(table_name, schema, overwrite_if_exists)
        if created and df.height > 0:
            if df.height > 1_000_000:
                step = max(df.height // num_splits, 1)
                for start in range(0, df.height, step):
                    self._insert_dataframe(table_name, df.slice(start, step))
            else:
                self._insert_dataframe(table_name, df)
        return created

    def insert(
        self,
        table_name: str,
        data: dict | list[dict],
    ) -> int:
        self._require_writable()
        if isinstance(data, dict):
            data = [data]
        if not data:
            return 0

        if len(data) == 1:
            row = data[0]
            columns_sql = ", ".join(row.keys())
            placeholders = ", ".join("?" for _ in row)
            query = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"
            self._conn().execute(query, list(row.values()))
            return 1

        columns = list(data[0].keys())
        self._insert_dataframe(table_name, pl.from_dicts(data))
        return len(data)

    def update(
        self,
        table_name: str,
        data: dict,
        where: dict,
    ) -> int:
        self._require_writable()
        set_clause = ", ".join(f"{col} = ?" for col in data)
        where_clause = " AND ".join(f"{col} = ?" for col in where)
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        conn = self._conn()
        cursor = conn.execute(query, params)
        # DuckDB returns the number of affected rows via the rows_modified attribute
        return cursor.rowcount if hasattr(cursor, "rowcount") else 0

    def delete(
        self,
        table_name: str,
        where: dict,
    ) -> int:
        self._require_writable()
        where_clause = " AND ".join(f"{col} = ?" for col in where)
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        conn = self._conn()
        cursor = conn.execute(query, list(where.values()))
        # DuckDB returns the number of affected rows via the rows_modified attribute
        return cursor.rowcount if hasattr(cursor, "rowcount") else 0

    def execute(
        self,
        query: str,
        params: tuple = (),
    ) -> list[dict]:
        try:
            cursor = self._conn().execute(query, params)
        except duckdb.Error as exc:
            raise QueryError(f"DuckDB error: {exc}") from exc

        result = cursor.fetchall()
        if result and cursor.description:
            column_names = [desc[0] for desc in cursor.description]
            return [dict(zip(column_names, row)) for row in result]
        return []
