"""SQLite implementation of the read (:class:`DataBackend`) and write
(:class:`DataSink`) contracts.

A single :class:`SQLiteSource` class serves both roles so the rest of the
project depends on one SQLite type. The serving path is still protected: with
``read_only=True`` (the default) the connection is
opened with ``mode=ro`` and the write methods are rejected, so it physically
cannot mutate the database. Ingestion code opts in with ``read_only=False``.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from .base import (
    ColumnInfo,
    DataSink,
    QueryError,
    TableSchema,
    is_read_only_sql,
)

# Load the project's .env and .secrets so SQLITE_DB_PATH is available even when
# not exported in the shell. Existing OS env vars take precedence.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[3] / ".secrets", override=False)


def _resolve_default_db_path() -> tuple[str, str]:
    """Resolve ``(directory, filename)`` for the default DB from ``SQLITEDB_PATH``.

    Evaluated lazily (never at import time) so importing this module never
    requires ``SQLITEDB_PATH`` to be configured.

    Raises:
        ValueError: if ``SQLITEDB_PATH`` is unset or does not end in ``.db``.
    """
    raw = os.getenv("SQLITEDB_PATH")
    if raw is None:
        raise ValueError("SQLITEDB_PATH environment variable is not set")
    name = os.path.basename(raw)
    if not name.endswith(".db"):
        raise ValueError(
            f"SQLITEDB_PATH environment variable must end with .db but found {name}"
        )
    return os.path.dirname(raw), name


class SQLiteSource(DataSink):
    """Read/write SQLite data source.

    Implements the read-only :class:`DataBackend` protocol (consumed by the
    server and query pipeline) *and* the writable :class:`DataSink` interface
    (consumed by ingestion scripts). Usable as a context manager::

        with SQLiteSource("company.db", read_only=False) as db:
            db.create_table("employees", {"id": "INTEGER PRIMARY KEY", "name": "TEXT"})
            db.insert("employees", {"id": 1, "name": "Alice"})
            rows = db.select("employees", where={"id": 1})

    Args:
        db_path: Path to the SQLite file, ``":memory:"`` for an in-memory db, or
            ``None`` to derive the location from the ``SQLITEDB_PATH`` env var.
        read_only: When ``True`` (default) the connection is opened read-only and
            mutating methods raise :class:`QueryError`.
    """

    name = "sqlite"

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
        self._connection: sqlite3.Connection | None = None
        self.connect()

    # ------------------------------------------------------------------
    # Default-path helpers (derived from SQLITEDB_PATH, resolved lazily)
    # ------------------------------------------------------------------

    @classmethod
    def default_db_dir(cls) -> str:
        """Directory portion of ``SQLITEDB_PATH``."""
        return _resolve_default_db_path()[0]

    @classmethod
    def default_db_name(cls) -> str:
        """Filename portion of ``SQLITEDB_PATH``."""
        return _resolve_default_db_path()[1]

    @classmethod
    def get_full_db_path(cls, db_name: str | None = None) -> str:
        default_dir, default_name = _resolve_default_db_path()
        if db_name is None:
            db_name = default_name
        return os.path.join(
            default_dir,
            f"{db_name}.db" if not db_name.endswith(".db") else db_name,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the connection if not already open.

        In read-only mode an existing on-disk file is opened with ``mode=ro``.
        In read-write mode the file (and any missing parent directories) is
        created on demand.
        """
        if self._connection is not None:
            return
        if self._read_only and self._db_path != ":memory:":
            path = Path(self._db_path)
            if not path.exists():
                raise QueryError(
                    f"SQLite database not found at {path}. " "Run the seeder first."
                )
            uri = f"file:{path.as_posix()}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            if self._db_path != ":memory:":
                os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "SQLiteSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError(
                "No active connection. Call connect() or use SQLiteSource as a context manager."
            )
        return self._connection

    def _require_writable(self) -> None:
        if self._read_only:
            raise QueryError(
                "This SQLiteSource is read-only; construct it with read_only=False to modify data."
            )

    # ------------------------------------------------------------------
    # DataBackend interface (read side)
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        cursor = self._conn().execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table: str) -> TableSchema:
        if table not in self.list_tables():
            raise QueryError(f"Unknown table: {table!r}")
        # PRAGMA does not support parameter binding for the table name; the name
        # is validated against the table list above, so this is safe.
        cursor = self._conn().execute(f'PRAGMA table_info("{table}")')
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
        except sqlite3.Error as exc:
            raise QueryError(f"SQL error: {exc}") from exc

        if cursor.description is None:
            return pl.DataFrame()

        column_names = [desc[0] for desc in cursor.description]
        # row_factory yields sqlite3.Row; convert to plain tuples for polars.
        rows = [tuple(row) for row in cursor.fetchall()]
        if not rows:
            return pl.DataFrame(schema=column_names)
        return pl.DataFrame(rows, schema=column_names, orient="row")

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
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            is not None
        )

        if already_exists and not overwrite_if_exists:
            return False

        if overwrite_if_exists:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.execute(f"CREATE TABLE {table_name} (\n    {columns_sql}\n);")
        conn.commit()
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
        return [dict(row) for row in cursor.fetchall()]

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
        columns_sql = ", ".join(data[0].keys())
        placeholders = ", ".join("?" for _ in data[0])
        query = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"
        conn = self._conn()
        conn.executemany(query, [list(row.values()) for row in data])
        conn.commit()
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
        conn.commit()
        return cursor.rowcount

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
        conn.commit()
        return cursor.rowcount

    def execute(
        self,
        query: str,
        params: tuple = (),
    ) -> list[dict]:
        try:
            cursor = self._conn().execute(query, params)
        except sqlite3.Error as exc:
            raise QueryError(f"SQL error: {exc}") from exc
        if not self._read_only:
            self._conn().commit()
        if cursor.description:
            return [dict(row) for row in cursor.fetchall()]
        return []
