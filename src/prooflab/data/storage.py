"""Storage and analytical querying helpers for market data.

Provides high-performance Parquet persistence and direct analytical
querying via DuckDB.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def write_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compression: str = "snappy",
) -> Path:
    """Write a pandas DataFrame to a Parquet file.

    Creates parent directories if they do not exist.

    Args:
        df: The pandas DataFrame to write.
        path: Target file destination path.
        compression: Compression algorithm (default: snappy).

    Returns:
        The resolved Path of the written Parquet file.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, engine="pyarrow", compression=compression, index=False)
    return target


def read_parquet(
    path: Path | str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame.

    Ensures that any 'timestamp' column is timezone-aware in UTC.

    Args:
        path: Source Parquet file path.
        columns: Optional list of column names to load.

    Returns:
        Loaded pandas DataFrame.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Parquet file not found: {target}")

    df = pd.read_parquet(target, engine="pyarrow", columns=columns)
    if "timestamp" in df.columns and not df.empty:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def read_parquet_bytes(path: Path | str) -> bytes:
    """Read the raw binary content of a Parquet file.

    Useful for checksum calculation and integrity validation.

    Args:
        path: Path to the Parquet file.

    Returns:
        File contents as raw bytes.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Parquet file not found: {target}")
    return target.read_bytes()


class DuckDBAccessLayer:
    """Analytical access layer using DuckDB.

    Allows querying datasets in-memory or directly against Parquet files
    on disk without loading full datasets into Python memory.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize DuckDB access layer.

        Args:
            db_path: Optional path to a persistent DuckDB database file,
                     or None / ':memory:' for in-memory operations.
        """
        self.db_path = str(db_path) if db_path is not None else ":memory:"

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Context manager providing a managed DuckDB connection."""
        conn = duckdb.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def query(
        self,
        sql: str,
        params: list[Any] | dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a pandas DataFrame.

        Args:
            sql: SQL statement to execute.
            params: Optional query parameters for parameterized queries.

        Returns:
            Result set as a pandas DataFrame.
        """
        with self.get_connection() as conn:
            if params is not None:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()

    def query_parquet(
        self,
        path: Path | str,
        select: str = "*",
        where: str | None = None,
        order_by: str | None = None,
    ) -> pd.DataFrame:
        """Query a Parquet file directly using DuckDB.

        Args:
            path: Path to the Parquet file or glob pattern.
            select: SQL select expression (default: '*').
            where: Optional SQL WHERE clause without the WHERE keyword.
            order_by: Optional SQL ORDER BY clause without the ORDER BY keyword.

        Returns:
            Query results as a pandas DataFrame.
        """
        # Convert path to posix format for DuckDB compatibility across platforms
        posix_path = Path(path).resolve().as_posix()
        query_str = f"SELECT {select} FROM read_parquet('{posix_path}')"
        if where:
            query_str += f" WHERE {where}"
        if order_by:
            query_str += f" ORDER BY {order_by}"
        return self.query(query_str)
