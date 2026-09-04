"""Unit tests for prooflab.data.storage."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from prooflab.data.storage import DuckDBAccessLayer, read_parquet, read_parquet_bytes, write_parquet


@pytest.fixture
def sample_df() -> pd.DataFrame:
    timestamps = [
        datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 10, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 10, 2, tzinfo=UTC),
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EURUSD", "EURUSD", "EURUSD"],
            "timeframe": ["M1", "M1", "M1"],
            "open": [1.1000, 1.1005, 1.1010],
            "high": [1.1010, 1.1015, 1.1020],
            "low": [1.0995, 1.1000, 1.1005],
            "close": [1.1005, 1.1010, 1.1015],
            "volume": [10.0, 15.0, 20.0],
            "tick_volume": [20.0, 30.0, 40.0],
            "spread": [1.0, 1.2, 1.1],
            "source": ["mt5", "mt5", "mt5"],
        }
    )


def test_write_and_read_parquet(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    parquet_path = tmp_path / "subdir" / "test.parquet"
    written_path = write_parquet(sample_df, parquet_path)
    assert written_path.exists()

    loaded_df = read_parquet(parquet_path)
    assert len(loaded_df) == len(sample_df)
    assert list(loaded_df.columns) == list(sample_df.columns)
    assert loaded_df["timestamp"].dt.tz == UTC
    assert loaded_df["close"].tolist() == [1.1005, 1.1010, 1.1015]


def test_read_parquet_column_subset(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    parquet_path = tmp_path / "test.parquet"
    write_parquet(sample_df, parquet_path)

    loaded_df = read_parquet(parquet_path, columns=["timestamp", "close"])
    assert list(loaded_df.columns) == ["timestamp", "close"]
    assert len(loaded_df) == 3


def test_read_parquet_bytes(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    parquet_path = tmp_path / "test.parquet"
    write_parquet(sample_df, parquet_path)

    raw_bytes = read_parquet_bytes(parquet_path)
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 0
    # Parquet files always start with magic bytes PAR1
    assert raw_bytes.startswith(b"PAR1")


def test_read_nonexistent_file_raises(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        read_parquet(non_existent)
    with pytest.raises(FileNotFoundError):
        read_parquet_bytes(non_existent)


def test_duckdb_query() -> None:
    layer = DuckDBAccessLayer()
    df = layer.query("SELECT 42 as answer, 'hello' as greeting")
    assert df.iloc[0]["answer"] == 42
    assert df.iloc[0]["greeting"] == "hello"


def test_duckdb_query_parameterized() -> None:
    layer = DuckDBAccessLayer()
    df = layer.query("SELECT ? as num, ? as text", params=[100, "prooflab"])
    assert df.iloc[0]["num"] == 100
    assert df.iloc[0]["text"] == "prooflab"


def test_duckdb_query_parquet(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    parquet_path = tmp_path / "data.parquet"
    write_parquet(sample_df, parquet_path)

    layer = DuckDBAccessLayer()
    result = layer.query_parquet(
        parquet_path,
        select="symbol, close, volume",
        where="close > 1.1008",
        order_by="close DESC",
    )
    assert len(result) == 2
    assert result.iloc[0]["close"] == 1.1015
    assert result.iloc[1]["close"] == 1.1010
