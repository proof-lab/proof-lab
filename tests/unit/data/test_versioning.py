"""Unit tests for prooflab.data.versioning and prooflab.data.repository."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe
from prooflab.data.versioning import (
    DatasetIntegrityError,
    compute_checksum,
    create_dataset_metadata,
    load_metadata,
    save_metadata,
    verify_dataset_integrity,
)


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


def test_compute_checksum(tmp_path: Path) -> None:
    data = b"prooflab market data bytes"
    file_path = tmp_path / "test.bin"
    file_path.write_bytes(data)

    hash_from_bytes = compute_checksum(data)
    hash_from_file = compute_checksum(file_path)
    assert hash_from_bytes == hash_from_file
    assert len(hash_from_bytes) == 64


def test_create_dataset_metadata(sample_df: pd.DataFrame) -> None:
    parquet_bytes = b"sample_parquet_bytes_mock"
    metadata = create_dataset_metadata(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        parquet_bytes=parquet_bytes,
    )
    assert metadata.source == "mt5"
    assert metadata.symbol == "EURUSD"
    assert metadata.timeframe == Timeframe.M1
    assert metadata.row_count == 3
    assert metadata.start_time == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    assert metadata.end_time == datetime(2026, 1, 1, 10, 2, tzinfo=UTC)
    assert metadata.feature_version is None
    assert metadata.checksum == compute_checksum(parquet_bytes)


def test_metadata_immutability(sample_df: pd.DataFrame) -> None:
    metadata = create_dataset_metadata(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        parquet_bytes=b"mock",
    )
    with pytest.raises(Exception):
        metadata.row_count = 100  # type: ignore[misc]


def test_verify_dataset_integrity(sample_df: pd.DataFrame) -> None:
    valid_bytes = b"real_parquet_payload"
    tampered_bytes = b"tampered_parquet_payload"
    metadata = create_dataset_metadata(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        parquet_bytes=valid_bytes,
    )
    assert verify_dataset_integrity(valid_bytes, metadata) is True

    with pytest.raises(DatasetIntegrityError, match="Dataset integrity violation"):
        verify_dataset_integrity(tampered_bytes, metadata)


def test_save_and_load_metadata(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    meta_path = tmp_path / "dataset1.meta.json"
    metadata = create_dataset_metadata(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        parquet_bytes=b"mock",
    )
    save_metadata(metadata, meta_path)
    assert meta_path.exists()

    loaded_meta = load_metadata(meta_path)
    assert loaded_meta.dataset_id == metadata.dataset_id
    assert loaded_meta.checksum == metadata.checksum
    assert loaded_meta.start_time == metadata.start_time

    # Immutable: cannot overwrite by default
    with pytest.raises(FileExistsError):
        save_metadata(metadata, meta_path, overwrite=False)


def test_parquet_repository_save_and_load(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    repo = ParquetRepository(base_dir=tmp_path / "repo")
    metadata = repo.save_dataset(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
    )
    assert metadata.dataset_id is not None

    loaded_df, loaded_meta = repo.load_dataset(metadata.dataset_id)
    assert len(loaded_df) == 3
    assert loaded_meta.dataset_id == metadata.dataset_id

    # Test get_metadata
    meta_only = repo.get_metadata(metadata.dataset_id)
    assert meta_only.dataset_id == metadata.dataset_id


def test_parquet_repository_tamper_detection(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    repo = ParquetRepository(base_dir=tmp_path / "repo")
    metadata = repo.save_dataset(
        df=sample_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
    )

    # Tamper with the parquet file on disk
    parquet_path, _ = repo._find_dataset_paths(metadata.dataset_id)
    parquet_path.write_bytes(b"tampered file content")

    with pytest.raises(DatasetIntegrityError):
        repo.load_dataset(metadata.dataset_id)


def test_parquet_repository_list(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    repo = ParquetRepository(base_dir=tmp_path / "repo")
    meta1 = repo.save_dataset(df=sample_df, source="mt5", symbol="EURUSD", timeframe=Timeframe.M1)
    meta2 = repo.save_dataset(df=sample_df, source="mt5", symbol="GBPUSD", timeframe=Timeframe.H1)

    all_datasets = repo.list_datasets()
    assert len(all_datasets) == 2

    eur_datasets = repo.list_datasets(symbol="EURUSD")
    assert len(eur_datasets) == 1
    assert eur_datasets[0].dataset_id == meta1.dataset_id

    h1_datasets = repo.list_datasets(timeframe=Timeframe.H1)
    assert len(h1_datasets) == 1
    assert h1_datasets[0].dataset_id == meta2.dataset_id
