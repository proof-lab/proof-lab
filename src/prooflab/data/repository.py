"""Data repository abstraction and Parquet-backed implementation.

Provides persistent dataset storage, versioning, indexing, and retrieval
behind a clean repository interface.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from prooflab.data.schema import Timeframe
from prooflab.data.storage import read_parquet, write_parquet
from prooflab.data.versioning import (
    DatasetMetadata,
    create_dataset_metadata,
    load_metadata,
    save_metadata,
    verify_dataset_integrity,
)


class DataRepository(ABC):
    """Abstract interface for dataset persistence and version retrieval."""

    @abstractmethod
    def save_dataset(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: Timeframe,
        feature_version: str | None = None,
    ) -> DatasetMetadata:
        """Persist a dataset and return its immutable metadata."""

    @abstractmethod
    def load_dataset(
        self,
        dataset_id: str,
    ) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Load a dataset by its unique ID, verifying its integrity."""

    @abstractmethod
    def get_metadata(
        self,
        dataset_id: str,
    ) -> DatasetMetadata:
        """Retrieve metadata for a dataset ID without loading the data."""

    @abstractmethod
    def list_datasets(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
    ) -> list[DatasetMetadata]:
        """List all stored dataset metadata matching optional filters."""


class ParquetRepository(DataRepository):
    """Filesystem-based Parquet and JSON sidecar repository implementation."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_dir(
        self,
        source: str,
        symbol: str,
        timeframe: str,
    ) -> Path:
        return self.base_dir / source / symbol / timeframe

    def _find_dataset_paths(
        self,
        dataset_id: str,
    ) -> tuple[Path, Path]:
        """Search for the parquet and metadata files of a given dataset_id."""
        matches = list(self.base_dir.glob(f"**/{dataset_id}.parquet"))
        if not matches:
            raise FileNotFoundError(f"Dataset with ID '{dataset_id}' not found.")
        parquet_path = matches[0]
        meta_path = parquet_path.with_name(f"{dataset_id}.meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing for dataset '{dataset_id}'.")
        return parquet_path, meta_path

    def save_dataset(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: Timeframe,
        feature_version: str | None = None,
    ) -> DatasetMetadata:
        """Persist a DataFrame as an immutable versioned Parquet dataset."""
        dataset_dir = self._get_dataset_dir(source, symbol, timeframe.value)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=dataset_dir) as tmp_dir:
            tmp_parquet = Path(tmp_dir) / "temp.parquet"
            write_parquet(df, tmp_parquet)
            parquet_bytes = tmp_parquet.read_bytes()

            metadata = create_dataset_metadata(
                df=df,
                source=source,
                symbol=symbol,
                timeframe=timeframe,
                parquet_bytes=parquet_bytes,
                feature_version=feature_version,
            )

            final_parquet = dataset_dir / f"{metadata.dataset_id}.parquet"
            final_meta = dataset_dir / f"{metadata.dataset_id}.meta.json"

            if final_parquet.exists() or final_meta.exists():
                raise FileExistsError(
                    f"Dataset ID collision or existing immutable file: {metadata.dataset_id}"
                )

            # Move parquet to destination and save metadata
            tmp_parquet.replace(final_parquet)
            save_metadata(metadata, final_meta, overwrite=False)

        return metadata

    def load_dataset(
        self,
        dataset_id: str,
    ) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Load and verify dataset by ID."""
        parquet_path, meta_path = self._find_dataset_paths(dataset_id)
        metadata = load_metadata(meta_path)
        verify_dataset_integrity(parquet_path, metadata)
        df = read_parquet(parquet_path)
        return df, metadata

    def get_metadata(
        self,
        dataset_id: str,
    ) -> DatasetMetadata:
        """Get dataset metadata without loading parquet payload."""
        _, meta_path = self._find_dataset_paths(dataset_id)
        return load_metadata(meta_path)

    def list_datasets(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
    ) -> list[DatasetMetadata]:
        """List metadata for all matching datasets."""
        meta_files = list(self.base_dir.glob("**/*.meta.json"))
        results: list[DatasetMetadata] = []
        for meta_file in meta_files:
            try:
                meta = load_metadata(meta_file)
                if symbol is not None and meta.symbol != symbol:
                    continue
                if timeframe is not None and meta.timeframe != timeframe:
                    continue
                results.append(meta)
            except Exception:
                continue
        return sorted(results, key=lambda m: m.created_at)
