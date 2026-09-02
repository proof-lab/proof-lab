"""Dataset versioning and metadata management.

Implements immutable dataset metadata tracking with SHA-256 checksums,
ensuring strict reproducibility and integrity across experiments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.data.schema import Timeframe


class DatasetIntegrityError(Exception):
    """Raised when a dataset's checksum does not match its recorded metadata."""


class DatasetMetadata(BaseModel):
    """Immutable metadata tracking a versioned dataset snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    symbol: str
    timeframe: Timeframe
    start_time: datetime
    end_time: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checksum: str
    row_count: int
    feature_version: str | None = None

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata to a formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> DatasetMetadata:
        """Construct DatasetMetadata from a JSON string."""
        data = json.loads(json_str)
        return cls.model_validate(data)


def compute_checksum(target: bytes | Path | str) -> str:
    """Calculate the SHA-256 hex digest of raw bytes or a file on disk.

    Args:
        target: Raw bytes or Path/str to a file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    hasher = hashlib.sha256()
    if isinstance(target, bytes):
        hasher.update(target)
    else:
        path = Path(target).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found for checksum: {path}")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def create_dataset_metadata(
    df: pd.DataFrame,
    source: str,
    symbol: str,
    timeframe: Timeframe,
    parquet_bytes: bytes,
    dataset_id: str | None = None,
    feature_version: str | None = None,
) -> DatasetMetadata:
    """Create a DatasetMetadata record from a DataFrame and its Parquet bytes.

    Args:
        df: DataFrame containing the dataset records.
        source: Data source identifier (e.g. 'mt5').
        symbol: Symbol identifier (e.g. 'EURUSD').
        timeframe: Bar timeframe.
        parquet_bytes: Raw binary bytes of the persisted Parquet file.
        dataset_id: Optional explicit dataset UUID string.
        feature_version: Reserved feature pipeline version tag (defaults to None).

    Returns:
        Immutable DatasetMetadata object.
    """
    if df.empty:
        raise ValueError("Cannot create dataset metadata for an empty DataFrame.")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    timestamps = pd.to_datetime(df["timestamp"])
    start_time = timestamps.min()
    end_time = timestamps.max()

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize(UTC)
    else:
        start_time = start_time.tz_convert(UTC)

    if end_time.tzinfo is None:
        end_time = end_time.tz_localize(UTC)
    else:
        end_time = end_time.tz_convert(UTC)

    checksum = compute_checksum(parquet_bytes)

    kwargs: dict[str, Any] = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_time": start_time.to_pydatetime(),
        "end_time": end_time.to_pydatetime(),
        "checksum": checksum,
        "row_count": len(df),
        "feature_version": feature_version,
    }
    if dataset_id is not None:
        kwargs["dataset_id"] = dataset_id

    return DatasetMetadata(**kwargs)


def verify_dataset_integrity(
    target: bytes | Path | str,
    metadata: DatasetMetadata,
) -> bool:
    """Verify that dataset contents match recorded metadata checksum.

    Args:
        target: Parquet bytes or path to Parquet file.
        metadata: The expected metadata for the dataset.

    Returns:
        True if checksum matches.

    Raises:
        DatasetIntegrityError: If checksums do not match.
    """
    actual_checksum = compute_checksum(target)
    if actual_checksum != metadata.checksum:
        raise DatasetIntegrityError(
            f"Dataset integrity violation for dataset {metadata.dataset_id}! "
            f"Expected checksum {metadata.checksum}, found {actual_checksum}."
        )
    return True


def save_metadata(
    metadata: DatasetMetadata,
    path: Path | str,
    overwrite: bool = False,
) -> Path:
    """Save metadata to a JSON sidecar file.

    Datasets are immutable once versioned; overwriting is forbidden by default.

    Args:
        metadata: DatasetMetadata object.
        path: Path to target .meta.json file.
        overwrite: Whether to allow overwriting an existing file (default: False).

    Returns:
        Resolved destination Path.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset metadata file already exists and is immutable: {target}"
        )
    target.write_text(metadata.to_json(), encoding="utf-8")
    return target


def load_metadata(path: Path | str) -> DatasetMetadata:
    """Load and parse metadata from a JSON sidecar file.

    Args:
        path: Path to .meta.json file.

    Returns:
        Parsed DatasetMetadata instance.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Metadata file not found: {target}")
    return DatasetMetadata.from_json(target.read_text(encoding="utf-8"))
