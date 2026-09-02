"""Proof Lab Data Engine.

Provides canonical schemas, immutable dataset versioning, high-performance
storage, quantitative validation, health diagnostics, and audit-logged cleaning.
"""

from prooflab.data.cleaner import (
    CleaningConfig,
    CleaningOperation,
    CleaningRecord,
    DataCleaner,
    ForwardFillConfig,
)
from prooflab.data.health import HealthReport, generate_health_report
from prooflab.data.repository import DataRepository, ParquetRepository
from prooflab.data.schema import (
    OHLCV_COLUMNS,
    TICK_COLUMNS,
    OHLCVBar,
    TickData,
    Timeframe,
)
from prooflab.data.storage import (
    DuckDBAccessLayer,
    read_parquet,
    read_parquet_bytes,
    write_parquet,
)
from prooflab.data.validator import (
    DataValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from prooflab.data.versioning import (
    DatasetIntegrityError,
    DatasetMetadata,
    compute_checksum,
    create_dataset_metadata,
    load_metadata,
    save_metadata,
    verify_dataset_integrity,
)

__all__ = [
    "CleaningConfig",
    "CleaningOperation",
    "CleaningRecord",
    "DataCleaner",
    "DataRepository",
    "DataValidator",
    "DatasetIntegrityError",
    "DatasetMetadata",
    "DuckDBAccessLayer",
    "ForwardFillConfig",
    "HealthReport",
    "OHLCVBar",
    "OHLCV_COLUMNS",
    "ParquetRepository",
    "TICK_COLUMNS",
    "TickData",
    "Timeframe",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "compute_checksum",
    "create_dataset_metadata",
    "generate_health_report",
    "load_metadata",
    "read_parquet",
    "read_parquet_bytes",
    "save_metadata",
    "verify_dataset_integrity",
    "write_parquet",
]
