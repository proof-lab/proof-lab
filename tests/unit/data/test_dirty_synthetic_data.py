"""Comprehensive unit tests exercising DataValidator and DataCleaner on dirty synthetic data.

Tests combinatoric anomalies, failure mode isolation, full dirty-to-clean transformations,
and repository integration.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from prooflab.data.cleaner import DataCleaner
from prooflab.data.health import generate_health_report
from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe
from prooflab.data.validator import DataValidator


def generate_deliberately_dirty_dataset(num_bars: int = 50) -> pd.DataFrame:
    """Generate synthetic market data with multiple deliberate defects."""
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_time + timedelta(minutes=i) for i in range(num_bars)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EURUSD"] * num_bars,
            "timeframe": ["M1"] * num_bars,
            "open": [1.1000 + (i * 0.0001) for i in range(num_bars)],
            "high": [1.1010 + (i * 0.0001) for i in range(num_bars)],
            "low": [1.0990 + (i * 0.0001) for i in range(num_bars)],
            "close": [1.1005 + (i * 0.0001) for i in range(num_bars)],
            "volume": [100.0] * num_bars,
            "tick_volume": [150.0] * num_bars,
            "spread": [1.2] * num_bars,
            "source": ["synthetic"] * num_bars,
        }
    )

    # Defect 1: Corrupted / NaN rows
    df.loc[5, "open"] = np.nan
    df.loc[12, "high"] = np.nan

    # Defect 2: Duplicate timestamps
    df.loc[8, "timestamp"] = df.loc[7, "timestamp"]
    df.loc[25, "timestamp"] = df.loc[24, "timestamp"]

    # Defect 3: Timestamp disorder
    temp_ts = df.loc[15, "timestamp"]
    df.loc[15, "timestamp"] = df.loc[16, "timestamp"]
    df.loc[16, "timestamp"] = temp_ts

    # Defect 4: Impossible OHLC bounds (High < Low)
    df.loc[20, "high"] = 1.0500  # Low is ~1.0990
    df.loc[21, "open"] = 1.2500  # High is ~1.1010

    # Defect 5: Negative price
    df.loc[30, "close"] = -1.1000

    # Defect 6: Negative volume
    df.loc[35, "volume"] = -500.0

    # Defect 7: Invalid negative spread
    df.loc[40, "spread"] = -2.0

    # Defect 8: Missing timeframe gap
    for i in range(45, num_bars):
        df.loc[i, "timestamp"] = df.loc[i, "timestamp"] + timedelta(minutes=15)

    return df


def test_validator_detects_all_anomalies_simultaneously() -> None:
    dirty_df = generate_deliberately_dirty_dataset(50)
    validator = DataValidator()
    result = validator.validate(dirty_df, timeframe=Timeframe.M1)

    assert result.is_valid is False
    assert result.has_errors is True

    detected_rules = {issue.rule for issue in result.issues}

    # Verify each specific failure mode was detected in the single pass
    assert "corrupted_rows" in detected_rules
    assert "duplicate_timestamps" in detected_rules
    assert "timestamp_disorder" in detected_rules
    assert "impossible_ohlc_relationships" in detected_rules
    assert "negative_prices_or_volume" in detected_rules
    assert "invalid_spreads" in detected_rules
    assert "missing_timestamps" in detected_rules
    assert "extreme_unexplained_gaps" in detected_rules


def test_cleaner_transforms_dirty_data_into_valid_dataset() -> None:
    # Generate dirty dataset with cleaner-supported defects (corrupted rows, duplicates, disorder)
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_time + timedelta(minutes=i) for i in range(30)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EURUSD"] * 30,
            "timeframe": ["M1"] * 30,
            "open": [1.1000 + (i * 0.0001) for i in range(30)],
            "high": [1.1010 + (i * 0.0001) for i in range(30)],
            "low": [1.0990 + (i * 0.0001) for i in range(30)],
            "close": [1.1005 + (i * 0.0001) for i in range(30)],
            "volume": [100.0] * 30,
            "tick_volume": [150.0] * 30,
            "spread": [1.2] * 30,
            "source": ["synthetic"] * 30,
        }
    )

    # Add defects
    df.loc[3, "close"] = np.nan
    df.loc[10, "timestamp"] = df.loc[9, "timestamp"]
    temp = df.loc[15, "timestamp"]
    df.loc[15, "timestamp"] = df.loc[16, "timestamp"]
    df.loc[16, "timestamp"] = temp

    validator = DataValidator()
    raw_result = validator.validate(df, timeframe=Timeframe.M1)
    assert raw_result.is_valid is False

    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df)

    assert record.rows_before == 30
    assert record.rows_after < 30
    assert len(record.operations) >= 3

    # Cleaned data must have sorted timestamps and no nulls/duplicates
    clean_result = validator.validate(cleaned_df, timeframe=Timeframe.M1)
    assert len(clean_result.get_issues_by_rule("corrupted_rows")) == 0
    assert len(clean_result.get_issues_by_rule("duplicate_timestamps")) == 0
    assert len(clean_result.get_issues_by_rule("timestamp_disorder")) == 0


def test_full_data_engine_end_to_end_pipeline(tmp_path: Path) -> None:
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_time + timedelta(minutes=i) for i in range(20)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EURUSD"] * 20,
            "timeframe": ["M1"] * 20,
            "open": [1.1000 + (i * 0.0001) for i in range(20)],
            "high": [1.1010 + (i * 0.0001) for i in range(20)],
            "low": [1.0990 + (i * 0.0001) for i in range(20)],
            "close": [1.1005 + (i * 0.0001) for i in range(20)],
            "volume": [100.0] * 20,
            "tick_volume": [150.0] * 20,
            "spread": [1.2] * 20,
            "source": ["mt5"] * 20,
        }
    )

    # Introduce duplicate row and disorder
    dup_row = df.iloc[[-1]]
    df = pd.concat([df, dup_row]).iloc[::-1].reset_index(drop=True)

    # 1. Health report on dirty data (disorder and duplicate)
    dirty_report = generate_health_report(df, timeframe=Timeframe.M1)
    assert dirty_report.is_valid is False
    assert dirty_report.duplicate_rows >= 1
    assert dirty_report.disordered_rows >= 1

    # 2. Clean data
    cleaner = DataCleaner()
    cleaned_df, cleaning_record = cleaner.clean(df)
    assert cleaning_record.rows_after == 20

    # 3. Health report on cleaned data
    clean_report = generate_health_report(cleaned_df, timeframe=Timeframe.M1)
    assert clean_report.is_valid is True
    assert clean_report.completeness == 1.0

    # 4. Version and persist in ParquetRepository
    repo = ParquetRepository(base_dir=tmp_path / "data_repo")
    metadata = repo.save_dataset(
        df=cleaned_df,
        source="mt5",
        symbol="EURUSD",
        timeframe=Timeframe.M1,
    )
    assert metadata.row_count == 20
    assert metadata.checksum is not None

    # 5. Load and verify integrity
    loaded_df, loaded_meta = repo.load_dataset(metadata.dataset_id)
    assert len(loaded_df) == 20
    assert loaded_meta.checksum == metadata.checksum
