"""Unit tests for prooflab.data.cleaner."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from prooflab.data.cleaner import CleaningConfig, DataCleaner, ForwardFillConfig


@pytest.fixture
def clean_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=i) for i in range(10)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EURUSD"] * 10,
            "timeframe": ["M1"] * 10,
            "open": [1.1000 + i * 0.0001 for i in range(10)],
            "high": [1.1010 + i * 0.0001 for i in range(10)],
            "low": [1.0990 + i * 0.0001 for i in range(10)],
            "close": [1.1005 + i * 0.0001 for i in range(10)],
            "volume": [100.0] * 10,
            "tick_volume": [150.0] * 10,
            "spread": [1.5] * 10,
            "source": ["mt5"] * 10,
        }
    )


def test_cleaner_clean_data(clean_df: pd.DataFrame) -> None:
    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(clean_df)

    assert len(cleaned_df) == 10
    assert record.rows_before == 10
    assert record.rows_after == 10
    assert record.imputation_applied is False
    assert len(record.operations) == 0


def test_cleaner_empty_data() -> None:
    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(pd.DataFrame())

    assert len(cleaned_df) == 0
    assert record.rows_before == 0
    assert record.rows_after == 0


def test_cleaner_removes_malformed_rows(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[2, "close"] = np.nan
    df.loc[7, "open"] = np.nan

    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df)

    assert len(cleaned_df) == 8
    assert record.rows_before == 10
    assert record.rows_after == 8
    assert any(op.name == "remove_malformed_rows" for op in record.operations)
    malformed_op = next(op for op in record.operations if op.name == "remove_malformed_rows")
    assert malformed_op.rows_affected == 2


def test_cleaner_removes_duplicates(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # Duplicate row 3
    dup_row = df.iloc[[3]]
    df = pd.concat([df.iloc[:4], dup_row, df.iloc[4:]]).reset_index(drop=True)
    assert len(df) == 11

    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df)

    assert len(cleaned_df) == 10
    assert record.rows_before == 11
    assert record.rows_after == 10
    assert any(op.name == "remove_duplicates" for op in record.operations)


def test_cleaner_sorts_timestamps(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # Reverse order
    df = df.iloc[::-1].reset_index(drop=True)

    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df)

    assert len(cleaned_df) == 10
    assert pd.to_datetime(cleaned_df["timestamp"]).is_monotonic_increasing
    assert any(op.name == "sort_chronologically" for op in record.operations)


def test_forward_fill_disabled_by_default(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # Create NaN in close, but disable remove_malformed_rows to test ffill behavior
    df.loc[3, "close"] = np.nan

    config = CleaningConfig(remove_malformed_rows=False)
    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df, config=config)

    assert pd.isna(cleaned_df.loc[3, "close"])
    assert record.imputation_applied is False
    assert not any(op.name == "forward_fill" for op in record.operations)


def test_forward_fill_requires_reason() -> None:
    with pytest.raises(ValidationError, match="explicit justification reason"):
        ForwardFillConfig(enabled=True, reason="")


def test_forward_fill_explicitly_configured(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[3, "close"] = np.nan

    config = CleaningConfig(
        remove_malformed_rows=False,
        forward_fill=ForwardFillConfig(
            enabled=True,
            max_gap_periods=1,
            reason="Market microstructure auction tick gap bridging.",
        ),
    )
    cleaner = DataCleaner()
    cleaned_df, record = cleaner.clean(df, config=config)

    # Value should be filled from row 2
    assert cleaned_df.loc[3, "close"] == clean_df.loc[2, "close"]
    assert record.imputation_applied is True
    assert any(op.name == "forward_fill" for op in record.operations)
    ffill_op = next(op for op in record.operations if op.name == "forward_fill")
    assert ffill_op.rows_affected == 1
    assert "Market microstructure" in ffill_op.description


def test_cleaning_record_summary(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[2, "close"] = np.nan
    cleaner = DataCleaner()
    _, record = cleaner.clean(df)

    summary_str = record.summary()
    assert "Data Cleaning Audit Record" in summary_str
    assert "Rows Before: 10" in summary_str
    assert "Rows After: 9" in summary_str
