"""Unit tests for prooflab.data.health."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.data.health import generate_health_report
from prooflab.data.schema import Timeframe


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


def test_generate_health_report_clean_data(clean_df: pd.DataFrame) -> None:
    report = generate_health_report(clean_df, timeframe=Timeframe.M1)

    assert report.is_valid is True
    assert report.row_count == 10
    assert report.symbols == ["EURUSD"]
    assert report.timeframes == ["M1"]
    assert report.sources == ["mt5"]
    assert report.missing_rows == 0
    assert report.duplicate_rows == 0
    assert report.invalid_rows == 0
    assert report.disordered_rows == 0
    assert report.missing_intervals == 0
    assert report.completeness == 1.0
    assert report.median_spread == 1.5
    assert report.max_spread == 1.5
    assert report.start_time == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    assert report.end_time == datetime(2026, 1, 1, 10, 9, tzinfo=UTC)


def test_health_report_json_and_summary(clean_df: pd.DataFrame) -> None:
    report = generate_health_report(clean_df, timeframe=Timeframe.M1)

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["row_count"] == 10
    assert parsed["is_valid"] is True

    summary_str = report.summary()
    assert "Dataset Health Report [PASSED]" in summary_str
    assert "EURUSD" in summary_str
    assert "Completeness: 100.00%" in summary_str


def test_generate_health_report_dirty_data(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # 1 corrupted row
    df.loc[2, "close"] = np.nan
    # 1 duplicate timestamp
    df.loc[4, "timestamp"] = df.loc[3, "timestamp"]
    # 1 invalid OHLC row
    df.loc[6, "high"] = 1.0800  # < low
    # Missing interval gap between 8 and 9 (shift 9 by 5 mins)
    df.loc[9, "timestamp"] = df.loc[9, "timestamp"] + timedelta(minutes=5)

    report = generate_health_report(df, timeframe=Timeframe.M1)

    assert report.is_valid is False
    assert report.missing_rows >= 1
    assert report.duplicate_rows >= 1
    assert report.invalid_rows >= 1
    assert report.missing_intervals >= 1
    assert report.completeness < 1.0


def test_generate_health_report_empty() -> None:
    report = generate_health_report(pd.DataFrame())
    assert report.is_valid is False
    assert report.row_count == 0
    assert report.completeness == 0.0
