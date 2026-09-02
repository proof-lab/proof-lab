"""Unit tests for prooflab.data.validator."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.data.schema import Timeframe
from prooflab.data.validator import DataValidator, ValidationSeverity


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
            "spread": [1.0] * 10,
            "source": ["mt5"] * 10,
        }
    )


def test_validator_clean_data(clean_df: pd.DataFrame) -> None:
    validator = DataValidator()
    result = validator.validate(clean_df, timeframe=Timeframe.M1)
    assert result.is_valid is True
    assert result.error_count == 0
    assert result.warning_count == 0
    assert len(result.issues) == 0


def test_validator_empty_data() -> None:
    validator = DataValidator()
    result = validator.validate(pd.DataFrame())
    assert result.is_valid is False
    assert result.has_errors is True


def test_validator_missing_columns() -> None:
    df = pd.DataFrame({"open": [1.1], "close": [1.2]})
    validator = DataValidator()
    result = validator.validate(df)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("missing_required_columns")
    assert len(issues) == 1


def test_validator_corrupted_rows(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[2, "close"] = np.nan
    df.loc[5, "high"] = np.nan

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("corrupted_rows")
    assert len(issues) == 1
    assert issues[0].count == 2
    assert 2 in issues[0].affected_indices
    assert 5 in issues[0].affected_indices


def test_validator_duplicate_timestamps(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[3, "timestamp"] = df.loc[2, "timestamp"]

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("duplicate_timestamps")
    assert len(issues) == 1
    assert issues[0].count == 2


def test_validator_timestamp_disorder(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # Swap row 3 and 4
    temp = df.loc[3, "timestamp"]
    df.loc[3, "timestamp"] = df.loc[4, "timestamp"]
    df.loc[4, "timestamp"] = temp

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("timestamp_disorder")
    assert len(issues) == 1


def test_validator_impossible_ohlc_high_low(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[1, "high"] = 1.0900  # Lower than low 1.0991

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("impossible_ohlc_relationships")
    assert len(issues) == 1
    assert 1 in issues[0].affected_indices


def test_validator_impossible_ohlc_open_close_bounds(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[2, "open"] = 1.2000  # Higher than high
    df.loc[3, "close"] = 1.0000  # Lower than low

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("impossible_ohlc_relationships")
    assert len(issues) == 1
    assert issues[0].count == 2


def test_validator_negative_prices_and_volume(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[2, "close"] = -1.1000
    df.loc[4, "volume"] = -50.0

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("negative_prices_or_volume")
    assert len(issues) == 1
    assert issues[0].count == 2


def test_validator_invalid_spreads(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    df.loc[3, "spread"] = -1.5

    validator = DataValidator()
    result = validator.validate(df, timeframe=Timeframe.M1)
    assert result.is_valid is False
    issues = result.get_issues_by_rule("invalid_spreads")
    assert len(issues) == 1
    assert 3 in issues[0].affected_indices


def test_validator_missing_timestamps_and_extreme_gaps(clean_df: pd.DataFrame) -> None:
    df = clean_df.copy()
    # Introduce a 20-minute gap between index 4 and 5 on M1 timeframe
    for i in range(5, 10):
        df.loc[i, "timestamp"] = df.loc[i, "timestamp"] + timedelta(minutes=20)

    validator = DataValidator(extreme_gap_multiplier=10.0)
    result = validator.validate(df, timeframe=Timeframe.M1)

    missing_issues = result.get_issues_by_rule("missing_timestamps")
    assert len(missing_issues) == 1
    assert missing_issues[0].severity == ValidationSeverity.ERROR

    extreme_issues = result.get_issues_by_rule("extreme_unexplained_gaps")
    assert len(extreme_issues) == 1
    assert extreme_issues[0].severity == ValidationSeverity.WARNING
