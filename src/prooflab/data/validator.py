"""Market data validation engine.

Detects corrupted rows, duplicate timestamps, impossible OHLC values,
negative prices/volumes, invalid spreads, ordering violations, and unexplained gaps.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.data.schema import Timeframe

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    Timeframe.M1.value: timedelta(minutes=1),
    Timeframe.M5.value: timedelta(minutes=5),
    Timeframe.M15.value: timedelta(minutes=15),
    Timeframe.M30.value: timedelta(minutes=30),
    Timeframe.H1.value: timedelta(hours=1),
    Timeframe.H4.value: timedelta(hours=4),
    Timeframe.D1.value: timedelta(days=1),
    Timeframe.W1.value: timedelta(weeks=1),
    Timeframe.MN1.value: timedelta(days=30),
}


class ValidationSeverity(StrEnum):
    """Severity of a detected validation issue."""

    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationIssue(BaseModel):
    """An individual data quality or integrity issue detected in a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    severity: ValidationSeverity
    description: str
    count: int
    affected_indices: list[int] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Overall outcome of running DataValidator over a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    total_rows: int = 0

    @property
    def has_errors(self) -> bool:
        """True if there are any ERROR-level issues."""
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        """True if there are any WARNING-level issues."""
        return self.warning_count > 0

    def get_issues_by_rule(self, rule_name: str) -> list[ValidationIssue]:
        """Filter issues by specific rule identifier."""
        return [issue for issue in self.issues if issue.rule == rule_name]


class DataValidator:
    """Validates market data DataFrames against quantitative correctness rules."""

    def __init__(self, extreme_gap_multiplier: float = 10.0) -> None:
        """Initialize DataValidator.

        Args:
            extreme_gap_multiplier: Factor over expected interval considered
                                    an extreme unexplained gap (default: 10.0).
        """
        self.extreme_gap_multiplier = extreme_gap_multiplier

    def validate(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe | str | None = None,
    ) -> ValidationResult:
        """Run all data quality checks on the provided DataFrame.

        Args:
            df: Market data DataFrame.
            timeframe: Expected bar timeframe (inferred from df if None).

        Returns:
            ValidationResult with all detected issues.
        """
        if df.empty:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        rule="empty_dataset",
                        severity=ValidationSeverity.ERROR,
                        description="Dataset contains no rows.",
                        count=0,
                    )
                ],
                error_count=1,
                warning_count=0,
                total_rows=0,
            )

        issues: list[ValidationIssue] = []

        # 1. Corrupted rows (Null / NaN values in essential columns)
        self._check_corrupted_rows(df, issues)

        # 2. Duplicate timestamps
        self._check_duplicate_timestamps(df, issues)

        # 3. Timestamp disorder (out-of-order sequence)
        self._check_timestamp_disorder(df, issues)

        # 4. Impossible OHLC relationships
        self._check_ohlc_relationships(df, issues)

        # 5. Negative prices or volume
        self._check_negative_values(df, issues)

        # 6. Invalid spreads
        self._check_invalid_spreads(df, issues)

        # 7 & 8. Missing timestamps and Extreme Gaps
        tf_str: str | None = None
        if timeframe is not None:
            tf_str = timeframe.value if isinstance(timeframe, Timeframe) else str(timeframe)
        elif "timeframe" in df.columns and not df["timeframe"].empty:
            tf_str = str(df["timeframe"].iloc[0])

        if tf_str and tf_str in TIMEFRAME_DELTAS and "timestamp" in df.columns:
            self._check_timestamp_gaps(df, tf_str, issues)

        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

        return ValidationResult(
            is_valid=(error_count == 0),
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
            total_rows=len(df),
        )

    def _check_corrupted_rows(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        required_cols = ["timestamp", "open", "high", "low", "close"]
        existing_required = [c for c in required_cols if c in df.columns]
        if len(existing_required) < len(required_cols):
            missing_cols = set(required_cols) - set(df.columns)
            issues.append(
                ValidationIssue(
                    rule="missing_required_columns",
                    severity=ValidationSeverity.ERROR,
                    description=f"DataFrame is missing required columns: {sorted(missing_cols)}",
                    count=len(missing_cols),
                )
            )
            return

        null_mask = df[existing_required].isna().any(axis=1)
        if null_mask.any():
            bad_indices = df.index[null_mask].tolist()
            issues.append(
                ValidationIssue(
                    rule="corrupted_rows",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Found {len(bad_indices)} row(s) with null/NaN values in required columns."
                    ),
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_duplicate_timestamps(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        if "timestamp" not in df.columns:
            return
        subset = ["timestamp"]
        if "symbol" in df.columns:
            subset.append("symbol")
        if "timeframe" in df.columns:
            subset.append("timeframe")

        dup_mask = df.duplicated(subset=subset, keep=False)
        if dup_mask.any():
            bad_indices = df.index[dup_mask].tolist()
            issues.append(
                ValidationIssue(
                    rule="duplicate_timestamps",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {len(bad_indices)} row(s) with duplicate timestamps.",
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_timestamp_disorder(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        if "timestamp" not in df.columns or len(df) <= 1:
            return

        ts = pd.to_datetime(df["timestamp"])
        diffs = ts.diff().dropna()
        disordered_mask = diffs < pd.Timedelta(0)
        if disordered_mask.any():
            # The indices where diff is negative (shifted by 1 for diff offset)
            bad_indices = df.index[1:][disordered_mask].tolist()
            issues.append(
                ValidationIssue(
                    rule="timestamp_disorder",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Found {len(bad_indices)} timestamp(s) not in ascending order."
                    ),
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_ohlc_relationships(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in cols):
            return

        # High must be >= Low
        high_low_violation = df["high"] < df["low"]
        # Open must be between Low and High
        open_high_violation = df["open"] > df["high"]
        open_low_violation = df["open"] < df["low"]
        # Close must be between Low and High
        close_high_violation = df["close"] > df["high"]
        close_low_violation = df["close"] < df["low"]

        ohlc_mask = (
            high_low_violation
            | open_high_violation
            | open_low_violation
            | close_high_violation
            | close_low_violation
        )

        if ohlc_mask.any():
            bad_indices = df.index[ohlc_mask].tolist()
            issues.append(
                ValidationIssue(
                    rule="impossible_ohlc_relationships",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Found {len(bad_indices)} row(s) with impossible OHLC relationships "
                        "(e.g. High < Low, or Open/Close outside High-Low range)."
                    ),
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_negative_values(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        vol_cols = [c for c in ["volume", "tick_volume"] if c in df.columns]

        neg_price_mask = pd.Series(False, index=df.index)
        for col in price_cols:
            neg_price_mask |= df[col] < 0

        neg_vol_mask = pd.Series(False, index=df.index)
        for col in vol_cols:
            neg_vol_mask |= df[col] < 0

        combined = neg_price_mask | neg_vol_mask
        if combined.any():
            bad_indices = df.index[combined].tolist()
            issues.append(
                ValidationIssue(
                    rule="negative_prices_or_volume",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Found {len(bad_indices)} row(s) with negative price or volume values."
                    ),
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_invalid_spreads(
        self,
        df: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        if "spread" not in df.columns:
            return
        invalid_spread_mask = df["spread"] < 0
        if invalid_spread_mask.any():
            bad_indices = df.index[invalid_spread_mask].tolist()
            issues.append(
                ValidationIssue(
                    rule="invalid_spreads",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {len(bad_indices)} row(s) with negative spread.",
                    count=len(bad_indices),
                    affected_indices=bad_indices,
                )
            )

    def _check_timestamp_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        issues: list[ValidationIssue],
    ) -> None:
        if len(df) <= 1:
            return

        expected_delta = TIMEFRAME_DELTAS[timeframe]
        ts = pd.to_datetime(df["timestamp"]).sort_values()
        diffs = ts.diff().dropna()

        # Missing timestamps: any gap > 1x expected delta
        missing_mask = diffs > expected_delta
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            total_missing_intervals = int(
                np.sum(np.floor(diffs[missing_mask] / expected_delta)) - missing_count
            )
            issues.append(
                ValidationIssue(
                    rule="missing_timestamps",
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Detected {missing_count} time gap(s) exceeding {timeframe} interval "
                        f"(~{total_missing_intervals} missing bar intervals)."
                    ),
                    count=missing_count,
                    details={"total_missing_intervals": total_missing_intervals},
                )
            )

        # Extreme gaps: gap > extreme_gap_multiplier * expected_delta
        extreme_threshold = expected_delta * self.extreme_gap_multiplier
        extreme_mask = diffs > extreme_threshold
        if extreme_mask.any():
            extreme_count = int(extreme_mask.sum())
            issues.append(
                ValidationIssue(
                    rule="extreme_unexplained_gaps",
                    severity=ValidationSeverity.WARNING,
                    description=(
                        f"Detected {extreme_count} extreme gap(s) exceeding "
                        f"{self.extreme_gap_multiplier}x expected timeframe interval."
                    ),
                    count=extreme_count,
                )
            )
