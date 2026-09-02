"""Dataset health report generator.

Produces quantitative diagnostics for a market dataset snapshot,
including completeness metrics, spread statistics, error classifications,
and temporal coverage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.data.schema import Timeframe
from prooflab.data.validator import DataValidator, ValidationIssue, ValidationResult


class HealthReport(BaseModel):
    """Structured health report containing quantitative diagnostics for a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: list[str]
    timeframes: list[str]
    sources: list[str]
    row_count: int
    missing_rows: int
    duplicate_rows: int
    invalid_rows: int
    disordered_rows: int
    start_time: datetime | None
    end_time: datetime | None
    missing_intervals: int
    median_spread: float | None
    max_spread: float | None
    completeness: float
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self, indent: int = 2) -> str:
        """Serialize the health report to formatted JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    def summary(self) -> str:
        """Return a formatted human-readable summary string."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Dataset Health Report [{status}] ===",
            f"Symbols: {', '.join(self.symbols) or 'N/A'}",
            f"Timeframes: {', '.join(self.timeframes) or 'N/A'}",
            f"Sources: {', '.join(self.sources) or 'N/A'}",
            f"Row Count: {self.row_count:,}",
            f"Time Span: {self.start_time} to {self.end_time}",
            f"Completeness: {self.completeness:.2%}",
            f"Missing Intervals: {self.missing_intervals:,}",
            f"Missing/Corrupted Rows: {self.missing_rows:,}",
            f"Duplicate Rows: {self.duplicate_rows:,}",
            f"Disordered Rows: {self.disordered_rows:,}",
            f"Invalid Value Rows: {self.invalid_rows:,}",
            f"Spread (Median / Max): {self.median_spread} / {self.max_spread}",
            f"Total Issues Detected: {len(self.issues)}",
        ]
        return "\n".join(lines)


def generate_health_report(
    df: pd.DataFrame,
    timeframe: Timeframe | str | None = None,
    validation_result: ValidationResult | None = None,
) -> HealthReport:
    """Generate a comprehensive HealthReport for a market data DataFrame.

    Args:
        df: Input DataFrame.
        timeframe: Expected timeframe (optional).
        validation_result: Precomputed ValidationResult (optional, validated if None).

    Returns:
        HealthReport instance.
    """
    if validation_result is None:
        validator = DataValidator()
        validation_result = validator.validate(df, timeframe=timeframe)

    row_count = len(df)
    symbols = (
        sorted(df["symbol"].dropna().unique().astype(str).tolist())
        if "symbol" in df.columns
        else []
    )
    timeframes = (
        sorted(df["timeframe"].dropna().unique().astype(str).tolist())
        if "timeframe" in df.columns
        else []
    )
    sources = (
        sorted(df["source"].dropna().unique().astype(str).tolist())
        if "source" in df.columns
        else []
    )

    start_time: datetime | None = None
    end_time: datetime | None = None
    if "timestamp" in df.columns and not df.empty:
        valid_ts = pd.to_datetime(df["timestamp"].dropna())
        if not valid_ts.empty:
            min_ts = valid_ts.min()
            max_ts = valid_ts.max()
            start_time = (
                min_ts.tz_localize(UTC) if min_ts.tzinfo is None else min_ts.tz_convert(UTC)
            ).to_pydatetime()
            end_time = (
                max_ts.tz_localize(UTC) if max_ts.tzinfo is None else max_ts.tz_convert(UTC)
            ).to_pydatetime()

    # Aggregate issue categories from validation result
    corrupted_count = sum(
        i.count for i in validation_result.get_issues_by_rule("corrupted_rows")
    )
    duplicate_count = sum(
        i.count for i in validation_result.get_issues_by_rule("duplicate_timestamps")
    )
    disorder_count = sum(
        i.count for i in validation_result.get_issues_by_rule("timestamp_disorder")
    )

    invalid_rules = [
        "impossible_ohlc_relationships",
        "negative_prices_or_volume",
        "invalid_spreads",
    ]
    invalid_count = sum(
        i.count
        for rule in invalid_rules
        for i in validation_result.get_issues_by_rule(rule)
    )

    missing_intervals = 0
    for issue in validation_result.get_issues_by_rule("missing_timestamps"):
        missing_intervals += issue.details.get("total_missing_intervals", issue.count)

    # Calculate spread statistics
    median_spread: float | None = None
    max_spread: float | None = None
    if "spread" in df.columns and not df.empty:
        spreads = pd.to_numeric(df["spread"], errors="coerce").dropna()
        if not spreads.empty:
            median_spread = round(float(spreads.median()), 4)
            max_spread = round(float(spreads.max()), 4)

    # Calculate completeness: valid rows / expected total intervals
    total_expected = row_count + missing_intervals
    completeness = round(row_count / total_expected, 4) if total_expected > 0 else 0.0

    return HealthReport(
        symbols=symbols,
        timeframes=timeframes,
        sources=sources,
        row_count=row_count,
        missing_rows=corrupted_count,
        duplicate_rows=duplicate_count,
        invalid_rows=invalid_count,
        disordered_rows=disorder_count,
        start_time=start_time,
        end_time=end_time,
        missing_intervals=missing_intervals,
        median_spread=median_spread,
        max_spread=max_spread,
        completeness=completeness,
        is_valid=validation_result.is_valid,
        issues=validation_result.issues,
    )
