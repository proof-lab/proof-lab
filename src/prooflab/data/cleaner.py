"""Market data cleaning pipeline.

Provides causal, audit-logged cleaning transformations including duplicate removal,
malformed row filtering, chronological ordering, and explicitly configured imputation.
Guarantees no silent forward-filling across market closures or look-ahead leakage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CleaningOperation(BaseModel):
    """Record of an individual cleaning transformation step applied to a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    rows_affected: int
    description: str
    details: dict[str, Any] = Field(default_factory=dict)


class CleaningRecord(BaseModel):
    """Complete audit log of all transformations applied during cleaning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_before: int
    rows_after: int
    operations: list[CleaningOperation] = Field(default_factory=list)
    imputation_applied: bool = False
    cleaned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self, indent: int = 2) -> str:
        """Serialize the cleaning record to JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    def summary(self) -> str:
        """Return a formatted human-readable summary string."""
        lines = [
            "=== Data Cleaning Audit Record ===",
            f"Rows Before: {self.rows_before:,}",
            f"Rows After: {self.rows_after:,}",
            f"Imputation Applied: {self.imputation_applied}",
            f"Operations Executed: {len(self.operations)}",
        ]
        for op in self.operations:
            lines.append(f"  - [{op.name}] Affected: {op.rows_affected:,} | {op.description}")
        return "\n".join(lines)


class ForwardFillConfig(BaseModel):
    """Configuration for missing data forward-fill imputation.

    Disabled by default. When enabled, requires an explicit human-readable
    reason to prevent accidental or silent look-ahead/leakage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    max_gap_periods: int = 1
    reason: str = ""

    @model_validator(mode="after")
    def validate_reason(self) -> ForwardFillConfig:
        if self.enabled and not self.reason.strip():
            raise ValueError(
                "An explicit justification reason is required when enabling forward-fill."
            )
        return self


class CleaningConfig(BaseModel):
    """Settings controlling the DataCleaner pipeline steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    remove_duplicates: bool = True
    remove_malformed_rows: bool = True
    sort_chronologically: bool = True
    forward_fill: ForwardFillConfig = Field(default_factory=ForwardFillConfig)


class DataCleaner:
    """Causal data cleaning pipeline for market datasets."""

    def clean(
        self,
        df: pd.DataFrame,
        config: CleaningConfig | None = None,
    ) -> tuple[pd.DataFrame, CleaningRecord]:
        """Execute the cleaning pipeline on a market data DataFrame.

        Transformations are strictly causal and logged in the returned CleaningRecord.

        Args:
            df: Input DataFrame.
            config: Optional CleaningConfig (uses strict defaults if None).

        Returns:
            Tuple of (cleaned_df, cleaning_record).
        """
        cfg = config or CleaningConfig()
        rows_before = len(df)
        if df.empty:
            return df.copy(), CleaningRecord(
                rows_before=0,
                rows_after=0,
                operations=[],
                imputation_applied=False,
            )

        cleaned = df.copy()
        operations: list[CleaningOperation] = []

        # 1. Remove malformed rows (Null values in required columns)
        if cfg.remove_malformed_rows:
            required_cols = [
                c for c in ["timestamp", "open", "high", "low", "close"] if c in cleaned.columns
            ]
            if required_cols:
                null_mask = cleaned[required_cols].isna().any(axis=1)
                malformed_count = int(null_mask.sum())
                if malformed_count > 0:
                    cleaned = cleaned.loc[~null_mask].copy()
                    operations.append(
                        CleaningOperation(
                            name="remove_malformed_rows",
                            rows_affected=malformed_count,
                            description=(
                                f"Removed {malformed_count} row(s) with null/NaN in "
                                f"required columns ({', '.join(required_cols)})."
                            ),
                        )
                    )

        # 2. Remove duplicate timestamps
        if cfg.remove_duplicates and "timestamp" in cleaned.columns and not cleaned.empty:
            subset = ["timestamp"]
            if "symbol" in cleaned.columns:
                subset.append("symbol")
            if "timeframe" in cleaned.columns:
                subset.append("timeframe")

            dup_mask = cleaned.duplicated(subset=subset, keep="first")
            dup_count = int(dup_mask.sum())
            if dup_count > 0:
                cleaned = cleaned.loc[~dup_mask].copy()
                operations.append(
                    CleaningOperation(
                        name="remove_duplicates",
                        rows_affected=dup_count,
                        description=(
                            f"Removed {dup_count} duplicate timestamp row(s) keeping first."
                        ),
                    )
                )

        # 3. Sort chronologically
        if cfg.sort_chronologically and "timestamp" in cleaned.columns and not cleaned.empty:
            ts_series = pd.to_datetime(cleaned["timestamp"])
            is_sorted = ts_series.is_monotonic_increasing
            if not is_sorted:
                cleaned = cleaned.sort_values(by="timestamp").reset_index(drop=True)
                operations.append(
                    CleaningOperation(
                        name="sort_chronologically",
                        rows_affected=len(cleaned),
                        description="Sorted records in ascending chronological order.",
                    )
                )
            else:
                cleaned = cleaned.reset_index(drop=True)

        # 4. Explicit forward-fill imputation (never silent)
        imputation_applied = False
        if cfg.forward_fill.enabled and not cleaned.empty:
            price_cols = [c for c in ["open", "high", "low", "close"] if c in cleaned.columns]
            nan_count_before = int(cleaned[price_cols].isna().sum().sum())
            if nan_count_before > 0:
                cleaned[price_cols] = cleaned[price_cols].ffill(
                    limit=cfg.forward_fill.max_gap_periods
                )
                nan_count_after = int(cleaned[price_cols].isna().sum().sum())
                filled_count = nan_count_before - nan_count_after
                if filled_count > 0:
                    imputation_applied = True
                    operations.append(
                        CleaningOperation(
                            name="forward_fill",
                            rows_affected=filled_count,
                            description=(
                                f"Forward-filled {filled_count} value(s) across price columns "
                                f"(max limit {cfg.forward_fill.max_gap_periods} periods). "
                                f"Justification: {cfg.forward_fill.reason}"
                            ),
                            details={
                                "max_gap_periods": cfg.forward_fill.max_gap_periods,
                                "reason": cfg.forward_fill.reason,
                            },
                        )
                    )

        record = CleaningRecord(
            rows_before=rows_before,
            rows_after=len(cleaned),
            operations=operations,
            imputation_applied=imputation_applied,
        )

        return cleaned, record
