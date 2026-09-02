"""Outcome definitions, canonical labels, and rich metadata container.

Represents both coarse-grained classification targets (BUY=1, SELL=-1, IGNORE=0)
and detailed diagnostic execution metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.labels.config import SetupConfig


class CanonicalLabel(IntEnum):
    """Canonical 3-class classification target."""

    BUY = 1
    SELL = -1
    IGNORE = 0


class BarrierOutcome(StrEnum):
    """Rich execution outcome classification."""

    TARGET_FIRST = "TARGET_FIRST"
    STOP_FIRST = "STOP_FIRST"
    TIMEOUT = "TIMEOUT"
    AMBIGUOUS = "AMBIGUOUS"
    EXCLUDED = "EXCLUDED"


class RichLabelOutcome(BaseModel):
    """Detailed metadata for a single evaluated setup instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_index: int
    entry_time: datetime
    entry_price: float
    target_price: float
    stop_price: float
    canonical_label: CanonicalLabel
    barrier_outcome: BarrierOutcome
    exit_index: int | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    bars_held: int = 0
    return_at_exit: float = 0.0
    was_ambiguous: bool = False
    ambiguity_policy_used: str | None = None


class LabelMatrix(BaseModel):
    """Container holding a collection of evaluated label outcomes and setup configuration."""

    outcomes: list[RichLabelOutcome] = Field(default_factory=list)
    setup_config: SetupConfig
    dataset_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def __len__(self) -> int:
        return len(self.outcomes)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all label outcomes to a structured pandas DataFrame.

        Returns:
            DataFrame containing canonical labels and full rich metadata.
        """
        if not self.outcomes:
            return pd.DataFrame(
                columns=[
                    "entry_index",
                    "entry_time",
                    "entry_price",
                    "target_price",
                    "stop_price",
                    "canonical_label",
                    "barrier_outcome",
                    "exit_index",
                    "exit_time",
                    "exit_price",
                    "bars_held",
                    "return_at_exit",
                    "was_ambiguous",
                    "ambiguity_policy_used",
                ]
            )

        records = [outcome.model_dump(mode="python") for outcome in self.outcomes]
        df = pd.DataFrame.from_records(records)
        df["canonical_label"] = df["canonical_label"].astype(int)
        return df

    def get_canonical_series(self) -> pd.Series:
        """Return a pandas Series of canonical integer labels indexed by entry timestamp."""
        if not self.outcomes:
            return pd.Series(dtype=int)
        df = self.to_dataframe()
        return pd.Series(
            data=df["canonical_label"].to_numpy(),
            index=pd.to_datetime(df["entry_time"]),
            name="label",
        )
