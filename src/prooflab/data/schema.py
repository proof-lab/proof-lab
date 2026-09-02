"""Canonical data schemas for Proof Lab.

Defines the core OHLCV and tick market data models and timeframes.
All timestamps are timezone-aware and stored internally in UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator


class Timeframe(StrEnum):
    """Standard market data timeframes."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


OHLCV_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tick_volume",
    "spread",
    "source",
]

TICK_COLUMNS: Final[list[str]] = [
    "timestamp",
    "symbol",
    "bid",
    "ask",
    "last",
    "volume",
]


def _ensure_utc_aware(value: object) -> datetime:
    """Validate that timestamp is timezone-aware and convert to UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise ValueError(f"Invalid timestamp type: {type(value)}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return value.astimezone(UTC)


class OHLCVBar(BaseModel):
    """Canonical OHLCV market data bar representation.

    Every bar represents price action over a fixed interval (timeframe).
    Timestamps must be timezone-aware and are normalized to UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_volume: float
    spread: float
    source: str

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> datetime:
        return _ensure_utc_aware(value)


class TickData(BaseModel):
    """Canonical tick data representation.

    Represents individual market quote / trade updates.
    Timestamps must be timezone-aware and are normalized to UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> datetime:
        return _ensure_utc_aware(value)
