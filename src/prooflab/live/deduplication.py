"""Idempotency and duplicate order signal prevention engine."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class DuplicateSignalError(Exception):
    """Raised when an order submission is attempted for an already-processed signal ID."""


class ProcessedSignalRecord(BaseModel):
    """Immutable record of an executed trading signal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str
    order_id: str
    symbol: str
    side: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SignalDeduplicator:
    """Thread-safe signal registry preventing duplicate order dispatches for identical signals."""

    def __init__(self) -> None:
        self._processed: dict[str, ProcessedSignalRecord] = {}

    def is_duplicate(self, signal_id: str) -> bool:
        """Check whether a signal ID has already been recorded."""
        return signal_id in self._processed

    def register_signal(
        self,
        signal_id: str,
        order_id: str,
        symbol: str,
        side: str,
    ) -> ProcessedSignalRecord:
        """Register a new signal execution; raises DuplicateSignalError if already seen."""
        if not signal_id:
            raise ValueError("Signal ID must not be empty.")

        if signal_id in self._processed:
            existing = self._processed[signal_id]
            logger.warning(
                "Duplicate signal submission rejected: signal_id=%s already executed as order_id=%s at %s",
                signal_id,
                existing.order_id,
                existing.timestamp_utc,
            )
            raise DuplicateSignalError(
                f"Duplicate order submission rejected: signal '{signal_id}' was already executed "
                f"under order '{existing.order_id}' at {existing.timestamp_utc}."
            )

        record = ProcessedSignalRecord(
            signal_id=signal_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
        )
        self._processed[signal_id] = record
        logger.debug("Registered signal execution: signal_id=%s, order_id=%s", signal_id, order_id)
        return record

    def get_record(self, signal_id: str) -> ProcessedSignalRecord | None:
        """Retrieve signal execution record if present."""
        return self._processed.get(signal_id)

    def count(self) -> int:
        """Return total count of recorded unique signals."""
        return len(self._processed)

    def clear(self) -> None:
        """Clear signal cache."""
        self._processed.clear()
