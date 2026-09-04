"""Unit tests for signal deduplication and idempotency."""

from __future__ import annotations

import pytest

from prooflab.live.deduplication import (
    DuplicateSignalError,
    SignalDeduplicator,
)


def test_signal_registration_and_duplicate_rejection() -> None:
    """Ensure duplicate signals are rejected with DuplicateSignalError."""
    dedup = SignalDeduplicator()

    assert not dedup.is_duplicate("SIG_100")
    record = dedup.register_signal("SIG_100", "ORD_100", "EURUSD", "BUY")

    assert record.signal_id == "SIG_100"
    assert record.order_id == "ORD_100"
    assert dedup.is_duplicate("SIG_100")
    assert dedup.count() == 1

    # Second submission of identical signal must raise DuplicateSignalError
    with pytest.raises(DuplicateSignalError) as exc_info:
        dedup.register_signal("SIG_100", "ORD_101", "EURUSD", "BUY")

    assert "SIG_100" in str(exc_info.value)
    assert "ORD_100" in str(exc_info.value)

    # Empty signal ID is rejected
    with pytest.raises(ValueError):
        dedup.register_signal("", "ORD_102", "EURUSD", "BUY")


def test_different_signals_succeed() -> None:
    """Ensure distinct signal IDs are successfully registered."""
    dedup = SignalDeduplicator()

    dedup.register_signal("SIG_1", "ORD_1", "EURUSD", "BUY")
    dedup.register_signal("SIG_2", "ORD_2", "GBPUSD", "SELL")
    dedup.register_signal("SIG_3", "ORD_3", "USDJPY", "BUY")

    assert dedup.count() == 3
    assert dedup.get_record("SIG_2") is not None
    assert dedup.get_record("SIG_2").symbol == "GBPUSD"  # type: ignore[union-attr]

    dedup.clear()
    assert dedup.count() == 0
