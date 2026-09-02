"""Unit tests for prooflab.data.schema."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from prooflab.data.schema import OHLCV_COLUMNS, TICK_COLUMNS, OHLCVBar, TickData, Timeframe


def test_timeframe_enum() -> None:
    assert Timeframe.M1 == "M1"
    assert Timeframe.H1 == "H1"
    assert Timeframe.D1 == "D1"
    assert len(Timeframe) == 9


def test_ohlcv_columns_constant() -> None:
    assert "open" in OHLCV_COLUMNS
    assert "close" in OHLCV_COLUMNS
    assert "spread" in OHLCV_COLUMNS
    assert "tick_volume" in OHLCV_COLUMNS


def test_tick_columns_constant() -> None:
    assert "bid" in TICK_COLUMNS
    assert "ask" in TICK_COLUMNS
    assert "last" in TICK_COLUMNS


def test_valid_ohlcv_bar() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    bar = OHLCVBar(
        timestamp=now,
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1010,
        volume=100.0,
        tick_volume=150.0,
        spread=1.5,
        source="mt5",
    )
    assert bar.symbol == "EURUSD"
    assert bar.open == 1.1000
    assert bar.timestamp == now


def test_naive_timestamp_rejected() -> None:
    naive_dt = datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        OHLCVBar(
            timestamp=naive_dt,
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            open=1.1000,
            high=1.1020,
            low=1.0990,
            close=1.1010,
            volume=100.0,
            tick_volume=150.0,
            spread=1.5,
            source="mt5",
        )


def test_timezone_conversion_to_utc() -> None:
    est = timezone(timedelta(hours=-5))
    dt_est = datetime(2026, 1, 1, 12, 0, tzinfo=est)
    bar = OHLCVBar(
        timestamp=dt_est,
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1010,
        volume=100.0,
        tick_volume=150.0,
        spread=1.5,
        source="mt5",
    )
    assert bar.timestamp.tzinfo == UTC
    assert bar.timestamp.hour == 17  # 12:00 EST is 17:00 UTC


def test_string_timestamp_iso() -> None:
    bar = OHLCVBar(
        timestamp="2026-01-01T12:00:00+00:00",  # type: ignore[arg-type]
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1010,
        volume=100.0,
        tick_volume=150.0,
        spread=1.5,
        source="mt5",
    )
    assert bar.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_ohlcv_bar_frozen() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    bar = OHLCVBar(
        timestamp=now,
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1010,
        volume=100.0,
        tick_volume=150.0,
        spread=1.5,
        source="mt5",
    )
    with pytest.raises(ValidationError):
        bar.open = 1.2000  # type: ignore[misc]


def test_ohlcv_extra_fields_forbidden() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        OHLCVBar(
            timestamp=now,
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            open=1.1000,
            high=1.1020,
            low=1.0990,
            close=1.1010,
            volume=100.0,
            tick_volume=150.0,
            spread=1.5,
            source="mt5",
            extra_field="invalid",  # type: ignore[call-arg]
        )


def test_valid_tick_data() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    tick = TickData(
        timestamp=now,
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1002,
        last=1.1001,
        volume=1.5,
    )
    assert tick.symbol == "EURUSD"
    assert tick.bid == 1.1000
    assert tick.ask == 1.1002


def test_tick_naive_timestamp_rejected() -> None:
    naive_dt = datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TickData(
            timestamp=naive_dt,
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=1.5,
        )
