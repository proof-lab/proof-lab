"""Unit tests for prooflab.paper.consumer (Live Market Data Consumer)."""

from datetime import UTC, datetime

from prooflab.paper.consumer import (
    ConsumerConfig,
    DataQualityIssue,
    LiveBar,
    LiveTick,
    MarketDataConsumer,
)


def test_market_data_consumer_valid_ticks_and_bars() -> None:
    consumer = MarketDataConsumer()
    t0 = datetime(2026, 3, 2, 10, 0, 0, tzinfo=UTC)

    # Valid tick
    tick1 = LiveTick(
        symbol="EURUSD",
        timestamp_utc=t0,
        bid=1.1000,
        ask=1.1001,
        volume=10.0,
    )
    ok_tick, issues_tick = consumer.process_tick(tick1, wall_clock_utc=t0)
    assert ok_tick is True
    assert len(issues_tick) == 0
    assert consumer.get_last_tick("EURUSD") == tick1

    # Valid bar
    bar1 = LiveBar(
        symbol="EURUSD",
        timestamp_utc=t0,
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1015,
        volume=500.0,
        spread=0.0001,
    )
    ok_bar, issues_bar = consumer.process_bar(bar1, wall_clock_utc=t0)
    assert ok_bar is True
    assert len(issues_bar) == 0
    assert consumer.get_last_bar("EURUSD") == bar1

    df = consumer.get_bars_dataframe("EURUSD")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 1.1015


def test_consumer_staleness_and_abnormal_spread() -> None:
    consumer = MarketDataConsumer(
        ConsumerConfig(max_staleness_seconds=60.0, max_spread_pips=3.0)
    )
    t0 = datetime(2026, 3, 2, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 3, 2, 10, 5, 0, tzinfo=UTC)  # 5 min later (stale)

    stale_tick = LiveTick(
        symbol="EURUSD",
        timestamp_utc=t0,
        bid=1.1000,
        ask=1.1005,  # 5 pips spread > 3 pips max
    )

    ok, issues = consumer.process_tick(stale_tick, wall_clock_utc=now)
    assert ok is False
    assert DataQualityIssue.STALE_DATA in issues
    assert DataQualityIssue.ABNORMAL_SPREAD in issues


def test_consumer_duplicate_and_out_of_order() -> None:
    consumer = MarketDataConsumer()
    t0 = datetime(2026, 3, 2, 10, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 2, 10, 0, 1, tzinfo=UTC)

    tick1 = LiveTick(symbol="EURUSD", timestamp_utc=t1, bid=1.1000, ask=1.1001)
    consumer.process_tick(tick1, wall_clock_utc=t1)

    # Duplicate tick
    ok_dup, issues_dup = consumer.process_tick(tick1, wall_clock_utc=t1)
    assert ok_dup is False
    assert DataQualityIssue.DUPLICATE_TIMESTAMP in issues_dup

    # Out of order tick (earlier timestamp)
    tick_old = LiveTick(symbol="EURUSD", timestamp_utc=t0, bid=1.1000, ask=1.1001)
    ok_ooo, issues_ooo = consumer.process_tick(tick_old, wall_clock_utc=t1)
    assert ok_ooo is False
    assert DataQualityIssue.OUT_OF_ORDER_TIMESTAMP in issues_ooo
