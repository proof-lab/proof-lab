"""Unit tests for prooflab.risk.safety (Automated Trading Pause Checks)."""

from datetime import UTC, datetime, timedelta

from prooflab.risk.safety import (
    SafetyCheckConfig,
    SafetyCheckResult,
    SafetyMonitor,
    SafetyPauseReason,
)


def test_safety_check_all_clear() -> None:
    monitor = SafetyMonitor()
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    res = monitor.check_safety(
        current_time=now,
        data_timestamp=now - timedelta(seconds=10),
        is_broker_connected=True,
        is_model_valid=True,
        features={"rsi": 55.0, "atr": 0.0015},
        current_spread_pips=1.2,
        model_confidence=0.68,
    )

    assert isinstance(res, SafetyCheckResult)
    assert res.is_safe is True
    assert len(res.pause_reasons) == 0


def test_safety_stale_data_pause() -> None:
    monitor = SafetyMonitor(SafetyCheckConfig(max_data_staleness_seconds=120.0))
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    stale_time = now - timedelta(seconds=300)

    res = monitor.check_safety(
        current_time=now,
        data_timestamp=stale_time,
    )

    assert res.is_safe is False
    assert SafetyPauseReason.MARKET_DATA_STALE in res.pause_reasons


def test_safety_spread_blowout_and_nan_features() -> None:
    monitor = SafetyMonitor(SafetyCheckConfig(max_spread_pips=4.0, max_spread_multiplier=2.5))
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    res = monitor.check_safety(
        current_time=now,
        current_spread_pips=6.5,
        normal_spread_pips=1.0,
        features={"rsi": float("nan"), "macd": 0.002},
    )

    assert res.is_safe is False
    assert SafetyPauseReason.UNEXPECTED_SPREAD_BLOWOUT in res.pause_reasons
    assert SafetyPauseReason.FEATURE_CALCULATION_FAILURE in res.pause_reasons


def test_safety_news_blackout() -> None:
    cfg = SafetyCheckConfig(news_blackout_pre_minutes=15, news_blackout_post_minutes=15)
    monitor = SafetyMonitor(cfg)
    now = datetime(2026, 3, 2, 13, 25, tzinfo=UTC)
    nfp_event = datetime(2026, 3, 2, 13, 30, tzinfo=UTC)

    res = monitor.check_safety(
        current_time=now,
        scheduled_news_events=[nfp_event],
    )

    assert res.is_safe is False
    assert SafetyPauseReason.NEWS_BLACKOUT_ACTIVE in res.pause_reasons


def test_safety_duplicate_signal_detection() -> None:
    monitor = SafetyMonitor()
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    res1 = monitor.check_safety(
        current_time=now,
        symbol="EURUSD",
        signal_id="sig-20260302-001",
    )
    assert res1.is_safe is True

    # Same signal sent again -> DUPLICATE_SIGNAL_DETECTED
    res2 = monitor.check_safety(
        current_time=now,
        symbol="EURUSD",
        signal_id="sig-20260302-001",
    )
    assert res2.is_safe is False
    assert SafetyPauseReason.DUPLICATE_SIGNAL_DETECTED in res2.pause_reasons
