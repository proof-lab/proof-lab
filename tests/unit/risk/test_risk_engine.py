"""Unit and integration tests for prooflab.risk.engine (Risk Engine Signal Interception)."""

from datetime import UTC, datetime, timedelta

from prooflab.risk.engine import RiskDecision, RiskDecisionAction, RiskEngine
from prooflab.risk.kill_switch import KillSwitch
from prooflab.risk.limits import LimitBreachReason, OpenPositionRecord, RiskLimitsConfig
from prooflab.risk.safety import SafetyCheckConfig, SafetyPauseReason


def test_risk_engine_approves_valid_signal() -> None:
    engine = RiskEngine(initial_equity=100000.0)
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    decision = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
        data_timestamp=now - timedelta(seconds=15),
        model_confidence=0.72,
    )

    assert isinstance(decision, RiskDecision)
    assert decision.action == RiskDecisionAction.APPROVED
    assert decision.is_approved is True
    assert decision.approved_lots == 2.0
    assert decision.risk_amount_dollars == 1000.0


def test_risk_engine_rejects_model_buy_on_daily_loss_breach() -> None:
    engine = RiskEngine(
        limits_config=RiskLimitsConfig(max_daily_loss_pct=0.03),
        initial_equity=100000.0,
    )
    now = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)

    # Simulate losing trades breaching 3% (,000) daily limit
    engine.record_closed_trade(-1600.0)
    engine.record_closed_trade(-1600.0)

    # Model gives a high-confidence BUY signal
    decision = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
        model_confidence=0.95,
    )

    assert decision.action == RiskDecisionAction.REJECTED
    assert decision.is_approved is False
    assert LimitBreachReason.MAX_DAILY_LOSS_BREACHED in decision.limit_breaches


def test_risk_engine_pauses_on_kill_switch() -> None:
    ks = KillSwitch()
    ks.activate(actor="RiskOfficer", reason="Emergency liquidity event")

    engine = RiskEngine(kill_switch=ks, initial_equity=100000.0)
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    # Model gives a SELL signal
    decision = engine.evaluate_signal(
        symbol="GBPUSD",
        side="SELL",
        entry_price=1.2500,
        stop_loss_price=1.2550,
        current_time=now,
    )

    assert decision.action == RiskDecisionAction.KILL_SWITCH_ACTIVE
    assert decision.is_approved is False
    assert "Kill switch" in str(decision.message)


def test_risk_engine_pauses_on_safety_stale_data() -> None:
    engine = RiskEngine(
        safety_config=SafetyCheckConfig(max_data_staleness_seconds=60.0),
        initial_equity=100000.0,
    )
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    stale_ts = now - timedelta(minutes=5)

    decision = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
        data_timestamp=stale_ts,
    )

    assert decision.action == RiskDecisionAction.PAUSED
    assert decision.is_approved is False
    assert SafetyPauseReason.MARKET_DATA_STALE in decision.safety_pauses


def test_risk_engine_rejects_on_max_open_positions() -> None:
    engine = RiskEngine(
        limits_config=RiskLimitsConfig(max_open_positions=1),
        initial_equity=100000.0,
    )
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    # Populate 1 active position
    engine.sync_open_positions([
        OpenPositionRecord(
            symbol="USDJPY",
            side="BUY",
            quantity=100000.0,
            nominal_exposure=100000.0,
        )
    ])

    # Model requests another position
    decision = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
    )

    assert decision.action == RiskDecisionAction.REJECTED
    assert LimitBreachReason.MAX_OPEN_POSITIONS_REACHED in decision.limit_breaches


def test_risk_engine_pauses_on_broker_disconnect_and_invalid_model() -> None:
    engine = RiskEngine(initial_equity=100000.0)
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)

    # Disconnected broker
    dec_disc = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
        is_broker_connected=False,
    )
    assert dec_disc.action == RiskDecisionAction.PAUSED
    assert SafetyPauseReason.BROKER_CONNECTION_LOST in dec_disc.safety_pauses

    # Invalid model
    dec_model = engine.evaluate_signal(
        symbol="EURUSD",
        side="SELL",
        entry_price=1.1000,
        stop_loss_price=1.1050,
        current_time=now,
        is_model_valid=False,
        model_error="Artifact checksum mismatch",
    )
    assert dec_model.action == RiskDecisionAction.PAUSED
    assert SafetyPauseReason.INVALID_MODEL_ARTIFACT in dec_model.safety_pauses


def test_risk_engine_weekly_loss_breach_and_json() -> None:
    engine = RiskEngine(
        limits_config=RiskLimitsConfig(max_weekly_loss_pct=0.06),
        initial_equity=100000.0,
    )
    now = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)

    # Accumulate -$7,000 weekly loss (exceeding 6% limit)
    engine.record_closed_trade(-3500.0)
    engine.record_closed_trade(-3600.0)

    decision = engine.evaluate_signal(
        symbol="EURUSD",
        side="BUY",
        entry_price=1.1000,
        stop_loss_price=1.0950,
        current_time=now,
    )

    assert decision.action == RiskDecisionAction.REJECTED
    assert LimitBreachReason.MAX_WEEKLY_LOSS_BREACHED in decision.limit_breaches

    # JSON export
    json_str = decision.to_json()
    assert "REJECTED" in json_str
    assert "EURUSD" in json_str
