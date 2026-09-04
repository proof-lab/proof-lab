"""Unit tests for prooflab.risk.limits (Exposure, Loss, and Streak Limits)."""

from datetime import UTC, datetime

from prooflab.risk.limits import (
    LimitBreachReason,
    LimitEvaluationResult,
    OpenPositionRecord,
    RiskLimitsConfig,
    RiskLimitsEvaluator,
    RiskStateTracker,
)


def test_risk_limits_happy_path() -> None:
    evaluator = RiskLimitsEvaluator(RiskLimitsConfig())
    t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    state = RiskStateTracker(initial_equity=100000.0, current_time=t0)

    res = evaluator.evaluate_new_order(
        symbol="EURUSD",
        requested_nominal_exposure=100000.0,
        risk_amount_dollars=1000.0,
        state=state,
    )

    assert isinstance(res, LimitEvaluationResult)
    assert res.allowed is True
    assert len(res.breach_reasons) == 0


def test_max_daily_loss_disables_new_trades() -> None:
    # 3% daily loss limit = -,000 on  equity
    evaluator = RiskLimitsEvaluator(RiskLimitsConfig(max_daily_loss_pct=0.03))
    t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    state = RiskStateTracker(initial_equity=100000.0, current_time=t0)

    # 3 losses of -,100 -> daily PnL = -,300
    state.record_closed_trade(-1100.0)
    state.record_closed_trade(-1100.0)
    state.record_closed_trade(-1100.0)
    state.current_equity = 96700.0

    res = evaluator.evaluate_new_order(
        symbol="EURUSD",
        requested_nominal_exposure=50000.0,
        risk_amount_dollars=500.0,
        state=state,
    )

    assert res.allowed is False
    assert LimitBreachReason.MAX_DAILY_LOSS_BREACHED in res.breach_reasons
    assert "Daily loss" in str(res.rejection_message)


def test_consecutive_losses_limit() -> None:
    evaluator = RiskLimitsEvaluator(RiskLimitsConfig(max_consecutive_losses=3))
    t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    state = RiskStateTracker(initial_equity=100000.0, current_time=t0)

    state.record_closed_trade(-100.0)
    state.record_closed_trade(-100.0)
    state.record_closed_trade(-100.0)

    res = evaluator.evaluate_new_order(
        symbol="EURUSD",
        requested_nominal_exposure=50000.0,
        risk_amount_dollars=500.0,
        state=state,
    )

    assert res.allowed is False
    assert LimitBreachReason.MAX_CONSECUTIVE_LOSSES_BREACHED in res.breach_reasons

    # A winning trade resets the streak
    state.record_closed_trade(200.0)
    assert state.consecutive_loss_streak == 0

    res2 = evaluator.evaluate_new_order(
        symbol="EURUSD",
        requested_nominal_exposure=50000.0,
        risk_amount_dollars=500.0,
        state=state,
    )
    assert res2.allowed is True


def test_symbol_and_total_exposure_limits() -> None:
    evaluator = RiskLimitsEvaluator(
        RiskLimitsConfig(max_open_positions=2, max_symbol_positions=1, max_total_leverage=2.0)
    )
    t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    state = RiskStateTracker(initial_equity=100000.0, current_time=t0)

    # 1 open position on EURUSD ( exposure -> 1.0x leverage)
    pos1 = OpenPositionRecord(
        symbol="EURUSD",
        side="BUY",
        quantity=100000.0,
        nominal_exposure=100000.0,
        unrealized_pnl=0.0,
    )
    state.set_open_positions([pos1])

    # Attempting 2nd EURUSD order fails symbol position limit
    res_sym = evaluator.evaluate_new_order(
        symbol="EURUSD",
        requested_nominal_exposure=50000.0,
        risk_amount_dollars=500.0,
        state=state,
    )
    assert res_sym.allowed is False
    assert LimitBreachReason.MAX_SYMBOL_POSITIONS_REACHED in res_sym.breach_reasons

    # Attempting GBPUSD order exceeding 2x total leverage ( +  =  > )
    res_lev = evaluator.evaluate_new_order(
        symbol="GBPUSD",
        requested_nominal_exposure=120000.0,
        risk_amount_dollars=500.0,
        state=state,
    )
    assert res_lev.allowed is False
    assert LimitBreachReason.MAX_TOTAL_LEVERAGE_EXCEEDED in res_lev.breach_reasons
