"""Integration tests for live execution loop, governance gating, and kill switch."""

from __future__ import annotations

import pytest

from prooflab.live.deduplication import DuplicateSignalError
from prooflab.live.engine import (
    LiveExecutionEngine,
    LiveTradingDisabledError,
)
from prooflab.live.mt5_adapter import MockMT5Adapter
from prooflab.live.orders import LiveOrderState
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState
from prooflab.risk.engine import RiskEngine
from prooflab.risk.kill_switch import KillSwitch
from prooflab.risk.limits import RiskLimitsConfig


def test_live_engine_governance_disabled_by_default() -> None:
    """Ensure live engine strictly rejects execution when not in LIVE_ENABLED state."""
    adapter = MockMT5Adapter()
    lifecycle = StrategyLifecycleManager("STRAT_M13", initial_state=StrategyLifecycleState.RESEARCH)
    risk_engine = RiskEngine()
    kill_switch = KillSwitch()

    engine = LiveExecutionEngine(
        adapter=adapter,
        lifecycle_manager=lifecycle,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
    )

    assert not engine.is_live_enabled

    # Must raise LiveTradingDisabledError
    with pytest.raises(LiveTradingDisabledError):
        engine.process_signal(
            signal_id="SIG_001",
            symbol="EURUSD",
            side="BUY",
            quantity=0.1,
            price=1.0850,
        )


def test_live_engine_full_execution_flow() -> None:
    """Test full execution pipeline when governance explicitly approves live trading."""
    adapter = MockMT5Adapter(initial_balance=50000.0)
    lifecycle = StrategyLifecycleManager("STRAT_M13", initial_state=StrategyLifecycleState.RESEARCH)

    # Progress through lifecycle gates to LIVE_ENABLED
    lifecycle.transition_to(StrategyLifecycleState.VALIDATED, reason="Passed proof")
    lifecycle.transition_to(StrategyLifecycleState.PAPER_TRADING, reason="Paper test")
    lifecycle.transition_to(StrategyLifecycleState.APPROVED, reason="Committee approved")
    lifecycle.transition_to(
        StrategyLifecycleState.LIVE_ENABLED,
        reason="Explicit human authorization",
        explicit_human_approval=True,
    )

    risk_engine = RiskEngine(
        limits_config=RiskLimitsConfig(
            max_risk_per_trade_pct=0.02,
            max_symbol_leverage=5.0,
            max_total_leverage=10.0,
        ),
        initial_equity=50000.0,
    )
    kill_switch = KillSwitch()

    engine = LiveExecutionEngine(
        adapter=adapter,
        lifecycle_manager=lifecycle,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
    )

    # Reconcile on startup
    report = engine.startup_and_reconcile()
    assert report.is_consistent
    assert adapter.is_connected()

    # Process live signal
    order = engine.process_signal(
        signal_id="SIG_LIVE_1",
        symbol="EURUSD",
        side="BUY",
        quantity=0.1,
        price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
    )

    assert order.status == LiveOrderState.FILLED
    assert order.broker_ticket is not None
    assert len(engine.active_orders) == 1
    assert len(engine.orders) == 1
    assert len(adapter.get_positions("EURUSD")) == 1

    # Duplicate signal must be rejected
    with pytest.raises(DuplicateSignalError):
        engine.process_signal(
            signal_id="SIG_LIVE_1",
            symbol="EURUSD",
            side="BUY",
            quantity=0.1,
            price=1.0850,
        )

    # Emergency close all positions
    closed_count = engine.close_all_positions(reason="Test close all")
    assert closed_count == 1
    assert len(adapter.get_positions()) == 0


def test_live_engine_risk_rejection_and_broker_error() -> None:
    """Test risk rejection and broker exception handling."""
    adapter = MockMT5Adapter(initial_balance=1000.0)
    lifecycle = StrategyLifecycleManager("STRAT_M13", initial_state=StrategyLifecycleState.APPROVED)
    lifecycle.transition_to(
        StrategyLifecycleState.LIVE_ENABLED,
        reason="Approved",
        explicit_human_approval=True,
    )

    # Risk limits with very low max risk to force rejection
    risk_engine = RiskEngine(
        limits_config=RiskLimitsConfig(max_risk_per_trade_pct=0.0001, max_total_leverage=0.1),
        initial_equity=1000.0,
    )
    kill_switch = KillSwitch()

    engine = LiveExecutionEngine(
        adapter=adapter,
        lifecycle_manager=lifecycle,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
    )
    adapter.connect()

    # Signal rejected by risk engine
    rejected_order = engine.process_signal(
        signal_id="SIG_TOO_RISKY",
        symbol="EURUSD",
        side="BUY",
        quantity=5.0,
        price=1.0850,
    )
    assert rejected_order.status == LiveOrderState.REJECTED

    # Disconnect adapter to force broker exception
    adapter.disconnect()
    risk_engine.limits_evaluator.config = RiskLimitsConfig(
        max_risk_per_trade_pct=0.05,
        max_symbol_leverage=50.0,
        max_total_leverage=50.0,
    )
    failed_order = engine.process_signal(
        signal_id="SIG_DISCONNECTED",
        symbol="EURUSD",
        side="BUY",
        quantity=0.1,
        price=1.0850,
    )
    assert failed_order.status == LiveOrderState.FAILED


def test_live_engine_kill_switch_blocking() -> None:
    """Test that active kill switch halts live order submissions."""
    adapter = MockMT5Adapter()
    lifecycle = StrategyLifecycleManager("STRAT_M13", initial_state=StrategyLifecycleState.APPROVED)
    lifecycle.transition_to(
        StrategyLifecycleState.LIVE_ENABLED,
        reason="Approved",
        explicit_human_approval=True,
    )
    risk_engine = RiskEngine()
    kill_switch = KillSwitch()
    kill_switch.activate(actor="human", reason="Operator emergency stop")

    engine = LiveExecutionEngine(
        adapter=adapter,
        lifecycle_manager=lifecycle,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
    )

    with pytest.raises(PermissionError) as exc:
        engine.process_signal(
            signal_id="SIG_KILL",
            symbol="EURUSD",
            side="BUY",
            quantity=0.1,
            price=1.0850,
        )
    assert "Kill switch is active" in str(exc.value)
