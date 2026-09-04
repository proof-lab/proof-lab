"""Unit tests for prooflab.risk.kill_switch (Kill Switch & Persistence)."""

from pathlib import Path

from prooflab.risk.kill_switch import (
    KillSwitch,
    KillSwitchAuditEvent,
    KillSwitchPolicy,
)


def test_kill_switch_activation_and_reset() -> None:
    ks = KillSwitch(default_policy=KillSwitchPolicy.CLOSE_ALL)
    assert ks.is_active is False

    open_pos = [
        {"position_id": "pos-1", "symbol": "EURUSD", "unrealized_pnl": 150.0},
        {"position_id": "pos-2", "symbol": "GBPUSD", "unrealized_pnl": -350.0},
    ]
    pending_orders = ["ord-101", "ord-102"]

    event = ks.activate(
        actor="RiskAdmin",
        reason="Abnormal market volatility detected",
        pending_order_ids=pending_orders,
        open_positions=open_pos,
    )

    assert isinstance(event, KillSwitchAuditEvent)
    assert event.action == "ACTIVATED"
    assert event.cancelled_orders == ["ord-101", "ord-102"]
    assert event.liquidated_positions == ["pos-1", "pos-2"]
    assert ks.is_active is True

    # Reset
    reset_event = ks.reset(actor="RiskAdmin", reason="Volatility subsided")
    assert reset_event.action == "RESET"
    assert ks.is_active is False


def test_kill_switch_close_losing_policy() -> None:
    ks = KillSwitch()
    open_pos = [
        {"position_id": "pos-win", "symbol": "EURUSD", "unrealized_pnl": 200.0},
        {"position_id": "pos-loss", "symbol": "GBPUSD", "unrealized_pnl": -100.0},
    ]

    event = ks.activate(
        actor="RiskAdmin",
        reason="Test policy",
        policy=KillSwitchPolicy.CLOSE_LOSING,
        open_positions=open_pos,
    )

    assert event.liquidated_positions == ["pos-loss"]


def test_kill_switch_state_persistence(tmp_path: Path) -> None:
    state_file = tmp_path / "kill_switch_state.json"
    ks1 = KillSwitch(state_file=state_file)
    assert ks1.is_active is False

    ks1.activate(actor="AutomatedMonitor", reason="Latency breach")
    assert state_file.exists()

    # Re-instantiate from persisted file
    ks2 = KillSwitch(state_file=state_file)
    assert ks2.is_active is True
    assert ks2.state.triggered_by == "AutomatedMonitor"
