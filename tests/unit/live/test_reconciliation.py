"""Unit tests for restart recovery and position reconciliation engine."""

from __future__ import annotations

import pytest

from prooflab.live.mt5_adapter import MockMT5Adapter
from prooflab.live.orders import LiveOrder, LiveOrderState
from prooflab.live.reconciliation import (
    DiscrepancyType,
    ReconciliationEngine,
)
from prooflab.risk.engine import RiskEngine
from prooflab.risk.limits import RiskLimitsConfig


def test_reconciliation_matching_positions() -> None:
    """Test reconciliation with perfectly matching local and broker positions."""
    adapter = MockMT5Adapter(initial_balance=20000.0)
    adapter.connect()

    order = LiveOrder(
        order_id="ORD_REC_1",
        signal_id="SIG_REC_1",
        symbol="EURUSD",
        side="BUY",
        quantity=1.0,
        price=1.0850,
    )
    filled_order = adapter.submit_order(order)

    risk_engine = RiskEngine(limits_config=RiskLimitsConfig(max_risk_per_trade_pct=0.01))

    report = ReconciliationEngine.reconcile(
        adapter=adapter,
        local_orders=[filled_order],
        risk_engine=risk_engine,
    )

    assert report.is_consistent
    assert report.broker_positions_count == 1
    assert report.local_positions_count == 1
    assert len(report.discrepancies) == 0
    assert report.account_equity > 0
    assert risk_engine.limits_evaluator.start_of_day_equity == report.account_equity


def test_reconciliation_orphan_broker_position() -> None:
    """Test detection of position that exists on broker but not in local ledger."""
    adapter = MockMT5Adapter()
    adapter.connect()

    # Create position directly on broker
    order = LiveOrder(
        order_id="ORD_EXTERNAL",
        signal_id="SIG_EXTERNAL",
        symbol="GBPUSD",
        side="SELL",
        quantity=0.5,
        price=1.2700,
    )
    adapter.submit_order(order)

    # Reconcile with empty local ledger
    report = ReconciliationEngine.reconcile(adapter=adapter, local_orders=[])

    assert not report.is_consistent
    assert report.broker_positions_count == 1
    assert report.local_positions_count == 0
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0].discrepancy_type == DiscrepancyType.ORPHAN_BROKER_POSITION


def test_reconciliation_missing_broker_position() -> None:
    """Test detection when local order thinks it's open, but broker closed it."""
    adapter = MockMT5Adapter()
    adapter.connect()

    # Local order with ticket that does not exist on broker
    local_order = LiveOrder(
        order_id="ORD_GHOST",
        signal_id="SIG_GHOST",
        symbol="EURUSD",
        side="BUY",
        quantity=1.0,
        price=1.0850,
        status=LiveOrderState.FILLED,
        broker_ticket=999999,
    )

    report = ReconciliationEngine.reconcile(adapter=adapter, local_orders=[local_order])

    assert not report.is_consistent
    assert report.broker_positions_count == 0
    assert report.local_positions_count == 1
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_BROKER_POSITION


def test_reconciliation_disconnected_adapter_error() -> None:
    """Test that reconciliation requires a connected adapter."""
    adapter = MockMT5Adapter()
    with pytest.raises(RuntimeError):
        ReconciliationEngine.reconcile(adapter=adapter, local_orders=[])
