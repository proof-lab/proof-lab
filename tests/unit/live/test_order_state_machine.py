"""Unit tests for live order state machine transitions and audit events."""

from __future__ import annotations

import pytest

from prooflab.live.orders import (
    InvalidOrderStateTransitionError,
    LiveOrder,
    LiveOrderState,
    OrderStateMachine,
)


def test_happy_path_order_lifecycle() -> None:
    """Test standard order transition path from CREATED to CLOSED."""
    order = LiveOrder(
        order_id="ORD_001",
        signal_id="SIG_001",
        symbol="EURUSD",
        side="BUY",
        order_type="MARKET",
        quantity=1.0,
        price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
    )

    assert order.status == LiveOrderState.CREATED
    assert order.is_active
    assert not order.is_terminal

    # 1. Transition to SUBMITTED
    ev1 = OrderStateMachine.transition(
        order, LiveOrderState.SUBMITTED, reason="Dispatched to MT5 broker"
    )
    assert order.status == LiveOrderState.SUBMITTED
    assert ev1.from_state == LiveOrderState.CREATED
    assert ev1.to_state == LiveOrderState.SUBMITTED

    # 2. Transition to ACKNOWLEDGED
    OrderStateMachine.transition(
        order,
        LiveOrderState.ACKNOWLEDGED,
        reason="Broker received and acknowledged order",
        broker_ticket=12345678,
    )
    assert order.status == LiveOrderState.ACKNOWLEDGED
    assert order.broker_ticket == 12345678

    # 3. Transition to FILLED
    OrderStateMachine.transition(
        order,
        LiveOrderState.FILLED,
        reason="Order fully executed by broker",
        filled_quantity=1.0,
        filled_price=1.0851,
        slippage=0.0001,
        commission=3.50,
    )
    assert order.status == LiveOrderState.FILLED
    assert order.filled_quantity == 1.0
    assert order.filled_price == 1.0851
    assert order.commission == 3.50
    assert order.is_active

    # 4. Transition to CLOSED
    OrderStateMachine.transition(
        order, LiveOrderState.CLOSED, reason="Position closed on TP target"
    )
    assert order.status == LiveOrderState.CLOSED
    assert order.is_terminal
    assert not order.is_active

    assert len(order.events) == 4


def test_rejection_and_terminal_paths() -> None:
    """Test order rejection, cancellation, and illegal state transitions."""
    order = LiveOrder(
        order_id="ORD_002",
        signal_id="SIG_002",
        symbol="GBPUSD",
        side="SELL",
        quantity=0.5,
        price=1.2700,
    )

    OrderStateMachine.transition(order, LiveOrderState.SUBMITTED, reason="Sent to broker")
    OrderStateMachine.transition(
        order, LiveOrderState.REJECTED, reason="Broker rejected: Insufficient margin"
    )

    assert order.status == LiveOrderState.REJECTED
    assert order.is_terminal

    # Attempting to transition from terminal REJECTED state must fail
    with pytest.raises(InvalidOrderStateTransitionError):
        OrderStateMachine.transition(
            order, LiveOrderState.FILLED, reason="Illegal fill after reject"
        )


def test_partial_fill_path() -> None:
    """Test PARTIALLY_FILLED -> FILLED progression."""
    order = LiveOrder(
        order_id="ORD_003",
        signal_id="SIG_003",
        symbol="USDJPY",
        side="BUY",
        quantity=2.0,
        price=150.00,
    )

    OrderStateMachine.transition(order, LiveOrderState.SUBMITTED, reason="Sent")
    OrderStateMachine.transition(
        order, LiveOrderState.ACKNOWLEDGED, reason="Acked", broker_ticket=999
    )
    OrderStateMachine.transition(
        order,
        LiveOrderState.PARTIALLY_FILLED,
        reason="Partial fill 1.0 lot",
        filled_quantity=1.0,
        filled_price=150.01,
    )

    assert order.status == LiveOrderState.PARTIALLY_FILLED
    assert order.filled_quantity == 1.0

    OrderStateMachine.transition(
        order,
        LiveOrderState.FILLED,
        reason="Second partial fill complete",
        filled_quantity=2.0,
        filled_price=150.02,
    )
    assert order.status == LiveOrderState.FILLED
    assert order.filled_quantity == 2.0
