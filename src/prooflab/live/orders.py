"""Live order state machine, models, and transition audit logging."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LiveOrderState(StrEnum):
    """Explicit lifecycle states for live broker orders."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class LiveOrderEvent(BaseModel):
    """Immutable audit record of an order lifecycle transition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    from_state: LiveOrderState
    to_state: LiveOrderState
    reason: str
    broker_ticket: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


ALLOWED_ORDER_TRANSITIONS: dict[LiveOrderState, set[LiveOrderState]] = {
    LiveOrderState.CREATED: {
        LiveOrderState.SUBMITTED,
        LiveOrderState.REJECTED,
        LiveOrderState.CANCELLED,
        LiveOrderState.FAILED,
    },
    LiveOrderState.SUBMITTED: {
        LiveOrderState.ACKNOWLEDGED,
        LiveOrderState.FILLED,
        LiveOrderState.REJECTED,
        LiveOrderState.FAILED,
    },
    LiveOrderState.ACKNOWLEDGED: {
        LiveOrderState.PARTIALLY_FILLED,
        LiveOrderState.FILLED,
        LiveOrderState.CANCELLED,
        LiveOrderState.EXPIRED,
        LiveOrderState.REJECTED,
        LiveOrderState.FAILED,
    },
    LiveOrderState.PARTIALLY_FILLED: {
        LiveOrderState.FILLED,
        LiveOrderState.CLOSED,
        LiveOrderState.CANCELLED,
        LiveOrderState.EXPIRED,
        LiveOrderState.FAILED,
    },
    LiveOrderState.FILLED: {
        LiveOrderState.CLOSED,
    },
    LiveOrderState.CLOSED: set(),
    LiveOrderState.REJECTED: set(),
    LiveOrderState.CANCELLED: set(),
    LiveOrderState.EXPIRED: set(),
    LiveOrderState.FAILED: set(),
}


class InvalidOrderStateTransitionError(Exception):
    """Raised when an illegal order state transition is attempted."""


class LiveOrder(BaseModel):
    """Canonical live order tracking model with full execution and audit history."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    signal_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str = "MARKET"  # "MARKET", "LIMIT", "STOP"
    quantity: float
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    status: LiveOrderState = LiveOrderState.CREATED
    filled_quantity: float = 0.0
    filled_price: float | None = None
    broker_ticket: int | None = None
    commission: float = 0.0
    swap: float = 0.0
    slippage: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    events: list[LiveOrderEvent] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Return True if order is currently active/open."""
        return self.status in {
            LiveOrderState.CREATED,
            LiveOrderState.SUBMITTED,
            LiveOrderState.ACKNOWLEDGED,
            LiveOrderState.PARTIALLY_FILLED,
            LiveOrderState.FILLED,
        }

    @property
    def is_terminal(self) -> bool:
        """Return True if order has reached a final immutable state."""
        return self.status in {
            LiveOrderState.CLOSED,
            LiveOrderState.REJECTED,
            LiveOrderState.CANCELLED,
            LiveOrderState.EXPIRED,
            LiveOrderState.FAILED,
        }


class OrderStateMachine:
    """State machine enforcing valid order progression and producing transition audit trails."""

    @staticmethod
    def transition(
        order: LiveOrder,
        target_state: LiveOrderState,
        reason: str,
        *,
        filled_quantity: float | None = None,
        filled_price: float | None = None,
        broker_ticket: int | None = None,
        commission: float | None = None,
        swap: float | None = None,
        slippage: float | None = None,
        timestamp: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> LiveOrderEvent:
        """Validate and apply a state transition to a LiveOrder."""
        if target_state == order.status:
            return LiveOrderEvent(
                timestamp_utc=timestamp or datetime.now(UTC),
                from_state=order.status,
                to_state=target_state,
                reason=f"No-op: already in state {target_state}",
                broker_ticket=broker_ticket or order.broker_ticket,
            )

        valid_targets = ALLOWED_ORDER_TRANSITIONS.get(order.status, set())
        if target_state not in valid_targets:
            raise InvalidOrderStateTransitionError(
                f"Illegal order transition for {order.order_id}: "
                f"{order.status} -> {target_state}. Allowed targets: {valid_targets}"
            )

        event = LiveOrderEvent(
            timestamp_utc=timestamp or datetime.now(UTC),
            from_state=order.status,
            to_state=target_state,
            reason=reason,
            broker_ticket=broker_ticket or order.broker_ticket,
            details=details or {},
        )

        order.status = target_state
        order.updated_at = event.timestamp_utc
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
        if filled_price is not None:
            order.filled_price = filled_price
        if broker_ticket is not None:
            order.broker_ticket = broker_ticket
        if commission is not None:
            order.commission = commission
        if swap is not None:
            order.swap = swap
        if slippage is not None:
            order.slippage = slippage

        order.events.append(event)
        return event
