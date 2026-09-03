"""Order and position lifecycle management for algorithmic backtesting.

Defines the complete order state machine, position tracking, intrabar barrier
evaluation (with gap protection and conservative stop priority), and trade outcome
record serialization.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

OrderSide = Literal["BUY", "SELL"]
OrderStatus = Literal["PENDING", "FILLED", "REJECTED", "CANCELLED", "CLOSED"]
ExitReason = Literal[
    "TAKE_PROFIT",
    "STOP_LOSS",
    "TIME_HORIZON",
    "SIGNAL_REVERSAL",
    "MANUAL_EXIT",
    "FORCE_CLOSE",
]


class OrderRecord(BaseModel):
    """Immutable audit record detailing the complete simulated lifecycle of an order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(default_factory=lambda: f"ORD-{uuid4().hex[:8].upper()}")
    timestamp: AwareDatetime
    symbol: str
    side: OrderSide
    requested_price: float
    fill_price: float | None = None
    quantity: float = Field(gt=0.0)
    stop: float | None = None
    target: float | None = None
    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap: float = 0.0
    status: OrderStatus = "PENDING"
    exit_reason: ExitReason | None = None
    fill_timestamp: AwareDatetime | None = None
    exit_timestamp: AwareDatetime | None = None
    exit_price: float | None = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pips: float = 0.0
    rejection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lifecycle_consistency(self) -> OrderRecord:
        if self.status == "FILLED" and self.fill_price is None:
            raise ValueError("FILLED status requires fill_price.")
        if self.status == "CLOSED":
            if self.fill_price is None or self.exit_price is None:
                raise ValueError("CLOSED status requires fill_price and exit_price.")
            if self.exit_reason is None:
                raise ValueError("CLOSED status requires exit_reason.")
        if self.status == "REJECTED" and self.rejection_reason is None:
            raise ValueError("REJECTED status requires a rejection_reason.")
        return self


class Position:
    """Active simulated trading position tracking open risk, mark-to-market, and barriers."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: float,
        entry_time: AwareDatetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        commission_paid: float = 0.0,
        entry_spread: float = 0.0,
        entry_slippage: float = 0.0,
        max_holding_bars: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.position_id = f"POS-{uuid4().hex[:8].upper()}"
        self.order_id = order_id
        self.symbol = symbol.upper()
        self.side = side
        self.quantity = float(quantity)
        self.entry_price = float(entry_price)
        self.entry_time = entry_time
        self.stop_loss = float(stop_loss) if stop_loss is not None else None
        self.take_profit = float(take_profit) if take_profit is not None else None
        self.commission_paid = float(commission_paid)
        self.entry_spread = float(entry_spread)
        self.entry_slippage = float(entry_slippage)
        self.max_holding_bars = max_holding_bars
        self.metadata = metadata or {}

        self.accumulated_swap: float = 0.0
        self.bars_held: int = 0
        self.is_closed: bool = False

    def increment_bar(self) -> None:
        """Advance holding duration by one bar."""
        self.bars_held += 1

    def apply_swap(self, swap_cost: float) -> None:
        """Accumulate financing/rollover swap charge (positive = cost)."""
        self.accumulated_swap += float(swap_cost)

    def check_intrabar_exit(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        bar_time: AwareDatetime,
    ) -> tuple[ExitReason, float] | None:
        """Evaluate whether bar price action triggered stop loss, take profit, or time exit.

        If both stop-loss and take-profit fall within the bar High/Low range,
        evaluates conservatively (assumes stop loss triggered first). Correctly handles
        market gaps beyond stop/target levels.

        Returns:
            Tuple of (ExitReason, executed_exit_price) or None if position remains open.
        """
        if self.is_closed:
            return None

        # 1. Check max holding time horizon at bar open
        if self.max_holding_bars is not None and self.bars_held >= self.max_holding_bars:
            return "TIME_HORIZON", open_

        # 2. Evaluate barriers for BUY (Long)
        if self.side == "BUY":
            hit_stop = self.stop_loss is not None and low <= self.stop_loss
            hit_target = self.take_profit is not None and high >= self.take_profit

            if hit_stop and hit_target:
                # Ambiguous intrabar: conservative resolution hits stop loss
                exit_price = min(open_, self.stop_loss) if self.stop_loss else open_
                return "STOP_LOSS", exit_price
            if hit_stop and self.stop_loss is not None:
                # If market gapped below stop loss on open, fill at open_
                exit_price = min(open_, self.stop_loss)
                return "STOP_LOSS", exit_price
            if hit_target and self.take_profit is not None:
                # If market gapped above take profit on open, fill at open_
                exit_price = max(open_, self.take_profit)
                return "TAKE_PROFIT", exit_price

        # 3. Evaluate barriers for SELL (Short)
        elif self.side == "SELL":
            hit_stop = self.stop_loss is not None and high >= self.stop_loss
            hit_target = self.take_profit is not None and low <= self.take_profit

            if hit_stop and hit_target:
                # Ambiguous intrabar: conservative resolution hits stop loss
                exit_price = max(open_, self.stop_loss) if self.stop_loss else open_
                return "STOP_LOSS", exit_price
            if hit_stop and self.stop_loss is not None:
                # If market gapped above stop loss on open, fill at open_
                exit_price = max(open_, self.stop_loss)
                return "STOP_LOSS", exit_price
            if hit_target and self.take_profit is not None:
                # If market gapped below take profit on open, fill at open_
                exit_price = min(open_, self.take_profit)
                return "TAKE_PROFIT", exit_price

        return None

    def calculate_unrealized_pnl(
        self,
        current_price: float,
        point_value: float = 1.0,
    ) -> float:
        """Calculate mark-to-market unrealized gross PnL."""
        if self.side == "BUY":
            return (current_price - self.entry_price) * self.quantity * point_value
        else:
            return (self.entry_price - current_price) * self.quantity * point_value

    def close(
        self,
        exit_price: float,
        exit_time: AwareDatetime,
        exit_reason: ExitReason,
        exit_commission: float = 0.0,
        exit_spread: float = 0.0,
        exit_slippage: float = 0.0,
        point_value: float = 1.0,
        pip_size: float = 0.0001,
    ) -> OrderRecord:
        """Close active position and return finalized OrderRecord with full cost audit."""
        self.is_closed = True

        if self.side == "BUY":
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price

        gross_pnl = price_diff * self.quantity * point_value
        pnl_pips = price_diff / pip_size if pip_size > 0 else 0.0

        total_commission = self.commission_paid + exit_commission
        total_spread = self.entry_spread + exit_spread
        total_slippage = self.entry_slippage + exit_slippage
        total_costs = total_commission + total_spread + total_slippage + self.accumulated_swap
        net_pnl = gross_pnl - total_costs

        return OrderRecord(
            order_id=self.order_id,
            timestamp=self.entry_time,
            symbol=self.symbol,
            side=self.side,
            requested_price=self.entry_price,
            fill_price=self.entry_price,
            quantity=self.quantity,
            stop=self.stop_loss,
            target=self.take_profit,
            spread=total_spread,
            commission=total_commission,
            slippage=total_slippage,
            swap=self.accumulated_swap,
            status="CLOSED",
            exit_reason=exit_reason,
            fill_timestamp=self.entry_time,
            exit_timestamp=exit_time,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_pips=pnl_pips,
            metadata={
                **self.metadata,
                "position_id": self.position_id,
                "bars_held": self.bars_held,
            },
        )
