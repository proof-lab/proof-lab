"""Paper execution engine and portfolio accounting simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.orders import ExitReason, OrderRecord, OrderSide
from prooflab.paper.consumer import LiveBar, LiveTick
from prooflab.paper.ledger import PaperTradeLedger
from prooflab.risk.limits import OpenPositionRecord


class PaperExecutionConfig(BaseModel):
    """Configuration governing simulated paper execution friction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float = Field(default=100000.0, gt=0.0)
    commission_per_unit: float = Field(default=0.00003, ge=0.0)
    slippage_pips: float = Field(default=0.5, ge=0.0)
    pip_size: float = Field(default=0.0001, gt=0.0)


@dataclass
class PaperPosition:
    """Active open paper trading position."""

    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    current_price: float = 0.0
    commission_paid: float = 0.0
    spread_cost_paid: float = 0.0
    slippage_cost_paid: float = 0.0

    @property
    def nominal_exposure(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.current_price <= 0:
            return 0.0
        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity


class PaperExecutionEngine:
    """Executes paper orders with realistic friction models and updates portfolio equity."""

    def __init__(
        self,
        config: PaperExecutionConfig | None = None,
        ledger: PaperTradeLedger | None = None,
    ) -> None:
        self.config = config or PaperExecutionConfig()
        self.ledger = ledger or PaperTradeLedger()

        self.cash: float = self.config.initial_capital
        self._positions: dict[str, PaperPosition] = {}

    @property
    def open_positions(self) -> list[PaperPosition]:
        return list(self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def current_equity(self) -> float:
        return self.cash + self.total_unrealized_pnl

    def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        spread_pips: float = 1.0,
    ) -> OrderRecord:
        """Fill a new paper market order, applying slippage, spread, and commission."""
        slippage_dist = self.config.slippage_pips * self.config.pip_size
        spread_dist = spread_pips * self.config.pip_size

        # Effective fill price
        if side == "BUY":
            fill_price = current_price + slippage_dist + (spread_dist / 2.0)
        else:
            fill_price = current_price - slippage_dist - (spread_dist / 2.0)

        commission = quantity * self.config.commission_per_unit
        spread_cost = quantity * spread_dist
        slippage_cost = quantity * slippage_dist

        # Deduct entry commission from cash
        self.cash -= commission

        pos_id = f"POS-{uuid4().hex[:8].upper()}"
        pos = PaperPosition(
            position_id=pos_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=fill_price,
            commission_paid=commission,
            spread_cost_paid=spread_cost,
            slippage_cost_paid=slippage_cost,
        )
        self._positions[pos_id] = pos

        order = OrderRecord(
            order_id=pos_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            requested_price=current_price,
            fill_price=round(fill_price, 5),
            quantity=quantity,
            stop=stop_loss,
            target=take_profit,
            spread=round(spread_cost, 2),
            commission=round(commission, 2),
            slippage=round(slippage_cost, 2),
            fill_timestamp=timestamp,
            status="FILLED",
        )
        return order

    def update_on_tick(self, tick: LiveTick) -> list[OrderRecord]:
        """Update mark-to-market prices and evaluate barrier stops/targets on live tick."""
        closed_orders: list[OrderRecord] = []
        to_close: list[tuple[str, float, ExitReason]] = []

        for pos_id, pos in self._positions.items():
            if pos.symbol != tick.symbol:
                continue

            pos.current_price = tick.bid if pos.side == "BUY" else tick.ask

            if pos.side == "BUY":
                if pos.stop_loss and tick.bid <= pos.stop_loss:
                    to_close.append((pos_id, tick.bid, "STOP_LOSS"))
                elif pos.take_profit and tick.bid >= pos.take_profit:
                    to_close.append((pos_id, tick.bid, "TAKE_PROFIT"))
            else:
                if pos.stop_loss and tick.ask >= pos.stop_loss:
                    to_close.append((pos_id, tick.ask, "STOP_LOSS"))
                elif pos.take_profit and tick.ask <= pos.take_profit:
                    to_close.append((pos_id, tick.ask, "TAKE_PROFIT"))

        for p_id, exit_p, reason in to_close:
            order = self._close_position(
                pos_id=p_id,
                exit_price=exit_p,
                exit_time=tick.timestamp_utc,
                reason=reason,
            )
            closed_orders.append(order)

        return closed_orders

    def update_on_bar(self, bar: LiveBar) -> list[OrderRecord]:
        """Update mark-to-market prices and evaluate barrier stops/targets on completed bar."""
        closed_orders: list[OrderRecord] = []
        to_close: list[tuple[str, float, ExitReason]] = []

        for pos_id, pos in self._positions.items():
            if pos.symbol != bar.symbol:
                continue

            pos.current_price = bar.close

            if pos.side == "BUY":
                if pos.stop_loss and bar.low <= pos.stop_loss:
                    to_close.append((pos_id, pos.stop_loss, "STOP_LOSS"))
                elif pos.take_profit and bar.high >= pos.take_profit:
                    to_close.append((pos_id, pos.take_profit, "TAKE_PROFIT"))
            else:
                if pos.stop_loss and bar.high >= pos.stop_loss:
                    to_close.append((pos_id, pos.stop_loss, "STOP_LOSS"))
                elif pos.take_profit and bar.low <= pos.take_profit:
                    to_close.append((pos_id, pos.take_profit, "TAKE_PROFIT"))

        for p_id, exit_p, reason in to_close:
            order = self._close_position(
                pos_id=p_id,
                exit_price=exit_p,
                exit_time=bar.timestamp_utc,
                reason=reason,
            )
            closed_orders.append(order)

        return closed_orders

    def close_position_manually(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason = "MANUAL_EXIT",
    ) -> OrderRecord:
        """Manually close an open paper position."""
        return self._close_position(position_id, exit_price, exit_time, reason)

    def _close_position(
        self,
        pos_id: str,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason,
    ) -> OrderRecord:
        pos = self._positions.pop(pos_id)

        # Gross PnL
        if pos.side == "BUY":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_pips = (exit_price - pos.entry_price) / self.config.pip_size
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_pips = (pos.entry_price - exit_price) / self.config.pip_size

        exit_comm = pos.quantity * self.config.commission_per_unit
        total_comm = pos.commission_paid + exit_comm
        net_pnl = gross_pnl - total_comm

        # Realize cash
        self.cash += (gross_pnl - exit_comm)

        order = OrderRecord(
            order_id=pos.position_id,
            timestamp=pos.entry_time,
            symbol=pos.symbol,
            side=pos.side,
            requested_price=pos.entry_price,
            fill_price=round(pos.entry_price, 5),
            quantity=pos.quantity,
            stop=pos.stop_loss,
            target=pos.take_profit,
            fill_timestamp=pos.entry_time,
            exit_timestamp=exit_time,
            exit_price=round(exit_price, 5),
            exit_reason=reason,
            spread=round(pos.spread_cost_paid, 2),
            commission=round(total_comm, 2),
            slippage=round(pos.slippage_cost_paid, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            pnl_pips=round(pnl_pips, 2),
            status="CLOSED",
        )

        self.ledger.record_trade(order)
        return order

    def get_open_position_records(self) -> list[OpenPositionRecord]:
        """Convert active positions to Risk Engine OpenPositionRecord snapshot."""
        records: list[OpenPositionRecord] = []
        for p in self._positions.values():
            records.append(
                OpenPositionRecord(
                    symbol=p.symbol,
                    side=p.side,
                    quantity=p.quantity,
                    nominal_exposure=round(p.nominal_exposure, 2),
                    unrealized_pnl=round(p.unrealized_pnl, 2),
                )
            )
        return records
