"""Portfolio accounting, risk-based position sizing, and equity curve generation.

Enforces risk-per-trade position sizing clamped strictly to broker limits, real-time
margin tracking, and mark-to-market gross vs net equity curve tracking.
"""

from __future__ import annotations

import math

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from prooflab.backtest.orders import Position


class BrokerLimitsConfig(BaseModel):
    """Configuration governing broker trading and leverage constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lot_unit_size: float = Field(default=100000.0, gt=0.0)  # 1.0 standard lot = 100,000 units
    min_lot_size: float = Field(default=0.01, gt=0.0)       # 0.01 micro lot
    max_lot_size: float = Field(default=50.0, gt=0.0)       # 50.0 standard lots
    lot_step: float = Field(default=0.01, gt=0.0)           # 0.01 lot increments
    max_leverage: float = Field(default=30.0, gt=0.0)       # 30:1 leverage
    max_open_positions: int = Field(default=5, ge=1)
    margin_call_level_pct: float = Field(default=100.0, ge=0.0)  # 100% margin level
    stop_out_level_pct: float = Field(default=50.0, ge=0.0)      # 50% liquidation stop-out

    @model_validator(mode="after")
    def validate_limits(self) -> BrokerLimitsConfig:
        if self.min_lot_size > self.max_lot_size:
            raise ValueError("min_lot_size cannot exceed max_lot_size.")
        if self.stop_out_level_pct > self.margin_call_level_pct:
            raise ValueError("stop_out_level_pct cannot exceed margin_call_level_pct.")
        return self


class PositionSizer:
    """Calculates risk-based position sizes adhering to equity risk and broker bounds."""

    def __init__(self, broker_limits: BrokerLimitsConfig | None = None) -> None:
        self.limits = broker_limits or BrokerLimitsConfig()

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float | None,
        risk_per_trade_pct: float = 0.01,
        *,
        point_value: float = 1.0,
    ) -> float:
        """Calculate the clamped trade quantity in units based on risk per trade.

        Formula:
            risk_amount = account_equity * risk_per_trade_pct
            stop_loss_distance = abs(entry_price - stop_loss_price)
            position_units = risk_amount / (stop_loss_distance * point_value)
            lots = clamp(round_down_to_step(position_units / lot_unit_size), min_lot, max_lot)

        Args:
            account_equity: Current net account equity.
            entry_price: Anticipated entry price.
            stop_loss_price: Anticipated protective stop loss price.
            risk_per_trade_pct: Fraction of equity to risk (e.g. 0.01 for 1%).
            point_value: Point monetary multiplier.

        Returns:
            Position size in units (e.g. 100,000 for 1.0 standard lot).
        """
        if account_equity <= 0:
            return 0.0

        if stop_loss_price is None or abs(entry_price - stop_loss_price) <= 1e-9:
            # Fallback when stop loss is unconfigured: size at 1x leverage maximum risk
            max_notional = account_equity * self.limits.max_leverage
            raw_units = max_notional / entry_price
        else:
            risk_amount = account_equity * float(risk_per_trade_pct)
            stop_distance = abs(entry_price - stop_loss_price)
            raw_units = risk_amount / (stop_distance * float(point_value))

        # Convert units to lots
        raw_lots = raw_units / self.limits.lot_unit_size

        # Round down to nearest allowed lot step (with floating-point precision defense)
        step = self.limits.lot_step
        stepped_lots = round(math.floor(round(raw_lots / step, 8)) * step, 4)

        # Check maximum leverage cap: notional <= equity * max_leverage
        max_notional_units = (account_equity * self.limits.max_leverage) / entry_price
        max_allowed_lots = max_notional_units / self.limits.lot_unit_size
        effective_max_lots = min(self.limits.max_lot_size, max_allowed_lots)

        # Clamp between min_lot and effective_max_lots
        if stepped_lots < self.limits.min_lot_size:
            if max_allowed_lots >= self.limits.min_lot_size:
                final_lots = self.limits.min_lot_size
            else:
                return 0.0  # Not enough margin even for minimum lot
        else:
            final_lots = min(stepped_lots, effective_max_lots)

        return float(round(final_lots * self.limits.lot_unit_size, 4))


class EquitySnapshot(BaseModel):
    """Immutable point-in-time portfolio valuation snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: AwareDatetime
    cash: float
    gross_equity: float
    net_equity: float
    unrealized_gross_pnl: float
    unrealized_net_pnl: float
    realized_gross_pnl: float
    realized_net_pnl: float
    margin_used: float
    free_margin: float
    margin_level_pct: float | None
    drawdown_gross: float
    drawdown_gross_pct: float
    drawdown_net: float
    drawdown_net_pct: float
    open_positions: int


class PortfolioAccountant:
    """Tracks cash, open margin, mark-to-market valuations, and equity curves."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        broker_limits: BrokerLimitsConfig | None = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.limits = broker_limits or BrokerLimitsConfig()

        self.realized_gross_pnl: float = 0.0
        self.realized_net_pnl: float = 0.0
        self.total_commission_paid: float = 0.0
        self.total_spread_paid: float = 0.0
        self.total_slippage_paid: float = 0.0
        self.total_swap_paid: float = 0.0

        self.peak_gross_equity: float = float(initial_capital)
        self.peak_net_equity: float = float(initial_capital)

        self.history: list[EquitySnapshot] = []

    def can_open_position(
        self,
        quantity: float,
        price: float,
        active_positions_count: int,
    ) -> tuple[bool, str | None]:
        """Verify broker limits (max open positions, available margin)."""
        if active_positions_count >= self.limits.max_open_positions:
            return False, f"Maximum open positions limit reached ({self.limits.max_open_positions})"

        required_margin = (quantity * price) / self.limits.max_leverage
        # Free margin check based on current net equity estimate
        current_net_equity = self.cash
        if required_margin > current_net_equity:
            return (
                False,
                "Insufficient margin: requires , free equity is ",
            )

        return True, None

    def record_trade_close(
        self,
        gross_pnl: float,
        net_pnl: float,
        commission: float,
        spread: float,
        slippage: float,
        swap: float,
    ) -> None:
        """Update cash and realized accounting upon closing a position."""
        self.realized_gross_pnl += gross_pnl
        self.realized_net_pnl += net_pnl
        self.cash += net_pnl  # Cash adjusts by the net proceeds

        self.total_commission_paid += commission
        self.total_spread_paid += spread
        self.total_slippage_paid += slippage
        self.total_swap_paid += swap

    def update_snapshot(
        self,
        timestamp: AwareDatetime,
        open_positions: list[Position],
        current_prices: dict[str, float],
        point_value: float = 1.0,
    ) -> EquitySnapshot:
        """Compute mark-to-market portfolio snapshot and append to history."""
        unrealized_gross = 0.0
        unrealized_swap = 0.0
        margin_used = 0.0

        for pos in open_positions:
            curr_p = current_prices.get(pos.symbol, pos.entry_price)
            unrealized_gross += pos.calculate_unrealized_pnl(curr_p, point_value=point_value)
            unrealized_swap += pos.accumulated_swap
            margin_used += (pos.quantity * curr_p) / self.limits.max_leverage

        unrealized_net = unrealized_gross - unrealized_swap

        gross_equity = self.initial_capital + self.realized_gross_pnl + unrealized_gross
        net_equity = self.cash + unrealized_net
        free_margin = max(0.0, net_equity - margin_used)

        margin_level_pct = (net_equity / margin_used * 100.0) if margin_used > 0 else None

        # Update peak equity
        self.peak_gross_equity = max(self.peak_gross_equity, gross_equity)
        self.peak_net_equity = max(self.peak_net_equity, net_equity)

        # Drawdowns
        dd_gross = self.peak_gross_equity - gross_equity
        dd_gross_pct = (
            (dd_gross / self.peak_gross_equity * 100.0) if self.peak_gross_equity > 0 else 0.0
        )

        dd_net = self.peak_net_equity - net_equity
        dd_net_pct = (dd_net / self.peak_net_equity * 100.0) if self.peak_net_equity > 0 else 0.0

        snapshot = EquitySnapshot(
            timestamp=timestamp,
            cash=self.cash,
            gross_equity=gross_equity,
            net_equity=net_equity,
            unrealized_gross_pnl=unrealized_gross,
            unrealized_net_pnl=unrealized_net,
            realized_gross_pnl=self.realized_gross_pnl,
            realized_net_pnl=self.realized_net_pnl,
            margin_used=margin_used,
            free_margin=free_margin,
            margin_level_pct=margin_level_pct,
            drawdown_gross=dd_gross,
            drawdown_gross_pct=dd_gross_pct,
            drawdown_net=dd_net,
            drawdown_net_pct=dd_net_pct,
            open_positions=len(open_positions),
        )

        self.history.append(snapshot)
        return snapshot

    def get_equity_curve_dataframe(self) -> pd.DataFrame:
        """Export the equity curve history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        records = [s.model_dump(mode="python") for s in self.history]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
