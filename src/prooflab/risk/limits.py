"""Exposure, daily/weekly drawdown, and consecutive loss limit enforcement."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LimitBreachReason(StrEnum):
    """Specific risk limit violation classification."""

    MAX_RISK_PER_TRADE_EXCEEDED = "MAX_RISK_PER_TRADE_EXCEEDED"
    MAX_OPEN_POSITIONS_REACHED = "MAX_OPEN_POSITIONS_REACHED"
    MAX_TOTAL_LEVERAGE_EXCEEDED = "MAX_TOTAL_LEVERAGE_EXCEEDED"
    MAX_SYMBOL_POSITIONS_REACHED = "MAX_SYMBOL_POSITIONS_REACHED"
    MAX_SYMBOL_LEVERAGE_EXCEEDED = "MAX_SYMBOL_LEVERAGE_EXCEEDED"
    MAX_DAILY_LOSS_BREACHED = "MAX_DAILY_LOSS_BREACHED"
    MAX_WEEKLY_LOSS_BREACHED = "MAX_WEEKLY_LOSS_BREACHED"
    MAX_CONSECUTIVE_LOSSES_BREACHED = "MAX_CONSECUTIVE_LOSSES_BREACHED"


class RiskLimitsConfig(BaseModel):
    """Configuration governing exposure, drawdown, and sequence loss limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_risk_per_trade_pct: float = Field(default=0.01, ge=0.0001, le=1.0)
    max_open_positions: int = Field(default=5, ge=1)
    max_total_leverage: float = Field(default=10.0, ge=0.1)

    max_symbol_positions: int = Field(default=2, ge=1)
    max_symbol_leverage: float = Field(default=3.0, ge=0.1)

    max_daily_loss_pct: float = Field(default=0.03, ge=0.001, le=1.0)
    max_weekly_loss_pct: float = Field(default=0.06, ge=0.001, le=1.0)
    max_consecutive_losses: int = Field(default=4, ge=1)


class LimitEvaluationResult(BaseModel):
    """Result of evaluating risk limits against a proposed order request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    breach_reasons: list[LimitBreachReason] = Field(default_factory=list)
    rejection_message: str | None = None
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class OpenPositionRecord(BaseModel):
    """Audit schema for active positions tracked by the Risk Engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: str
    quantity: float
    nominal_exposure: float
    unrealized_pnl: float = 0.0


class RiskStateTracker:
    """Stateful portfolio risk tracker recording daily/weekly PnL and loss streaks."""

    def __init__(
        self,
        initial_equity: float,
        current_time: datetime | None = None,
    ) -> None:
        self.current_equity = initial_equity
        self.start_of_day_equity = initial_equity
        self.start_of_week_equity = initial_equity

        self.daily_realized_pnl = 0.0
        self.weekly_realized_pnl = 0.0
        self.consecutive_loss_streak = 0

        self.current_date = current_time.date() if current_time else None
        self.current_iso_week = (
            current_time.isocalendar()[:2] if current_time else None
        )
        self.open_positions: list[OpenPositionRecord] = []

    def check_period_rollover(self, timestamp: datetime, current_equity: float) -> None:
        """Update equity and reset daily/weekly tracking when calendar boundaries pass."""
        self.current_equity = current_equity
        new_date = timestamp.date()
        new_iso_week = timestamp.isocalendar()[:2]

        if self.current_date is None or new_date > self.current_date:
            self.current_date = new_date
            self.start_of_day_equity = current_equity
            self.daily_realized_pnl = 0.0

        if self.current_iso_week is None or new_iso_week > self.current_iso_week:
            self.current_iso_week = new_iso_week
            self.start_of_week_equity = current_equity
            self.weekly_realized_pnl = 0.0

    def record_closed_trade(self, net_pnl: float) -> None:
        """Record trade realization and update consecutive loss count."""
        self.daily_realized_pnl += net_pnl
        self.weekly_realized_pnl += net_pnl

        if net_pnl < 0:
            self.consecutive_loss_streak += 1
        elif net_pnl > 0:
            self.consecutive_loss_streak = 0

    def set_open_positions(self, positions: list[OpenPositionRecord]) -> None:
        """Update active positions snapshot."""
        self.open_positions = list(positions)


class RiskLimitsEvaluator:
    """Enforces sovereign hard limits on exposure, drawdown, and loss streaks."""

    def __init__(self, config: RiskLimitsConfig | None = None) -> None:
        self.config = config or RiskLimitsConfig()

    def evaluate_new_order(
        self,
        symbol: str,
        requested_nominal_exposure: float,
        risk_amount_dollars: float,
        state: RiskStateTracker,
    ) -> LimitEvaluationResult:
        """Evaluate order against all configured risk limits."""
        breaches: list[LimitBreachReason] = []
        messages: list[str] = []

        equity = state.current_equity
        if equity <= 0:
            return LimitEvaluationResult(
                allowed=False,
                breach_reasons=[LimitBreachReason.MAX_TOTAL_LEVERAGE_EXCEEDED],
                rejection_message="Account equity is zero or negative",
            )

        # 1. Max Risk Per Trade Check
        max_risk_dollars = equity * self.config.max_risk_per_trade_pct
        if risk_amount_dollars > max_risk_dollars * 1.001:  # small epsilon for rounding
            breaches.append(LimitBreachReason.MAX_RISK_PER_TRADE_EXCEEDED)
            messages.append(
                f"Risk  exceeds {self.config.max_risk_per_trade_pct * 100:.1f}% "
                f"limit ()"
            )

        # 2. Max Open Positions Check
        current_open_count = len(state.open_positions)
        if current_open_count >= self.config.max_open_positions:
            breaches.append(LimitBreachReason.MAX_OPEN_POSITIONS_REACHED)
            messages.append(
                f"Open positions ({current_open_count}) reached maximum "
                f"({self.config.max_open_positions})"
            )

        # 3. Max Symbol Positions Check
        sym_open_count = sum(1 for p in state.open_positions if p.symbol == symbol)
        if sym_open_count >= self.config.max_symbol_positions:
            breaches.append(LimitBreachReason.MAX_SYMBOL_POSITIONS_REACHED)
            messages.append(
                f"Open positions for {symbol} ({sym_open_count}) reached maximum "
                f"({self.config.max_symbol_positions})"
            )

        # 4. Total Leverage Check
        existing_total_exposure = sum(p.nominal_exposure for p in state.open_positions)
        new_total_exposure = existing_total_exposure + requested_nominal_exposure
        implied_total_lev = new_total_exposure / equity

        if implied_total_lev > self.config.max_total_leverage:
            breaches.append(LimitBreachReason.MAX_TOTAL_LEVERAGE_EXCEEDED)
            messages.append(
                f"Total leverage {implied_total_lev:.2f}x exceeds limit "
                f"{self.config.max_total_leverage:.2f}x"
            )

        # 5. Symbol Leverage Check
        existing_sym_exposure = sum(
            p.nominal_exposure for p in state.open_positions if p.symbol == symbol
        )
        new_sym_exposure = existing_sym_exposure + requested_nominal_exposure
        implied_sym_lev = new_sym_exposure / equity

        if implied_sym_lev > self.config.max_symbol_leverage:
            breaches.append(LimitBreachReason.MAX_SYMBOL_LEVERAGE_EXCEEDED)
            messages.append(
                f"Symbol leverage for {symbol} ({implied_sym_lev:.2f}x) exceeds limit "
                f"{self.config.max_symbol_leverage:.2f}x"
            )

        # 6. Max Daily Loss Check
        total_unrealized = sum(p.unrealized_pnl for p in state.open_positions)
        daily_total_pnl = state.daily_realized_pnl + total_unrealized
        sod_equity = state.start_of_day_equity
        max_daily_loss_dollars = sod_equity * self.config.max_daily_loss_pct

        if daily_total_pnl <= -max_daily_loss_dollars:
            breaches.append(LimitBreachReason.MAX_DAILY_LOSS_BREACHED)
            messages.append(
                f"Daily loss  breached {self.config.max_daily_loss_pct * 100:.1f}% "
                f"daily limit ()"
            )

        # 7. Max Weekly Loss Check
        sow_equity = state.start_of_week_equity
        weekly_total_pnl = state.weekly_realized_pnl + total_unrealized
        max_weekly_loss_dollars = sow_equity * self.config.max_weekly_loss_pct

        if weekly_total_pnl <= -max_weekly_loss_dollars:
            breaches.append(LimitBreachReason.MAX_WEEKLY_LOSS_BREACHED)
            messages.append(
                f"Weekly loss  breached {self.config.max_weekly_loss_pct * 100:.1f}% "
                f"weekly limit ()"
            )

        # 8. Max Consecutive Losses Check
        if state.consecutive_loss_streak >= self.config.max_consecutive_losses:
            breaches.append(LimitBreachReason.MAX_CONSECUTIVE_LOSSES_BREACHED)
            messages.append(
                f"Consecutive loss streak ({state.consecutive_loss_streak}) reached "
                f"threshold ({self.config.max_consecutive_losses})"
            )

        allowed = len(breaches) == 0
        rejection_msg = "; ".join(messages) if messages else None

        metrics = {
            "current_equity": round(equity, 2),
            "daily_loss_dollars": round(daily_total_pnl, 2),
            "weekly_loss_dollars": round(weekly_total_pnl, 2),
            "consecutive_losses": state.consecutive_loss_streak,
            "open_positions_count": current_open_count,
            "implied_total_leverage": round(implied_total_lev, 2),
        }

        return LimitEvaluationResult(
            allowed=allowed,
            breach_reasons=breaches,
            rejection_message=rejection_msg,
            metrics_snapshot=metrics,
        )
