"""Sovereign Risk Engine intercepting model signals and enforcing portfolio risk gates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.portfolio import BrokerLimitsConfig
from prooflab.risk.kill_switch import KillSwitch
from prooflab.risk.limits import (
    LimitBreachReason,
    OpenPositionRecord,
    RiskLimitsConfig,
    RiskLimitsEvaluator,
    RiskStateTracker,
)
from prooflab.risk.safety import (
    SafetyCheckConfig,
    SafetyMonitor,
    SafetyPauseReason,
)
from prooflab.risk.sizing import RiskPositionSizer


class RiskDecisionAction(StrEnum):
    """Final decision outcome rendered by the Risk Engine."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAUSED = "PAUSED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


class RiskDecision(BaseModel):
    """Immutable audit record of a risk evaluation decision for a trade signal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: RiskDecisionAction
    is_approved: bool
    symbol: str
    order_side: str
    approved_lots: float = Field(default=0.0, ge=0.0)
    approved_units: float = Field(default=0.0, ge=0.0)
    risk_amount_dollars: float = Field(default=0.0, ge=0.0)
    rejection_reasons: list[str] = Field(default_factory=list)
    limit_breaches: list[LimitBreachReason] = Field(default_factory=list)
    safety_pauses: list[SafetyPauseReason] = Field(default_factory=list)
    message: str | None = None
    evaluated_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class RiskEngine:
    """Central risk coordinator with absolute sovereign veto power over model predictions."""

    def __init__(
        self,
        limits_config: RiskLimitsConfig | None = None,
        broker_limits: BrokerLimitsConfig | None = None,
        safety_config: SafetyCheckConfig | None = None,
        kill_switch: KillSwitch | None = None,
        initial_equity: float = 100000.0,
        current_time: datetime | None = None,
    ) -> None:
        self.limits_evaluator = RiskLimitsEvaluator(limits_config or RiskLimitsConfig())
        self.position_sizer = RiskPositionSizer(broker_limits or BrokerLimitsConfig())
        self.safety_monitor = SafetyMonitor(safety_config or SafetyCheckConfig())
        self.kill_switch = kill_switch or KillSwitch()
        self.state_tracker = RiskStateTracker(
            initial_equity=initial_equity,
            current_time=current_time or datetime.now(UTC),
        )

    def evaluate_signal(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        entry_price: float,
        stop_loss_price: float,
        current_time: datetime,
        risk_per_trade_pct: float | None = None,
        signal_id: str | None = None,
        bar_timestamp: datetime | None = None,
        data_timestamp: datetime | None = None,
        is_broker_connected: bool = True,
        last_broker_heartbeat: datetime | None = None,
        is_model_valid: bool = True,
        model_error: str | None = None,
        features: dict[str, float] | None = None,
        current_spread_pips: float | None = None,
        normal_spread_pips: float = 1.0,
        scheduled_news_events: list[datetime] | None = None,
        model_confidence: float | None = None,
    ) -> RiskDecision:
        """Evaluate trade signal through risk, safety, and kill switch gates."""
        # 0. Rollover calendar tracking if day or week has elapsed
        self.state_tracker.check_period_rollover(
            timestamp=current_time,
            current_equity=self.state_tracker.current_equity,
        )

        # 1. Kill Switch Check (Highest Priority Sovereign Gate)
        if self.kill_switch.is_active:
            ks_state = self.kill_switch.state
            return RiskDecision(
                action=RiskDecisionAction.KILL_SWITCH_ACTIVE,
                is_approved=False,
                symbol=symbol,
                order_side=side,
                rejection_reasons=["Emergency kill switch is currently ACTIVE"],
                message=(
                    f"Kill switch triggered by {ks_state.triggered_by}: "
                    f"{ks_state.reason}"
                ),
            )

        # 2. Safety Monitor Checks (Automated Pauses)
        safety_res = self.safety_monitor.check_safety(
            current_time=current_time,
            data_timestamp=data_timestamp,
            is_broker_connected=is_broker_connected,
            last_broker_heartbeat=last_broker_heartbeat,
            is_model_valid=is_model_valid,
            model_error=model_error,
            features=features,
            current_spread_pips=current_spread_pips,
            normal_spread_pips=normal_spread_pips,
            scheduled_news_events=scheduled_news_events,
            model_confidence=model_confidence,
            symbol=symbol,
            signal_id=signal_id,
            bar_timestamp=bar_timestamp,
        )

        if not safety_res.is_safe:
            return RiskDecision(
                action=RiskDecisionAction.PAUSED,
                is_approved=False,
                symbol=symbol,
                order_side=side,
                safety_pauses=safety_res.pause_reasons,
                rejection_reasons=[str(r.value) for r in safety_res.pause_reasons],
                message=safety_res.rejection_message,
            )

        # 3. Position Sizing Gate
        risk_pct = risk_per_trade_pct or self.limits_evaluator.config.max_risk_per_trade_pct
        sizing_res = self.position_sizer.calculate_position_size(
            account_equity=self.state_tracker.current_equity,
            risk_per_trade_pct=risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )

        if not sizing_res.is_valid:
            return RiskDecision(
                action=RiskDecisionAction.REJECTED,
                is_approved=False,
                symbol=symbol,
                order_side=side,
                rejection_reasons=[str(sizing_res.rejection_reason)],
                message=sizing_res.rejection_reason,
            )

        # 4. Exposure & Drawdown Limits Gate
        limit_res = self.limits_evaluator.evaluate_new_order(
            symbol=symbol,
            requested_nominal_exposure=sizing_res.nominal_exposure_dollars,
            risk_amount_dollars=sizing_res.risk_amount_dollars,
            state=self.state_tracker,
        )

        if not limit_res.allowed:
            return RiskDecision(
                action=RiskDecisionAction.REJECTED,
                is_approved=False,
                symbol=symbol,
                order_side=side,
                limit_breaches=limit_res.breach_reasons,
                rejection_reasons=[str(r.value) for r in limit_res.breach_reasons],
                message=limit_res.rejection_message,
            )

        # 5. All gates passed -> APPROVE order
        return RiskDecision(
            action=RiskDecisionAction.APPROVED,
            is_approved=True,
            symbol=symbol,
            order_side=side,
            approved_lots=sizing_res.calculated_lots,
            approved_units=sizing_res.calculated_units,
            risk_amount_dollars=sizing_res.risk_amount_dollars,
            message="Order approved by Risk Engine",
        )

    def record_closed_trade(self, net_pnl: float) -> None:
        """Update tracker with closed trade result."""
        self.state_tracker.record_closed_trade(net_pnl)
        self.state_tracker.current_equity += net_pnl

    def sync_open_positions(self, positions: list[OpenPositionRecord]) -> None:
        """Synchronize current open positions snapshot."""
        self.state_tracker.set_open_positions(positions)
