"""Sovereign Risk Engine and Emergency Kill Switch endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from prooflab.api.dependencies import get_kill_switch, get_risk_engine
from prooflab.api.schemas import (
    EvaluateSignalRequest,
    KillSwitchActivateRequest,
    KillSwitchResetRequest,
    RiskDecisionResponse,
)
from prooflab.api.security import verify_admin_key, verify_api_key
from prooflab.risk.engine import RiskEngine
from prooflab.risk.kill_switch import KillSwitch, KillSwitchPolicy

router = APIRouter(prefix="/api/risk", tags=["Risk"])


@router.get("/limits", status_code=status.HTTP_200_OK)
async def get_risk_limits(
    engine: Annotated[RiskEngine, Depends(get_risk_engine)],
    _token: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Inspect current hard risk limits and kill-switch status."""
    return {
        "kill_switch_active": engine.kill_switch.is_active,
        "account_equity": engine.state_tracker.current_equity,
        "max_daily_loss_pct": engine.limits_evaluator.config.max_daily_loss_pct,
        "max_weekly_loss_pct": engine.limits_evaluator.config.max_weekly_loss_pct,
        "max_concurrent_trades": engine.limits_evaluator.config.max_open_positions,
        "max_symbol_risk_pct": engine.limits_evaluator.config.max_risk_per_trade_pct,
    }


@router.post("/evaluate-signal", response_model=RiskDecisionResponse)
async def evaluate_order_signal(
    req: EvaluateSignalRequest,
    engine: Annotated[RiskEngine, Depends(get_risk_engine)],
    _token: Annotated[str, Depends(verify_api_key)],
) -> RiskDecisionResponse:
    """Evaluate a trading signal against the sovereign Risk Engine."""
    decision = engine.evaluate_signal(
        symbol=req.symbol,
        side=req.side.upper(),
        entry_price=req.entry_price,
        stop_loss_price=req.stop_loss_price,
        current_time=datetime.now(UTC),
        model_confidence=req.confidence,
    )
    return RiskDecisionResponse(
        action=decision.action.value,
        is_approved=decision.is_approved,
        symbol=decision.symbol,
        order_side=decision.order_side,
        approved_units=decision.approved_units,
        message=decision.message,
        rejection_reasons=decision.rejection_reasons,
    )


@router.post("/kill-switch/activate", status_code=status.HTTP_200_OK)
async def activate_kill_switch(
    req: KillSwitchActivateRequest,
    kill_switch: Annotated[KillSwitch, Depends(get_kill_switch)],
    _token: Annotated[str, Depends(verify_admin_key)],
) -> dict[str, Any]:
    """Trigger emergency trading halt across all execution paths."""
    policy = (
        KillSwitchPolicy.CLOSE_ALL
        if req.policy.upper() == "CLOSE_ALL"
        else KillSwitchPolicy.HOLD_OPEN
    )
    event = kill_switch.activate(
        reason=req.reason,
        actor=req.actor,
        policy=policy,
    )
    return {
        "status": "HALTED",
        "is_active": kill_switch.is_active,
        "reason": event.reason,
        "actor": event.actor,
        "timestamp_utc": event.timestamp_utc,
    }


@router.post("/kill-switch/reset", status_code=status.HTTP_200_OK)
async def reset_kill_switch(
    req: KillSwitchResetRequest,
    kill_switch: Annotated[KillSwitch, Depends(get_kill_switch)],
    _token: Annotated[str, Depends(verify_admin_key)],
) -> dict[str, Any]:
    """Reset emergency kill switch after human safety clearance."""
    if not kill_switch.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kill switch is not currently active",
        )
    event = kill_switch.reset(
        actor=req.actor,
        reason=req.reason,
    )
    return {
        "status": "NORMAL",
        "is_active": kill_switch.is_active,
        "reason": event.reason,
        "actor": event.actor,
        "timestamp_utc": event.timestamp_utc,
    }
