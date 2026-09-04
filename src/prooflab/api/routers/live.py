"""Live and Paper trading governance endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from prooflab.api.dependencies import get_lifecycle_manager
from prooflab.api.schemas import LiveEnableRequest
from prooflab.api.security import verify_admin_key, verify_api_key
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState

router = APIRouter(prefix="/api/live", tags=["Live"])

# Hard global safety flag: live execution disabled by default across entire system
_LIVE_MODE_ENABLED: bool = False


@router.get("/status", status_code=status.HTTP_200_OK)
async def get_live_status(
    lifecycle: Annotated[StrategyLifecycleManager, Depends(get_lifecycle_manager)],
    _token: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Inspect current live and paper trading operational state."""
    return {
        "live_trading_enabled": _LIVE_MODE_ENABLED,
        "paper_strategy_state": lifecycle.current_state.value,
        "strategy_id": lifecycle.strategy_id,
        "mode": "PAPER" if not _LIVE_MODE_ENABLED else "LIVE",
    }


@router.post("/enable", status_code=status.HTTP_200_OK)
async def enable_live_trading(
    req: LiveEnableRequest,
    lifecycle: Annotated[StrategyLifecycleManager, Depends(get_lifecycle_manager)],
    _token: Annotated[str, Depends(verify_admin_key)],
) -> dict[str, Any]:
    """Explicit human approval gate to enable live capital execution."""
    global _LIVE_MODE_ENABLED

    if not req.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Live trading requires explicit confirmation boolean",
        )

    # Sovereign safety check: must be approved or in paper trading before live enablement
    if lifecycle.current_state not in (
        StrategyLifecycleState.APPROVED,
        StrategyLifecycleState.PAPER_TRADING,
    ):
        st_val = lifecycle.current_state.value
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot enable live mode for strategy in state: {st_val}",
        )

    _LIVE_MODE_ENABLED = True
    return {
        "status": "LIVE_ENABLED",
        "strategy_id": req.strategy_id,
        "authorized_by": req.authorized_by,
        "reason": req.reason,
        "warning": "Live execution active. Hard risk gates and kill switch remain sovereign.",
    }
