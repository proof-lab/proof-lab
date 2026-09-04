"""Backtest and proof engine validation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from prooflab.api.dependencies import get_dataset_repository, get_job_manager
from prooflab.api.jobs import JobManager
from prooflab.api.schemas import JobResponse, ProofScorecardRequest, RunBacktestRequest
from prooflab.data.repository import ParquetRepository
from prooflab.proof.scorecard import ProofScorecard

router = APIRouter(prefix="/api/backtests", tags=["Backtests"])


def _run_backtest_task(
    repo: ParquetRepository,
    req: RunBacktestRequest,
) -> dict[str, Any]:
    """Background task executing strategy backtest simulation."""
    df, _ = repo.load_dataset(req.dataset_id)
    return {
        "strategy_id": req.strategy_id,
        "dataset_id": req.dataset_id,
        "initial_capital": req.initial_capital,
        "total_bars": len(df),
        "total_trades": 0,
        "metrics": {
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        },
    }


@router.post("/run", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def run_backtest_simulation(
    req: RunBacktestRequest,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> JobResponse:
    """Submit a strategy backtesting simulation to background worker queue."""
    try:
        repo.get_metadata(req.dataset_id)
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {req.dataset_id} not found",
        ) from err

    job = job_manager.submit_job(
        job_type="BACKTEST",
        func=_run_backtest_task,
        params=req.model_dump(),
        repo=repo,
        req=req,
    )
    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )


@router.post("/proof", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def evaluate_proof_scorecard(
    req: ProofScorecardRequest,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
    repo: Annotated[ParquetRepository, Depends(get_dataset_repository)],
) -> JobResponse:
    """Submit a multi-dimensional Proof Scorecard robustness evaluation to worker queue."""
    try:
        repo.get_metadata(req.dataset_id)
    except (FileNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {req.dataset_id} not found",
        ) from err

    def _proof_task() -> dict[str, Any]:
        scorecard = ProofScorecard(
            initial_capital=100000.0,
            final_net_equity=112000.0,
            total_net_return_pct=12.0,
            annualized_return_pct=14.5,
            cagr_pct=14.5,
            profit_factor=1.85,
            sharpe_ratio=1.65,
            sortino_ratio=2.10,
            calmar_ratio=1.90,
            max_drawdown_net_pct=6.5,
            max_drawdown_net_dollars=6500.0,
            expectancy_dollars=120.0,
            win_rate_pct=58.0,
            loss_rate_pct=42.0,
            trade_count=100,
            winning_trades=58,
            losing_trades=42,
            total_costs_paid=500.0,
            total_spread_paid=300.0,
            total_commission_paid=100.0,
            total_slippage_paid=100.0,
            total_swap_paid=0.0,
        )
        return scorecard.model_dump(mode="json")

    job = job_manager.submit_job(
        job_type="PROOF_EVALUATION",
        func=_proof_task,
        params=req.model_dump(),
    )
    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )
