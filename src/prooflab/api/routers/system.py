"""System health, versioning, and background job inspection endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from prooflab.api.dependencies import get_job_manager
from prooflab.api.jobs import JobManager, JobStatus
from prooflab.api.schemas import JobResponse, SystemHealthResponse, SystemVersionResponse

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> SystemHealthResponse:
    """Check API server health and active job load."""
    running_jobs = len(job_manager.list_jobs(status=JobStatus.RUNNING))
    return SystemHealthResponse(
        status="OK",
        version="0.1.0",
        timestamp_utc=datetime.now(UTC),
        active_jobs=running_jobs,
        environment="production",
    )


@router.get("/version", response_model=SystemVersionResponse)
async def get_system_version() -> SystemVersionResponse:
    """Retrieve application version and build specifications."""
    return SystemVersionResponse(
        version="0.1.0",
        api_version="v1",
        quantitative_core="0.1.0",
    )


@router.get("/jobs", response_model=list[JobResponse])
async def list_background_jobs(
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
    limit: int = Query(default=50, ge=1, le=200),
    job_type: str | None = None,
    status_filter: JobStatus | None = Query(default=None, alias="status"),
) -> list[JobResponse]:
    """List background quantitative task executions."""
    jobs = job_manager.list_jobs(limit=limit, job_type=job_type, status=status_filter)
    return [
        JobResponse(
            job_id=j.job_id,
            job_type=j.job_type,
            status=j.status,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
            progress=j.progress,
            result=j.result,
            error=j.error,
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_background_job(
    job_id: str,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> JobResponse:
    """Retrieve detailed execution status and output for a background job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
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


@router.delete("/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def cancel_background_job(
    job_id: str,
    job_manager: Annotated[JobManager, Depends(get_job_manager)],
) -> dict[str, Any]:
    """Attempt cancellation of a pending background job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} could not be cancelled or has already completed",
        )
    return {"message": f"Job {job_id} cancelled successfully", "job_id": job_id}
