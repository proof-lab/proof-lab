"""Asynchronous background job manager for non-blocking quantitative tasks."""

from __future__ import annotations

import concurrent.futures
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

logger = logging.getLogger("prooflab.api.jobs")


class JobStatus(StrEnum):
    """Lifecycle states for background quantitative jobs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobRecord(BaseModel):
    """Execution state and output payload of a background quantitative job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(default_factory=lambda: f"job-{uuid.uuid4().hex[:12]}")
    job_type: str
    status: JobStatus = JobStatus.PENDING
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    progress: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Manages asynchronous execution and tracking of long-running quantitative tasks."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, concurrent.futures.Future[Any]] = {}

    def submit_job(
        self,
        job_type: str,
        func: Callable[..., dict[str, Any]],
        params: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> JobRecord:
        """Enqueue a quantitative task for asynchronous background execution."""
        record = JobRecord(
            job_type=job_type,
            status=JobStatus.PENDING,
            params=params or {},
        )
        self._jobs[record.job_id] = record

        def _runner(job_id: str) -> None:
            # Mark running
            current = self._jobs[job_id]
            self._jobs[job_id] = JobRecord(
                job_id=current.job_id,
                job_type=current.job_type,
                status=JobStatus.RUNNING,
                created_at=current.created_at,
                started_at=datetime.now(UTC),
                params=current.params,
                progress=0.1,
            )
            try:
                result = func(*args, **kwargs)
                self._jobs[job_id] = JobRecord(
                    job_id=current.job_id,
                    job_type=current.job_type,
                    status=JobStatus.COMPLETED,
                    created_at=current.created_at,
                    started_at=self._jobs[job_id].started_at,
                    completed_at=datetime.now(UTC),
                    progress=1.0,
                    params=current.params,
                    result=result,
                )
            except Exception as exc:
                logger.exception("Job %s failed: %s", job_id, exc)
                self._jobs[job_id] = JobRecord(
                    job_id=current.job_id,
                    job_type=current.job_type,
                    status=JobStatus.FAILED,
                    created_at=current.created_at,
                    started_at=self._jobs[job_id].started_at,
                    completed_at=datetime.now(UTC),
                    progress=1.0,
                    params=current.params,
                    error=str(exc),
                )

        future = self._executor.submit(_runner, record.job_id)
        self._futures[record.job_id] = future
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        """Retrieve the current state of a background job."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        limit: int = 50,
        job_type: str | None = None,
        status: JobStatus | None = None,
    ) -> list[JobRecord]:
        """List tracked jobs matching optional filter criteria."""
        jobs = list(self._jobs.values())
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a pending job."""
        future = self._futures.get(job_id)
        if future and future.cancel():
            current = self._jobs[job_id]
            self._jobs[job_id] = JobRecord(
                job_id=current.job_id,
                job_type=current.job_type,
                status=JobStatus.CANCELLED,
                created_at=current.created_at,
                completed_at=datetime.now(UTC),
                params=current.params,
                error="Cancelled by user",
            )
            return True
        return False

    def shutdown(self, wait: bool = False) -> None:
        """Gracefully shutdown background executor."""
        self._executor.shutdown(wait=wait)
