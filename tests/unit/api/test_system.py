"""Unit tests for /api/system health, version, and background job queue endpoints."""

import time

from fastapi.testclient import TestClient

from prooflab.api.app import app
from prooflab.api.jobs import JobManager, JobStatus


def test_system_health_and_version() -> None:
    client = TestClient(app)

    # Health check
    resp_health = client.get("/api/system/health")
    assert resp_health.status_code == 200
    data = resp_health.json()
    assert data["status"] == "OK"
    assert data["version"] == "0.1.0"
    assert "timestamp_utc" in data

    # Version check
    resp_ver = client.get("/api/system/version")
    assert resp_ver.status_code == 200
    data_ver = resp_ver.json()
    assert data_ver["version"] == "0.1.0"
    assert data_ver["api_version"] == "v1"


def test_job_manager_lifecycle() -> None:
    mgr = JobManager(max_workers=2)

    def dummy_task(x: int) -> dict[str, int]:
        return {"squared": x * x}

    job = mgr.submit_job("TEST_JOB", dummy_task, params={"x": 5}, x=5)
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    # Wait for completion
    time.sleep(0.1)
    updated = mgr.get_job(job.job_id)
    assert updated is not None
    assert updated.status == JobStatus.COMPLETED
    assert updated.result == {"squared": 25}

    # List jobs
    all_jobs = mgr.list_jobs(job_type="TEST_JOB")
    assert len(all_jobs) >= 1
    assert all_jobs[0].job_id == job.job_id

    mgr.shutdown(wait=False)


def test_system_jobs_endpoints() -> None:
    client = TestClient(app)

    # List jobs
    resp = client.get("/api/system/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Non-existent job
    resp_404 = client.get("/api/system/jobs/non-existent-id")
    assert resp_404.status_code == 404
