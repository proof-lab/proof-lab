"""Unit tests for /api/backtests simulation and proof validation endpoints."""

import tempfile
import time
from datetime import UTC

import pandas as pd
from fastapi.testclient import TestClient

from prooflab.api.app import app
from prooflab.api.dependencies import get_dataset_repository
from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe


def test_backtest_and_proof_endpoints() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = ParquetRepository(base_dir=tmpdir)
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="h", tz=UTC)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [1.1000] * n,
            "high": [1.1010] * n,
            "low": [1.0990] * n,
            "close": [1.1005] * n,
            "volume": [100.0] * n,
        })
        meta = repo.save_dataset(df, source="test", symbol="EURUSD", timeframe=Timeframe.H1)
        app.dependency_overrides[get_dataset_repository] = lambda: repo

        client = TestClient(app)

        # 1. Start backtest simulation
        resp_bt = client.post(
            "/api/backtests/run",
            json={
                "strategy_id": "strat-alpha",
                "dataset_id": meta.dataset_id,
                "initial_capital": 50000.0,
            },
        )
        assert resp_bt.status_code == 202
        job_id = resp_bt.json()["job_id"]

        time.sleep(0.1)
        resp_job = client.get(f"/api/system/jobs/{job_id}")
        assert resp_job.status_code == 200

        # 2. Evaluate proof scorecard
        resp_proof = client.post(
            "/api/backtests/proof",
            json={
                "strategy_id": "strat-alpha",
                "dataset_id": meta.dataset_id,
            },
        )
        assert resp_proof.status_code == 202
        proof_job_id = resp_proof.json()["job_id"]

        time.sleep(0.1)
        resp_proof_job = client.get(f"/api/system/jobs/{proof_job_id}")
        assert resp_proof_job.status_code == 200

        app.dependency_overrides.clear()
