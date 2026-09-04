"""Unit tests for /api/experiments model training endpoints."""

import tempfile
import time
from datetime import UTC

import pandas as pd
from fastapi.testclient import TestClient

from prooflab.api.app import app
from prooflab.api.dependencies import get_dataset_repository
from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe


def _make_training_repo(tmpdir: str) -> tuple[ParquetRepository, str]:
    repo = ParquetRepository(base_dir=tmpdir)
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="h", tz=UTC)
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["EURUSD"] * n,
        "timeframe": ["H1"] * n,
        "open": [1.1000 + (i % 10) * 0.0005 for i in range(n)],
        "high": [1.1020 + (i % 10) * 0.0005 for i in range(n)],
        "low": [1.0980 + (i % 10) * 0.0005 for i in range(n)],
        "close": [1.1005 + (i % 10) * 0.0005 for i in range(n)],
        "volume": [200.0] * n,
        "tick_volume": [200] * n,
        "spread": [0.0001] * n,
        "source": ["test"] * n,
    })
    meta = repo.save_dataset(df, source="test", symbol="EURUSD", timeframe=Timeframe.H1)
    return repo, meta.dataset_id


def test_start_training_experiment_and_poll_job() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        repo, dataset_id = _make_training_repo(tmpdir)
        app.dependency_overrides[get_dataset_repository] = lambda: repo

        client = TestClient(app)

        # 1. Submit training job
        payload = {
            "model_name": "majority_classifier",
            "dataset_id": dataset_id,
            "feature_preset": "PRICE_ONLY",
            "target_pips": 20.0,
            "stop_pips": 10.0,
            "horizon_bars": 5,
        }
        resp = client.post("/api/experiments/train", json=payload)
        assert resp.status_code == 202
        job_data = resp.json()
        job_id = job_data["job_id"]
        assert job_data["job_type"] == "MODEL_TRAINING"

        # 2. Poll job until completion
        completed = False
        for _ in range(40):
            time.sleep(0.1)
            resp_job = client.get(f"/api/system/jobs/{job_id}")
            assert resp_job.status_code == 200
            st = resp_job.json()["status"]
            if st == "COMPLETED":
                completed = True
                assert resp_job.json()["result"] is not None
                break
            elif st == "FAILED":
                break

        assert completed is True

        # Clean overrides
        app.dependency_overrides.clear()
