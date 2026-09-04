"""Unit tests for /api/data endpoints."""

import tempfile
from datetime import UTC, datetime

import pandas as pd
from fastapi.testclient import TestClient

from prooflab.api.app import app
from prooflab.api.dependencies import get_dataset_repository
from prooflab.data.repository import ParquetRepository
from prooflab.data.schema import Timeframe


def _make_dummy_repo(tmpdir: str) -> tuple[ParquetRepository, str]:
    repo = ParquetRepository(base_dir=tmpdir)
    n = 30
    dates = pd.date_range("2026-01-01", periods=n, freq="h", tz=UTC)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [1.1000 + i * 0.0001 for i in range(n)],
        "high": [1.1005 + i * 0.0001 for i in range(n)],
        "low": [1.0995 + i * 0.0001 for i in range(n)],
        "close": [1.1002 + i * 0.0001 for i in range(n)],
        "volume": [100.0] * n,
    })
    meta = repo.save_dataset(df, source="test", symbol="EURUSD", timeframe=Timeframe.H1)
    return repo, meta.dataset_id


def test_data_routes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        repo, dataset_id = _make_dummy_repo(tmpdir)
        app.dependency_overrides[get_dataset_repository] = lambda: repo

        client = TestClient(app)

        # 1. List datasets
        resp_list = client.get("/api/data/datasets")
        assert resp_list.status_code == 200
        data_list = resp_list.json()
        assert len(data_list) == 1
        assert data_list[0]["dataset_id"] == dataset_id

        # 2. Get specific dataset
        resp_get = client.get(f"/api/data/datasets/{dataset_id}")
        assert resp_get.status_code == 200
        assert resp_get.json()["symbol"] == "EURUSD"

        # 3. Non-existent dataset
        resp_404 = client.get("/api/data/datasets/unknown-id")
        assert resp_404.status_code == 404

        # 4. Get dataset health
        resp_health = client.get(f"/api/data/health/{dataset_id}")
        assert resp_health.status_code == 200
        assert resp_health.json()["is_valid"] is True

        # 5. Validate raw bars endpoint
        bars = [
            {
                "timestamp": datetime(2026, 1, 1, 0, 0, tzinfo=UTC).isoformat(),
                "open": 1.1000,
                "high": 1.1010,
                "low": 1.0990,
                "close": 1.1005,
                "volume": 100.0,
            }
        ]
        resp_val = client.post(
            "/api/data/validate",
            json={"symbol": "EURUSD", "timeframe": "H1", "bars": bars},
        )
        assert resp_val.status_code == 200
        assert resp_val.json()["is_valid"] is True

        # Clean overrides
        app.dependency_overrides.clear()
