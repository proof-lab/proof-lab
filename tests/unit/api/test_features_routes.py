"""Unit tests for /api/features catalog and preset endpoints."""

from fastapi.testclient import TestClient

from prooflab.api.app import app


def test_features_endpoints() -> None:
    client = TestClient(app)

    # 1. Feature catalog list
    resp = client.get("/api/features")
    assert resp.status_code == 200
    features = resp.json()
    assert len(features) > 0
    assert any(f["feature_name"] == "return_1" for f in features)

    # 2. Feature presets list
    resp_presets = client.get("/api/features/presets")
    assert resp_presets.status_code == 200
    presets = resp_presets.json()
    assert len(presets) >= 4
    preset_names = [p["preset_name"] for p in presets]
    assert "PRICE_ONLY" in preset_names
    assert "ALL_STANDARD" in preset_names
