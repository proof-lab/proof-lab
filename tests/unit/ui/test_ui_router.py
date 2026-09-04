"""Integration unit tests for UI router endpoints, template serving, and safety gates."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prooflab.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_ui_serves_html_page_and_assets(client: TestClient) -> None:
    """Test root and /ui paths serve HTML and static CSS/JS files."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers["content-type"]
    assert "PROOF LAB" in res_root.text
    assert "Data Studio" in res_root.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Quant Laboratory" in res_ui.text

    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"]

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers["content-type"]


def test_data_studio_extraction_endpoint(client: TestClient) -> None:
    """Test Data Studio extract & validate history endpoint."""
    payload = {
        "symbol": "EURUSD",
        "broker": "MetaQuotes-Demo",
        "timeframe": "H1",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "data_source": "MT5",
    }
    res = client.post("/api/v1/ui/data-studio/extract", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EURUSD"
    assert data["health_status"] == "HEALTHY"
    assert data["rows_retained"] > 0
    assert data["completeness_pct"] > 90.0


def test_quant_lab_training_endpoint(client: TestClient) -> None:
    """Test Quant Lab train model endpoint."""
    payload = {
        "instrument": "EURUSD",
        "direction": "BOTH",
        "target_pips": 25.0,
        "stop_pips": 15.0,
        "horizon_bars": 12,
        "label_policy": "FIRST_TOUCH",
    }
    res = client.post("/api/v1/ui/quant-lab/train", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["current_stage"] == "COMPLETED"
    assert data["progress_pct"] == 100.0
    assert len(data["stage_logs"]) > 0


def test_proof_engine_view_endpoint(client: TestClient) -> None:
    """Test Proof Engine scorecard and warnings view endpoint."""
    res = client.get("/api/v1/ui/proof-engine/EURUSD_M15_CHAMPION")
    assert res.status_code == 200
    data = res.json()
    assert data["strategy_id"] == "EURUSD_M15_CHAMPION"
    assert data["proof_status"] == "ROBUST"
    assert data["scorecard"]["sharpe_ratio"] > 1.0
    assert data["scorecard"]["win_rate"] > 0.5


def test_live_dashboard_view_endpoint(client: TestClient) -> None:
    """Test Live Dashboard telemetry, AI signal, and model votes endpoint."""
    res = client.get("/api/v1/ui/live-dashboard")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EURUSD"
    assert data["ai_direction"] in ("BUY", "SELL", "IGNORE")
    assert len(data["model_votes"]) == 3
    assert data["account_equity"] > 0


def test_safeguards_kill_switch_endpoint(client: TestClient) -> None:
    """Test Emergency Kill Switch activation endpoint."""
    res = client.post("/api/v1/ui/safeguards/kill-switch")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "KILL_SWITCH_ACTIVATED"


def test_autopilot_live_confirmation_endpoint(client: TestClient) -> None:
    """Test Live Auto-Pilot explicit confirmation endpoint."""
    valid_payload = {
        "strategy_id": "EURUSD_PROD",
        "operator_name": "lead_trader",
        "acknowledged_proof_status": "ROBUST",
        "paper_trading_confirmed_days": 10,
        "max_allowed_drawdown_pct": 10.0,
        "explicit_live_risk_acknowledgement": True,
    }
    res_ok = client.post("/api/v1/ui/autopilot/confirm-live", json=valid_payload)
    assert res_ok.status_code == 200
    assert res_ok.json()["status"] == "LIVE_ENABLED"

    # Reject without risk acknowledgement
    invalid_payload = {**valid_payload, "explicit_live_risk_acknowledgement": False}
    with pytest.raises(PermissionError):
        client.post("/api/v1/ui/autopilot/confirm-live", json=invalid_payload)


def test_copilot_order_submission_endpoint(client: TestClient) -> None:
    """Test Co-Pilot manual order submission endpoint."""
    payload = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume_lots": 0.5,
        "take_profit_price": 1.08700,
        "stop_loss_price": 1.08300,
        "explicit_confirmation": True,
    }
    res = client.post("/api/v1/ui/copilot/submit-order", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "SUBMITTED"
    assert res.json()["volume_lots"] == 0.5
