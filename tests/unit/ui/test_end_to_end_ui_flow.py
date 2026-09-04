"""Simulated complete end-to-end research-to-execution workflow via UI layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prooflab.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_complete_end_to_end_ui_workflow(client: TestClient) -> None:
    """Simulate complete lifecycle workflow exclusively through the UI presentation layer:

    1. Data Studio: Ingest & validate historical market data.
    2. Quant Lab: Define target/stop setup, select causal features, configure ensemble, and train.
    3. Proof Engine: Inspect metrics scorecard, walk-forward results, and proof status.
    4. Paper Trading / Live Dashboard: Monitor live probability, model consensus, and risk.
    5. Safeguards: Update risk limits and verify Kill Switch.
    6. Co-Pilot Pad: Submit trader-assisted order with pre-trade risk validation.
    7. Auto-Pilot Gate: Explicitly confirm and arm live execution mode.
    """
    # -------------------------------------------------------------------------
    # Step 1: Data Studio Ingestion & Validation
    # -------------------------------------------------------------------------
    data_res = client.post(
        "/api/v1/ui/data-studio/extract",
        json={
            "symbol": "EURUSD",
            "broker": "MetaQuotes-Demo",
            "timeframe": "H1",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "data_source": "MT5",
        },
    )
    assert data_res.status_code == 200
    data_summary = data_res.json()
    assert data_summary["health_status"] == "HEALTHY"
    assert data_summary["rows_retained"] >= 9000

    # -------------------------------------------------------------------------
    # Step 2: Quant Lab Setup Definition & Training
    # -------------------------------------------------------------------------
    train_res = client.post(
        "/api/v1/ui/quant-lab/train",
        json={
            "instrument": "EURUSD",
            "direction": "BOTH",
            "target_pips": 20.0,
            "stop_pips": 15.0,
            "horizon_bars": 12,
            "label_policy": "FIRST_TOUCH",
        },
    )
    assert train_res.status_code == 200
    train_progress = train_res.json()
    strategy_id = train_progress["strategy_id"]
    assert train_progress["current_stage"] == "COMPLETED"

    # -------------------------------------------------------------------------
    # Step 3: Proof Engine Verification
    # -------------------------------------------------------------------------
    proof_res = client.get(f"/api/v1/ui/proof-engine/{strategy_id}")
    assert proof_res.status_code == 200
    proof_data = proof_res.json()
    assert proof_data["proof_status"] == "ROBUST"
    assert proof_data["scorecard"]["sharpe_ratio"] >= 1.5
    assert proof_data["scorecard"]["profit_factor"] >= 1.5

    # -------------------------------------------------------------------------
    # Step 4: Live Dashboard & Telemetry Inspection
    # -------------------------------------------------------------------------
    dash_res = client.get("/api/v1/ui/live-dashboard")
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert dash_data["ai_direction"] in ("BUY", "SELL", "IGNORE")
    assert len(dash_data["model_votes"]) >= 2
    assert dash_data["risk_status"] == "NORMAL"

    # -------------------------------------------------------------------------
    # Step 5: Co-Pilot Manual Order Pad Execution
    # -------------------------------------------------------------------------
    order_res = client.post(
        "/api/v1/ui/copilot/submit-order",
        json={
            "symbol": "EURUSD",
            "direction": "BUY",
            "volume_lots": 0.2,
            "take_profit_price": 1.08600,
            "stop_loss_price": 1.08250,
            "explicit_confirmation": True,
        },
    )
    assert order_res.status_code == 200
    assert order_res.json()["status"] == "SUBMITTED"

    # -------------------------------------------------------------------------
    # Step 6: Live Auto-Pilot Confirmation Gate
    # -------------------------------------------------------------------------
    live_gate_res = client.post(
        "/api/v1/ui/autopilot/confirm-live",
        json={
            "strategy_id": strategy_id,
            "operator_name": "chief_risk_officer",
            "acknowledged_proof_status": "ROBUST",
            "paper_trading_confirmed_days": 14,
            "max_allowed_drawdown_pct": 10.0,
            "explicit_live_risk_acknowledgement": True,
        },
    )
    assert live_gate_res.status_code == 200
    assert live_gate_res.json()["status"] == "LIVE_ENABLED"

    # -------------------------------------------------------------------------
    # Step 7: Emergency Kill Switch Disarmament
    # -------------------------------------------------------------------------
    kill_res = client.post("/api/v1/ui/safeguards/kill-switch")
    assert kill_res.status_code == 200
    assert kill_res.json()["status"] == "KILL_SWITCH_ACTIVATED"
