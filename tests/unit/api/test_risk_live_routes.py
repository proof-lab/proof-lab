"""Unit tests for /api/risk and /api/live governance and emergency kill switch endpoints."""

from fastapi.testclient import TestClient

from prooflab.api.app import app
from prooflab.api.dependencies import get_lifecycle_manager
from prooflab.paper.lifecycle import StrategyLifecycleManager, StrategyLifecycleState


def test_risk_and_live_endpoints() -> None:
    client = TestClient(app)
    headers = {"X-API-Key": "prooflab-dev-key"}
    admin_headers = {"X-Admin-Key": "prooflab-admin-key"}

    # 1. Get risk limits
    resp_limits = client.get("/api/risk/limits", headers=headers)
    assert resp_limits.status_code == 200
    limits_data = resp_limits.json()
    assert "max_daily_loss_pct" in limits_data
    assert limits_data["kill_switch_active"] is False

    # 2. Evaluate signal
    signal_payload = {
        "symbol": "EURUSD",
        "side": "BUY",
        "entry_price": 1.1000,
        "stop_loss_price": 1.0950,
        "confidence": 0.70,
    }
    resp_sig = client.post("/api/risk/evaluate-signal", json=signal_payload, headers=headers)
    assert resp_sig.status_code == 200
    sig_data = resp_sig.json()
    assert sig_data["is_approved"] is True
    assert sig_data["approved_units"] > 0

    # 3. Emergency Kill Switch activation
    resp_ks = client.post(
        "/api/risk/kill-switch/activate",
        json={"reason": "Flash Crash Emergency", "actor": "CRO"},
        headers=admin_headers,
    )
    assert resp_ks.status_code == 200
    assert resp_ks.json()["status"] == "HALTED"
    assert resp_ks.json()["is_active"] is True

    # Check signal rejected when kill switch active
    resp_sig_halted = client.post("/api/risk/evaluate-signal", json=signal_payload, headers=headers)
    assert resp_sig_halted.status_code == 200
    assert resp_sig_halted.json()["is_approved"] is False

    # 4. Emergency Kill Switch reset
    resp_reset = client.post(
        "/api/risk/kill-switch/reset",
        json={"reason": "Normal Market Restored", "actor": "CRO"},
        headers=admin_headers,
    )
    assert resp_reset.status_code == 200
    assert resp_reset.json()["status"] == "NORMAL"

    # 5. Live status
    resp_live = client.get("/api/live/status", headers=headers)
    assert resp_live.status_code == 200
    assert resp_live.json()["live_trading_enabled"] is False

    # 6. Enable live mode with admin key (when strategy is in APPROVED state)
    lifecycle = StrategyLifecycleManager(
        strategy_id="strat-alpha",
        initial_state=StrategyLifecycleState.APPROVED,
    )
    app.dependency_overrides[get_lifecycle_manager] = lambda: lifecycle

    resp_enable = client.post(
        "/api/live/enable",
        json={
            "strategy_id": "strat-alpha",
            "confirm": True,
            "authorized_by": "HeadOfTrading",
            "reason": "Passed validation and proof engine gates",
        },
        headers=admin_headers,
    )
    assert resp_enable.status_code == 200
    assert resp_enable.json()["status"] == "LIVE_ENABLED"

    app.dependency_overrides.clear()
