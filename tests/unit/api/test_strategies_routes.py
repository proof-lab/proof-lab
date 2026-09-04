"""Unit tests for /api/strategies export and secure import endpoints."""

import io
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from prooflab.api.app import app


def test_strategy_export_and_import_endpoints() -> None:
    client = TestClient(app)
    headers = {"X-API-Key": "prooflab-dev-key"}
    admin_headers = {
        "X-Admin-Key": "prooflab-admin-key",
        "Content-Type": "application/octet-stream",
    }

    # 1. Export strategy package
    export_payload = {
        "strategy_id": "strat-export-demo",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "target_pips": 20.0,
        "stop_pips": 10.0,
        "horizon_bars": 5,
        "output_filename": "demo_export.plb",
    }
    resp_exp = client.post("/api/strategies/export", json=export_payload, headers=headers)
    assert resp_exp.status_code == 200
    export_data = resp_exp.json()
    assert export_data["status"] == "SUCCESS"
    package_path = export_data["package_file"]
    assert Path(package_path).exists()

    # 2. Import valid strategy package
    with open(package_path, "rb") as f:
        file_bytes = f.read()

    resp_imp = client.post("/api/strategies/import", content=file_bytes, headers=admin_headers)
    assert resp_imp.status_code == 200
    imp_data = resp_imp.json()
    assert imp_data["status"] == "SUCCESS"
    assert imp_data["strategy_id"] == "strat-export-demo"

    # 3. Import unauthorized
    resp_unauth = client.post("/api/strategies/import", content=file_bytes)
    assert resp_unauth.status_code == 403

    # 4. Import malicious zip (contains evil.py)
    bad_buf = io.BytesIO()
    with ZipFile(bad_buf, "w") as z:
        z.writestr("manifest.json", b"{}")
        z.writestr("checksums/sha256.json", b"{}")
        z.writestr("models/evil.py", b"import os; os.system('echo hacked')")
    bad_buf.seek(0)

    resp_bad = client.post(
        "/api/strategies/import",
        content=bad_buf.getvalue(),
        headers=admin_headers,
    )
    assert resp_bad.status_code == 400
    assert "Security/Integrity check failed" in resp_bad.json()["detail"]
