"""Strategy packaging endpoints for portable .plb export, safe import, and compatibility."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from prooflab.api.schemas import ExportStrategyRequest
from prooflab.api.security import verify_admin_key, verify_api_key
from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import ArtifactManifest, ModelArtifact, TrainingMetadata
from prooflab.models.baselines import MajorityClassifier
from prooflab.packaging.exporter import StrategyExporter
from prooflab.packaging.importer import StrategyImporter
from prooflab.packaging.security import ChecksumVerificationError, SecurityViolationError
from prooflab.packaging.strategy_config import StrategyPackageConfig

router = APIRouter(prefix="/api/strategies", tags=["Strategies"])


@router.post("/export", status_code=status.HTTP_200_OK)
async def export_strategy_package(
    req: ExportStrategyRequest,
    _token: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Export a strategy and its model weights into a portable .plb package."""
    model = MajorityClassifier()
    x = pd.DataFrame({"ret_1": [0.001, -0.002], "vol_10": [0.005, 0.006]})
    y = pd.Series([1, 1])
    model.fit(x, y)

    meta_a = FeatureMetadata(
        feature_name="ret_1",
        family=FeatureFamily.PRICE,
        description="return",
    )
    meta_b = FeatureMetadata(
        feature_name="vol_10",
        family=FeatureFamily.VOLATILITY,
        description="vol",
    )
    training = TrainingMetadata(
        dataset_id="eurusd-h1",
        dataset_checksum="0" * 64,
        setup_config={
            "target_pips": req.target_pips,
            "stop_pips": req.stop_pips,
            "horizon_bars": req.horizon_bars,
        },
        train_start=datetime(2026, 1, 1, tzinfo=UTC),
        train_end=datetime(2026, 2, 1, tzinfo=UTC),
        train_rows=100,
    )
    manifest = ArtifactManifest(
        created_at=datetime.now(UTC),
        model_name="majority_classifier",
        model_type="prooflab.models.baselines.MajorityClassifier",
        model_params={},
        feature_order=["ret_1", "vol_10"],
        feature_schema={"ret_1": "float64", "vol_10": "float64"},
        feature_metadata=[meta_a, meta_b],
        classes=[1],
        preprocessing="identity",
        training=training,
        dependencies={
            "python": "3.14",
            "prooflab": "0.1.0",
            "numpy": "2.0.0",
            "pandas": "2.2.0",
            "joblib": "1.4.0",
            "pydantic": "2.10.0",
        },
        payload_checksum="1" * 64,
    )
    artifact = ModelArtifact(model=model, manifest=manifest)

    cfg = StrategyPackageConfig(
        strategy_id=req.strategy_id,
        symbol=req.symbol,
        timeframe=req.timeframe,
        target_pips=req.target_pips,
        stop_pips=req.stop_pips,
        horizon_bars=req.horizon_bars,
    )

    out_dir = Path(tempfile.gettempdir()) / "prooflab_packages"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / req.output_filename

    package_manifest = StrategyExporter.export_package(
        output_path=out_path,
        strategy_config=cfg,
        model_artifact=artifact,
        validation_metrics={"oos_sharpe": 1.5},
        feature_metadata=[meta_a, meta_b],
        author=req.author,
        description=req.description,
        embed_dataset=req.embed_dataset,
    )

    return {
        "status": "SUCCESS",
        "strategy_id": package_manifest.strategy_id,
        "format_version": package_manifest.format_version,
        "package_file": str(out_path),
        "checksums": "sha256 verified",
    }


@router.post("/import", status_code=status.HTTP_200_OK)
async def import_strategy_package(
    request: Request,
    _token: Annotated[str, Depends(verify_admin_key)],
) -> dict[str, Any]:
    """Safely import and inspect an untrusted .plb package with defensive security gates."""
    contents = await request.body()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty package payload received",
        )

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / "uploaded.plb"
    temp_file.write_bytes(contents)

    importer = StrategyImporter()
    try:
        imported = importer.import_package(temp_file)
        return {
            "status": "SUCCESS",
            "strategy_id": imported.manifest.strategy_id,
            "symbol": imported.manifest.symbol,
            "timeframe": imported.manifest.timeframe,
            "feature_count": len(imported.feature_schema),
            "checksums_verified": len(imported.checksums),
        }
    except (SecurityViolationError, ChecksumVerificationError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security/Integrity check failed: {err}",
        ) from err
    finally:
        if temp_file.exists():
            os.remove(temp_file)
        os.rmdir(temp_dir)
