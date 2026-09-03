"""End-to-end integration tests for exporting, inspecting, and importing .plb packages."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import ArtifactManifest, ModelArtifact, TrainingMetadata
from prooflab.models.baselines import MajorityClassifier
from prooflab.packaging.exporter import StrategyExporter
from prooflab.packaging.importer import StrategyImporter
from prooflab.packaging.strategy_config import StrategyPackageConfig


def _create_dummy_artifact() -> ModelArtifact:
    model = MajorityClassifier()
    # Fit model on synthetic data
    x_df = pd.DataFrame({
        "ret_1": [0.001, -0.002, 0.003, -0.001],
        "vol_10": [0.005, 0.006, 0.004, 0.005],
    })
    y_ser = pd.Series([1, 1, 1, 1])
    model.fit(x_df, y_ser)

    meta_a = FeatureMetadata(
        feature_name="ret_1",
        family=FeatureFamily.PRICE,
        description="1-period return",
    )
    meta_b = FeatureMetadata(
        feature_name="vol_10",
        family=FeatureFamily.VOLATILITY,
        description="10-period volatility",
    )

    training_meta = TrainingMetadata(
        dataset_id="eurusd-h1-v1",
        dataset_checksum="a" * 64,
        setup_config={"target_pips": 20.0, "stop_pips": 10.0, "horizon_bars": 5},
        train_start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        train_end=datetime(2026, 2, 1, 0, 0, tzinfo=UTC),
        train_rows=1000,
    )

    deps = {
        "python": "3.14",
        "prooflab": "0.1.0",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "joblib": "1.4.0",
        "pydantic": "2.10.0",
    }

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
        training=training_meta,
        dependencies=deps,
        payload_checksum="b" * 64,
    )

    return ModelArtifact(model=model, manifest=manifest)


def test_export_and_import_pipeline_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "demo_strategy.plb"

        cfg = StrategyPackageConfig(
            strategy_id="strat-alpha-h1",
            symbol="EURUSD",
            timeframe="H1",
            target_pips=25.0,
            stop_pips=15.0,
            horizon_bars=5,
            risk_per_trade_pct=0.01,
            min_confidence=0.55,
            parameters={"regime": "trend"},
        )

        artifact = _create_dummy_artifact()
        validation_metrics = {
            "sharpe_ratio": 1.85,
            "win_rate": 0.58,
            "brier_score": 0.21,
            "total_trades": 150,
        }

        # 1. Export package
        manifest = StrategyExporter.export_package(
            output_path=package_path,
            strategy_config=cfg,
            model_artifact=artifact,
            validation_metrics=validation_metrics,
            feature_metadata=artifact.manifest.feature_metadata,
            author="ProofLab Lead",
            description="Production H1 Alpha Strategy",
            embed_dataset=False,  # default
        )

        assert package_path.exists()
        assert manifest.strategy_id == "strat-alpha-h1"
        assert manifest.models == ["models/model_artifact.bin"]

        # Verify historical dataset is NOT embedded
        with ZipFile(package_path, "r") as archive:
            files = archive.namelist()
            assert "metadata/embedded_dataset.parquet" not in files
            assert "manifest.json" in files
            assert "strategy/strategy.yaml" in files
            assert "features/schema.json" in files
            assert "validation/metrics.json" in files
            assert "checksums/sha256.json" in files

        # 2. Inspect package safely
        importer = StrategyImporter()
        inspected_manifest = importer.inspect_package(package_path)
        assert inspected_manifest.strategy_id == "strat-alpha-h1"
        assert inspected_manifest.symbol == "EURUSD"

        # 3. Import package safely
        imported = importer.import_package(package_path)
        assert imported.strategy_config.strategy_id == "strat-alpha-h1"
        assert imported.validation_metrics["sharpe_ratio"] == 1.85
        assert imported.feature_schema["feature_order"] == ["ret_1", "vol_10"]

        # 4. Model loading requires explicit trusted=True
        with pytest.raises(PermissionError, match="requires an explicit trusted=True"):
            imported.load_model(trusted=False)

        loaded_artifact = imported.load_model(trusted=True)
        assert loaded_artifact.model.is_fitted is True
        assert loaded_artifact.manifest.model_name == "majority_classifier"

        # Verify inference on loaded model
        eval_df = pd.DataFrame({"ret_1": [0.001], "vol_10": [0.005]})
        preds = loaded_artifact.model.predict_proba(eval_df)
        assert preds.shape[0] == 1


def test_export_with_optional_embedded_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = Path(tmpdir) / "embedded_strategy.plb"

        cfg = StrategyPackageConfig(
            strategy_id="strat-with-data",
            symbol="EURUSD",
            timeframe="H1",
            target_pips=20.0,
            stop_pips=10.0,
            horizon_bars=5,
        )
        artifact = _create_dummy_artifact()

        StrategyExporter.export_package(
            output_path=package_path,
            strategy_config=cfg,
            model_artifact=artifact,
            validation_metrics={},
            feature_metadata=[],
            embed_dataset=True,
            dataset_content=b"dummy parquet bytes",
        )

        with ZipFile(package_path, "r") as archive:
            files = archive.namelist()
            assert "metadata/embedded_dataset.parquet" in files

        # Safe importer verifies checksum including embedded dataset
        importer = StrategyImporter()
        imported = importer.import_package(package_path)
        assert "metadata/embedded_dataset.parquet" in imported.checksums
