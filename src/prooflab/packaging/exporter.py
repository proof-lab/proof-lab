"""Exporter for bundling quantitative trading strategies into portable .plb packages."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import joblib

from prooflab.models.artifacts import ModelArtifact
from prooflab.packaging.manifest import CompatibilityDeclaration, PackageManifest
from prooflab.packaging.strategy_config import StrategyPackageConfig


class StrategyExporter:
    """Packages strategy rules, trained models, schemas, and metrics into a .plb container."""

    @staticmethod
    def export_package(
        output_path: Path | str,
        strategy_config: StrategyPackageConfig,
        model_artifact: ModelArtifact,
        validation_metrics: dict[str, Any],
        feature_metadata: list[dict[str, Any]] | list[Any],
        calibration_data: dict[str, Any] | None = None,
        author: str = "",
        description: str = "",
        embed_dataset: bool = False,
        dataset_content: bytes | None = None,
        app_version: str = "0.1.0",
    ) -> PackageManifest:
        """Assemble all strategy components, compute checksums, and write the .plb package."""
        target = Path(output_path)
        if not target.name.endswith(".plb"):
            target = target.with_suffix(".plb")

        if not model_artifact.model.is_fitted:
            raise ValueError("Cannot package an unfitted model artifact.")

        # Ensure directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        files_to_pack: dict[str, bytes] = {}

        # 1. Strategy Configuration YAML
        files_to_pack["strategy/strategy.yaml"] = strategy_config.to_yaml().encode("utf-8")

        # 2. Features Schema JSON
        schema_dict = {
            "feature_order": model_artifact.manifest.feature_order,
            "feature_schema": model_artifact.manifest.feature_schema,
            "feature_metadata": [
                m if isinstance(m, dict) else (
                    m.model_dump(mode="json") if hasattr(m, "model_dump") else str(m)
                )
                for m in feature_metadata
            ],
        }
        files_to_pack["features/schema.json"] = json.dumps(schema_dict, indent=2).encode("utf-8")

        # 3. Model Artifact Payload
        model_payload_buffer = io.BytesIO()
        preprocessing = getattr(model_artifact.model, "preprocessor", "identity")
        joblib.dump(
            {"model": model_artifact.model, "preprocessing": preprocessing},
            model_payload_buffer,
        )
        files_to_pack["models/model_artifact.bin"] = model_payload_buffer.getvalue()
        manifest_json = model_artifact.manifest.model_dump_json(indent=2).encode("utf-8")
        files_to_pack["models/model_manifest.json"] = manifest_json

        # 4. Calibration Data JSON
        calib_dict = calibration_data or {
            "is_calibrated": getattr(model_artifact.model, "is_calibrated", False),
            "calibrator_type": getattr(model_artifact.model, "calibrator_type", None),
        }
        calib_bytes = json.dumps(calib_dict, indent=2).encode("utf-8")
        files_to_pack["calibration/calibration.json"] = calib_bytes

        # 5. Validation Metrics JSON
        val_bytes = json.dumps(validation_metrics, indent=2).encode("utf-8")
        files_to_pack["validation/metrics.json"] = val_bytes

        # 6. Optional Embedded Dataset (Default False!)
        if embed_dataset and dataset_content is not None:
            files_to_pack["metadata/embedded_dataset.parquet"] = dataset_content

        # 7. Build Canonical Package Manifest
        compat_decl = CompatibilityDeclaration(
            symbol=strategy_config.symbol,
            timeframe=strategy_config.timeframe,
            feature_names=model_artifact.manifest.feature_order,
            feature_parameters=strategy_config.parameters,
            min_app_version=app_version,
            target_pips=strategy_config.target_pips,
            stop_pips=strategy_config.stop_pips,
            horizon_bars=strategy_config.horizon_bars,
            extra_parameters={},
        )

        dataset_meta = {}
        if model_artifact.manifest.training:
            t = model_artifact.manifest.training
            dataset_meta = {
                "dataset_id": t.dataset_id,
                "dataset_checksum": t.dataset_checksum,
                "train_start": t.train_start.isoformat(),
                "train_end": t.train_end.isoformat(),
                "train_rows": t.train_rows,
            }

        manifest = PackageManifest(
            format_version="1.0.0",
            strategy_id=strategy_config.strategy_id,
            symbol=strategy_config.symbol,
            timeframe=strategy_config.timeframe,
            feature_version="1.0.0",
            model_version="1.0.0",
            app_version=app_version,
            min_app_version=app_version,
            compatibility=compat_decl,
            models=["models/model_artifact.bin"],
            description=description,
            author=author,
            dataset_metadata=dataset_meta,
        )
        files_to_pack["manifest.json"] = manifest.to_json(indent=2).encode("utf-8")

        # 8. Compute SHA-256 Checksums for all package components
        checksums: dict[str, str] = {}
        for rel_path, content in files_to_pack.items():
            checksums[rel_path] = hashlib.sha256(content).hexdigest()

        files_to_pack["checksums/sha256.json"] = json.dumps(
            checksums, indent=2
        ).encode("utf-8")

        # 9. Pack into ZIP container
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
            for rel_path, content in files_to_pack.items():
                archive.writestr(rel_path, content)

        return manifest
