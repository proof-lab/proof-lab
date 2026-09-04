"""Secure importer and validator for .plb strategy packages."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import joblib
from pydantic import BaseModel, ConfigDict, Field

from prooflab.models.artifacts import ArtifactManifest, ModelArtifact
from prooflab.packaging.manifest import PackageManifest
from prooflab.packaging.security import PackageSecurityValidator
from prooflab.packaging.strategy_config import StrategyPackageConfig


class ImportedPackage(BaseModel):
    """Represent an inspected and securely loaded .plb strategy package."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    manifest: PackageManifest
    strategy_config: StrategyPackageConfig
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    feature_schema: dict[str, Any] = Field(default_factory=dict)
    calibration_data: dict[str, Any] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    package_path: str
    raw_model_bin: bytes | None = None
    raw_model_manifest: bytes | None = None

    def load_model(self, trusted: bool = False) -> ModelArtifact:
        """Load the inner model artifact payload. Requires explicit trusted=True assertion."""
        if not trusted:
            raise PermissionError(
                "Loading model artifact requires an explicit trusted=True confirmation."
            )

        if self.raw_model_bin is None or self.raw_model_manifest is None:
            # Load from package path
            with ZipFile(self.package_path, "r") as archive:
                raw_bin = archive.read("models/model_artifact.bin")
                raw_manifest = archive.read("models/model_manifest.json")
        else:
            raw_bin = self.raw_model_bin
            raw_manifest = self.raw_model_manifest

        model_manifest = ArtifactManifest.model_validate_json(raw_manifest.decode("utf-8"))
        payload_buffer = io.BytesIO(raw_bin)
        loaded = joblib.load(payload_buffer)
        model = loaded["model"]

        if not getattr(model, "is_fitted", False):
            raise ValueError("Loaded model wrapper is not fitted.")

        return ModelArtifact(model=model, manifest=model_manifest)


class StrategyImporter:
    """Safely inspects and imports portable .plb strategy packages."""

    def __init__(self, validator: PackageSecurityValidator | None = None) -> None:
        self.validator = validator or PackageSecurityValidator()

    def inspect_package(self, package_path: Path | str) -> PackageManifest:
        """Inspect and validate package structure and manifest without executing model code."""
        path = Path(package_path)
        if not path.exists():
            raise FileNotFoundError(f"Package file not found: {path}")

        with ZipFile(path, "r") as archive:
            # 1. Structural security checks
            self.validator.validate_zip_structure(archive)

            # 2. Checksum verification
            self.validator.verify_checksums(archive)

            # 3. Read manifest safely
            manifest_raw = archive.read("manifest.json")
            return PackageManifest.from_json(manifest_raw.decode("utf-8"))

    def import_package(
        self,
        package_path: Path | str,
    ) -> ImportedPackage:
        """Import all strategy configurations, metrics, and schemas safely."""
        path = Path(package_path)
        if not path.exists():
            raise FileNotFoundError(f"Package file not found: {path}")

        with ZipFile(path, "r") as archive:
            # 1. Structural security validation
            self.validator.validate_zip_structure(archive)

            # 2. Checksum validation
            checksums = self.validator.verify_checksums(archive)

            # 3. Safe Parsing of Components
            manifest_raw = archive.read("manifest.json")
            manifest = PackageManifest.from_json(manifest_raw.decode("utf-8"))

            strategy_raw = archive.read("strategy/strategy.yaml")
            strategy_config = StrategyPackageConfig.from_yaml(strategy_raw.decode("utf-8"))

            metrics_raw = archive.read("validation/metrics.json")
            validation_metrics = json.loads(metrics_raw.decode("utf-8"))

            schema_raw = archive.read("features/schema.json")
            feature_schema = json.loads(schema_raw.decode("utf-8"))

            calib_raw = archive.read("calibration/calibration.json")
            calibration_data = json.loads(calib_raw.decode("utf-8"))

            model_bin = archive.read("models/model_artifact.bin")
            model_manifest_raw = archive.read("models/model_manifest.json")

        return ImportedPackage(
            manifest=manifest,
            strategy_config=strategy_config,
            validation_metrics=validation_metrics,
            feature_schema=feature_schema,
            calibration_data=calibration_data,
            checksums=checksums,
            package_path=str(path),
            raw_model_bin=model_bin,
            raw_model_manifest=model_manifest_raw,
        )
