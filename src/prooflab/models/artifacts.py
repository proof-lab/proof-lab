"""Complete, immutable local model artifacts for trusted research workflows.

The JSON manifest can be inspected without executing a model. The joblib payload
is pickle-based: loading requires an explicit trusted=True assertion from the
caller. This is not a .plb strategy-package loader; .plb paths are rejected.
Checksums detect corruption, not malicious replacement or authenticity.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_STORED, ZipFile

import joblib
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from prooflab.features.base import FeatureMetadata
from prooflab.models.base import BaseModelWrapper


class TrainingMetadata(BaseModel):
    """Caller-supplied training provenance; split safety is enforced by training."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    dataset_id: str = Field(min_length=1)
    dataset_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_config: dict[str, Any]
    train_start: AwareDatetime
    train_end: AwareDatetime
    train_rows: int = Field(gt=0)
    validation_start: AwareDatetime | None = None
    validation_end: AwareDatetime | None = None
    validation_rows: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_intervals(self) -> TrainingMetadata:
        if self.train_end < self.train_start:
            raise ValueError("Training interval is reversed.")
        if self.validation_rows:
            if (
                self.validation_start is None or self.validation_end is None
                or not self.train_end < self.validation_start <= self.validation_end
            ):
                raise ValueError("Validation must follow training without overlap.")
        elif self.validation_start is not None or self.validation_end is not None:
            raise ValueError("Validation timestamps require validation rows.")
        return self


class ArtifactManifest(BaseModel):
    """Inspectable inference contract and reproducibility information."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal[1] = 1
    created_at: AwareDatetime
    model_name: str
    model_type: str
    model_params: dict[str, Any]
    feature_order: list[str]
    feature_schema: dict[str, str]
    feature_metadata: list[FeatureMetadata]
    classes: list[int]
    preprocessing: Literal["identity", "pipeline", "preprocessor"]
    training: TrainingMetadata
    training_history: list[dict[str, Any]] = Field(default_factory=list)
    fit_details: dict[str, Any] = Field(default_factory=dict)
    best_epoch: int | None = None
    stopped_epoch: int | None = None
    dependencies: dict[str, str]
    payload_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_schema(self) -> ArtifactManifest:
        if (
            not self.feature_order or len(set(self.feature_order)) != len(self.feature_order)
            or set(self.feature_schema) != set(self.feature_order)
            or [item.feature_name for item in self.feature_metadata] != self.feature_order
        ):
            raise ValueError("Artifact feature metadata, schema, and order must agree.")
        if not self.classes or self.classes != sorted(set(self.classes)):
            raise ValueError("Artifact classes must be unique and sorted.")
        if not set(self.classes).issubset({-1, 0, 1}):
            raise ValueError("Artifact classes must be canonical.")
        required = {"python", "prooflab", "numpy", "pandas", "joblib", "pydantic"}
        if not required.issubset(self.dependencies):
            raise ValueError("Artifact dependency metadata is incomplete.")
        return self


@dataclass(frozen=True)
class ModelArtifact:
    model: BaseModelWrapper
    manifest: ArtifactManifest


def _artifact_path(path: Path | str) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".plmodel":
        raise ValueError("Native model artifacts require .plmodel; .plb packages are unsupported.")
    return target


def _preprocessing(model: BaseModelWrapper) -> str:
    if getattr(model, "pipeline", None) is not None:
        return "pipeline"
    if getattr(model, "preprocessor", None) is not None:
        return "preprocessor"
    if model.model_name in {"random_classifier", "majority_classifier"}:
        return "identity"
    raise ValueError("Model does not declare a complete preprocessing pipeline.")


def _dependencies(model: BaseModelWrapper) -> dict[str, str]:
    packages = ["prooflab", "numpy", "pandas", "joblib", "pydantic"]
    if model.model_name in {"logistic_regression", "xgboost", "neural_network", "svm"}:
        packages.extend(["scikit-learn", "scipy"])
    if model.model_name == "xgboost":
        packages.append("xgboost")
    if model.model_name == "neural_network":
        packages.append("torch")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def save_artifact(
    model: BaseModelWrapper, path: Path | str, *, training: TrainingMetadata,
    feature_metadata: list[FeatureMetadata],
) -> ArtifactManifest:
    """Persist a fitted wrapper, including estimator weights and preprocessing.

    Serialization and schema validation complete before opening the destination.
    Exclusive creation prevents replacement of an existing artifact.
    """
    target = _artifact_path(path)
    if not model.is_fitted:
        raise ValueError("Cannot save an unfitted model.")
    preprocessing = _preprocessing(model)
    payload_buffer = io.BytesIO()
    joblib.dump({"model": model, "preprocessing": preprocessing}, payload_buffer)
    payload = payload_buffer.getvalue()
    manifest = ArtifactManifest.model_validate({
        "created_at": datetime.now(UTC), "model_name": model.model_name,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_params": model.get_params(), "feature_order": model.feature_names,
        "feature_schema": model.feature_schema, "feature_metadata": feature_metadata,
        "classes": model.classes_, "preprocessing": preprocessing, "training": training,
        "training_history": getattr(model, "history_", []),
        "fit_details": model.fit_details_,
        "best_epoch": getattr(model, "best_epoch_", None),
        "stopped_epoch": getattr(model, "stopped_epoch_", None),
        "dependencies": _dependencies(model),
        "payload_checksum": hashlib.sha256(payload).hexdigest(),
    })
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_STORED) as archive:
        archive.writestr("manifest.json", manifest.model_dump_json(indent=2))
        archive.writestr("model.joblib", payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(archive_buffer.getvalue())
    return manifest


def inspect_artifact(path: Path | str) -> ArtifactManifest:
    """Read and validate JSON without deserializing executable Python objects."""
    with ZipFile(_artifact_path(path)) as archive:
        if sorted(archive.namelist()) != ["manifest.json", "model.joblib"]:
            raise ValueError("Artifact must contain exactly a manifest and model payload.")
        return ArtifactManifest.model_validate_json(archive.read("manifest.json"))


def load_artifact(
    path: Path | str, *, trusted: bool = False,
    expected_feature_order: list[str] | None = None,
) -> ModelArtifact:
    """Load only a trusted native artifact under matching dependency versions."""
    target = _artifact_path(path)
    if not trusted:
        raise ValueError("Pickle-based artifacts require explicit trusted=True.")
    manifest = inspect_artifact(target)
    if expected_feature_order is not None and expected_feature_order != manifest.feature_order:
        raise ValueError("Artifact feature order does not match the requested schema.")
    for name, recorded in manifest.dependencies.items():
        current = platform.python_version() if name == "python" else version(name)
        if current != recorded:
            raise ValueError(f"Artifact dependency mismatch for {name}: {recorded} != {current}")
    with ZipFile(target) as archive:
        payload = archive.read("model.joblib")
    if hashlib.sha256(payload).hexdigest() != manifest.payload_checksum:
        raise ValueError("Artifact payload checksum mismatch.")
    content = joblib.load(io.BytesIO(payload))
    model = content.get("model") if isinstance(content, dict) else None
    if not isinstance(model, BaseModelWrapper) or not model.is_fitted:
        raise ValueError("Artifact payload is not a fitted model wrapper.")
    normalized_params = json.loads(json.dumps(model.get_params()))
    if (
        model.model_name != manifest.model_name
        or f"{type(model).__module__}.{type(model).__qualname__}" != manifest.model_type
        or model.feature_names != manifest.feature_order
        or model.feature_schema != manifest.feature_schema
        or model.classes_ != manifest.classes
        or normalized_params != manifest.model_params
        or _preprocessing(model) != manifest.preprocessing
        or content.get("preprocessing") != manifest.preprocessing
        or _dependencies(model) != manifest.dependencies
        or getattr(model, "history_", []) != manifest.training_history
        or model.fit_details_ != manifest.fit_details
        or getattr(model, "best_epoch_", None) != manifest.best_epoch
        or getattr(model, "stopped_epoch_", None) != manifest.stopped_epoch
    ):
        raise ValueError("Artifact manifest does not match its model payload.")
    return ModelArtifact(model=model, manifest=manifest)
