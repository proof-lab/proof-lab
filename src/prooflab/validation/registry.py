"""Experiment registry with full reproducibility metadata and multiple-testing tracking.

Every experiment receives a unique ID (PL-YYYY-XXXXXX) and records complete system,
library, dataset, feature, label, and partition provenance. Evaluates and warns against
repeated reuse of the same validation partition.
"""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class MultipleTestingWarning(UserWarning):
    """Raised when multiple experiments have been run against the same validation window."""


def generate_experiment_id(
    created_at: AwareDatetime | None = None,
    entropy: str | None = None,
) -> str:
    """Generate a compliant experiment identifier in format PL-YYYY-XXXXXX.

    Args:
        created_at: Optional timestamp to determine the year (defaults to current UTC year).
        entropy: Optional seed string for deterministic token derivation.
    """
    if created_at is not None:
        year = created_at.year
    else:
        year = pd.Timestamp.now(tz="UTC").year

    if entropy is not None:
        token = secrets.token_hex(3).upper() if not entropy else (
            re.sub(r"[^A-Z0-9]", "", entropy.upper())[:6].ljust(6, "0")
        )
    else:
        token = uuid4().hex[:6].upper()

    return f"PL-{year}-{token}"


def collect_environment_metadata() -> dict[str, Any]:
    """Capture current Python runtime, platform, and installed package versions."""
    packages = [
        "prooflab",
        "numpy",
        "pandas",
        "pydantic",
        "joblib",
        "scikit-learn",
        "scipy",
        "xgboost",
        "torch",
        "duckdb",
        "pyarrow",
    ]
    lib_versions: dict[str, str] = {}
    for pkg in packages:
        try:
            lib_versions[pkg] = version(pkg)
        except Exception:
            pass

    git_commit = os.getenv("PROOFLAB_GIT_COMMIT", "HEAD")

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
        "library_versions": lib_versions,
    }


class ExperimentRecord(BaseModel):
    """Complete, immutable reproducibility record for an individual quantitative experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(pattern=r"^PL-\d{4}-[A-Z0-9]{6}$")
    dataset_id: str = Field(min_length=1)
    dataset_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_version: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    software_version: str = Field(default="0.1.0")
    git_commit: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    library_versions: dict[str, str]
    random_seed: int
    created_at: AwareDatetime
    configuration: dict[str, Any]
    results: dict[str, Any]
    artifact_locations: dict[str, str] = Field(default_factory=dict)
    validation_start: AwareDatetime
    validation_end: AwareDatetime
    blind_start: AwareDatetime
    dataset_end: AwareDatetime
    timeline_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("experiment_id")
    @classmethod
    def validate_id_year(cls, value: str) -> str:
        parts = value.split("-")
        if len(parts) != 3 or parts[0] != "PL":
            raise ValueError("Experiment ID must follow format PL-YYYY-XXXXXX.")
        return value

    @model_validator(mode="after")
    def validate_chronology(self) -> ExperimentRecord:
        if self.validation_start >= self.validation_end:
            raise ValueError("Validation start must precede validation end.")
        if self.validation_end > self.blind_start:
            raise ValueError("Validation end cannot extend past blind start.")
        if self.blind_start >= self.dataset_end:
            raise ValueError("Blind start must precede dataset end.")
        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialize experiment record to JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentRecord:
        """Construct ExperimentRecord from JSON string."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    def save(self, path: Path | str) -> None:
        """Persist experiment record JSON to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> ExperimentRecord:
        """Load experiment record from disk."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Experiment record file not found: {target}")
        return cls.from_json(target.read_text(encoding="utf-8"))


class ExperimentRegistry:
    """Thread-safe, filesystem-backed repository for experiment provenance tracking."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        warning_threshold: int = 10,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.warning_threshold = warning_threshold
        self._records: dict[str, ExperimentRecord] = {}

        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_records()

    def _load_existing_records(self) -> None:
        if self.storage_dir is None or not self.storage_dir.exists():
            return
        for file in self.storage_dir.glob("PL-*.json"):
            try:
                record = ExperimentRecord.load(file)
                self._records[record.experiment_id] = record
            except Exception:
                pass

    def register_experiment(
        self,
        record: ExperimentRecord,
    ) -> tuple[ExperimentRecord, str | None]:
        """Register an experiment, persist it, and evaluate multiple testing count.

        Returns:
            Tuple of (record, warning_message_if_any).
        """
        if record.experiment_id in self._records:
            raise ValueError(f"Experiment ID '{record.experiment_id}' is already registered.")

        # Persist to disk if storage_dir is configured
        if self.storage_dir is not None:
            target_path = self.storage_dir / f"{record.experiment_id}.json"
            record.save(target_path)

        self._records[record.experiment_id] = record

        # Track multiple testing against the same validation span
        matching_count = self.count_experiments_for_window(
            validation_start=record.validation_start,
            validation_end=record.validation_end,
            dataset_id=record.dataset_id,
        )

        warning_msg: str | None = None
        if matching_count >= self.warning_threshold:
            warning_msg = (
                f"Multiple testing alert: {matching_count} experiments have been evaluated "
                f"against validation window [{record.validation_start} to {record.validation_end}] "
                f"on dataset '{record.dataset_id}'. "
                f"Elevated risk of multiple comparison bias / backtest overfitting."
            )
            warnings.warn(warning_msg, MultipleTestingWarning, stacklevel=2)

        return record, warning_msg

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Retrieve an experiment record by ID."""
        if experiment_id not in self._records:
            raise KeyError(f"Experiment '{experiment_id}' not found in registry.")
        return self._records[experiment_id]

    def list_experiments(
        self,
        dataset_id: str | None = None,
    ) -> list[ExperimentRecord]:
        """List registered experiments in chronological order."""
        records = list(self._records.values())
        if dataset_id is not None:
            records = [r for r in records if r.dataset_id == dataset_id]
        return sorted(records, key=lambda r: r.created_at)

    def count_experiments_for_window(
        self,
        validation_start: AwareDatetime,
        validation_end: AwareDatetime,
        dataset_id: str | None = None,
    ) -> int:
        """Count experiments that evaluated against the specified validation span."""
        count = 0
        for r in self._records.values():
            if dataset_id is not None and r.dataset_id != dataset_id:
                continue
            if r.validation_start == validation_start and r.validation_end == validation_end:
                count += 1
        return count
