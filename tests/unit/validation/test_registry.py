"""Unit tests for prooflab.validation.registry (Experiment Registry & Provenance Tracking)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from prooflab.validation.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    MultipleTestingWarning,
    collect_environment_metadata,
    generate_experiment_id,
)


@pytest.fixture
def base_record_data() -> dict:
    env = collect_environment_metadata()
    return {
        "experiment_id": "PL-2026-ABC123",
        "dataset_id": "ds-eurusd-h1-v1",
        "dataset_checksum": "a" * 64,
        "feature_version": "1.0.0",
        "label_version": "1.0.0",
        "software_version": "0.1.0",
        "git_commit": env["git_commit"],
        "python_version": env["python_version"],
        "library_versions": env["library_versions"],
        "random_seed": 42,
        "created_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "configuration": {"model": "xgboost", "max_depth": 4},
        "results": {"val_brier_score": 0.18, "val_accuracy": 0.58},
        "artifact_locations": {"model": "models/PL-2026-ABC123.plmodel"},
        "validation_start": datetime(2023, 1, 1, 0, 0, tzinfo=UTC),
        "validation_end": datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        "blind_start": datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        "dataset_end": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "timeline_checksum": "b" * 64,
    }


def test_generate_experiment_id() -> None:
    eid = generate_experiment_id()
    assert eid.startswith(f"PL-{datetime.now(UTC).year}-")
    assert len(eid) == 14

    custom_year_dt = datetime(2025, 5, 10, tzinfo=UTC)
    eid_2025 = generate_experiment_id(created_at=custom_year_dt)
    assert eid_2025.startswith("PL-2025-")

    # Entropy determinism
    eid_entropy1 = generate_experiment_id(custom_year_dt, entropy="SEED123")
    eid_entropy2 = generate_experiment_id(custom_year_dt, entropy="SEED123")
    assert eid_entropy1 == eid_entropy2


def test_experiment_record_creation_and_validation(base_record_data: dict) -> None:
    record = ExperimentRecord.model_validate(base_record_data)
    assert record.experiment_id == "PL-2026-ABC123"
    assert record.software_version == "0.1.0"
    assert "numpy" in record.library_versions

    # Rejection of invalid ID
    bad_id = {**base_record_data, "experiment_id": "INVALID-ID"}
    with pytest.raises(Exception, match="pattern"):
        ExperimentRecord.model_validate(bad_id)

    # Rejection of reversed chronology
    bad_chrono = {
        **base_record_data,
        "validation_start": datetime(2025, 1, 1, tzinfo=UTC),
        "validation_end": datetime(2023, 1, 1, tzinfo=UTC),
    }
    with pytest.raises(ValueError, match="Validation start must precede validation end"):
        ExperimentRecord.model_validate(bad_chrono)

    # Rejection of validation extending past blind start
    bad_blind = {
        **base_record_data,
        "validation_end": datetime(2025, 1, 1, tzinfo=UTC),
        "blind_start": datetime(2024, 1, 1, tzinfo=UTC),
    }
    with pytest.raises(ValueError, match="cannot extend past blind start"):
        ExperimentRecord.model_validate(bad_blind)


def test_experiment_record_save_load_roundtrip(base_record_data: dict, tmp_path: Path) -> None:
    record = ExperimentRecord.model_validate(base_record_data)
    target_file = tmp_path / "PL-2026-ABC123.json"
    record.save(target_file)
    assert target_file.exists()

    loaded = ExperimentRecord.load(target_file)
    assert loaded == record
    assert loaded.to_json() == record.to_json()


def test_registry_registration_and_query(base_record_data: dict, tmp_path: Path) -> None:
    registry = ExperimentRegistry(storage_dir=tmp_path, warning_threshold=3)
    record1 = ExperimentRecord.model_validate(base_record_data)

    reg_rec, warning = registry.register_experiment(record1)
    assert reg_rec == record1
    assert warning is None
    assert (tmp_path / "PL-2026-ABC123.json").exists()

    # Query
    assert registry.get_experiment("PL-2026-ABC123") == record1
    assert len(registry.list_experiments()) == 1
    assert len(registry.list_experiments(dataset_id="ds-eurusd-h1-v1")) == 1
    assert len(registry.list_experiments(dataset_id="other-ds")) == 0

    # Duplicate registration rejected
    with pytest.raises(ValueError, match="already registered"):
        registry.register_experiment(record1)


def test_registry_multiple_testing_warning(base_record_data: dict, tmp_path: Path) -> None:
    registry = ExperimentRegistry(storage_dir=tmp_path, warning_threshold=2)

    # Register 1st experiment on validation window [2023, 2024]
    rec1 = ExperimentRecord.model_validate({**base_record_data, "experiment_id": "PL-2026-AAA111"})
    _, warn1 = registry.register_experiment(rec1)
    assert warn1 is None

    # Register 2nd experiment on the exact same validation window -> Triggers warning!
    rec2 = ExperimentRecord.model_validate({**base_record_data, "experiment_id": "PL-2026-BBB222"})
    with pytest.warns(MultipleTestingWarning, match="Multiple testing alert"):
        _, warn2 = registry.register_experiment(rec2)
    assert warn2 is not None
    assert "Multiple testing alert: 2 experiments" in warn2

    # Verify count for window
    count = registry.count_experiments_for_window(
        validation_start=base_record_data["validation_start"],
        validation_end=base_record_data["validation_end"],
    )
    assert count == 2
