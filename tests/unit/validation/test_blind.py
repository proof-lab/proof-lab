"""Unit tests for prooflab.validation.blind (Blind Test Set Protection & Audit Gates)."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prooflab.models.baselines import MajorityClassifier
from prooflab.validation.blind import (
    BlindAccessViolationError,
    BlindEvaluationGate,
    BlindMultipleTestingWarning,
)


@pytest.fixture
def trained_model() -> MajorityClassifier:
    train_x = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0]})
    train_y = pd.Series([1, 1, 0, 1])
    model = MajorityClassifier()
    model.fit(train_x, train_y)
    return model


@pytest.fixture
def blind_data() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    features = pd.DataFrame({"feat_1": np.arange(20, dtype=float)}, index=idx)
    labels = pd.Series([1, 0] * 10, index=idx)
    return features, labels


def test_blind_evaluation_blocked_without_confirmation(
    trained_model: MajorityClassifier,
    blind_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, labels = blind_data
    gate = BlindEvaluationGate()

    with pytest.raises(BlindAccessViolationError, match="strictly protected"):
        gate.evaluate(
            model=trained_model,
            features=features,
            labels=labels,
            experiment_id="PL-2026-ABC123",
            dataset_id="ds-eurusd-h1",
            blind_start=datetime(2024, 1, 1, tzinfo=UTC),
            dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
            confirm_blind_evaluation=False,
            confirmation_reason="Formal final audit evaluation",
        )


def test_blind_evaluation_requires_meaningful_reason(
    trained_model: MajorityClassifier,
    blind_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, labels = blind_data
    gate = BlindEvaluationGate()

    with pytest.raises(ValueError, match="at least 10 characters"):
        gate.evaluate(
            model=trained_model,
            features=features,
            labels=labels,
            experiment_id="PL-2026-ABC123",
            dataset_id="ds-eurusd-h1",
            blind_start=datetime(2024, 1, 1, tzinfo=UTC),
            dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
            confirm_blind_evaluation=True,
            confirmation_reason="short",
        )


def test_blind_evaluation_rejects_out_of_bounds_data(
    trained_model: MajorityClassifier,
    blind_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, labels = blind_data
    gate = BlindEvaluationGate()

    # Data starts before blind_start
    with pytest.raises(ValueError, match="before blind start"):
        gate.evaluate(
            model=trained_model,
            features=features,
            labels=labels,
            experiment_id="PL-2026-ABC123",
            dataset_id="ds-eurusd-h1",
            blind_start=datetime(2024, 1, 10, tzinfo=UTC),  # later than features[0]
            dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
            confirm_blind_evaluation=True,
            confirmation_reason="Formal final audit evaluation",
        )


def test_blind_evaluation_success_and_ledger_logging(
    trained_model: MajorityClassifier,
    blind_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    features, labels = blind_data
    ledger_file = tmp_path / "blind_audits.jsonl"
    gate = BlindEvaluationGate(ledger_path=ledger_file, max_permitted_evaluations=1)

    result = gate.evaluate(
        model=trained_model,
        features=features,
        labels=labels,
        experiment_id="PL-2026-ABC123",
        dataset_id="ds-eurusd-h1",
        blind_start=datetime(2024, 1, 1, tzinfo=UTC),
        dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
        confirm_blind_evaluation=True,
        confirmation_reason="Final pre-deployment out-of-sample verification",
    )

    assert result["experiment_id"] == "PL-2026-ABC123"
    assert result["metrics"]["rows"] == 20
    assert result["metrics"]["accuracy"] == 0.5
    assert ledger_file.exists()

    history = gate.get_audit_history()
    assert len(history) == 1
    assert history[0].operator_confirmed is True


def test_repeated_blind_evaluation_triggers_warning(
    trained_model: MajorityClassifier,
    blind_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    features, labels = blind_data
    ledger_file = tmp_path / "blind_audits.jsonl"
    gate = BlindEvaluationGate(ledger_path=ledger_file, max_permitted_evaluations=1)

    # 1st evaluation -> Success, no warning
    gate.evaluate(
        model=trained_model,
        features=features,
        labels=labels,
        experiment_id="PL-2026-AAA111",
        dataset_id="ds-eurusd-h1",
        blind_start=datetime(2024, 1, 1, tzinfo=UTC),
        dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
        confirm_blind_evaluation=True,
        confirmation_reason="Initial formal blind evaluation run",
    )

    # 2nd evaluation on the same blind holdout -> Triggers warning!
    with pytest.warns(BlindMultipleTestingWarning, match="Repeated blind test evaluation alert"):
        gate.evaluate(
            model=trained_model,
            features=features,
            labels=labels,
            experiment_id="PL-2026-BBB222",
            dataset_id="ds-eurusd-h1",
            blind_start=datetime(2024, 1, 1, tzinfo=UTC),
            dataset_end=datetime(2024, 2, 1, tzinfo=UTC),
            confirm_blind_evaluation=True,
            confirmation_reason="Second requested blind evaluation run",
        )

    assert (
        gate.count_blind_evaluations(
            "ds-eurusd-h1",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )
        == 2
    )
