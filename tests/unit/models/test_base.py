"""Unit tests for prooflab.models.base (Common Model Interface)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from prooflab.models.base import BaseModelWrapper, ModelNotFittedError


class DummyModel(BaseModelWrapper):
    """Dummy model implementation for testing base interface contracts."""

    def __init__(self, constant_pred: int = 1) -> None:
        super().__init__(model_name="dummy_model")
        self.constant_pred = constant_pred

    def _fit_internal(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        pass

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.constant_pred)

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        n_classes = len(self.classes_)
        prob = 1.0 / n_classes
        return np.full((len(features), n_classes), prob)

    def get_params(self) -> dict[str, Any]:
        return {"constant_pred": self.constant_pred}


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0, 4.0],
            "feat_2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    labels = pd.Series([1, -1, 1, 0])
    return features, labels


def test_not_fitted_error(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, _ = sample_data
    model = DummyModel()
    assert model.is_fitted is False

    with pytest.raises(ModelNotFittedError, match="is not fitted yet"):
        model.predict(features)

    with pytest.raises(ModelNotFittedError, match="is not fitted yet"):
        model.predict_proba(features)


def test_fit_and_predict(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = sample_data
    model = DummyModel(constant_pred=1)
    model.fit(features, labels)

    assert model.is_fitted is True
    assert model.feature_names == ["feat_1", "feat_2"]
    assert model.classes_ == [-1, 0, 1]

    preds = model.predict(features)
    assert len(preds) == len(features)
    assert (preds == 1).all()

    proba = model.predict_proba(features)
    assert proba.shape == (4, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_feature_alignment_and_missing_column(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, labels = sample_data
    model = DummyModel()
    model.fit(features, labels)

    # Inverted columns should be aligned without error
    features_inverted = pd.DataFrame(
        {
            "feat_2": [10.0, 20.0],
            "feat_1": [1.0, 2.0],
        }
    )
    preds = model.predict(features_inverted)
    assert len(preds) == 2

    # Missing column raises ValueError
    features_missing = pd.DataFrame({"feat_1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing required feature columns"):
        model.predict(features_missing)


def test_validation_input_errors() -> None:
    model = DummyModel()
    # Empty features
    with pytest.raises(ValueError, match="cannot be empty"):
        model.fit(pd.DataFrame(), np.array([]))

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        model.fit(pd.DataFrame({"a": [1, 2]}), np.array([1]))


@pytest.mark.parametrize("labels", [np.array(1), [[1], [0]], [1, 2], [1, np.nan],
                                   [True, False], ["1", "0"]])
def test_invalid_labels(labels: Any) -> None:
    with pytest.raises(ValueError, match="Labels"):
        DummyModel().fit(pd.DataFrame({"a": [1, 2]}), labels)


@pytest.mark.parametrize("features", [
    pd.DataFrame([[1, 2]], columns=["a", "a"]),
    pd.DataFrame({"a": [float("inf")]}),
    pd.DataFrame({"a": [float("nan")]}),
    pd.DataFrame({"a": ["text"]}),
    pd.DataFrame({"a": [1, 2]}, index=[0, 0]),
    pd.DataFrame({1: [1]}),
])
def test_invalid_feature_matrices(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        DummyModel().fit(features, np.ones(len(features)))


def test_series_alignment_and_failed_refit(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = sample_data
    model = DummyModel().fit(features, labels)
    with pytest.raises(ValueError, match="indices"):
        model.fit(features, labels.iloc[::-1])
    with pytest.raises(ModelNotFittedError):
        model.predict(features)


def test_validation_schema(sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
    features, labels = sample_data
    with pytest.raises(ValueError, match="Missing required"):
        DummyModel().fit(features, labels, (features[["feat_1"]], labels))
    with pytest.raises(ValueError, match="Unexpected"):
        DummyModel().fit(features, labels).predict(features.assign(extra=0))
    DummyModel().fit(features, labels, (features.iloc[:, ::-1], labels))


@pytest.mark.parametrize("probabilities", [
    np.ones((1, 3)) / 3,
    np.ones((4, 3)),
    np.full((4, 3), np.nan),
    np.tile([-0.1, 0.5, 0.6], (4, 1)),
])
def test_invalid_probability_outputs(
    sample_data: tuple[pd.DataFrame, pd.Series], probabilities: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = sample_data
    model = DummyModel().fit(features, labels)
    monkeypatch.setattr(model, "_predict_proba_internal", lambda _: probabilities)
    with pytest.raises(ValueError):
        model.predict_proba(features)


def test_internal_fit_failure_and_invalid_predictions(
    sample_data: tuple[pd.DataFrame, pd.Series], monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = sample_data
    model = DummyModel(99).fit(features, labels)
    with pytest.raises(ValueError, match="known class"):
        model.predict(features)

    def fail(*args: Any) -> None:
        raise RuntimeError("Training failed")

    monkeypatch.setattr(model, "_fit_internal", fail)
    with pytest.raises(RuntimeError):
        model.fit(features, labels)
    assert not model.is_fitted
