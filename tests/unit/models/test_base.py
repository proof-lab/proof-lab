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
