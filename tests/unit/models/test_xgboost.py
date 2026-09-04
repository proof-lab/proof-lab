"""Real XGBoost fits, canonical mapping, and training-only class weights."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from prooflab.models.xgboost import XGBoostConfig, XGBoostModel


@pytest.mark.parametrize("classes", [[-1, 1], [0, 1], [-1, 0, 1]])
def test_class_mapping_and_repeatability(classes):
    labels = np.repeat(classes, 12)
    features = pd.DataFrame({"signal": labels.astype(float), "position": np.arange(len(labels))})
    config = XGBoostConfig(n_estimators=8, max_depth=2, learning_rate=0.3)
    first = XGBoostModel(config).fit(features, labels)
    second = XGBoostModel(config).fit(features, labels, (features * 999, labels[::-1]))
    assert first.classes_ == classes
    np.testing.assert_array_equal(first.predict(features), labels)
    np.testing.assert_allclose(first.predict_proba(features), second.predict_proba(features))
    np.testing.assert_allclose(first.predict_proba(features).sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(first.predict(features.iloc[:, ::-1]), labels)


def test_weights_and_parameter_propagation():
    labels = np.array([-1] * 3 + [1] * 9)
    features = pd.DataFrame({"signal": labels})
    config = XGBoostConfig(n_estimators=2, class_weight="balanced", subsample=0.8,
                           colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=2)
    model = XGBoostModel(config).fit(features, labels)
    assert model.training_class_weights_ == {-1: 2, 1: 2 / 3}
    params = model.pipeline["model"].get_params()
    for key in ["subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "n_estimators"]:
        assert params[key] == getattr(config, key)
    assert model.get_params() == config.model_dump()
    weighted = XGBoostModel(XGBoostConfig(n_estimators=2, class_weight={-1: 3, 1: 1}))
    weighted.fit(features, labels)
    assert weighted.training_class_weights_ == {-1: 3, 1: 1}


@pytest.mark.parametrize("params", [
    {"max_depth": 0}, {"subsample": 0}, {"colsample_bytree": 2}, {"reg_lambda": -1},
    {"learning_rate": float("nan")}, {"n_estimators": 0}, {"class_weight": {2: 1}},
])
def test_invalid_config(params):
    with pytest.raises(ValueError):
        XGBoostConfig(**params)


def test_single_class_rejected():
    with pytest.raises(ValueError, match="two training classes"):
        XGBoostModel().fit(pd.DataFrame({"a": [1, 2]}), np.array([0, 0]))
