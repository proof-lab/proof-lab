"""Scaled logistic baseline behavior and training isolation."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from prooflab.models.logistic import LogisticRegressionBaseline, LogisticRegressionConfig


@pytest.fixture
def data():
    x = pd.DataFrame({"signal": [-3., -2., -1., 1., 2., 3.], "constant": [5.] * 6})
    return x, np.array([-1, -1, -1, 1, 1, 1])


def test_fit_scaling_and_inference(data):
    x, y = data
    first = LogisticRegressionBaseline().fit(x, y)
    second = LogisticRegressionBaseline().fit(x, y, (x * 10000, -y))
    assert first.classes_ == [-1, 1]
    np.testing.assert_array_equal(first.predict(x), y)
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))
    np.testing.assert_allclose(first.pipeline["scaler"].mean_, x.mean().to_numpy())
    np.testing.assert_allclose(first.predict_proba(x), first.predict_proba(x.iloc[:, ::-1]))
    assert first.predict_proba(x).shape == (6, 2)


def test_multiclass_and_parameters(data):
    x, _ = data
    config = LogisticRegressionConfig(c_param=2, max_iter=50, class_weight={-1: 2, 0: 1, 1: 3})
    model = LogisticRegressionBaseline(config).fit(x, np.array([-1, -1, 0, 0, 1, 1]))
    assert model.classes_ == [-1, 0, 1]
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
    assert model.pipeline["model"].C == 2
    assert model.pipeline["model"].class_weight == config.class_weight
    assert model.get_params() == config.model_dump()


@pytest.mark.parametrize("params", [
    {"c_param": 0}, {"c_param": float("inf")}, {"max_iter": 0}, {"unknown": 1},
    {"class_weight": {7: 1}}, {"class_weight": {1: -1}}, {"class_weight": {1: float("nan")}},
])
def test_invalid_config(params):
    with pytest.raises(ValueError):
        LogisticRegressionConfig(**params)


def test_single_class_rejected(data):
    x, _ = data
    model = LogisticRegressionBaseline()
    with pytest.raises(ValueError, match="two training classes"):
        model.fit(x, np.zeros(len(x)))
    assert not model.is_fitted
