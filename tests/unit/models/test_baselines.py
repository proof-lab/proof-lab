"""Behavioral checks for training-only baseline estimates."""

import numpy as np
import pandas as pd
import pytest

from prooflab.models.baselines import MajorityClassifier, RandomClassifier


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray]:
    return pd.DataFrame({"signal": range(8)}), np.array([-1, 0, 1, 1, 1, 1, 0, 1])


@pytest.mark.parametrize("model_type", [RandomClassifier, MajorityClassifier])
def test_empirical_priors_and_validation_isolation(model_type, data) -> None:
    features, labels = data
    first = model_type().fit(features, labels)
    second = model_type().fit(features, labels, (features * 100, np.full(8, -1)))
    expected = np.tile([1 / 8, 2 / 8, 5 / 8], (8, 1))
    np.testing.assert_allclose(first.predict_proba(features), expected)
    np.testing.assert_array_equal(first.predict(features), second.predict(features))
    np.testing.assert_array_equal(second.predict_proba(features), expected)


def test_majority_and_ties(data) -> None:
    features, labels = data
    model = MajorityClassifier().fit(features, labels)
    assert (model.predict(features) == 1).all()
    model.fit(features.iloc[:2], np.array([1, -1]))
    assert (model.predict(features) == -1).all()
    assert model.get_params() == {}


@pytest.mark.parametrize("model_type", [RandomClassifier, MajorityClassifier])
def test_single_class_and_refitting(model_type, data) -> None:
    features, labels = data
    model = model_type().fit(features, labels)
    model.fit(features, np.zeros(8, dtype=int))
    assert model.classes_ == [0]
    assert model.class_priors_ == {0: 1.0}
    assert (model.predict(features) == 0).all()
    np.testing.assert_array_equal(model.predict_proba(features), np.ones((8, 1)))


@pytest.mark.parametrize("strategy", ["prior", "uniform"])
def test_random_repeatability_and_sampling(strategy, data) -> None:
    features, labels = data
    model = RandomClassifier(strategy=strategy, random_state=19).fit(features, labels)
    many = pd.DataFrame({"signal": np.zeros(20_000)})
    first = model.predict(many)
    model.fit(features, labels)
    model.predict_proba(many)  # Probability queries must not advance the RNG.
    np.testing.assert_array_equal(model.predict(many), first)
    expected = [1 / 3] * 3 if strategy == "uniform" else [1 / 8, 2 / 8, 5 / 8]
    observed = [(first == cls).mean() for cls in model.classes_]
    np.testing.assert_allclose(observed, expected, atol=0.02)
    assert model.get_params() == {"strategy": strategy, "random_state": 19}


def test_invalid_random_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        RandomClassifier(strategy="unknown")
