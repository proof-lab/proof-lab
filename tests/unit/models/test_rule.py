"""Deterministic threshold actions, configured vocabulary, and artifact identity."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import TrainingMetadata, load_artifact, save_artifact
from prooflab.models.rule import SimpleRuleConfig, SimpleRuleStrategy


def config(**updates):
    return SimpleRuleConfig(**{
        "feature_col": "signal", "lower_threshold": -1, "upper_threshold": 1,
        "direction": "LONG", "mode": "mean_reversion", **updates,
    })


@pytest.mark.parametrize("direction,mode,expected", [
    ("LONG", "mean_reversion", [1, 1, 0, 0, 0]),
    ("LONG", "momentum", [0, 0, 0, 1, 1]),
    ("SHORT", "mean_reversion", [0, 0, 0, -1, -1]),
    ("SHORT", "momentum", [-1, -1, 0, 0, 0]),
])
def test_inclusive_thresholds_and_one_hot_actions(direction, mode, expected):
    x = pd.DataFrame({"signal": [-2., -1., 0., 1., 2.]})
    model = SimpleRuleStrategy(config(direction=direction, mode=mode)).fit(x, np.zeros(5))
    np.testing.assert_array_equal(model.predict(x), expected)
    one_hot = model.predict_proba(x)
    np.testing.assert_array_equal(one_hot.sum(axis=1), np.ones(5))
    np.testing.assert_array_equal(np.asarray(model.classes_)[one_hot.argmax(axis=1)], expected)
    assert model.fit_details_["thresholds_fitted"] is False


def test_labels_and_validation_do_not_learn_thresholds():
    x = pd.DataFrame({"signal": [-2., 0., 2.]})
    model = SimpleRuleStrategy(config())
    first = model.fit(x, np.array([0, 0, 0])).predict(x)
    second = model.fit(x, np.array([1, 1, 1]), (x * 10000, np.array([0, 0, 0]))).predict(x)
    np.testing.assert_array_equal(first, second)
    assert model.classes_ == [0, 1]
    assert model.get_params() == config().model_dump(mode="json")


@pytest.mark.parametrize("updates", [
    {"lower_threshold": 1}, {"upper_threshold": float("nan")}, {"lower_threshold": float("inf")},
    {"mode": "automatic"}, {"feature_col": ""}, {"direction": "BOTH"},
])
def test_invalid_configuration(updates):
    with pytest.raises(ValueError):
        config(**updates)


def test_missing_settings_features_and_wrong_direction():
    with pytest.raises(ValueError):
        SimpleRuleConfig()
    model = SimpleRuleStrategy(config())
    with pytest.raises(ValueError, match="missing"):
        model.fit(pd.DataFrame({"other": [1]}), np.array([0]))
    with pytest.raises(ValueError, match="direction"):
        model.fit(pd.DataFrame({"signal": [1]}), np.array([-1]))


def test_artifact_round_trip(tmp_path):
    x = pd.DataFrame({"signal": [-2., 0., 2.]})
    model = SimpleRuleStrategy(config()).fit(x, np.zeros(3))
    path = tmp_path / "rule.plmodel"
    save_artifact(model, path, training=TrainingMetadata(
        dataset_id="synthetic", dataset_checksum="a" * 64,
        setup_config={"direction": "LONG"}, train_rows=3,
        train_start=datetime(2020, 1, 1, tzinfo=UTC), train_end=datetime(2020, 1, 3, tzinfo=UTC),
    ), feature_metadata=[FeatureMetadata(feature_name="signal", family=FeatureFamily.PRICE,
                                         description="Synthetic rule signal")])
    artifact = load_artifact(path, trusted=True)
    np.testing.assert_array_equal(model.predict_proba(x), artifact.model.predict_proba(x))
    assert artifact.manifest.preprocessing == "identity"
    assert artifact.manifest.fit_details["probability_semantics"] == "deterministic_action_encoding"
