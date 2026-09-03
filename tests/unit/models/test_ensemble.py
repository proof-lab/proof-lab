"""Analytical combination checks with deterministic test-only model outputs."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import ModelArtifact, TrainingMetadata, save_artifact
from prooflab.models.base import BaseModelWrapper
from prooflab.models.ensemble import DirectionalEnsemble, EnsembleConfig


class FixedModel(BaseModelWrapper):
    def __init__(self, action, probability, vote):
        super().__init__("simple_rule")
        self.action, self.probability, self.vote = action, probability, vote

    def _fit_internal(self, features, labels, val_data=None):
        self.classes_ = sorted([0, self.action])

    def _predict_internal(self, features):
        return np.full(len(features), self.vote)

    def _predict_proba_internal(self, features):
        return np.tile([self.probability if cls == self.action else 1 - self.probability
                        for cls in self.classes_], (len(features), 1))

    def get_params(self):
        return {"action": self.action, "probability": self.probability, "vote": self.vote}


@pytest.fixture
def member_factory(tmp_path):
    counter = 0

    def make(action=1, probability=0.8, vote=None):
        nonlocal counter
        counter += 1
        x = pd.DataFrame({"signal": [0., 1.]})
        model = FixedModel(action, probability, action if vote is None else vote)
        model.fit(x, [0, action])
        manifest = save_artifact(model, tmp_path / f"member-{counter}.plmodel", training=
            TrainingMetadata(
                dataset_id="synthetic", dataset_checksum="a" * 64,
                setup_config={"direction": "LONG" if action == 1 else "SHORT",
                              "target_distance": 1, "stop_distance": 1, "horizon_bars": 2},
                train_start=datetime(2020, 1, 1, tzinfo=UTC),
                train_end=datetime(2020, 1, 2, tzinfo=UTC), train_rows=2,
                details={"training_partition": {
                    "last_complete_horizon": "2020-01-03T00:00:00Z"}},
            ), feature_metadata=[FeatureMetadata(feature_name="signal", family=FeatureFamily.PRICE,
                                                description="Test-only score")])
        return ModelArtifact(model, manifest)

    return make


def config(direction="LONG", **kwargs):
    return EnsembleConfig(direction=direction, blind_start="2020-02-01T00:00:00Z", **kwargs)


@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_hard_vote_tie_and_majority(member_factory, action, direction):
    members = {"a": member_factory(action), "b": member_factory(action, vote=0)}
    x = pd.DataFrame({"signal": [0., 2.]})
    tied = DirectionalEnsemble(members, config(direction)).evaluate(x)
    np.testing.assert_array_equal(tied.predictions, [0, 0])
    np.testing.assert_allclose(tied.probabilities, 0.5)
    members["c"] = member_factory(action)
    ensemble = DirectionalEnsemble(members, config(direction))
    batch = ensemble.evaluate(x)
    np.testing.assert_array_equal(batch.predictions, [action, action])
    np.testing.assert_allclose(batch.probabilities[:, ensemble.classes_.index(action)], 2 / 3)
    assert set(batch.model_votes) == set(members)
    assert ensemble.fit_details_["probability_semantics"] == "raw_vote_fraction"


def test_frozen_members_and_invalid_contracts(member_factory):
    artifact = member_factory()
    ensemble = DirectionalEnsemble({"a": artifact}, config())
    artifact.model.vote = 0
    assert ensemble.predict(pd.DataFrame({"signal": [1.]}))[0] == 1
    with pytest.raises(NotImplementedError):
        ensemble.fit(pd.DataFrame({"signal": [1.]}), [1])
    with pytest.raises(ValueError, match="named"):
        DirectionalEnsemble({}, config())
    with pytest.raises(ValueError, match="direction"):
        DirectionalEnsemble({"a": member_factory(-1)}, config())
    with pytest.raises(ValueError, match="complete-horizon"):
        artifact.manifest.training.details.clear()
        DirectionalEnsemble({"a": artifact}, config())


def test_member_validation_horizons_count_and_blind_boundary(member_factory):
    artifact = member_factory()
    training = artifact.manifest.training.model_copy(update={
        "validation_start": datetime(2020, 1, 4, tzinfo=UTC),
        "validation_end": datetime(2020, 1, 5, tzinfo=UTC), "validation_rows": 2,
    })
    training.details["validation_partition"] = {
        "last_complete_horizon": "2020-02-01T00:00:00Z"}
    artifact = ModelArtifact(artifact.model,
                             artifact.manifest.model_copy(update={"training": training}))
    with pytest.raises(ValueError, match="blind"):
        DirectionalEnsemble({"a": artifact}, config())


@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_probability_average_uses_class_mapping_not_votes(member_factory, action, direction):
    members = {"a": member_factory(action, 0.2), "b": member_factory(action, 0.4)}
    ensemble = DirectionalEnsemble(members, config(direction, method="probability_average"))
    batch = ensemble.evaluate(pd.DataFrame({"signal": [1.]}))
    assert batch.predictions[0] == 0
    assert all(vote[0] == action for vote in batch.model_votes.values())
    assert batch.probabilities[0, ensemble.classes_.index(action)] == pytest.approx(0.3)
    assert ensemble.fit_details_["probability_semantics"] == "uncalibrated_probability"


def test_missing_action_class_is_zero_padded(member_factory):
    member = member_factory()
    member.model.classes_ = [0]
    member.model.probability = 0
    member.model.vote = 0
    member = ModelArtifact(member.model, member.manifest.model_copy(update={"classes": [0]}))
    ensemble = DirectionalEnsemble({"a": member}, config(method="probability_average"))
    np.testing.assert_array_equal(ensemble.predict_proba(pd.DataFrame({"signal": [1.]})), [[1, 0]])


@pytest.mark.parametrize("action,direction", [(1, "LONG"), (-1, "SHORT")])
def test_weighted_average_named_weights_and_extreme_values(member_factory, action, direction):
    members = {"a": member_factory(action, 0.2), "b": member_factory(action, 0.8)}
    cfg = config(direction, method="weighted_average", weights={"b": 3e307, "a": 1e307})
    ensemble = DirectionalEnsemble(members, cfg)
    result = ensemble.evaluate(pd.DataFrame({"signal": [1.]}))
    assert result.probabilities[0, ensemble.classes_.index(action)] == pytest.approx(0.65)
    assert result.predictions[0] == action
    cfg.weights["a"] = 3e307
    assert ensemble.get_params()["weights"]["a"] == 1e307


@pytest.mark.parametrize("weights", [{}, {"a": 0}, {"a": -1}, {"a": float("nan")},
                                     {"a": float("inf")}])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        config(method="weighted_average", weights=weights)


def test_weight_configuration_and_zero_weight(member_factory):
    with pytest.raises(ValueError, match="only valid"):
        config(weights={"a": 1})
    with pytest.raises(ValueError, match="every member"):
        DirectionalEnsemble({"a": member_factory()}, config(method="weighted_average",
                                                           weights={"b": 1}))
    ensemble = DirectionalEnsemble({"a": member_factory(probability=0.1),
                                   "b": member_factory(probability=0.9)},
                                  config(method="weighted_average", weights={"a": 0, "b": 2}))
    assert ensemble.predict_proba(pd.DataFrame({"signal": [1.]}))[0, 1] == pytest.approx(0.9)
