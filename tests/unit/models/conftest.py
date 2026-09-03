"""Analytical combination checks with deterministic test-only model outputs."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, FeatureMetadata
from prooflab.models.artifacts import ModelArtifact, TrainingMetadata, save_artifact
from prooflab.models.base import BaseModelWrapper


class FixedModel(BaseModelWrapper):
    def __init__(self, action, probability, vote):
        super().__init__("simple_rule")
        self.action, self.probability, self.vote = action, probability, vote

    def _fit_internal(self, features, labels, val_data=None):
        self.classes_ = sorted([0, self.action])

    def _predict_internal(self, features):
        if self.probability is None:
            return np.where(features.signal > 0.5, self.action, 0)
        return np.full(len(features), self.vote)

    def _predict_proba_internal(self, features):
        if self.probability is None:
            return np.column_stack([features.signal if cls == self.action else 1 - features.signal
                                    for cls in self.classes_])
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


