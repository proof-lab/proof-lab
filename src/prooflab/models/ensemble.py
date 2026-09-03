"""Direction-specific combinations of frozen, already fitted M04 artifacts.

Hard-vote fractions are raw scores, not calibrated confidence. A single batch
captures votes once, including for stochastic members, so outputs stay coherent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict

from prooflab.labels.config import Direction, SetupConfig
from prooflab.models.artifacts import ModelArtifact
from prooflab.models.base import BaseModelWrapper


class EnsembleConfig(BaseModel):
    """Fixed combination configuration; no weight or method search."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    direction: Direction
    method: Literal["hard_vote", "probability_average"] = "hard_vote"
    blind_start: AwareDatetime


@dataclass(frozen=True)
class EnsembleBatch:
    """Columns follow ensemble.classes_; votes remain separate from probabilities."""

    predictions: np.ndarray
    probabilities: np.ndarray
    model_votes: dict[str, np.ndarray]


class DirectionalEnsemble(BaseModelWrapper):
    """Snapshot fitted members; never refit them when combining their outputs."""

    def __init__(self, members: dict[str, ModelArtifact], config: EnsembleConfig) -> None:
        super().__init__("directional_ensemble")
        if not members or any(not name.strip() for name in members):
            raise ValueError("An ensemble requires named members.")
        self.config = config.model_copy(deep=True)
        self._members = deepcopy(members)
        self.classes_ = sorted([0, self.action])
        first = next(iter(self._members.values()))
        self.setup = SetupConfig.model_validate(first.manifest.training.setup_config)
        self.feature_names = list(first.model.feature_names)
        self.feature_schema = dict(first.model.feature_schema)
        for artifact in self._members.values():
            model, manifest = artifact.model, artifact.manifest
            setup = SetupConfig.model_validate(manifest.training.setup_config)
            if setup != self.setup or setup.direction != config.direction:
                raise ValueError("Members must share the configured direction and setup.")
            if (
                not model.is_fitted or not model.classes_
                or not set(model.classes_).issubset(self.classes_)
                or model.classes_ != manifest.classes
            ):
                raise ValueError("Members must be fitted directional classifiers.")
            if (
                model.feature_names != self.feature_names
                or model.feature_schema != self.feature_schema
                or manifest.feature_order != self.feature_names
                or manifest.feature_schema != self.feature_schema
                or manifest.feature_metadata != first.manifest.feature_metadata
            ):
                raise ValueError("Members must share the feature schema and feature definitions.")
        self.information_end = max(self._information_end(a) for a in self._members.values())
        if self.information_end >= config.blind_start:
            raise ValueError("Member information reaches the blind period.")
        self.fit_details_ = {
            "direction": config.direction.value,
            "information_end": self.information_end.isoformat(),
            "blind_start": config.blind_start.isoformat(),
            "members": {name: a.manifest.model_dump(mode="json")
                        for name, a in self._members.items()},
            "probability_semantics": ("raw_vote_fraction" if config.method == "hard_vote"
                                      else "uncalibrated_probability"),
        }
        self.is_fitted = True

    @property
    def action(self) -> int:
        return 1 if self.config.direction == Direction.LONG else -1

    def _information_end(self, artifact: ModelArtifact) -> pd.Timestamp:
        training = artifact.manifest.training
        details = training.details
        ends = []
        for key, last_entry in [("training_partition", training.train_end),
                                ("validation_partition", training.validation_end)]:
            if last_entry is None:
                continue
            try:
                end = pd.Timestamp(details[key]["last_complete_horizon"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Member artifacts require complete-horizon provenance.") from exc
            if pd.isna(end) or end.tzinfo is None or end <= last_entry:
                raise ValueError("Invalid member complete-horizon provenance.")
            ends.append(end)
        original_blind = details.get("training_config", {}).get("blind_start")
        if original_blind is not None and pd.Timestamp(original_blind) != self.config.blind_start:
            raise ValueError("Cannot change a member's recorded blind boundary.")
        return max(ends)

    def fit(
        self, features: pd.DataFrame, labels: pd.Series | np.ndarray,
        val_data: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        *, horizon_end_times: pd.Series | None = None,
    ) -> Self:
        raise NotImplementedError("Construct ensembles from fitted artifacts; members are frozen.")

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        raise NotImplementedError("Ensembles do not refit members.")

    def evaluate(self, features: pd.DataFrame) -> EnsembleBatch:
        aligned = self._validate_features(features)
        votes = {name: artifact.model.predict(aligned)
                 for name, artifact in self._members.items()}
        action_score = np.mean([vote == self.action for vote in votes.values()], axis=0)
        if self.config.method == "probability_average":
            member_scores = []
            for artifact in self._members.values():
                model = artifact.model
                proba = model.predict_proba(aligned)
                member_scores.append(proba[:, model.classes_.index(self.action)]
                                     if self.action in model.classes_ else np.zeros(len(aligned)))
            action_score = np.mean(member_scores, axis=0)
        probabilities = np.column_stack([
            action_score if cls == self.action else 1 - action_score for cls in self.classes_
        ])
        # An exact tie always abstains, regardless of direction or member order.
        predictions = np.where(action_score > 0.5, self.action, 0)
        return EnsembleBatch(predictions, probabilities, votes)

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        return self.evaluate(features).predictions

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        return self.evaluate(features).probabilities

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")
