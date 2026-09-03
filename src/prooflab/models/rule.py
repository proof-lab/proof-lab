"""Explicit deterministic threshold baseline for one setup direction."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prooflab.labels.config import Direction
from prooflab.models.base import BaseModelWrapper


class SimpleRuleConfig(BaseModel):
    """All action-defining settings must be supplied, never learned from labels."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    feature_col: str = Field(min_length=1)
    lower_threshold: float
    upper_threshold: float
    direction: Direction
    mode: Literal["mean_reversion", "momentum"]

    @model_validator(mode="after")
    def ordered_thresholds(self) -> SimpleRuleConfig:
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("lower_threshold must be strictly below upper_threshold.")
        return self


class SimpleRuleStrategy(BaseModelWrapper):
    """Emit IGNORE or the configured direction's action using inclusive thresholds.

    The fixed class vocabulary comes from the rule configuration, not observed
    outcomes. predict_proba is one-hot action encoding, never a success estimate.
    """

    def __init__(self, config: SimpleRuleConfig) -> None:
        super().__init__("simple_rule")
        self.config = config.model_copy(deep=True)

    @property
    def action(self) -> int:
        return 1 if self.config.direction == Direction.LONG else -1

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        if self.config.feature_col not in features:
            raise ValueError(f"Required rule feature is missing: {self.config.feature_col}")
        if not np.isin(labels, [0, self.action]).all():
            raise ValueError("Rule labels must match the configured setup direction.")
        self.classes_ = sorted([0, self.action])
        self.fit_details_ = {
            "thresholds_fitted": False,
            "probability_semantics": "deterministic_action_encoding",
        }

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        values = features[self.config.feature_col].to_numpy()
        lower_action = (
            (self.config.direction == Direction.LONG and self.config.mode == "mean_reversion")
            or (self.config.direction == Direction.SHORT and self.config.mode == "momentum")
        )
        triggered = (values <= self.config.lower_threshold if lower_action
                     else values >= self.config.upper_threshold)
        return np.where(triggered, self.action, 0)

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self._predict_internal(features)
        return np.asarray(predictions[:, None] == np.asarray(self.classes_)[None, :], dtype=float)

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")
