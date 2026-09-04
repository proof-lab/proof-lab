"""Configurable XGBoost classifier and standalone tree baseline."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from prooflab.models.base import BaseModelWrapper


class XGBoostConfig(BaseModel):
    """Tree configuration, independent of validation and blind observations."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    max_depth: int = Field(default=3, gt=0, strict=True)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    n_estimators: int = Field(default=100, gt=0, strict=True)
    subsample: float = Field(default=1, gt=0, le=1)
    colsample_bytree: float = Field(default=1, gt=0, le=1)
    reg_alpha: float = Field(default=0, ge=0)
    reg_lambda: float = Field(default=1, ge=0)
    min_child_weight: float = Field(default=1, ge=0)
    class_weight: Literal["balanced"] | dict[int, float] | None = None
    random_state: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    n_jobs: int = Field(default=1, gt=0, strict=True)

    @field_validator("class_weight")
    @classmethod
    def validate_weights(
        cls, value: Literal["balanced"] | dict[int, float] | None,
    ) -> Literal["balanced"] | dict[int, float] | None:
        if isinstance(value, dict) and (
            not value or not set(value).issubset({-1, 0, 1})
            or any(not np.isfinite(weight) or weight <= 0 for weight in value.values())
        ):
            raise ValueError("Class weights require canonical keys and finite positive values.")
        return value


class XGBoostModel(BaseModelWrapper):
    """Map observed canonical classes to XGBoost's contiguous integer targets.

    No tuning or early stopping is performed here. Class weights are derived
    solely from training labels, and the identity preprocessor is persisted.
    """

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        super().__init__("xgboost")
        self.config = (config or XGBoostConfig()).model_copy(deep=True)
        self.pipeline: Pipeline | None = None
        self.training_class_weights_: dict[int, float] = {}

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        if len(self.classes_) < 2:
            raise ValueError("XGBoost requires at least two training classes.")
        encoded = np.searchsorted(self.classes_, labels)
        weights = compute_sample_weight(self.config.class_weight, labels)
        self.training_class_weights_ = {
            cls: float(weights[labels == cls][0]) for cls in self.classes_
        }
        params = self.config.model_dump(exclude={"version", "class_weight"})
        if len(self.classes_) == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", eval_metric="mlogloss",
                          num_class=len(self.classes_))
        self.pipeline = Pipeline([
            ("preprocessor", FunctionTransformer(validate=False)),
            ("model", XGBClassifier(**params, tree_method="hist")),
        ])
        self.pipeline.fit(features, encoded, model__sample_weight=weights)

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        encoded = np.asarray(self.pipeline.predict(features), dtype=int)
        return np.asarray(self.classes_, dtype=int)[encoded]

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return np.asarray(self.pipeline.predict_proba(features))

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump()
