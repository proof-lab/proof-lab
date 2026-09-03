"""Logistic regression baseline with training-fitted standardization."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prooflab.models.base import BaseModelWrapper


class LogisticRegressionConfig(BaseModel):
    """Versioned L2 logistic regression parameters; unknown settings are errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    c_param: float = Field(default=1.0, gt=0)
    class_weight: Literal["balanced"] | dict[int, float] | None = "balanced"
    max_iter: int = Field(default=1000, gt=0, strict=True)
    tol: float = Field(default=1e-4, gt=0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)

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


class LogisticRegressionBaseline(BaseModelWrapper):
    """L2 classifier; validation data never enters its estimator or scaler."""

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        super().__init__("logistic_regression")
        self.config = (config or LogisticRegressionConfig()).model_copy(deep=True)
        self.pipeline: Pipeline | None = None

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        if len(self.classes_) < 2:
            raise ValueError("Logistic regression requires at least two training classes.")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=self.config.c_param, class_weight=self.config.class_weight,
                max_iter=self.config.max_iter, tol=self.config.tol,
                random_state=self.config.random_state, solver="lbfgs",
            )),
        ])
        self.pipeline.fit(features, labels)

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return np.asarray(self.pipeline.predict(features))

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return np.asarray(self.pipeline.predict_proba(features))

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump()
