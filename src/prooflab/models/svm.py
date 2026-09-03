"""Directional SVM with a chronologically fitted native-style probability link.

The SVM's binary native mechanism uses a sigmoid link with smoothed targets.
Here that link sees a later, purged training holdout rather than shuffled folds.
It is the approved M04 SVM compatibility exception, not the M05 calibration API.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from prooflab.models.base import BaseModelWrapper


class SVMConfig(BaseModel):
    """Explicit training boundaries; all ends are exclusive."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    c_param: float = Field(default=1, gt=0)
    gamma: Literal["scale", "auto"] | float = "scale"
    degree: int = Field(default=3, ge=0, strict=True)
    coef0: float = 0
    class_weight: Literal["balanced"] | dict[int, float] | None = "balanced"
    probability: bool = True
    probability_start: AwareDatetime | None = None
    training_end: AwareDatetime
    tol: float = Field(default=1e-3, gt=0)

    @field_validator("gamma")
    @classmethod
    def positive_gamma(cls, value: str | float) -> str | float:
        if isinstance(value, float) and (not np.isfinite(value) or value <= 0):
            raise ValueError("Numeric gamma must be finite and positive.")
        return value

    @field_validator("class_weight")
    @classmethod
    def positive_weights(
        cls, value: Literal["balanced"] | dict[int, float] | None,
    ) -> Literal["balanced"] | dict[int, float] | None:
        if isinstance(value, dict) and (
            not value or not set(value).issubset({-1, 0, 1})
            or any(not np.isfinite(weight) or weight <= 0 for weight in value.values())
        ):
            raise ValueError("Class weights require canonical keys and finite positive values.")
        return value

    @model_validator(mode="after")
    def ordered_boundaries(self) -> SVMConfig:
        if self.probability and (
            self.probability_start is None or self.probability_start >= self.training_end
        ):
            raise ValueError("Probability fitting requires a start before training_end.")
        if not self.probability and self.probability_start is not None:
            raise ValueError("Disable probability_start when probability fitting is disabled.")
        return self


class SVMModel(BaseModelWrapper):
    """Binary setup model; probability rows never refit support vectors or scaling.

    Directional setup labels contain IGNORE and one action class. Pooling both
    setup directions into a three-class SVM is outside the approved M04 pipeline.
    """

    def __init__(self, config: SVMConfig) -> None:
        super().__init__("svm")
        self.config = config.model_copy(deep=True)
        self.pipeline: Pipeline | None = None
        self.probability_coefficients_: tuple[float, float] | None = None

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        self.pipeline = None
        self.probability_coefficients_ = None
        ends = self._horizon_end_times
        if ends is None:
            raise ValueError("SVM fitting requires full horizon_end_times.")
        if not (ends < self.config.training_end).all():
            raise ValueError("A training label horizon reaches training_end.")
        if len(self.classes_) != 2 or 0 not in self.classes_:
            raise ValueError("SVM requires IGNORE and one action class from a single direction.")
        if self.config.probability:
            boundary = self.config.probability_start
            assert boundary is not None
            fit_mask = np.asarray((features.index < boundary) & (ends < boundary))
            probability_mask = np.asarray(features.index >= boundary)
        else:
            fit_mask = np.ones(len(features), dtype=bool)
            probability_mask = np.zeros(len(features), dtype=bool)
        if sorted(np.unique(labels[fit_mask]).tolist()) != self.classes_:
            raise ValueError("Earlier SVM subpartition must contain both training classes.")
        if self.config.probability and (
            sorted(np.unique(labels[probability_mask]).tolist()) != self.classes_
        ):
            raise ValueError("Later probability subpartition must contain both training classes.")
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel=self.config.kernel, C=self.config.c_param, gamma=self.config.gamma,
                degree=self.config.degree, coef0=self.config.coef0,
                class_weight=self.config.class_weight, tol=self.config.tol,
            )),
        ])
        self.pipeline.fit(features.loc[fit_mask], labels[fit_mask])
        self.fit_details_ = {
            "svm_fit_start": features.index[fit_mask][0].isoformat(),
            "svm_fit_last_entry": features.index[fit_mask][-1].isoformat(),
            "svm_fit_last_horizon": ends.loc[fit_mask].max().isoformat(),
            "svm_fit_rows": int(fit_mask.sum()),
            "purged_rows": int((~fit_mask & ~probability_mask).sum()),
            "training_end": self.config.training_end.isoformat(),
            "probability_method": "disabled",
        }
        if self.config.probability:
            assert self.config.probability_start is not None
            scores = np.asarray(self.pipeline.decision_function(features.loc[probability_mask]))
            self._fit_native_probability(scores, labels[probability_mask])
            self.fit_details_.update({
                "probability_method": "svm_native_sigmoid_chronological_v1",
                "probability_start": self.config.probability_start.isoformat(),
                "probability_first_entry": features.index[probability_mask][0].isoformat(),
                "probability_last_horizon": ends.loc[probability_mask].max().isoformat(),
                "probability_rows": int(probability_mask.sum()),
                "m04_compatibility_exception": True,
            })

    def _fit_native_probability(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit only the binary native-style sigmoid objective on the later holdout.

        Fixed smoothed targets match the native binary SVM probability mechanism.
        No selectable calibration methods, CV, or general estimator adapter exist.
        """
        positive = labels == self.classes_[1]
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        targets = np.where(positive, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        scale = max(1.0, float(np.max(np.abs(scores))))
        design = np.column_stack([scores / scale, np.ones(len(scores))])

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ parameters
            loss = float(np.mean(np.logaddexp(0, logits) - targets * logits))
            gradient = design.T @ (expit(logits) - targets) / len(targets)
            return loss, np.asarray(gradient)

        result = minimize(
            objective, np.array([0.0, np.log((n_pos + 1) / (n_neg + 1))]),
            jac=True, method="L-BFGS-B", options={"gtol": 1e-8, "maxiter": 1000},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("SVM probability estimation did not converge.")
        self.probability_coefficients_ = (float(result.x[0] / scale), float(result.x[1]))

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return np.asarray(self.pipeline.predict(features))

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        if self.probability_coefficients_ is None:
            raise NotImplementedError("SVM probability estimation is disabled.")
        assert self.pipeline is not None
        slope, intercept = self.probability_coefficients_
        scores = np.asarray(self.pipeline.decision_function(features))
        positive = np.clip(expit(slope * scores + intercept), 1e-7, 1 - 1e-7)
        return np.column_stack([1 - positive, positive])

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")
