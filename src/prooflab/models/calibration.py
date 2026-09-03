"""Formal M05 calibration on a later, purged, pre-blind holdout.

The base ensemble is frozen. Its member training AND early-stopping horizons
must finish before calibration begins. The blind period is never a fit input.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator
from scipy.optimize import minimize
from scipy.special import expit

from prooflab.models.base import BaseModelWrapper
from prooflab.models.ensemble import DirectionalEnsemble, EnsembleBatch
from prooflab.validation.calibration import probability_quality


class CalibrationConfig(BaseModel):
    """Explicit exclusive end; method selected before evaluating held-out quality."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    method: Literal["platt"] = "platt"
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def ordered_window(self) -> CalibrationConfig:
        if self.start >= self.end:
            raise ValueError("Calibration start must precede end.")
        return self


class _PlattLink:
    """Sigmoid of raw action probability (or vote fraction), with Platt targets.

    This is a two-parameter, unregularized binary sigmoid fit. Smoothed targets
    avoid infinite coefficients on separable data; no member is trained here.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        positives = int(labels.sum())
        negatives = len(labels) - positives
        target = np.where(labels == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))
        design = np.column_stack([scores, np.ones(len(scores))])

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            logits = design @ coefficients
            loss = float(np.mean(np.logaddexp(0, logits) - target * logits))
            gradient = np.asarray(design.T @ (expit(logits) - target) / len(target))
            return loss, gradient

        result = minimize(objective, [0., np.log((positives + 1) / (negatives + 1))],
                          jac=True, method="L-BFGS-B", options={"gtol": 1e-9, "maxiter": 1000})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("Platt calibration did not converge.")
        self.coefficients = np.asarray(result.x)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(expit(self.coefficients[0] * scores + self.coefficients[1]))


class CalibratedEnsemble(BaseModelWrapper):
    """Fit a formal probability link without changing the frozen ensemble.

    Hard voting retains its majority decision even if calibration changes the
    most probable class. Soft combinations decide from calibrated probabilities.
    """

    def __init__(self, ensemble: DirectionalEnsemble, config: CalibrationConfig) -> None:
        super().__init__("calibrated_ensemble")
        if not ensemble.is_fitted:
            raise ValueError("Calibration requires a fitted ensemble.")
        if config.start <= ensemble.information_end:
            raise ValueError("Calibration must follow all member training and validation horizons.")
        if config.end > ensemble.config.blind_start:
            raise ValueError("Calibration window reaches the blind period.")
        self._ensemble = deepcopy(ensemble)
        self.config = config.model_copy(deep=True)
        self._link: _PlattLink | None = None

    @property
    def action(self) -> int:
        return self._ensemble.action

    @property
    def information_end(self) -> pd.Timestamp:
        if not self.is_fitted:
            raise ValueError("Calibration has not been fitted.")
        return pd.Timestamp(self.fit_details_["last_complete_horizon"])

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        self._link = None
        if val_data is not None:
            raise ValueError("Supply only calibration data; evaluation is a separate operation.")
        if self._horizon_end_times is None:
            raise ValueError("Calibration requires full horizon_end_times.")
        if (
            (features.index < self.config.start).any()
            or (self._horizon_end_times >= self.config.end).any()
        ):
            raise ValueError("Calibration samples and full horizons must stay within its window.")
        if set(labels) != {0, self.action}:
            raise ValueError("Calibration requires both classes from the configured direction.")
        if self.feature_schema != self._ensemble.feature_schema:
            raise ValueError("Calibration features must match the member feature schema.")
        self.feature_names = list(self._ensemble.feature_names)
        self.classes_ = list(self._ensemble.classes_)
        batch = self._ensemble.evaluate(features)
        scores = batch.probabilities[:, self.classes_.index(self.action)]
        self._link = _PlattLink(scores, (labels == self.action).astype(int))
        self.fit_details_ = {
            "method": self.config.method, "calibration_framework": "m05_v1",
            "score_transform": "raw_action_probability_or_vote_fraction",
            "calibration_start": self.config.start.isoformat(),
            "calibration_end": self.config.end.isoformat(),
            "first_entry": features.index[0].isoformat(),
            "last_entry": features.index[-1].isoformat(),
            "last_complete_horizon": self._horizon_end_times.max().isoformat(),
            "rows": len(features), "blind_accessed": False,
            "ensemble": deepcopy(self._ensemble.fit_details_),
            "probability_semantics": "calibrated_probability",
        }

    def evaluate(self, features: pd.DataFrame) -> EnsembleBatch:
        aligned = self._validate_features(features)
        raw = self._ensemble.evaluate(aligned)
        return self._calibrate_batch(raw)

    def _calibrate_batch(self, raw: EnsembleBatch) -> EnsembleBatch:
        assert self._link is not None
        scores = raw.probabilities[:, self.classes_.index(self.action)]
        action_probability = self._link.predict(scores)
        probabilities = np.column_stack([
            action_probability if cls == self.action else 1 - action_probability
            for cls in self.classes_
        ])
        predictions = (raw.predictions if self._ensemble.config.method == "hard_vote" else
                       np.where(action_probability > 0.5, self.action, 0))
        return EnsembleBatch(predictions, probabilities, raw.model_votes)

    def evaluate_quality(
        self, features: pd.DataFrame, labels: pd.Series,
        *, horizon_end_times: pd.Series, n_bins: int = 10,
    ) -> dict[str, Any]:
        """Compare raw/calibrated quality strictly after calibration, before blind.

        This report measures quality; it never chooses a method or changes a fit.
        Both reports use the same raw batch, even for stochastic voters.
        """
        aligned = self._validate_features(features)
        _, y = self._validate_inputs(aligned, labels)
        self._validate_horizons(aligned, horizon_end_times)
        if (
            (aligned.index < self.config.end).any()
            or (horizon_end_times >= self._ensemble.config.blind_start).any()
            or not np.isin(y, self.classes_).all()
        ):
            raise ValueError(
                "Quality evaluation must be directional, after calibration, pre-blind.")
        # Validate metric configuration before invoking any member prediction.
        probability_quality(np.array([0.5]), np.array([0]), n_bins=n_bins)
        raw = self._ensemble.evaluate(aligned)
        calibrated = self._calibrate_batch(raw)
        column = self.classes_.index(self.action)
        binary = (y == self.action).astype(int)
        return {
            "raw": probability_quality(raw.probabilities[:, column], binary, n_bins=n_bins),
            "calibrated": probability_quality(calibrated.probabilities[:, column], binary,
                                               n_bins=n_bins),
            "first_entry": aligned.index[0].isoformat(),
            "last_complete_horizon": horizon_end_times.max().isoformat(),
            "calibration_method": self.config.method, "blind_accessed": False,
        }

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        return self.evaluate(features).predictions

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        return self.evaluate(features).probabilities

    def get_params(self) -> dict[str, Any]:
        return {"calibration": self.config.model_dump(mode="json"),
                "ensemble": self._ensemble.get_params()}
