"""Common model interface and base classes for Proof Lab predictive models.

Defines the universal BaseModelWrapper contract required for all baseline,
tabular, neural network, and statistical setup classifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd


class ModelNotFittedError(ValueError):
    """Raised when predict or predict_proba is called before fitting."""


class BaseModelWrapper(ABC):
    """Abstract base class establishing the standard model contract across Proof Lab."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.feature_names: list[str] = []
        self.feature_schema: dict[str, str] = {}
        self.classes_: list[int] = []
        self.is_fitted: bool = False
        self.fit_details_: dict[str, Any] = {}
        self._horizon_end_times: pd.Series | None = None

    def _validate_inputs(
        self,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Validate feature matrix and label vector dimensions."""
        self._validate_matrix(features)
        y_arr = np.asarray(labels)
        if y_arr.ndim != 1:
            raise ValueError("Labels must be one-dimensional.")
        if len(features) != len(y_arr):
            raise ValueError(
                f"Length mismatch: features has {len(features)} rows "
                f"but labels has {len(y_arr)} elements."
            )
        if isinstance(labels, pd.Series) and not labels.index.equals(features.index):
            raise ValueError("Feature and label indices must match exactly.")
        if y_arr.dtype.kind not in "iuf" or not np.isin(y_arr, [-1, 0, 1]).all():
            raise ValueError("Labels must contain only canonical classes -1, 0, 1.")
        return features, y_arr.astype(np.int64)

    @staticmethod
    def _validate_matrix(features: pd.DataFrame) -> None:
        """Require an unambiguous, finite numeric feature matrix."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"Features must be a pandas DataFrame, got {type(features).__name__}."
            )

        if features.empty:
            raise ValueError("Feature matrix cannot be empty.")

        if not features.columns.is_unique or not all(
            isinstance(name, str) and name for name in features.columns
        ):
            raise ValueError("Feature names must be unique nonempty strings.")
        if not features.index.is_unique:
            raise ValueError("Feature row indices must be unique.")
        if any(dtype.kind not in "iuf" for dtype in features.dtypes):
            raise ValueError("Features must be real numeric values.")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Features must contain only finite values.")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        self._validate_matrix(features)
        missing = [col for col in self.feature_names if col not in features.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        extra = [col for col in features.columns if col not in self.feature_names]
        if extra:
            raise ValueError(f"Unexpected feature columns: {extra}")
        return features[self.feature_names]

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate that test/inference features contain all required columns in expected order."""
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"Model '{self.model_name}' is not fitted yet. Call 'fit' before predicting."
            )

        return self._align_features(features)

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        val_data: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        *,
        horizon_end_times: pd.Series | None = None,
    ) -> Self:
        """Fit training data; the caller must enforce chronological split isolation.

        Array labels are positional; Series labels must have matching indices.
        Validation may contain an unseen canonical class, but never expands the
        training-derived class vocabulary. Failed fitting invalidates the model.
        """
        self.is_fitted = False
        self.feature_names = []
        self.feature_schema = {}
        self.classes_ = []
        self.fit_details_ = {}
        self._horizon_end_times = None
        clean_features, y_arr = self._validate_inputs(features, labels)
        if horizon_end_times is not None:
            self._validate_horizons(clean_features, horizon_end_times)
        self.feature_names = list(clean_features.columns)
        self.feature_schema = {name: str(dtype) for name, dtype in clean_features.dtypes.items()}

        unique_classes = sorted(np.unique(y_arr).tolist())
        self.classes_ = unique_classes

        # These checks validate shape/schema, not chronological separation.
        val_clean: tuple[pd.DataFrame, np.ndarray] | None = None
        if val_data is not None:
            val_x, val_y = val_data
            val_x_clean, val_y_arr = self._validate_inputs(val_x, val_y)
            val_clean = (self._align_features(val_x_clean), val_y_arr)

        self._horizon_end_times = horizon_end_times
        try:
            self._fit_internal(clean_features, y_arr, val_clean)
        finally:
            # Boundary evidence is recorded in fit_details_; retain no training rows.
            self._horizon_end_times = None
        self.is_fitted = True
        return self

    @staticmethod
    def _validate_horizons(features: pd.DataFrame, ends: pd.Series) -> None:
        """Validate full-horizon timestamps supplied by the chronological pipeline."""
        if (
            not isinstance(features.index, pd.DatetimeIndex)
            or str(features.index.tz) != "UTC"
            or not features.index.is_monotonic_increasing
        ):
            raise ValueError("Horizon context requires ordered UTC feature timestamps.")
        if (
            not isinstance(ends, pd.Series) or not ends.index.equals(features.index)
            or not isinstance(ends.dtype, pd.DatetimeTZDtype)
            or str(ends.dt.tz) != "UTC" or ends.isna().any()
            or not (ends > features.index).all()
        ):
            raise ValueError("Full horizon ends must be aligned UTC timestamps after each entry.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate discrete class predictions for input samples."""
        aligned_features = self._validate_features(features)
        preds = self._predict_internal(aligned_features)
        result = np.asarray(preds)
        if result.shape != (len(features),) or not np.isin(result, self.classes_).all():
            raise ValueError("Predictions must be one known class per input row.")
        return result.astype(np.int64)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Generate class probability distributions (shape: (N, n_classes))."""
        aligned_features = self._validate_features(features)
        proba = self._predict_proba_internal(aligned_features)
        proba_arr = np.asarray(proba)

        if proba_arr.shape != (len(features), len(self.classes_)):
            raise ValueError(
                f"Predicted probabilities shape {proba_arr.shape} does not match "
                f"expected (N, {len(self.classes_)})."
            )

        if (
            proba_arr.dtype.kind not in "iuf"
            or not np.isfinite(proba_arr).all()
            or (proba_arr < 0).any()
            or (proba_arr > 1).any()
            or not np.allclose(proba_arr.sum(axis=1), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError("Probabilities must be finite, in [0, 1], and sum to one.")
        return proba_arr

    @abstractmethod
    def _fit_internal(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        """Internal model-specific training logic."""

    @abstractmethod
    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        """Internal model-specific discrete prediction logic."""

    @abstractmethod
    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        """Internal model-specific probability prediction logic."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return dictionary of model parameters and hyperparameters."""
