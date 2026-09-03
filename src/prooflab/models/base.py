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
        self.classes_: list[int] = []
        self.is_fitted: bool = False

    def _validate_inputs(
        self,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Validate feature matrix and label vector dimensions."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"Features must be a pandas DataFrame, got {type(features).__name__}."
            )

        if features.empty:
            raise ValueError("Feature matrix cannot be empty.")

        y_arr = np.asarray(labels)
        if len(features) != len(y_arr):
            raise ValueError(
                f"Length mismatch: features has {len(features)} rows "
                f"but labels has {len(y_arr)} elements."
            )

        return features, y_arr

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate that test/inference features contain all required columns in expected order."""
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"Model '{self.model_name}' is not fitted yet. Call 'fit' before predicting."
            )

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"Features must be a pandas DataFrame, got {type(features).__name__}."
            )

        missing = [col for col in self.feature_names if col not in features.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns for model '{self.model_name}': {missing}"
            )

        # Enforce exact column order as seen during training
        return features[self.feature_names]

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        val_data: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
    ) -> Self:
        """Fit the model on training data, optionally evaluating on validation data."""
        clean_features, y_arr = self._validate_inputs(features, labels)
        self.feature_names = list(clean_features.columns)

        unique_classes = sorted(np.unique(y_arr).tolist())
        self.classes_ = unique_classes

        # Validate val_data if provided (ensuring split separation)
        val_clean: tuple[pd.DataFrame, np.ndarray] | None = None
        if val_data is not None:
            val_x, val_y = val_data
            val_x_clean, val_y_arr = self._validate_inputs(val_x, val_y)
            val_clean = (val_x_clean[self.feature_names], val_y_arr)

        self._fit_internal(clean_features, y_arr, val_clean)
        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate discrete class predictions for input samples."""
        aligned_features = self._validate_features(features)
        preds = self._predict_internal(aligned_features)
        return np.asarray(preds)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Generate class probability distributions (shape: (N, n_classes))."""
        aligned_features = self._validate_features(features)
        proba = self._predict_proba_internal(aligned_features)
        proba_arr = np.asarray(proba)

        if proba_arr.ndim != 2 or proba_arr.shape[1] != len(self.classes_):
            raise ValueError(
                f"Predicted probabilities shape {proba_arr.shape} does not match "
                f"expected (N, {len(self.classes_)})."
            )

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
