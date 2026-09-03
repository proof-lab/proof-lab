"""Baseline classifiers for Proof Lab model benchmarking.

Random predictions are seeded draws; probability outputs expose the sampling
distribution. Majority probabilities are empirical training class frequencies.
Ties select the smallest canonical class. Neither baseline fits validation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prooflab.models.base import BaseModelWrapper


class RandomClassifier(BaseModelWrapper):
    """Generates random predictions based on class priors or uniform distributions."""

    def __init__(
        self,
        strategy: str = "prior",
        random_state: int | None = 42,
    ) -> None:
        super().__init__(model_name="random_classifier")
        if strategy not in ("prior", "uniform"):
            raise ValueError(f"Unknown strategy '{strategy}'. Must be 'prior' or 'uniform'.")
        self.strategy = strategy
        self.random_state = random_state
        self.class_priors_: dict[int, float] = {}
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def _fit_internal(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self.class_priors_.clear()
        n_samples = len(labels)

        if self.strategy == "prior":
            for cls_val in self.classes_:
                count = (labels == cls_val).sum()
                self.class_priors_[cls_val] = float(count / n_samples)
        else:
            uniform_prob = 1.0 / len(self.classes_)
            for cls_val in self.classes_:
                self.class_priors_[cls_val] = uniform_prob

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        priors = [self.class_priors_[cls_val] for cls_val in self.classes_]
        return self._rng.choice(self.classes_, size=len(features), p=priors)

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        priors = np.array([self.class_priors_[cls_val] for cls_val in self.classes_])
        return np.tile(priors, (len(features), 1))

    def get_params(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "random_state": self.random_state}


class MajorityClassifier(BaseModelWrapper):
    """Always predicts the most frequent class observed in the training split."""

    def __init__(self) -> None:
        super().__init__(model_name="majority_classifier")
        self.majority_class_: int = 0
        self.class_priors_: dict[int, float] = {}

    def _fit_internal(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        unique, counts = np.unique(labels, return_counts=True)
        max_idx = int(np.argmax(counts))
        self.majority_class_ = int(unique[max_idx])

        self.class_priors_.clear()
        n_samples = len(labels)
        for cls_val in self.classes_:
            count = (labels == cls_val).sum()
            self.class_priors_[cls_val] = float(count / n_samples)

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.majority_class_, dtype=int)

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        # Returns empirical training class prior probabilities
        priors = np.array([self.class_priors_[cls_val] for cls_val in self.classes_])
        return np.tile(priors, (len(features), 1))

    def get_params(self) -> dict[str, Any]:
        return {}

