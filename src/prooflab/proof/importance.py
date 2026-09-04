"""Global feature importance analyzer with strict separation from local trade explanations."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class FeatureImportanceEntry(BaseModel):
    """Normalized importance record for a single feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    importance_score: float
    relative_importance_pct: float
    rank: int


class FeatureImportanceResult(BaseModel):
    """Container for global feature importance rankings across strategy features."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    importance_type: str  # "gain", "split", "permutation", "coefficient"
    scope: str = "GLOBAL"  # Clear distinction from per-trade local explanations
    features: list[FeatureImportanceEntry]
    top_features: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Export importance rankings as a DataFrame sorted by rank."""
        if not self.features:
            return pd.DataFrame()
        records = [f.model_dump(mode="python") for f in self.features]
        df = pd.DataFrame(records)
        df.sort_values("rank", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def to_dict(self) -> dict[str, float]:
        """Export as feature_name -> relative_importance_pct mapping."""
        return {f.feature_name: f.relative_importance_pct for f in self.features}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class FeatureImportanceAnalyzer:
    """Computes global feature importances via model attributes or permutation testing."""

    @staticmethod
    def calculate_tree_importance(
        model: Any,
        feature_names: list[str],
        importance_type: str = "gain",
    ) -> FeatureImportanceResult:
        """Extract global feature importance from tree-based estimators (e.g. XGBoost)."""
        raw_scores: list[float] = []

        # Scikit-learn or XGBoost sklearn-wrapper
        if hasattr(model, "feature_importances_"):
            raw_scores = [float(x) for x in model.feature_importances_]
        elif hasattr(model, "coef_"):
            # Linear model coefficients (magnitude)
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                coefs = np.mean(coefs, axis=0)
            raw_scores = [float(x) for x in coefs]
        else:
            # Fallback uniform importance if model lacks internal attribution
            raw_scores = [1.0 / max(1, len(feature_names))] * len(feature_names)

        total_sum = sum(raw_scores)
        if total_sum <= 1e-12:
            pct_shares = [0.0] * len(raw_scores)
        else:
            pct_shares = [(s / total_sum) * 100.0 for s in raw_scores]

        # Pair with names and sort descending
        paired = list(zip(feature_names, raw_scores, pct_shares, strict=False))
        paired.sort(key=lambda x: x[1], reverse=True)

        entries = [
            FeatureImportanceEntry(
                feature_name=name,
                importance_score=round(score, 6),
                relative_importance_pct=round(pct, 4),
                rank=i + 1,
            )
            for i, (name, score, pct) in enumerate(paired)
        ]

        top_names = [e.feature_name for e in entries[: min(10, len(entries))]]

        return FeatureImportanceResult(
            importance_type=importance_type,
            scope="GLOBAL",
            features=entries,
            top_features=top_names,
        )

    @staticmethod
    def calculate_permutation_importance(
        predict_fn: Callable[[np.ndarray], np.ndarray],
        x_val: np.ndarray | pd.DataFrame,
        y_val: np.ndarray,
        feature_names: list[str],
        scoring_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        n_repeats: int = 5,
        random_seed: int = 42,
    ) -> FeatureImportanceResult:
        """Compute model-agnostic global permutation feature importance.

        Measures baseline score degradation when each feature column is randomly shuffled.
        """
        rng = np.random.default_rng(random_seed)
        x_arr = x_val.to_numpy() if isinstance(x_val, pd.DataFrame) else np.array(x_val)
        y_arr = np.array(y_val)

        if scoring_fn is None:
            # Default accuracy score
            def scoring_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(np.mean(y_true == y_pred))

        # Baseline evaluation
        baseline_preds = predict_fn(x_arr)
        base_score = scoring_fn(y_arr, baseline_preds)

        raw_drops: list[float] = []
        n_cols = x_arr.shape[1]

        for col_idx in range(n_cols):
            col_drops: list[float] = []
            for _ in range(n_repeats):
                x_shuffled = x_arr.copy()
                shuffled_col = x_shuffled[:, col_idx].copy()
                rng.shuffle(shuffled_col)
                x_shuffled[:, col_idx] = shuffled_col

                shuffled_preds = predict_fn(x_shuffled)
                shuffled_score = scoring_fn(y_arr, shuffled_preds)
                drop = max(0.0, base_score - shuffled_score)
                col_drops.append(drop)

            raw_drops.append(float(np.mean(col_drops)))

        total_drop = sum(raw_drops)
        if total_drop <= 1e-12:
            pct_shares = [100.0 / max(1, len(raw_drops))] * len(raw_drops)
        else:
            pct_shares = [(d / total_drop) * 100.0 for d in raw_drops]

        paired = list(zip(feature_names, raw_drops, pct_shares, strict=False))
        paired.sort(key=lambda x: x[1], reverse=True)

        entries = [
            FeatureImportanceEntry(
                feature_name=name,
                importance_score=round(score, 6),
                relative_importance_pct=round(pct, 4),
                rank=i + 1,
            )
            for i, (name, score, pct) in enumerate(paired)
        ]

        top_names = [e.feature_name for e in entries[: min(10, len(entries))]]

        return FeatureImportanceResult(
            importance_type="permutation",
            scope="GLOBAL",
            features=entries,
            top_features=top_names,
        )
