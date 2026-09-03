"""Unit tests for prooflab.proof.importance (Global Feature Importance Analyzer)."""

import numpy as np
import pytest

from prooflab.proof.importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceResult,
)


class DummyTreeModel:
    """Mock model exposing feature_importances_."""

    def __init__(self, importances: list[float]) -> None:
        self.feature_importances_ = np.array(importances)


def test_tree_feature_importance() -> None:
    model = DummyTreeModel([0.10, 0.50, 0.40])
    feature_names = ["feat_A", "feat_B", "feat_C"]

    res = FeatureImportanceAnalyzer.calculate_tree_importance(
        model, feature_names, importance_type="gain"
    )

    assert isinstance(res, FeatureImportanceResult)
    assert res.scope == "GLOBAL"
    assert len(res.features) == 3
    assert res.top_features[0] == "feat_B"
    assert res.features[0].feature_name == "feat_B"
    assert pytest.approx(res.features[0].relative_importance_pct) == 50.0
    assert res.features[0].rank == 1

    # DataFrame export
    df = res.to_dataframe()
    assert len(df) == 3
    assert "feature_name" in df.columns
    assert "relative_importance_pct" in df.columns


def test_permutation_feature_importance() -> None:
    # Synthetic dataset where Y = 1 if x[0] > 0 else 0 (col 0 is predictive, col 1 & 2 are noise)
    rng = np.random.default_rng(42)
    x_mat = rng.normal(size=(100, 3))
    y = np.where(x_mat[:, 0] > 0, 1, 0)
    feature_names = ["signal_feature", "noise_1", "noise_2"]

    def oracle_predictor(x_arr: np.ndarray) -> np.ndarray:
        return np.where(x_arr[:, 0] > 0, 1, 0)

    res = FeatureImportanceAnalyzer.calculate_permutation_importance(
        predict_fn=oracle_predictor,
        x_val=x_mat,
        y_val=y,
        feature_names=feature_names,
        n_repeats=3,
        random_seed=42,
    )

    assert isinstance(res, FeatureImportanceResult)
    assert res.scope == "GLOBAL"
    assert res.top_features[0] == "signal_feature"
    assert res.features[0].rank == 1
    assert res.features[0].importance_score > res.features[1].importance_score
