"""Deliberately leaking features and fitted transforms with measurable advantages."""

import numpy as np
import pandas as pd
import pytest

from prooflab.validation.leakage import (
    audit_availability,
    audit_feature_causality,
    audit_fit_rows,
    audit_plan,
)
from prooflab.validation.splits import SplitConfig, chronological_split


@pytest.fixture
def history():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"shock": rng.choice([-1., 1.], 100)},
                        index=pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC"))


def test_detect_oracle_feature_with_obvious_accuracy_gain(history):
    labels = history.shock.shift(-1).iloc[:-1]
    causal_predictions = history.shock.iloc[:-1]
    leaked_predictions = history.shock.shift(-1).iloc[:-1]
    assert (leaked_predictions == labels).mean() == 1
    assert (causal_predictions == labels).mean() < 0.65

    def leaking(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.shift(-1)

    report = audit_feature_causality(history, leaking, blind_start=pd.Timestamp("2021", tz="UTC"))
    assert not report.passed
    assert any("future_sensitive_feature" in issue for issue in report.issues)
    with pytest.raises(ValueError, match="Leakage detected"):
        report.require_pass()
    causal = audit_feature_causality(history, lambda frame: frame.rolling(3).mean(),
                                     blind_start=pd.Timestamp("2021", tz="UTC"))
    assert causal.passed


def test_detect_whole_history_scaling_and_future_availability(history):
    result = audit_feature_causality(
        history,
        lambda frame: (frame - frame.mean()) / frame.std(),
        blind_start=pd.Timestamp("2021", tz="UTC"),
        cutoffs=(25, 50, 75),
    )
    assert not result.passed
    available = pd.DataFrame({"causal": history.index,
                              "leaked_label": history.index + pd.Timedelta("1h")},
                             index=history.index)
    assert audit_availability(available).issues == ("future_feature_availability: leaked_label",)
    assert audit_fit_rows(history.index[:50], history.index[:50]).passed
    assert not audit_fit_rows(history.index, history.index[:50]).passed


def test_detect_forged_fold_and_embargo_removal(history):
    plan = chronological_split(history.index, SplitConfig(validation_start=history.index[60],
                                blind_start="2020-01-06T00:00:00Z", max_label_horizon=3),
                               dataset_end=pd.Timestamp("2022", tz="UTC"))
    assert audit_plan(plan, history.index).passed
    assert not audit_plan(plan.model_copy(update={"train_indices": (0, 59, 61)}),
                          history.index).passed
    assert not audit_plan(plan.model_copy(update={"embargo_intervals": ()}), history.index).passed


def test_blind_history_rejected_before_transform(history):
    with pytest.raises(ValueError, match="blind"):
        audit_feature_causality(history, lambda _: pytest.fail("Blind history transformed"),
                                blind_start=history.index[50])
