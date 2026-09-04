"""Unit tests for feature, prediction, and performance drift detection."""

from __future__ import annotations

import numpy as np

from prooflab.monitoring.drift import (
    DriftCoordinator,
    DriftStatus,
    FeatureDriftDetector,
    PerformanceDriftDetector,
    PredictionDriftDetector,
)


def test_feature_drift_detection() -> None:
    """Test PSI and KS-test drift evaluation under stable and shifted distributions."""
    rng = np.random.default_rng(42)

    # Reference training distribution
    ref_samples = rng.normal(0.0, 1.0, size=1000)

    # Current sample from same distribution -> NORMAL
    cur_stable = rng.normal(0.02, 1.01, size=500)
    res_stable = FeatureDriftDetector.evaluate_feature("rsi_14", ref_samples, cur_stable)
    assert res_stable.status == DriftStatus.NORMAL
    assert res_stable.psi_score < 0.10

    # Current sample with major covariate shift -> SUSPENDED
    cur_shifted = rng.normal(2.5, 0.5, size=500)
    res_shifted = FeatureDriftDetector.evaluate_feature("rsi_14", ref_samples, cur_shifted)
    assert res_shifted.status == DriftStatus.SUSPENDED
    assert res_shifted.psi_score >= 0.25


def test_prediction_drift_detection() -> None:
    """Test prediction distribution divergence and confidence monitoring."""
    # Balanced nominal predictions
    nominal_preds = ["BUY"] * 20 + ["SELL"] * 20 + ["IGNORE"] * 60
    nominal_confs = [0.75] * 20 + [0.75] * 20 + [0.70] * 60
    res_nominal = PredictionDriftDetector.evaluate_predictions(nominal_preds, nominal_confs)
    assert res_nominal.status == DriftStatus.NORMAL

    # Shifted predictions (100% BUY signals - anomalous)
    anomalous_preds = ["BUY"] * 100
    anomalous_confs = [0.75] * 100
    res_shifted = PredictionDriftDetector.evaluate_predictions(anomalous_preds, anomalous_confs)
    assert res_shifted.status in {DriftStatus.WARNING, DriftStatus.SUSPENDED}


def test_performance_drift_detection() -> None:
    """Test win rate degradation and drawdown limit breaches."""
    # Nominal performance
    profitable_trades = [150.0, -100.0, 200.0, 120.0, -80.0, 300.0] * 3
    res_perf_ok = PerformanceDriftDetector.evaluate_performance(
        live_trade_pnls=profitable_trades,
        benchmark_win_rate=0.55,
        benchmark_profit_factor=1.5,
        current_equity=11000.0,
        peak_equity=11000.0,
    )
    assert res_perf_ok.status == DriftStatus.NORMAL

    # Severe drawdown breach -> SUSPENDED
    res_drawdown = PerformanceDriftDetector.evaluate_performance(
        live_trade_pnls=[-500.0, -500.0, -1000.0],
        current_equity=8000.0,
        peak_equity=10000.0,  # 20% DD > 15% limit
        max_drawdown_limit=0.15,
    )
    assert res_drawdown.status == DriftStatus.SUSPENDED
    assert res_drawdown.current_drawdown == 0.20


def test_drift_coordinator_rollup() -> None:
    """Test aggregated drift report rollup."""
    feat_res = {
        "f1": FeatureDriftDetector.evaluate_feature(
            "f1", np.array([1.0, 2.0, 3.0] * 50), np.array([1.0, 2.0, 3.0] * 50)
        )
    }
    pred_res = PredictionDriftDetector.evaluate_predictions(["BUY"] * 10, [0.45] * 10)
    perf_res = PerformanceDriftDetector.evaluate_performance(
        [], current_equity=10000.0, peak_equity=10000.0
    )


    report = DriftCoordinator.evaluate(
        feature_results=feat_res,
        prediction_result=pred_res,
        performance_result=perf_res,
    )

    assert report.overall_status in {DriftStatus.WARNING, DriftStatus.SUSPENDED}
    assert "f1" in report.feature_drift
