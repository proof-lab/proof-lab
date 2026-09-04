"""Statistical drift detection monitoring feature distributions, predictions, and performance."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class DriftStatus(StrEnum):
    """Drift evaluation classification state."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    SUSPENDED = "SUSPENDED"


class FeatureDriftResult(BaseModel):
    """Distribution drift evaluation for an individual feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    status: DriftStatus
    message: str


class PredictionDriftResult(BaseModel):
    """Drift evaluation of model classification output distribution and confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    buy_ratio: float
    sell_ratio: float
    ignore_ratio: float
    mean_confidence: float
    distribution_divergence: float
    status: DriftStatus
    message: str


class PerformanceDriftResult(BaseModel):
    """Drift evaluation comparing rolling live execution performance against proof benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    live_win_rate: float
    benchmark_win_rate: float
    win_rate_drop_pct: float
    live_profit_factor: float
    current_drawdown: float
    max_allowed_drawdown: float
    status: DriftStatus
    message: str


class DriftReport(BaseModel):
    """Unified comprehensive drift evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_status: DriftStatus
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    feature_drift: dict[str, FeatureDriftResult] = Field(default_factory=dict)
    prediction_drift: PredictionDriftResult | None = None
    performance_drift: PerformanceDriftResult | None = None
    summary: str


class FeatureDriftDetector:
    """Detects covariate shift between training feature distributions and live incoming data."""

    @staticmethod
    def calculate_psi(
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index (PSI) between reference and current samples."""
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        # Bin edges based on reference quantiles
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref, percentiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        # Ensure bins are strictly monotonic
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        # Avoid zero division with smoothing epsilon
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
        cur_pct = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(round(max(0.0, float(psi)), 4))

    @classmethod
    def evaluate_feature(
        cls,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
        psi_warning: float = 0.10,
        psi_critical: float = 0.25,
    ) -> FeatureDriftResult:
        """Evaluate KS-test and PSI for a single feature."""
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]

        if len(ref) < 5 or len(cur) < 5:
            return FeatureDriftResult(
                feature_name=feature_name,
                psi_score=0.0,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                status=DriftStatus.NORMAL,
                message="Insufficient samples for drift evaluation",
            )

        psi = cls.calculate_psi(ref, cur)
        ks_res = stats.ks_2samp(ref, cur)
        ks_stat = float(round(float(ks_res.statistic), 4))
        ks_p = float(round(float(ks_res.pvalue), 4))

        if psi >= psi_critical:
            status = DriftStatus.SUSPENDED
            msg = f"Critical feature drift: PSI={psi:.4f} >= {psi_critical}"
        elif psi >= psi_warning or (ks_p < 0.01 and ks_stat > 0.15):
            status = DriftStatus.WARNING
            msg = f"Moderate feature drift: PSI={psi:.4f}, KS-stat={ks_stat:.4f}"
        else:
            status = DriftStatus.NORMAL
            msg = f"Feature distribution stable (PSI={psi:.4f})"

        return FeatureDriftResult(
            feature_name=feature_name,
            psi_score=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            status=status,
            message=msg,
        )


class PredictionDriftDetector:
    """Monitors model output class distribution and calibrated confidence drift."""

    @staticmethod
    def evaluate_predictions(
        predictions: list[str],
        confidences: list[float],
        reference_distribution: dict[str, float] | None = None,
        divergence_warning: float = 0.15,
        divergence_critical: float = 0.30,
        min_confidence_warning: float = 0.52,
    ) -> PredictionDriftResult:
        """Evaluate class frequencies and confidence stability against reference baseline."""
        if not predictions:
            return PredictionDriftResult(
                buy_ratio=0.0,
                sell_ratio=0.0,
                ignore_ratio=1.0,
                mean_confidence=0.0,
                distribution_divergence=0.0,
                status=DriftStatus.NORMAL,
                message="No prediction samples",
            )

        total = len(predictions)
        buy_r = round(predictions.count("BUY") / total, 4)
        sell_r = round(predictions.count("SELL") / total, 4)
        ignore_r = round(predictions.count("IGNORE") / total, 4)

        mean_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        ref = reference_distribution or {"BUY": 0.20, "SELL": 0.20, "IGNORE": 0.60}
        divergence = round(
            float(
                0.5
                * (
                    abs(buy_r - ref.get("BUY", 0.0))
                    + abs(sell_r - ref.get("SELL", 0.0))
                    + abs(ignore_r - ref.get("IGNORE", 0.0))
                )
            ),
            4,
        )

        if divergence >= divergence_critical or (mean_conf < min_confidence_warning and len(confidences) >= 10):
            status = DriftStatus.SUSPENDED
            msg = f"Critical prediction drift: divergence={divergence:.4f}, mean_conf={mean_conf:.4f}"
        elif divergence >= divergence_warning:
            status = DriftStatus.WARNING
            msg = f"Moderate prediction shift: divergence={divergence:.4f}"
        else:
            status = DriftStatus.NORMAL
            msg = f"Prediction distribution nominal (divergence={divergence:.4f})"

        return PredictionDriftResult(
            buy_ratio=buy_r,
            sell_ratio=sell_r,
            ignore_ratio=ignore_r,
            mean_confidence=mean_conf,
            distribution_divergence=divergence,
            status=status,
            message=msg,
        )


class PerformanceDriftDetector:
    """Compares real-time execution metrics against benchmark proof metrics."""

    @staticmethod
    def evaluate_performance(
        live_trade_pnls: list[float],
        benchmark_win_rate: float = 0.55,
        benchmark_profit_factor: float = 1.60,
        current_equity: float = 10000.0,
        peak_equity: float = 10000.0,
        max_drawdown_limit: float = 0.15,
        win_rate_drop_warning: float = 0.15,
    ) -> PerformanceDriftResult:
        """Calculate real-time drawdown and win rate degradation."""
        # Calculate current drawdown
        dd = 0.0
        if peak_equity > 0:
            dd = max(0.0, (peak_equity - current_equity) / peak_equity)
        dd = round(dd, 4)

        if not live_trade_pnls:
            return PerformanceDriftResult(
                live_win_rate=benchmark_win_rate,
                benchmark_win_rate=benchmark_win_rate,
                win_rate_drop_pct=0.0,
                live_profit_factor=benchmark_profit_factor,
                current_drawdown=dd,
                max_allowed_drawdown=max_drawdown_limit,
                status=DriftStatus.SUSPENDED if dd >= max_drawdown_limit else DriftStatus.NORMAL,
                message="No live trades completed",
            )

        wins = [p for p in live_trade_pnls if p > 0]
        losses = [p for p in live_trade_pnls if p < 0]
        win_rate = round(len(wins) / len(live_trade_pnls), 4)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        drop_pct = round(max(0.0, (benchmark_win_rate - win_rate) / benchmark_win_rate), 4) if benchmark_win_rate > 0 else 0.0

        if dd >= max_drawdown_limit:
            status = DriftStatus.SUSPENDED
            msg = f"Drawdown limit exceeded: {dd * 100:.1f}% >= {max_drawdown_limit * 100:.1f}%"
        elif drop_pct >= win_rate_drop_warning and len(live_trade_pnls) >= 10:
            status = DriftStatus.WARNING
            msg = f"Performance degradation: win rate dropped {drop_pct * 100:.1f}% (PF={profit_factor:.2f})"
        else:
            status = DriftStatus.NORMAL
            msg = f"Performance nominal (WR={win_rate * 100:.1f}%, DD={dd * 100:.1f}%)"

        return PerformanceDriftResult(
            live_win_rate=win_rate,
            benchmark_win_rate=benchmark_win_rate,
            win_rate_drop_pct=drop_pct,
            live_profit_factor=profit_factor,
            current_drawdown=dd,
            max_allowed_drawdown=max_drawdown_limit,
            status=status,
            message=msg,
        )


class DriftCoordinator:
    """Unified drift evaluator aggregating feature, prediction, and performance monitors."""

    @staticmethod
    def evaluate(
        feature_results: dict[str, FeatureDriftResult] | None = None,
        prediction_result: PredictionDriftResult | None = None,
        performance_result: PerformanceDriftResult | None = None,
    ) -> DriftReport:
        """Combine all individual drift evaluations into a consolidated DriftReport."""
        all_statuses: list[DriftStatus] = []

        feat_map = feature_results or {}
        for r in feat_map.values():
            all_statuses.append(r.status)
        if prediction_result:
            all_statuses.append(prediction_result.status)
        if performance_result:
            all_statuses.append(performance_result.status)

        if DriftStatus.SUSPENDED in all_statuses:
            overall = DriftStatus.SUSPENDED
        elif DriftStatus.WARNING in all_statuses:
            overall = DriftStatus.WARNING
        else:
            overall = DriftStatus.NORMAL

        summary = (
            f"Drift assessment {overall.value}: {len(feat_map)} features evaluated, "
            f"prediction_status={prediction_result.status.value if prediction_result else 'N/A'}, "
            f"performance_status={performance_result.status.value if performance_result else 'N/A'}."
        )

        return DriftReport(
            overall_status=overall,
            feature_drift=feat_map,
            prediction_drift=prediction_result,
            performance_drift=performance_result,
            summary=summary,
        )
