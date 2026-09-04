"""Explicit quantitative research warning triggers and diagnostic detection."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prooflab.proof.monte_carlo import MonteCarloResult
from prooflab.proof.regime import RegimeAnalysisResult
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.sensitivity import ParameterSensitivityResult
from prooflab.proof.stress import ExecutionStressResult


class ResearchWarningCode(StrEnum):
    """Canonical research warning classifications."""

    LOW_TRADE_COUNT = "LOW_TRADE_COUNT"
    HIGH_PARAMETER_SENSITIVITY = "HIGH_PARAMETER_SENSITIVITY"
    HIGH_CLASS_IMBALANCE = "HIGH_CLASS_IMBALANCE"
    HIGH_OUT_OF_SAMPLE_DEGRADATION = "HIGH_OUT_OF_SAMPLE_DEGRADATION"
    PERFORMANCE_DEPENDS_ON_LOW_SPREAD = "PERFORMANCE_DEPENDS_ON_LOW_SPREAD"
    POSSIBLE_OVERFITTING = "POSSIBLE_OVERFITTING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    MODEL_DRIFT_DETECTED = "MODEL_DRIFT_DETECTED"


class ResearchWarning(BaseModel):
    """Structured diagnostic warning record emitted during strategy verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ResearchWarningCode
    severity: str = Field(default="WARNING")  # "INFO", "WARNING", "CRITICAL"
    message: str
    context: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class ResearchWarningDetector:
    """Evaluates strategy metrics against quantitative hazard thresholds."""

    @staticmethod
    def detect_warnings(
        scorecard: ProofScorecard,
        sensitivity_result: ParameterSensitivityResult | None = None,
        stress_result: ExecutionStressResult | None = None,
        monte_carlo_result: MonteCarloResult | None = None,
        regime_result: RegimeAnalysisResult | None = None,
        train_sharpe: float | None = None,
        class_balance_fraction: float | None = None,
    ) -> list[ResearchWarning]:
        """Inspect all strategy dimensions and emit active research warnings."""
        warnings: list[ResearchWarning] = []

        # 1. LOW_TRADE_COUNT
        if scorecard.trade_count < 50:
            sev = "CRITICAL" if scorecard.trade_count < 20 else "WARNING"
            warnings.append(
                ResearchWarning(
                    code=ResearchWarningCode.LOW_TRADE_COUNT,
                    severity=sev,
                    message=(
                        f"Sample size of {scorecard.trade_count} trades is too small "
                        "for reliable statistical confidence."
                    ),
                    context={"trade_count": scorecard.trade_count},
                )
            )

        # 2. HIGH_PARAMETER_SENSITIVITY
        if sensitivity_result and sensitivity_result.is_fragile:
            warnings.append(
                ResearchWarning(
                    code=ResearchWarningCode.HIGH_PARAMETER_SENSITIVITY,
                    severity="CRITICAL" if sensitivity_result.has_cliff_effect else "WARNING",
                    message=(
                        "Strategy exhibits high parameter sensitivity or cliff effects across "
                        "perturbed stop/target grids."
                    ),
                    context={
                        "profitable_cells_pct": sensitivity_result.profitable_cells_pct,
                        "has_cliff_effect": sensitivity_result.has_cliff_effect,
                    },
                )
            )

        # 3. PERFORMANCE_DEPENDS_ON_LOW_SPREAD
        if stress_result and stress_result.depends_on_low_spread:
            warnings.append(
                ResearchWarning(
                    code=ResearchWarningCode.PERFORMANCE_DEPENDS_ON_LOW_SPREAD,
                    severity="CRITICAL",
                    message=(
                        "Profitability collapses under realistic conservative or "
                        "stress spread tiers."
                    ),
                    context={
                        "normal_return_pct": stress_result.normal_return_pct,
                        "conservative_return_pct": stress_result.conservative_return_pct,
                        "stress_return_pct": stress_result.stress_return_pct,
                    },
                )
            )

        # 4. POSSIBLE_OVERFITTING / HIGH_OUT_OF_SAMPLE_DEGRADATION
        if train_sharpe is not None and train_sharpe > 0:
            test_sharpe = scorecard.sharpe_ratio
            if test_sharpe < train_sharpe * 0.40:
                warnings.append(
                    ResearchWarning(
                        code=ResearchWarningCode.HIGH_OUT_OF_SAMPLE_DEGRADATION,
                        severity="WARNING",
                        message=(
                            f"Out-of-sample Sharpe ({test_sharpe:.2f}) degraded by > 60% "
                            f"compared to in-sample ({train_sharpe:.2f})."
                        ),
                        context={
                            "train_sharpe": train_sharpe,
                            "test_sharpe": test_sharpe,
                        },
                    )
                )

        # 5. HIGH_CLASS_IMBALANCE
        if class_balance_fraction is not None and (
            class_balance_fraction < 0.10 or class_balance_fraction > 0.90
        ):
            imb_pct = class_balance_fraction * 100.0
            warnings.append(
                ResearchWarning(
                    code=ResearchWarningCode.HIGH_CLASS_IMBALANCE,
                    severity="WARNING",
                    message=(
                        f"Target class distribution is severely imbalanced ({imb_pct:.1f}%)."
                    ),
                    context={"class_balance_fraction": class_balance_fraction},
                )
            )

        # 6. MODEL_DRIFT_DETECTED
        if (
            regime_result
            and not regime_result.all_years_profitable
            and regime_result.total_years_count >= 3
        ):
            tot_yrs = regime_result.total_years_count
            prof_yrs = regime_result.profitable_years_count
            unprofitable_count = tot_yrs - prof_yrs
            if unprofitable_count >= 2:
                warnings.append(
                    ResearchWarning(
                        code=ResearchWarningCode.MODEL_DRIFT_DETECTED,
                        severity="WARNING",
                        message=(
                            f"Strategy suffered net losses in {unprofitable_count} "
                            f"out of {tot_yrs} calendar years."
                        ),
                        context={
                            "profitable_years": prof_yrs,
                            "total_years": tot_yrs,
                        },
                    )
                )

        return warnings
