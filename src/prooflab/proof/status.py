"""Explicit deterministic Proof Status rules evaluating strategy robustness."""

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


class ProofStatus(StrEnum):
    """Deterministic validation tiers for quantitative strategies."""

    NOT_PROVEN = "NOT_PROVEN"
    WEAK = "WEAK"
    PROMISING = "PROMISING"
    ROBUST = "ROBUST"


class ProofStatusThresholds(BaseModel):
    """Configurable thresholds governing explicit Proof Status rule gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_trade_count_weak: int = Field(default=20, ge=1)
    min_trade_count_promising: int = Field(default=50, ge=1)
    min_trade_count_robust: int = Field(default=100, ge=1)

    min_sharpe_promising: float = Field(default=0.5)
    min_sharpe_robust: float = Field(default=1.0)

    min_profit_factor_robust: float = Field(default=1.20)

    max_drawdown_promising_pct: float = Field(default=30.0)
    max_drawdown_robust_pct: float = Field(default=20.0)

    max_ruin_prob_promising_pct: float = Field(default=5.0)
    max_ruin_prob_robust_pct: float = Field(default=1.0)


class RuleEvaluationGate(BaseModel):
    """Record of a single rule gate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_name: str
    passed: bool
    description: str
    actual_value: Any
    threshold_value: Any


class ProofStatusEvaluation(BaseModel):
    """Result of deterministic strategy proof status assessment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ProofStatus
    status_reason: str
    is_validated: bool
    rule_gates: list[RuleEvaluationGate]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class ProofStatusEvaluator:
    """Evaluates strategy artifacts against explicit deterministic Proof Status rules."""

    def __init__(self, thresholds: ProofStatusThresholds | None = None) -> None:
        self.thresholds = thresholds or ProofStatusThresholds()

    def evaluate(
        self,
        scorecard: ProofScorecard,
        sensitivity_result: ParameterSensitivityResult | None = None,
        stress_result: ExecutionStressResult | None = None,
        monte_carlo_result: MonteCarloResult | None = None,
        regime_result: RegimeAnalysisResult | None = None,
        has_leakage: bool = False,
        blind_test_completed: bool = True,
    ) -> ProofStatusEvaluation:
        """Apply explicit rule gates to determine Proof Status."""
        gates: list[RuleEvaluationGate] = []

        # Gate 1: Leakage Check
        no_leakage = not has_leakage
        gates.append(
            RuleEvaluationGate(
                rule_name="No Data Leakage",
                passed=no_leakage,
                description="Zero lookahead or feature data leakage detected",
                actual_value=has_leakage,
                threshold_value=False,
            )
        )

        # Gate 2: Blind Test Completion
        gates.append(
            RuleEvaluationGate(
                rule_name="Blind Test Completed",
                passed=blind_test_completed,
                description="Strategy evaluated on untouched out-of-sample blind test set",
                actual_value=blind_test_completed,
                threshold_value=True,
            )
        )

        # Gate 3: Net Positive Profitability
        is_profitable = scorecard.total_net_return_pct > 0
        gates.append(
            RuleEvaluationGate(
                rule_name="Positive Net Return",
                passed=is_profitable,
                description="Total simulated net return after all costs is positive",
                actual_value=scorecard.total_net_return_pct,
                threshold_value=0.0,
            )
        )

        # Gate 4: Minimum Sample Size (Weak)
        min_weak = self.thresholds.min_trade_count_weak
        has_min_trades = scorecard.trade_count >= min_weak
        gates.append(
            RuleEvaluationGate(
                rule_name="Minimum Trade Count (Weak)",
                passed=has_min_trades,
                description=f"At least {min_weak} closed trades",
                actual_value=scorecard.trade_count,
                threshold_value=min_weak,
            )
        )

        # 1. NOT_PROVEN Evaluation
        if (
            not no_leakage
            or not blind_test_completed
            or not is_profitable
            or not has_min_trades
            or scorecard.profit_factor < 1.0
        ):
            reasons: list[str] = []
            if not no_leakage:
                reasons.append("Data leakage detected")
            if not blind_test_completed:
                reasons.append("Blind test evaluation not completed")
            if not is_profitable:
                reasons.append("Negative net return")
            if not has_min_trades:
                reasons.append(
                    f"Insufficient sample size ({scorecard.trade_count} < {min_weak})"
                )
            if scorecard.profit_factor < 1.0:
                reasons.append("Profit factor below 1.0")

            return ProofStatusEvaluation(
                status=ProofStatus.NOT_PROVEN,
                status_reason="; ".join(reasons),
                is_validated=False,
                rule_gates=gates,
            )

        # Evaluate Robustness Rules
        sens_ok = (
            (not sensitivity_result.is_fragile) if sensitivity_result else True
        )
        gates.append(
            RuleEvaluationGate(
                rule_name="Parameter Stability",
                passed=sens_ok,
                description=(
                    "Strategy stable across stop/target perturbations without cliff effects"
                ),
                actual_value=not sensitivity_result.is_fragile if sensitivity_result else True,
                threshold_value=True,
            )
        )

        cons_spread_ok = (
            stress_result.survives_conservative if stress_result else True
        )
        stress_spread_ok = stress_result.survives_stress if stress_result else True
        gates.append(
            RuleEvaluationGate(
                rule_name="Conservative Spread Survival (1.5x)",
                passed=cons_spread_ok,
                description="Strategy remains profitable under 1.5x conservative spread multiplier",
                actual_value=stress_result.conservative_return_pct if stress_result else 0.0,
                threshold_value=0.0,
            )
        )
        gates.append(
            RuleEvaluationGate(
                rule_name="Stress Spread Survival (2.5x)",
                passed=stress_spread_ok,
                description="Strategy remains profitable under 2.5x stress spread multiplier",
                actual_value=stress_result.stress_return_pct if stress_result else 0.0,
                threshold_value=0.0,
            )
        )

        max_prom_ruin = self.thresholds.max_ruin_prob_promising_pct
        max_rob_ruin = self.thresholds.max_ruin_prob_robust_pct
        ruin_ok_promising = (
            (monte_carlo_result.probability_of_ruin_pct <= max_prom_ruin)
            if monte_carlo_result
            else True
        )
        ruin_ok_robust = (
            (monte_carlo_result.probability_of_ruin_pct <= max_rob_ruin)
            if monte_carlo_result
            else True
        )
        actual_ruin = monte_carlo_result.probability_of_ruin_pct if monte_carlo_result else 0.0
        gates.append(
            RuleEvaluationGate(
                rule_name="Monte Carlo Ruin Risk (Promising)",
                passed=ruin_ok_promising,
                description=f"Probability of ruin <= {max_prom_ruin}%",
                actual_value=actual_ruin,
                threshold_value=max_prom_ruin,
            )
        )

        # 2. WEAK Evaluation
        min_prom_trades = self.thresholds.min_trade_count_promising
        max_prom_dd = self.thresholds.max_drawdown_promising_pct
        min_prom_sharpe = self.thresholds.min_sharpe_promising

        is_weak = (
            scorecard.trade_count < min_prom_trades
            or not cons_spread_ok
            or not sens_ok
            or not ruin_ok_promising
            or scorecard.max_drawdown_net_pct > max_prom_dd
            or scorecard.sharpe_ratio < min_prom_sharpe
        )

        if is_weak:
            weak_reasons: list[str] = []
            if scorecard.trade_count < min_prom_trades:
                weak_reasons.append(
                    f"Modest trade count ({scorecard.trade_count} < {min_prom_trades})"
                )
            if not cons_spread_ok:
                weak_reasons.append("Loses money under 1.5x conservative spread")
            if not sens_ok:
                weak_reasons.append("High parameter sensitivity / cliff effect detected")
            if not ruin_ok_promising:
                weak_reasons.append("Elevated Monte Carlo ruin risk")
            if scorecard.max_drawdown_net_pct > max_prom_dd:
                weak_reasons.append("Drawdown exceeds promising limit")
            if scorecard.sharpe_ratio < min_prom_sharpe:
                weak_reasons.append("Sharpe ratio below 0.5")

            return ProofStatusEvaluation(
                status=ProofStatus.WEAK,
                status_reason="; ".join(weak_reasons),
                is_validated=False,
                rule_gates=gates,
            )

        # 3. ROBUST Evaluation
        all_years_ok = regime_result.all_years_profitable if regime_result else True
        min_rob_trades = self.thresholds.min_trade_count_robust
        min_rob_pf = self.thresholds.min_profit_factor_robust
        min_rob_sharpe = self.thresholds.min_sharpe_robust
        max_rob_dd = self.thresholds.max_drawdown_robust_pct

        is_robust = (
            scorecard.trade_count >= min_rob_trades
            and scorecard.profit_factor >= min_rob_pf
            and scorecard.sharpe_ratio >= min_rob_sharpe
            and scorecard.max_drawdown_net_pct <= max_rob_dd
            and stress_spread_ok
            and sens_ok
            and ruin_ok_robust
            and all_years_ok
        )

        if is_robust:
            return ProofStatusEvaluation(
                status=ProofStatus.ROBUST,
                status_reason=(
                    "Strategy passed all rigorous statistical, stress, and robustness gates"
                ),
                is_validated=True,
                rule_gates=gates,
            )

        # 4. PROMISING Evaluation (Passed all Promising gates, but short of full ROBUST standards)
        promising_reasons: list[str] = []
        if scorecard.trade_count < min_rob_trades:
            promising_reasons.append(
                f"Trade count ({scorecard.trade_count}) below robust threshold ({min_rob_trades})"
            )
        if not stress_spread_ok:
            promising_reasons.append("Unprofitable under severe 2.5x spread stress")
        if not ruin_ok_robust:
            promising_reasons.append("Monte Carlo ruin probability > 1.0%")
        if not all_years_ok:
            promising_reasons.append("Experienced one or more negative calendar years")

        reason_str = (
            "; ".join(promising_reasons)
            if promising_reasons
            else "Demonstrates positive edge across tested conditions"
        )

        return ProofStatusEvaluation(
            status=ProofStatus.PROMISING,
            status_reason=f"Passed core validation: {reason_str}",
            is_validated=True,
            rule_gates=gates,
        )
