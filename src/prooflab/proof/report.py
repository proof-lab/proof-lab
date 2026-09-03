"""Proof Engine orchestrator generating comprehensive evidence-based research reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.engine import BacktestConfig, BacktestEngine
from prooflab.proof.importance import FeatureImportanceAnalyzer, FeatureImportanceResult
from prooflab.proof.monte_carlo import MonteCarloConfig, MonteCarloEngine, MonteCarloResult
from prooflab.proof.regime import RegimeAnalysisResult, RegimeAnalyzer
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.sensitivity import (
    ParameterSensitivityAnalyzer,
    ParameterSensitivityConfig,
    ParameterSensitivityResult,
)
from prooflab.proof.status import (
    ProofStatusEvaluation,
    ProofStatusEvaluator,
    ProofStatusThresholds,
)
from prooflab.proof.stress import (
    ExecutionStressAnalyzer,
    ExecutionStressConfig,
    ExecutionStressResult,
)
from prooflab.proof.warnings import (
    ResearchWarning,
    ResearchWarningDetector,
)


class ProofReport(BaseModel):
    """Complete evidence-based quantitative research report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_name: str
    symbol: str
    timeframe: str
    generated_at_utc: str

    proof_status: ProofStatusEvaluation
    scorecard: ProofScorecard
    feature_importance: FeatureImportanceResult | None = None
    parameter_sensitivity: ParameterSensitivityResult | None = None
    execution_stress: ExecutionStressResult | None = None
    monte_carlo: MonteCarloResult | None = None
    regime_analysis: RegimeAnalysisResult | None = None
    warnings: list[ResearchWarning] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    def to_markdown(self) -> str:
        """Render human-readable markdown research summary."""
        sc = self.scorecard
        lines = [
            f"# Proof Lab Research Report — {self.strategy_name}",
            (
                f"**Symbol:** `{self.symbol}` | "
                f"**Timeframe:** `{self.timeframe}` | "
                f"**Generated:** `{self.generated_at_utc}`"
            ),
            "",
            f"## Proof Status: **{self.proof_status.status.value}**",
            f"> {self.proof_status.status_reason}",
            "",
            "### Performance Scorecard",
            f"- **Net Return:** {sc.total_net_return_pct:.2f}% (CAGR: {sc.cagr_pct:.2f}%)",
            f"- **Profit Factor:** {sc.profit_factor:.2f}",
            f"- **Sharpe:** {sc.sharpe_ratio:.2f} | **Sortino:** {sc.sortino_ratio:.2f}",
            (
                f"- **Max Drawdown:** {sc.max_drawdown_net_pct:.2f}% "
                f"(${sc.max_drawdown_net_dollars:,.2f})"
            ),
            f"- **Win Rate:** {sc.win_rate_pct:.1f}% ({sc.trade_count} trades)",
            f"- **Expectancy:** ${sc.expectancy_dollars:.2f} / trade",
            f"- **Total Costs:** ${sc.total_costs_paid:,.2f}",
            "",
        ]

        if self.warnings:
            lines.append("### Active Research Warnings")
            for w in self.warnings:
                lines.append(f"- **[{w.severity}] {w.code.value}:** {w.message}")
            lines.append("")

        if self.parameter_sensitivity:
            ps = self.parameter_sensitivity
            lines.append("### Parameter Sensitivity")
            lines.append(
                f"- **Profitable Grid Cells:** {ps.profitable_cells_pct:.1f}%"
            )
            lines.append(
                f"- **Cliff Effect Detected:** {ps.has_cliff_effect}"
            )
            lines.append("")

        if self.execution_stress:
            lines.append("### Execution Stress Testing")
            lines.append(
                f"- **Normal (1.0x):** {self.execution_stress.normal_return_pct:.2f}%"
            )
            lines.append(
                f"- **Conservative (1.5x):** {self.execution_stress.conservative_return_pct:.2f}%"
            )
            lines.append(
                f"- **Stress (2.5x):** {self.execution_stress.stress_return_pct:.2f}%"
            )
            lines.append(
                f"- **Depends on Low Spread:** {self.execution_stress.depends_on_low_spread}"
            )
            lines.append("")

        if self.monte_carlo:
            mc = self.monte_carlo
            lines.append("### Monte Carlo Simulation (1,000+ Runs)")
            lines.append(
                f"- **Median Return:** {mc.median_return_pct:.2f}% "
                f"(5th: {mc.percentile_5_return_pct:.2f}%, "
                f"95th: {mc.percentile_95_return_pct:.2f}%)"
            )
            lines.append(
                f"- **Median Max Drawdown:** {mc.median_max_drawdown_pct:.2f}% "
                f"(95th: {mc.percentile_95_max_drawdown_pct:.2f}%)"
            )
            lines.append(
                f"- **Probability of Loss:** {mc.probability_of_loss_pct:.1f}%"
            )
            lines.append(
                f"- **Probability of Ruin:** {mc.probability_of_ruin_pct:.1f}%"
            )
            lines.append("")

        if self.regime_analysis and self.regime_analysis.yearly_performance:
            lines.append("### Yearly Performance")
            for y in self.regime_analysis.yearly_performance:
                lines.append(
                    f"- **{y.year}:** PnL: ${y.net_pnl_dollars:,.2f} | "
                    f"Win Rate: {y.win_rate_pct:.1f}% | Trades: {y.trade_count}"
                )
            lines.append("")

        return "\n".join(lines)


class ProofEngine:
    """Central trust coordinator conducting end-to-end strategy verification."""

    def __init__(
        self,
        status_thresholds: ProofStatusThresholds | None = None,
        sensitivity_config: ParameterSensitivityConfig | None = None,
        stress_config: ExecutionStressConfig | None = None,
        monte_carlo_config: MonteCarloConfig | None = None,
    ) -> None:
        self.status_evaluator = ProofStatusEvaluator(status_thresholds)
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer(sensitivity_config)
        self.stress_analyzer = ExecutionStressAnalyzer(stress_config)
        self.monte_carlo_engine = MonteCarloEngine(monte_carlo_config)
        self.regime_analyzer = RegimeAnalyzer()

    def evaluate(
        self,
        strategy_name: str,
        base_backtest_config: BacktestConfig,
        data: pd.DataFrame,
        predictions: list[dict[str, Any]] | pd.DataFrame,
        symbol: str = "EURUSD",
        timeframe: str = "1h",
        feature_names: list[str] | None = None,
        model: Any | None = None,
        has_leakage: bool = False,
        blind_test_completed: bool = True,
        train_sharpe: float | None = None,
        class_balance_fraction: float | None = None,
    ) -> ProofReport:
        """Run full Proof Engine pipeline and produce comprehensive research report."""
        # 1. Base Backtest & Scorecard
        engine = BacktestEngine(base_backtest_config)
        backtest_res = engine.run(data, predictions, symbol=symbol)
        scorecard = ProofScorecard.from_backtest_result(backtest_res)

        # 2. Global Feature Importance (if model provided)
        importance_res: FeatureImportanceResult | None = None
        if model is not None and feature_names:
            importance_res = FeatureImportanceAnalyzer.calculate_tree_importance(
                model=model,
                feature_names=feature_names,
            )

        # 3. Parameter Sensitivity
        sensitivity_res = self.sensitivity_analyzer.run_sensitivity_grid(
            base_backtest_config=base_backtest_config,
            data=data,
            predictions=predictions,
            symbol=symbol,
        )

        # 4. Execution Friction Stress
        stress_res = self.stress_analyzer.run_stress_tests(
            base_backtest_config=base_backtest_config,
            data=data,
            predictions=predictions,
            symbol=symbol,
        )

        # 5. Monte Carlo Reshuffling
        mc_res = self.monte_carlo_engine.run_simulation(trades=backtest_res.trades)

        # 6. Regime Breakdown
        regime_res = self.regime_analyzer.analyze(
            trades=backtest_res.trades,
            data=data,
            initial_capital=base_backtest_config.initial_capital,
        )

        # 7. Proof Status Determination
        status_eval = self.status_evaluator.evaluate(
            scorecard=scorecard,
            sensitivity_result=sensitivity_res,
            stress_result=stress_res,
            monte_carlo_result=mc_res,
            regime_result=regime_res,
            has_leakage=has_leakage,
            blind_test_completed=blind_test_completed,
        )

        # 8. Research Warnings
        active_warnings = ResearchWarningDetector.detect_warnings(
            scorecard=scorecard,
            sensitivity_result=sensitivity_res,
            stress_result=stress_res,
            monte_carlo_result=mc_res,
            regime_result=regime_res,
            train_sharpe=train_sharpe,
            class_balance_fraction=class_balance_fraction,
        )

        return ProofReport(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            generated_at_utc=datetime.now(UTC).isoformat(),
            proof_status=status_eval,
            scorecard=scorecard,
            feature_importance=importance_res,
            parameter_sensitivity=sensitivity_res,
            execution_stress=stress_res,
            monte_carlo=mc_res,
            regime_analysis=regime_res,
            warnings=active_warnings,
        )
