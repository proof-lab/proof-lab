"""Execution friction stress testing across spread and slippage scenarios."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.engine import BacktestConfig, BacktestEngine


class ExecutionStressConfig(BaseModel):
    """Configuration defining execution friction stress testing tiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    normal_spread_mult: float = Field(default=1.0, ge=0.0)
    conservative_spread_mult: float = Field(default=1.5, ge=0.0)
    stress_spread_mult: float = Field(default=2.5, ge=0.0)
    extreme_spread_mult: float = Field(default=3.5, ge=0.0)

    normal_slip_mult: float = Field(default=1.0, ge=0.0)
    conservative_slip_mult: float = Field(default=1.5, ge=0.0)
    stress_slip_mult: float = Field(default=2.0, ge=0.0)
    extreme_slip_mult: float = Field(default=3.0, ge=0.0)


class StressScenarioResult(BaseModel):
    """Execution metrics under a specific friction stress tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_name: str
    spread_multiplier: float
    slippage_multiplier: float
    total_net_return_pct: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_net_pct: float
    total_costs_paid: float
    is_profitable: bool


class ExecutionStressResult(BaseModel):
    """Aggregated stress testing evaluation across friction scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    normal_return_pct: float
    conservative_return_pct: float
    stress_return_pct: float
    extreme_return_pct: float
    scenarios: list[StressScenarioResult]

    survives_conservative: bool
    survives_stress: bool
    survives_extreme: bool
    depends_on_low_spread: bool  # True if profitable at normal friction but loses under stress

    def to_dataframe(self) -> pd.DataFrame:
        """Export scenario evaluations as a structured DataFrame."""
        if not self.scenarios:
            return pd.DataFrame()
        records = [s.model_dump(mode="python") for s in self.scenarios]
        return pd.DataFrame(records)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class ExecutionStressAnalyzer:
    """Evaluates strategy viability under elevated spread and slippage execution stress."""

    def __init__(self, config: ExecutionStressConfig | None = None) -> None:
        self.config = config or ExecutionStressConfig()

    def run_stress_tests(
        self,
        base_backtest_config: BacktestConfig,
        data: pd.DataFrame,
        predictions: list[dict[str, Any]] | pd.DataFrame,
        symbol: str = "EURUSD",
    ) -> ExecutionStressResult:
        """Execute strategy simulation across Normal, Conservative, Stress, and Extreme tiers."""
        tiers = [
            ("Normal (1.0x)", self.config.normal_spread_mult, self.config.normal_slip_mult),
            (
                "Conservative (1.5x)",
                self.config.conservative_spread_mult,
                self.config.conservative_slip_mult,
            ),
            ("Stress (2.5x)", self.config.stress_spread_mult, self.config.stress_slip_mult),
            ("Extreme (3.5x)", self.config.extreme_spread_mult, self.config.extreme_slip_mult),
        ]

        scenario_results: list[StressScenarioResult] = []

        for name, sp_mult, sl_mult in tiers:
            # Scale spread configuration
            base_costs = base_backtest_config.costs
            cons_m = (
                sp_mult if sp_mult <= 1.5 else base_costs.spread.conservative_multiplier
            )
            stress_m = sp_mult if sp_mult > 1.5 else base_costs.spread.stress_multiplier
            stressed_spread = base_costs.spread.model_copy(
                update={
                    "fixed_pips": base_costs.spread.fixed_pips * sp_mult,
                    "conservative_multiplier": cons_m,
                    "stress_multiplier": stress_m,
                }
            )
            # Scale slippage configuration
            stressed_slip = base_costs.slippage.model_copy(
                update={
                    "fixed_pips": base_costs.slippage.fixed_pips * sl_mult,
                    "atr_fraction": base_costs.slippage.atr_fraction * sl_mult,
                }
            )
            stressed_costs = base_costs.model_copy(
                update={"spread": stressed_spread, "slippage": stressed_slip}
            )
            stressed_config = base_backtest_config.model_copy(
                update={"costs": stressed_costs}
            )

            engine = BacktestEngine(stressed_config)
            res = engine.run(data, predictions, symbol=symbol)

            net_ret = res.metrics.returns.total_return_net_pct
            scenario_results.append(
                StressScenarioResult(
                    scenario_name=name,
                    spread_multiplier=sp_mult,
                    slippage_multiplier=sl_mult,
                    total_net_return_pct=round(net_ret, 4),
                    sharpe_ratio=round(res.metrics.risk_adjusted.sharpe_ratio, 4),
                    profit_factor=round(res.metrics.trading.profit_factor, 4),
                    max_drawdown_net_pct=round(
                        res.metrics.risk.max_drawdown_net_pct, 4
                    ),
                    total_costs_paid=round(
                        res.metrics.costs.total_execution_costs, 2
                    ),
                    is_profitable=net_ret > 0,
                )
            )

        norm_ret = scenario_results[0].total_net_return_pct
        cons_ret = scenario_results[1].total_net_return_pct
        stress_ret = scenario_results[2].total_net_return_pct
        ext_ret = scenario_results[3].total_net_return_pct

        surv_cons = cons_ret > 0
        surv_stress = stress_ret > 0
        surv_ext = ext_ret > 0

        # Flag dependency on unrealistically low friction
        depends_on_low_spread = (norm_ret > 0) and (not surv_cons or not surv_stress)

        return ExecutionStressResult(
            normal_return_pct=norm_ret,
            conservative_return_pct=cons_ret,
            stress_return_pct=stress_ret,
            extreme_return_pct=ext_ret,
            scenarios=scenario_results,
            survives_conservative=surv_cons,
            survives_stress=surv_stress,
            survives_extreme=surv_ext,
            depends_on_low_spread=depends_on_low_spread,
        )
