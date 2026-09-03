"""Parameter sensitivity analysis evaluating target/stop parameter stability and cliff effects."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.backtest.engine import BacktestConfig, BacktestEngine


class ParameterSensitivityConfig(BaseModel):
    """Configuration governing parameter perturbation grids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop_multipliers: list[float] = Field(
        default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2]
    )
    target_multipliers: list[float] = Field(
        default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2]
    )
    max_acceptable_degradation_pct: float = Field(default=50.0, ge=0.0)
    min_profitable_cells_pct: float = Field(default=60.0, ge=0.0, le=100.0)


class SensitivityGridCell(BaseModel):
    """Single cell in parameter perturbation grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stop_multiplier: float
    target_multiplier: float
    stop_pips: float
    target_pips: float
    total_net_return_pct: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_net_pct: float
    trade_count: int


class ParameterSensitivityResult(BaseModel):
    """Container for parameter robustness and cliff effect analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_stop_pips: float
    base_target_pips: float
    base_net_return_pct: float
    grid_cells: list[SensitivityGridCell]

    profitable_cells_pct: float
    avg_perturbed_return_pct: float
    worst_perturbed_return_pct: float
    return_std_pct: float
    has_cliff_effect: bool
    is_fragile: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Export grid cells to a structured DataFrame."""
        if not self.grid_cells:
            return pd.DataFrame()
        records = [c.model_dump(mode="python") for c in self.grid_cells]
        return pd.DataFrame(records)

    def to_pivot_table(self, value_col: str = "total_net_return_pct") -> pd.DataFrame:
        """Export grid as a 2D pivot table of stop vs target multipliers."""
        df = self.to_dataframe()
        if df.empty or value_col not in df.columns:
            return pd.DataFrame()
        return df.pivot(
            index="stop_multiplier",
            columns="target_multiplier",
            values=value_col,
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class ParameterSensitivityAnalyzer:
    """Evaluates strategy stability across adjacent stop loss and take profit parameters."""

    def __init__(self, config: ParameterSensitivityConfig | None = None) -> None:
        self.config = config or ParameterSensitivityConfig()

    def run_sensitivity_grid(
        self,
        base_backtest_config: BacktestConfig,
        data: pd.DataFrame,
        predictions: list[dict[str, Any]] | pd.DataFrame,
        symbol: str = "EURUSD",
    ) -> ParameterSensitivityResult:
        """Execute backtest grid across stop and target multiplier variations."""
        base_stop = base_backtest_config.default_stop_pips
        base_target = base_backtest_config.default_target_pips

        # 1. Run base backtest
        base_engine = BacktestEngine(base_backtest_config)
        base_res = base_engine.run(data, predictions, symbol=symbol)
        base_return = base_res.metrics.returns.total_return_net_pct

        grid_cells: list[SensitivityGridCell] = []
        return_values: list[float] = []

        # 2. Iterate through parameter grid
        for s_mult in self.config.stop_multipliers:
            for t_mult in self.config.target_multipliers:
                stop_p = base_stop * s_mult
                target_p = base_target * t_mult

                perturbed_config = base_backtest_config.model_copy(
                    update={
                        "default_stop_pips": stop_p,
                        "default_target_pips": target_p,
                    }
                )

                engine = BacktestEngine(perturbed_config)
                res = engine.run(data, predictions, symbol=symbol)

                net_ret = res.metrics.returns.total_return_net_pct
                grid_cells.append(
                    SensitivityGridCell(
                        stop_multiplier=s_mult,
                        target_multiplier=t_mult,
                        stop_pips=round(stop_p, 2),
                        target_pips=round(target_p, 2),
                        total_net_return_pct=round(net_ret, 4),
                        sharpe_ratio=round(res.metrics.risk_adjusted.sharpe_ratio, 4),
                        profit_factor=round(res.metrics.trading.profit_factor, 4),
                        max_drawdown_net_pct=round(
                            res.metrics.risk.max_drawdown_net_pct, 4
                        ),
                        trade_count=res.metrics.trading.trade_count,
                    )
                )
                return_values.append(net_ret)

        # 3. Analyze Stability & Cliff Effects
        profitable_count = sum(1 for r in return_values if r > 0)
        total_cells = len(return_values)
        profit_pct = (profitable_count / total_cells * 100.0) if total_cells > 0 else 0.0

        avg_return = float(np.mean(return_values)) if return_values else 0.0
        worst_return = float(np.min(return_values)) if return_values else 0.0
        std_return = float(np.std(return_values)) if return_values else 0.0

        # Cliff effect: if base is profitable (> 2%), but any immediate ±10% neighbor drops
        # by more than max_acceptable_degradation_pct or becomes negative.
        has_cliff = False
        if base_return > 2.0:
            for cell in grid_cells:
                if (
                    abs(cell.stop_multiplier - 1.0) <= 0.15
                    and abs(cell.target_multiplier - 1.0) <= 0.15
                ):
                    drop_pct = (base_return - cell.total_net_return_pct) / base_return * 100.0
                    exceeds_deg = drop_pct > self.config.max_acceptable_degradation_pct
                    if exceeds_deg or cell.total_net_return_pct <= 0:
                        has_cliff = True
                        break

        is_fragile = (profit_pct < self.config.min_profitable_cells_pct) or has_cliff

        return ParameterSensitivityResult(
            base_stop_pips=base_stop,
            base_target_pips=base_target,
            base_net_return_pct=base_return,
            grid_cells=grid_cells,
            profitable_cells_pct=round(profit_pct, 2),
            avg_perturbed_return_pct=round(avg_return, 4),
            worst_perturbed_return_pct=round(worst_return, 4),
            return_std_pct=round(std_return, 4),
            has_cliff_effect=has_cliff,
            is_fragile=is_fragile,
        )
