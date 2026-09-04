"""Proof Engine strategy performance scorecard and equity curve extraction."""

from __future__ import annotations

import json

import pandas as pd
from pydantic import BaseModel, ConfigDict

from prooflab.backtest.engine import BacktestResult
from prooflab.backtest.metrics import BacktestMetricsReport
from prooflab.backtest.portfolio import EquitySnapshot


class ProofScorecard(BaseModel):
    """Core evaluation scorecard summarizing strategy viability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float
    final_net_equity: float
    total_net_return_pct: float
    annualized_return_pct: float
    cagr_pct: float

    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_net_pct: float
    max_drawdown_net_dollars: float

    expectancy_dollars: float
    win_rate_pct: float
    loss_rate_pct: float
    trade_count: int
    winning_trades: int
    losing_trades: int

    total_costs_paid: float
    total_spread_paid: float
    total_commission_paid: float
    total_slippage_paid: float
    total_swap_paid: float

    @classmethod
    def from_metrics_report(cls, metrics: BacktestMetricsReport) -> ProofScorecard:
        """Construct scorecard from standard BacktestMetricsReport."""
        return cls(
            initial_capital=metrics.initial_capital,
            final_net_equity=metrics.final_net_equity,
            total_net_return_pct=metrics.returns.total_return_net_pct,
            annualized_return_pct=metrics.returns.annualized_return_net_pct,
            cagr_pct=metrics.returns.cagr_pct,
            profit_factor=metrics.trading.profit_factor,
            sharpe_ratio=metrics.risk_adjusted.sharpe_ratio,
            sortino_ratio=metrics.risk_adjusted.sortino_ratio,
            calmar_ratio=metrics.risk_adjusted.calmar_ratio,
            max_drawdown_net_pct=metrics.risk.max_drawdown_net_pct,
            max_drawdown_net_dollars=metrics.risk.max_drawdown_net_dollars,
            expectancy_dollars=metrics.trading.expectancy_dollars,
            win_rate_pct=metrics.trading.win_rate_pct,
            loss_rate_pct=metrics.trading.loss_rate_pct,
            trade_count=metrics.trading.trade_count,
            winning_trades=metrics.trading.winning_trades,
            losing_trades=metrics.trading.losing_trades,
            total_costs_paid=metrics.costs.total_execution_costs,
            total_spread_paid=metrics.costs.total_spread_paid,
            total_commission_paid=metrics.costs.total_commission_paid,
            total_slippage_paid=metrics.costs.total_slippage_paid,
            total_swap_paid=metrics.costs.total_swap_paid,
        )

    @classmethod
    def from_backtest_result(cls, result: BacktestResult) -> ProofScorecard:
        """Construct scorecard directly from BacktestResult."""
        return cls.from_metrics_report(result.metrics)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


class EquityCurveData(BaseModel):
    """Container for time-indexed equity snapshots and underwater drawdown series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshots: list[EquitySnapshot]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to a time-indexed pandas DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        records = [s.model_dump(mode="python") for s in self.snapshots]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df

    def get_drawdown_series(self) -> pd.Series:
        """Return net drawdown percentage series indexed by timestamp."""
        df = self.to_dataframe()
        if df.empty or "drawdown_net_pct" not in df.columns:
            return pd.Series(dtype="float64")
        return df["drawdown_net_pct"]
