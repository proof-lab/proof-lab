"""Comprehensive quantitative backtest performance and risk metrics suite.

Calculates returns (Total, Annualized, CAGR), risk (Drawdown, Volatility, VaR, CVaR),
risk-adjusted ratios (Sharpe, Sortino, Calmar), trade stats (Win rate, Profit factor,
Expectancy), and complete execution friction breakdowns.
"""

from __future__ import annotations

import json
import math

import numpy as np
from pydantic import BaseModel, ConfigDict

from prooflab.backtest.orders import OrderRecord
from prooflab.backtest.portfolio import EquitySnapshot


class ReturnsMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    total_return_gross_pct: float
    total_return_net_pct: float
    annualized_return_net_pct: float
    cagr_pct: float


class RiskMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_drawdown_gross_dollars: float
    max_drawdown_gross_pct: float
    max_drawdown_net_dollars: float
    max_drawdown_net_pct: float
    avg_drawdown_net_pct: float
    max_drawdown_duration_bars: int
    annualized_volatility_pct: float
    var_95_pct: float
    var_99_pct: float
    cvar_95_pct: float
    cvar_99_pct: float


class RiskAdjustedMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


class TradingMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trade_count: int
    winning_trades: int
    losing_trades: int
    break_even_trades: int
    win_rate_pct: float
    loss_rate_pct: float
    avg_win_dollars: float
    avg_loss_dollars: float
    win_loss_ratio: float
    profit_factor: float
    expectancy_dollars: float
    avg_holding_time_bars: float


class CostMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    total_spread_paid: float
    total_commission_paid: float
    total_slippage_paid: float
    total_swap_paid: float
    total_execution_costs: float


class BacktestMetricsReport(BaseModel):
    """Complete quantitative performance report for backtested strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float
    final_net_equity: float
    final_gross_equity: float
    start_time: str
    end_time: str
    duration_days: float

    returns: ReturnsMetrics
    risk: RiskMetrics
    risk_adjusted: RiskAdjustedMetrics
    trading: TradingMetrics
    costs: CostMetrics

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent)


def calculate_backtest_metrics(
    trades: list[OrderRecord],
    equity_snapshots: list[EquitySnapshot],
    initial_capital: float = 100000.0,
    risk_free_rate_pct: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestMetricsReport:
    """Compute the full quantitative metrics suite from simulated trades and equity snapshots."""
    if not equity_snapshots:
        raise ValueError("Cannot calculate metrics without equity snapshots.")

    first_snap = equity_snapshots[0]
    last_snap = equity_snapshots[-1]

    # 1. Timeline & Duration
    start_ts = first_snap.timestamp
    end_ts = last_snap.timestamp
    delta_seconds = max(1.0, (end_ts - start_ts).total_seconds())
    duration_days = max(1.0 / 24.0, delta_seconds / 86400.0)
    duration_years = duration_days / 365.25

    # 2. Returns Calculations
    final_net = last_snap.net_equity
    final_gross = last_snap.gross_equity

    total_return_net = (final_net - initial_capital) / initial_capital
    total_return_gross = (final_gross - initial_capital) / initial_capital

    if duration_years > 0 and final_net > 0:
        cagr = (final_net / initial_capital) ** (1.0 / duration_years) - 1.0
        annualized_return_net = (1.0 + total_return_net) ** (1.0 / duration_years) - 1.0
    else:
        cagr = total_return_net
        annualized_return_net = total_return_net

    # 3. Periodic Returns & Volatility
    net_equity_series = np.array([s.net_equity for s in equity_snapshots])
    if len(net_equity_series) > 1:
        periodic_returns = np.diff(net_equity_series) / net_equity_series[:-1]
    else:
        periodic_returns = np.array([0.0])

    if len(periodic_returns) > 1 and np.std(periodic_returns) > 1e-12:
        vol_periodic = float(np.std(periodic_returns, ddof=1))
        annualized_vol = vol_periodic * math.sqrt(periods_per_year)

        # Downside deviation (for Sortino)
        downside_returns = periodic_returns[periodic_returns < 0]
        if len(downside_returns) > 0:
            downside_dev = (
                float(np.sqrt(np.mean(downside_returns ** 2))) * math.sqrt(periods_per_year)
            )
        else:
            downside_dev = 1e-6

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = float(np.percentile(periodic_returns, 5.0))
        var_99 = float(np.percentile(periodic_returns, 1.0))

        cvar_95_tail = periodic_returns[periodic_returns <= var_95]
        cvar_95 = float(np.mean(cvar_95_tail)) if len(cvar_95_tail) > 0 else var_95

        cvar_99_tail = periodic_returns[periodic_returns <= var_99]
        cvar_99 = float(np.mean(cvar_99_tail)) if len(cvar_99_tail) > 0 else var_99
    else:
        annualized_vol = 0.0
        downside_dev = 1e-6
        var_95 = 0.0
        var_99 = 0.0
        cvar_95 = 0.0
        cvar_99 = 0.0

    # 4. Drawdowns & Duration
    max_dd_gross_dollars = max((s.drawdown_gross for s in equity_snapshots), default=0.0)
    max_dd_gross_pct = max((s.drawdown_gross_pct for s in equity_snapshots), default=0.0)
    max_dd_net_dollars = max((s.drawdown_net for s in equity_snapshots), default=0.0)
    max_dd_net_pct = max((s.drawdown_net_pct for s in equity_snapshots), default=0.0)

    dd_net_pcts = [s.drawdown_net_pct for s in equity_snapshots if s.drawdown_net_pct > 0]
    avg_dd_net_pct = float(np.mean(dd_net_pcts)) if dd_net_pcts else 0.0

    # Drawdown duration in bars
    max_dd_duration_bars = 0
    current_dd_bars = 0
    for s in equity_snapshots:
        if s.drawdown_net > 0.01:
            current_dd_bars += 1
            max_dd_duration_bars = max(max_dd_duration_bars, current_dd_bars)
        else:
            current_dd_bars = 0

    # 5. Risk-Adjusted Metrics
    rf = risk_free_rate_pct / 100.0
    excess_return = annualized_return_net - rf

    sharpe = (excess_return / annualized_vol) if annualized_vol > 1e-9 else 0.0
    sortino = (excess_return / downside_dev) if downside_dev > 1e-9 else 0.0

    calmar_denom = max_dd_net_pct / 100.0
    calmar = (annualized_return_net / calmar_denom) if calmar_denom > 1e-6 else 0.0

    # 6. Trading Statistics
    closed_trades = [t for t in trades if t.status == "CLOSED"]
    trade_count = len(closed_trades)

    winning = [t for t in closed_trades if t.net_pnl > 0]
    losing = [t for t in closed_trades if t.net_pnl < 0]
    break_even = [t for t in closed_trades if t.net_pnl == 0]

    win_count = len(winning)
    loss_count = len(losing)
    be_count = len(break_even)

    win_rate = (win_count / trade_count * 100.0) if trade_count > 0 else 0.0
    loss_rate = (loss_count / trade_count * 100.0) if trade_count > 0 else 0.0

    avg_win = float(np.mean([t.net_pnl for t in winning])) if winning else 0.0
    avg_loss = float(np.mean([t.net_pnl for t in losing])) if losing else 0.0

    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-9 else 0.0

    gross_gains = sum(t.net_pnl for t in winning)
    gross_losses = sum(abs(t.net_pnl) for t in losing)
    profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-9 else (
        float("inf") if gross_gains > 0 else 0.0
    )
    if math.isinf(profit_factor):
        profit_factor = 999.99

    expectancy = (
        (win_rate / 100.0 * avg_win) + (loss_rate / 100.0 * avg_loss)
    ) if trade_count > 0 else 0.0

    holding_times = [
        int(t.metadata.get("bars_held", 0)) for t in closed_trades if t.metadata
    ]
    avg_holding = float(np.mean(holding_times)) if holding_times else 0.0

    # 7. Costs Aggregation
    tot_spread = sum(t.spread for t in closed_trades)
    tot_comm = sum(t.commission for t in closed_trades)
    tot_slip = sum(t.slippage for t in closed_trades)
    tot_swap = sum(t.swap for t in closed_trades)
    tot_exec = tot_spread + tot_comm + tot_slip + tot_swap

    return BacktestMetricsReport(
        initial_capital=initial_capital,
        final_net_equity=final_net,
        final_gross_equity=final_gross,
        start_time=start_ts.isoformat(),
        end_time=end_ts.isoformat(),
        duration_days=round(duration_days, 2),
        returns=ReturnsMetrics(
            total_return_gross_pct=round(total_return_gross * 100.0, 4),
            total_return_net_pct=round(total_return_net * 100.0, 4),
            annualized_return_net_pct=round(annualized_return_net * 100.0, 4),
            cagr_pct=round(cagr * 100.0, 4),
        ),
        risk=RiskMetrics(
            max_drawdown_gross_dollars=round(max_dd_gross_dollars, 2),
            max_drawdown_gross_pct=round(max_dd_gross_pct, 4),
            max_drawdown_net_dollars=round(max_dd_net_dollars, 2),
            max_drawdown_net_pct=round(max_dd_net_pct, 4),
            avg_drawdown_net_pct=round(avg_dd_net_pct, 4),
            max_drawdown_duration_bars=max_dd_duration_bars,
            annualized_volatility_pct=round(annualized_vol * 100.0, 4),
            var_95_pct=round(var_95 * 100.0, 4),
            var_99_pct=round(var_99 * 100.0, 4),
            cvar_95_pct=round(cvar_95 * 100.0, 4),
            cvar_99_pct=round(cvar_99 * 100.0, 4),
        ),
        risk_adjusted=RiskAdjustedMetrics(
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
        ),
        trading=TradingMetrics(
            trade_count=trade_count,
            winning_trades=win_count,
            losing_trades=loss_count,
            break_even_trades=be_count,
            win_rate_pct=round(win_rate, 2),
            loss_rate_pct=round(loss_rate, 2),
            avg_win_dollars=round(avg_win, 2),
            avg_loss_dollars=round(avg_loss, 2),
            win_loss_ratio=round(win_loss_ratio, 4),
            profit_factor=round(profit_factor, 4),
            expectancy_dollars=round(expectancy, 2),
            avg_holding_time_bars=round(avg_holding, 2),
        ),
        costs=CostMetrics(
            total_spread_paid=round(tot_spread, 2),
            total_commission_paid=round(tot_comm, 2),
            total_slippage_paid=round(tot_slip, 2),
            total_swap_paid=round(tot_swap, 2),
            total_execution_costs=round(tot_exec, 2),
        ),
    )
