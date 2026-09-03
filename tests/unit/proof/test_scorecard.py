"""Unit tests for prooflab.proof.scorecard (ProofScorecard and EquityCurveData)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from prooflab.backtest.metrics import (
    BacktestMetricsReport,
    CostMetrics,
    ReturnsMetrics,
    RiskAdjustedMetrics,
    RiskMetrics,
    TradingMetrics,
)
from prooflab.backtest.portfolio import EquitySnapshot
from prooflab.proof.scorecard import EquityCurveData, ProofScorecard


@pytest.fixture
def sample_metrics_report() -> BacktestMetricsReport:
    return BacktestMetricsReport(
        initial_capital=100000.0,
        final_net_equity=112500.0,
        final_gross_equity=113000.0,
        start_time="2025-01-01T00:00:00+00:00",
        end_time="2025-12-31T00:00:00+00:00",
        duration_days=365.0,
        returns=ReturnsMetrics(
            total_return_gross_pct=13.0,
            total_return_net_pct=12.5,
            annualized_return_net_pct=12.5,
            cagr_pct=12.5,
        ),
        risk=RiskMetrics(
            max_drawdown_gross_dollars=4500.0,
            max_drawdown_gross_pct=4.5,
            max_drawdown_net_dollars=5000.0,
            max_drawdown_net_pct=5.0,
            avg_drawdown_net_pct=2.1,
            max_drawdown_duration_bars=18,
            annualized_volatility_pct=8.5,
            var_95_pct=-1.2,
            var_99_pct=-2.1,
            cvar_95_pct=-1.8,
            cvar_99_pct=-2.7,
        ),
        risk_adjusted=RiskAdjustedMetrics(
            sharpe_ratio=1.47,
            sortino_ratio=2.15,
            calmar_ratio=2.50,
        ),
        trading=TradingMetrics(
            trade_count=120,
            winning_trades=72,
            losing_trades=48,
            break_even_trades=0,
            win_rate_pct=60.0,
            loss_rate_pct=40.0,
            avg_win_dollars=350.0,
            avg_loss_dollars=-260.0,
            win_loss_ratio=1.35,
            profit_factor=2.02,
            expectancy_dollars=106.0,
            avg_holding_time_bars=14.5,
        ),
        costs=CostMetrics(
            total_spread_paid=300.0,
            total_commission_paid=120.0,
            total_slippage_paid=50.0,
            total_swap_paid=30.0,
            total_execution_costs=500.0,
        ),
    )


@pytest.fixture
def sample_snapshots() -> list[EquitySnapshot]:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        EquitySnapshot(
            timestamp=t0 + timedelta(days=i),
            cash=100000.0 + (i * 500.0),
            gross_equity=100000.0 + (i * 550.0),
            net_equity=100000.0 + (i * 500.0),
            unrealized_gross_pnl=0.0,
            unrealized_net_pnl=0.0,
            realized_gross_pnl=i * 550.0,
            realized_net_pnl=i * 500.0,
            margin_used=0.0,
            free_margin=100000.0 + (i * 500.0),
            margin_level_pct=None,
            drawdown_gross=0.0,
            drawdown_gross_pct=0.0,
            drawdown_net=0.0,
            drawdown_net_pct=0.0,
            open_positions=0,
        )
        for i in range(10)
    ]


def test_proof_scorecard_from_metrics(sample_metrics_report: BacktestMetricsReport) -> None:
    card = ProofScorecard.from_metrics_report(sample_metrics_report)

    assert card.initial_capital == 100000.0
    assert card.final_net_equity == 112500.0
    assert card.total_net_return_pct == 12.5
    assert card.profit_factor == 2.02
    assert card.sharpe_ratio == 1.47
    assert card.sortino_ratio == 2.15
    assert card.max_drawdown_net_pct == 5.0
    assert card.expectancy_dollars == 106.0
    assert card.win_rate_pct == 60.0
    assert card.trade_count == 120
    assert card.total_costs_paid == 500.0

    # JSON export
    json_str = card.to_json()
    assert "total_net_return_pct" in json_str
    assert "profit_factor" in json_str


def test_equity_curve_data_to_dataframe(sample_snapshots: list[EquitySnapshot]) -> None:
    data = EquityCurveData(snapshots=sample_snapshots)
    df = data.to_dataframe()

    assert not df.empty
    assert len(df) == 10
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "net_equity" in df.columns
    assert "drawdown_net_pct" in df.columns

    dd_series = data.get_drawdown_series()
    assert isinstance(dd_series, pd.Series)
    assert len(dd_series) == 10
