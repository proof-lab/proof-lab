"""Unit tests for prooflab.backtest.metrics (Quantitative Performance Metrics Suite)."""

from datetime import UTC, datetime, timedelta

import pytest

from prooflab.backtest.metrics import (
    BacktestMetricsReport,
    calculate_backtest_metrics,
)
from prooflab.backtest.orders import OrderRecord
from prooflab.backtest.portfolio import EquitySnapshot


@pytest.fixture
def sample_trades_and_equity() -> tuple[list[OrderRecord], list[EquitySnapshot]]:
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

    # 4 sample trades (2 winners: +, +; 2 losers: -, -)
    trades = [
        OrderRecord(
            timestamp=base_time + timedelta(days=1),
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1000,
            fill_price=1.1000,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="TAKE_PROFIT",
            fill_timestamp=base_time + timedelta(days=1),
            exit_timestamp=base_time + timedelta(days=2),
            exit_price=1.1100,
            gross_pnl=1000.0,
            net_pnl=950.0,  #  -  costs
            spread=20.0,
            commission=10.0,
            slippage=10.0,
            swap=10.0,
            metadata={"bars_held": 24},
        ),
        OrderRecord(
            timestamp=base_time + timedelta(days=3),
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1050,
            fill_price=1.1050,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="STOP_LOSS",
            fill_timestamp=base_time + timedelta(days=3),
            exit_timestamp=base_time + timedelta(days=4),
            exit_price=1.1010,
            gross_pnl=-400.0,
            net_pnl=-450.0,  # - -  costs
            spread=20.0,
            commission=10.0,
            slippage=10.0,
            swap=10.0,
            metadata={"bars_held": 12},
        ),
        OrderRecord(
            timestamp=base_time + timedelta(days=5),
            symbol="EURUSD",
            side="SELL",
            requested_price=1.1000,
            fill_price=1.1000,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="TAKE_PROFIT",
            fill_timestamp=base_time + timedelta(days=5),
            exit_timestamp=base_time + timedelta(days=6),
            exit_price=1.0950,
            gross_pnl=500.0,
            net_pnl=450.0,
            spread=20.0,
            commission=10.0,
            slippage=10.0,
            swap=10.0,
            metadata={"bars_held": 18},
        ),
        OrderRecord(
            timestamp=base_time + timedelta(days=7),
            symbol="EURUSD",
            side="SELL",
            requested_price=1.0950,
            fill_price=1.0950,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="STOP_LOSS",
            fill_timestamp=base_time + timedelta(days=7),
            exit_timestamp=base_time + timedelta(days=8),
            exit_price=1.0970,
            gross_pnl=-200.0,
            net_pnl=-250.0,
            spread=20.0,
            commission=10.0,
            slippage=10.0,
            swap=10.0,
            metadata={"bars_held": 6},
        ),
    ]

    # Equity snapshots over 10 days
    equity_values = [100000.0, 100950.0, 100500.0, 100950.0, 100700.0]
    snapshots = []
    for i, eq in enumerate(equity_values):
        snapshots.append(
            EquitySnapshot(
                timestamp=base_time + timedelta(days=i * 2),
                cash=eq,
                gross_equity=eq + 100.0,
                net_equity=eq,
                unrealized_gross_pnl=0.0,
                unrealized_net_pnl=0.0,
                realized_gross_pnl=eq - 100000.0,
                realized_net_pnl=eq - 100000.0,
                margin_used=0.0,
                free_margin=eq,
                margin_level_pct=None,
                drawdown_gross=0.0,
                drawdown_gross_pct=0.0,
                drawdown_net=max(0.0, 100950.0 - eq),
                drawdown_net_pct=max(0.0, (100950.0 - eq) / 100950.0 * 100.0),
                open_positions=0,
            )
        )

    return trades, snapshots


def test_calculate_backtest_metrics(
    sample_trades_and_equity: tuple[list[OrderRecord], list[EquitySnapshot]],
) -> None:
    trades, snapshots = sample_trades_and_equity

    report = calculate_backtest_metrics(
        trades=trades,
        equity_snapshots=snapshots,
        initial_capital=100000.0,
    )

    assert isinstance(report, BacktestMetricsReport)

    # 1. Trading stats: 4 trades, 2 wins (50%), 2 losses (50%)
    assert report.trading.trade_count == 4
    assert report.trading.winning_trades == 2
    assert report.trading.losing_trades == 2
    assert report.trading.win_rate_pct == 50.0
    assert report.trading.loss_rate_pct == 50.0
    # Avg win: (950 + 450) / 2 = 700.0
    assert pytest.approx(report.trading.avg_win_dollars) == 700.0
    # Avg loss: (-450 + -250) / 2 = -350.0
    assert pytest.approx(report.trading.avg_loss_dollars) == -350.0
    # Win/loss ratio = 700 / 350 = 2.0
    assert pytest.approx(report.trading.win_loss_ratio) == 2.0
    # Profit factor: (950 + 450) / (450 + 250) = 1400 / 700 = 2.0
    assert pytest.approx(report.trading.profit_factor) == 2.0
    # Expectancy: (0.5 * 700) + (0.5 * -350) = 350 - 175 = .00
    assert pytest.approx(report.trading.expectancy_dollars) == 175.00
    # Avg holding time: (24 + 12 + 18 + 6) / 4 = 15.0 bars
    assert pytest.approx(report.trading.avg_holding_time_bars) == 15.0

    # 2. Costs stats
    assert pytest.approx(report.costs.total_spread_paid) == 80.0
    assert pytest.approx(report.costs.total_commission_paid) == 40.0
    assert pytest.approx(report.costs.total_slippage_paid) == 40.0
    assert pytest.approx(report.costs.total_swap_paid) == 40.0
    assert pytest.approx(report.costs.total_execution_costs) == 200.0

    # 3. Risk stats
    assert report.risk.max_drawdown_net_dollars > 0.0
    assert report.risk.max_drawdown_net_pct > 0.0

    # 4. JSON serialization
    json_str = report.to_json()
    assert "total_return_net_pct" in json_str
    assert "sharpe_ratio" in json_str
