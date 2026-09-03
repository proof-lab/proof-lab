"""Unit tests for prooflab.proof.regime (Yearly, Volatility, and Session Breakdown)."""

from datetime import UTC, datetime

import pytest

from prooflab.backtest.orders import OrderRecord
from prooflab.proof.regime import (
    RegimeAnalysisResult,
    RegimeAnalyzer,
)


@pytest.fixture
def multi_year_trades() -> list[OrderRecord]:
    return [
        # Year 2025: 2 trades (both winners -> +)
        OrderRecord(
            timestamp=datetime(2025, 4, 10, 9, 30, tzinfo=UTC),
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1000,
            fill_price=1.1000,
            exit_price=1.1060,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="TAKE_PROFIT",
            gross_pnl=600.0,
            net_pnl=550.0,
        ),
        OrderRecord(
            timestamp=datetime(2025, 8, 15, 14, 0, tzinfo=UTC),
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1050,
            fill_price=1.1050,
            exit_price=1.1100,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="TAKE_PROFIT",
            gross_pnl=500.0,
            net_pnl=450.0,
        ),
        # Year 2026: 2 trades (1 win, 1 loss -> +$100)
        OrderRecord(
            timestamp=datetime(2026, 2, 5, 2, 0, tzinfo=UTC),  # Asian session
            symbol="EURUSD",
            side="SELL",
            requested_price=1.0900,
            fill_price=1.0900,
            exit_price=1.0860,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="TAKE_PROFIT",
            gross_pnl=400.0,
            net_pnl=350.0,
        ),
        OrderRecord(
            timestamp=datetime(2026, 6, 20, 15, 0, tzinfo=UTC),  # NY session
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1000,
            fill_price=1.1000,
            exit_price=1.0980,
            quantity=100000.0,
            status="CLOSED",
            exit_reason="STOP_LOSS",
            gross_pnl=-200.0,
            net_pnl=-250.0,
        ),
    ]


def test_regime_analyzer(multi_year_trades: list[OrderRecord]) -> None:
    analyzer = RegimeAnalyzer()
    res = analyzer.analyze(multi_year_trades)

    assert isinstance(res, RegimeAnalysisResult)
    assert len(res.yearly_performance) == 2
    assert res.yearly_performance[0].year == 2025
    assert res.yearly_performance[0].trade_count == 2
    assert res.yearly_performance[0].net_pnl_dollars == 1000.0
    assert res.yearly_performance[0].is_profitable is True

    assert res.yearly_performance[1].year == 2026
    assert res.yearly_performance[1].trade_count == 2
    assert res.yearly_performance[1].net_pnl_dollars == 100.0
    assert res.yearly_performance[1].is_profitable is True

    assert res.all_years_profitable is True

    # Session breakdown
    session_names = [s.regime_name for s in res.session_regimes]
    assert any("Asian" in n for n in session_names)
    assert any("New York" in n for n in session_names)

    # DataFrame export
    df = res.to_dataframe()
    assert len(df) == 2
    assert "year" in df.columns
    assert "net_pnl_dollars" in df.columns
