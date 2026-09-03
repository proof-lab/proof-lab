"""Unit tests for prooflab.backtest.portfolio (Position Sizing & Portfolio Accounting)."""

from datetime import UTC, datetime

import pytest

from prooflab.backtest.orders import Position
from prooflab.backtest.portfolio import (
    BrokerLimitsConfig,
    EquitySnapshot,
    PortfolioAccountant,
    PositionSizer,
)


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 3, 2, 12, 0, tzinfo=UTC)


def test_broker_limits_validation() -> None:
    # Invalid: min_lot > max_lot
    with pytest.raises(ValueError, match="min_lot_size cannot exceed"):
        BrokerLimitsConfig(min_lot_size=10.0, max_lot_size=5.0)

    # Invalid: stop_out > margin_call
    with pytest.raises(ValueError, match="stop_out_level_pct cannot exceed"):
        BrokerLimitsConfig(stop_out_level_pct=120.0, margin_call_level_pct=100.0)


def test_position_sizer_risk_math() -> None:
    sizer = PositionSizer(
        BrokerLimitsConfig(
            lot_unit_size=100000.0,
            min_lot_size=0.01,
            max_lot_size=50.0,
            lot_step=0.01,
        )
    )

    # Equity: $100,000, Risk: 1% ($1,000), Entry: 1.1000, Stop: 1.0950 (50 pips = 0.0050)
    # Target units = $1,000 / 0.0050 = 200,000 units (2.00 standard lots)
    units = sizer.calculate_position_size(
        account_equity=100000.0,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        risk_per_trade_pct=0.01,
    )
    assert pytest.approx(units) == 200000.0


def test_position_sizer_clamping_and_stepping() -> None:
    sizer = PositionSizer(
        BrokerLimitsConfig(
            lot_unit_size=100000.0,
            min_lot_size=0.01,
            max_lot_size=5.0,
            lot_step=0.01,
        )
    )

    # Small account ($500) -> clamped to min_lot 0.01 = 1,000
    units_min = sizer.calculate_position_size(
        account_equity=500.0,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        risk_per_trade_pct=0.01,
    )
    assert pytest.approx(units_min) == 1000.0

    # Large account (,000,000) -> clamped to max_lot 5.0 = 500,000
    units_max = sizer.calculate_position_size(
        account_equity=10000000.0,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        risk_per_trade_pct=0.01,
    )
    assert pytest.approx(units_max) == 500000.0


def test_portfolio_accountant_lifecycle_and_mark_to_market(now: datetime) -> None:
    accountant = PortfolioAccountant(initial_capital=100000.0)

    # 1. Initial state
    assert accountant.cash == 100000.0
    can_open, reason = accountant.can_open_position(100000.0, 1.1000, 0)
    assert can_open is True
    assert reason is None

    # Max positions limit check
    can_open_limit, reason_limit = accountant.can_open_position(100000.0, 1.1000, 5)
    assert can_open_limit is False
    assert "Maximum open positions limit reached" in (reason_limit or "")

    # 2. Open Position and Take Snapshot
    pos = Position(
        order_id="ORD-001",
        symbol="EURUSD",
        side="BUY",
        quantity=100000.0,
        entry_price=1.1000,
        entry_time=now,
        stop_loss=1.0950,
        take_profit=1.1100,
    )
    pos.apply_swap(5.00)

    # Price moves to 1.1050 (+50 pips = + gross)
    snap = accountant.update_snapshot(
        timestamp=now,
        open_positions=[pos],
        current_prices={"EURUSD": 1.1050},
    )

    assert isinstance(snap, EquitySnapshot)
    assert pytest.approx(snap.unrealized_gross_pnl) == 500.0
    assert pytest.approx(snap.unrealized_net_pnl) == 495.0  #  -  swap
    assert pytest.approx(snap.gross_equity) == 100500.0
    assert pytest.approx(snap.net_equity) == 100495.0
    assert snap.open_positions == 1
    assert snap.drawdown_gross == 0.0

    # 3. Close Trade with  gross PnL, .50 total costs -> .50 net
    accountant.record_trade_close(
        gross_pnl=1000.0,
        net_pnl=960.50,
        commission=7.00,
        spread=20.0,
        slippage=10.0,
        swap=2.50,
    )

    assert pytest.approx(accountant.cash) == 100960.50
    assert pytest.approx(accountant.realized_net_pnl) == 960.50
    assert pytest.approx(accountant.total_spread_paid) == 20.0

    # 4. DataFrame export
    df = accountant.get_equity_curve_dataframe()
    assert not df.empty
    assert "net_equity" in df.columns
    assert "gross_equity" in df.columns
