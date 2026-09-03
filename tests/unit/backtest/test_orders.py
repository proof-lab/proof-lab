"""Unit tests for prooflab.backtest.orders (Order & Position Lifecycle)."""

from datetime import UTC, datetime

import pytest

from prooflab.backtest.orders import OrderRecord, Position


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 3, 2, 12, 0, tzinfo=UTC)


def test_order_record_validation(now: datetime) -> None:
    # 1. Closed order
    closed_order = OrderRecord(
        timestamp=now,
        symbol="EURUSD",
        side="BUY",
        requested_price=1.1000,
        fill_price=1.1002,
        quantity=1.0,
        stop=1.0950,
        target=1.1100,
        status="CLOSED",
        exit_reason="TAKE_PROFIT",
        fill_timestamp=now,
        exit_timestamp=now,
        exit_price=1.1100,
        gross_pnl=980.0,
        net_pnl=960.0,
    )
    assert closed_order.status == "CLOSED"
    assert closed_order.exit_reason == "TAKE_PROFIT"

    # 2. Rejected order requires rejection_reason
    with pytest.raises(ValueError, match="rejection_reason"):
        OrderRecord(
            timestamp=now,
            symbol="EURUSD",
            side="BUY",
            requested_price=1.1000,
            quantity=1.0,
            status="REJECTED",
        )


def test_long_position_intrabar_barriers(now: datetime) -> None:
    pos = Position(
        order_id="ORD-001",
        symbol="EURUSD",
        side="BUY",
        quantity=1.0,
        entry_price=1.1000,
        entry_time=now,
        stop_loss=1.0950,
        take_profit=1.1100,
        max_holding_bars=10,
    )

    # 1. Normal bar within bounds -> stays open
    assert pos.check_intrabar_exit(1.1000, 1.1050, 1.0980, 1.1020, now) is None

    # 2. Take profit hit exactly
    res_tp = pos.check_intrabar_exit(1.1020, 1.1120, 1.1010, 1.1110, now)
    assert res_tp == ("TAKE_PROFIT", 1.1100)

    # 3. Take profit on Gap Up (Open = 1.1150 > target 1.1100) -> fills at Open
    res_gap_tp = pos.check_intrabar_exit(1.1150, 1.1180, 1.1140, 1.1160, now)
    assert res_gap_tp == ("TAKE_PROFIT", 1.1150)

    # 4. Stop loss hit exactly
    res_sl = pos.check_intrabar_exit(1.0980, 1.0990, 1.0940, 1.0945, now)
    assert res_sl == ("STOP_LOSS", 1.0950)

    # 5. Stop loss on Gap Down (Open = 1.0920 < stop 1.0950) -> fills at Open
    res_gap_sl = pos.check_intrabar_exit(1.0920, 1.0930, 1.0900, 1.0910, now)
    assert res_gap_sl == ("STOP_LOSS", 1.0920)

    # 6. Ambiguous bar (both SL and TP hit in high-volatility bar) -> Conservative SL hit
    res_ambig = pos.check_intrabar_exit(1.1000, 1.1200, 1.0900, 1.1050, now)
    assert res_ambig == ("STOP_LOSS", 1.0950)


def test_short_position_intrabar_barriers(now: datetime) -> None:
    pos = Position(
        order_id="ORD-002",
        symbol="EURUSD",
        side="SELL",
        quantity=1.0,
        entry_price=1.1000,
        entry_time=now,
        stop_loss=1.1050,
        take_profit=1.0900,
    )

    # 1. Take profit hit
    res_tp = pos.check_intrabar_exit(1.0950, 1.0960, 1.0880, 1.0890, now)
    assert res_tp == ("TAKE_PROFIT", 1.0900)

    # 2. Stop loss hit
    res_sl = pos.check_intrabar_exit(1.1020, 1.1070, 1.1010, 1.1060, now)
    assert res_sl == ("STOP_LOSS", 1.1050)

    # 3. Gap up stop loss (Open = 1.1080 > stop 1.1050) -> fills at Open
    res_gap_sl = pos.check_intrabar_exit(1.1080, 1.1100, 1.1070, 1.1090, now)
    assert res_gap_sl == ("STOP_LOSS", 1.1080)


def test_time_horizon_exit(now: datetime) -> None:
    pos = Position(
        order_id="ORD-003",
        symbol="EURUSD",
        side="BUY",
        quantity=1.0,
        entry_price=1.1000,
        entry_time=now,
        max_holding_bars=3,
    )
    for _ in range(3):
        pos.increment_bar()

    res = pos.check_intrabar_exit(1.1025, 1.1040, 1.1010, 1.1030, now)
    assert res == ("TIME_HORIZON", 1.1025)


def test_position_close_and_pnl_accounting(now: datetime) -> None:
    pos = Position(
        order_id="ORD-004",
        symbol="EURUSD",
        side="BUY",
        quantity=100000.0,  # 1 standard lot
        entry_price=1.1000,
        entry_time=now,
        stop_loss=1.0950,
        take_profit=1.1100,
        commission_paid=3.50,
        entry_spread=10.0,
        entry_slippage=5.0,
    )
    pos.apply_swap(2.50)
    pos.increment_bar()

    # Unrealized PnL at 1.1050 (+50 pips = )
    unrealized = pos.calculate_unrealized_pnl(1.1050)
    assert pytest.approx(unrealized) == 500.0

    # Close at target 1.1100 (+100 pips =  gross)
    record = pos.close(
        exit_price=1.1100,
        exit_time=now,
        exit_reason="TAKE_PROFIT",
        exit_commission=3.50,
        exit_spread=10.0,
        exit_slippage=5.0,
    )

    assert record.status == "CLOSED"
    assert record.exit_reason == "TAKE_PROFIT"
    assert pytest.approx(record.gross_pnl) == 1000.0
    # Total costs: commission (7.0) + spread (20.0) + slippage (10.0) + swap (2.50) = 39.50
    assert pytest.approx(record.net_pnl) == 960.50
    assert pytest.approx(record.pnl_pips) == 100.0
    assert pos.is_closed is True
