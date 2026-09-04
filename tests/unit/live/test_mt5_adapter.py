"""Unit tests for MockMT5Adapter and MT5Adapter."""

from __future__ import annotations

import pytest

from prooflab.data.schema import Timeframe
from prooflab.live.base import BrokerCredentials
from prooflab.live.mt5_adapter import (
    MockMT5Adapter,
    MT5Adapter,
    MT5ConnectionError,
)
from prooflab.live.orders import (
    LiveOrder,
    LiveOrderState,
)


def test_mock_mt5_connection_lifecycle() -> None:
    """Test connection lifecycle and error when not connected."""
    adapter = MockMT5Adapter(initial_balance=50000.0)
    assert not adapter.is_connected()

    # Operations when disconnected must fail
    with pytest.raises(MT5ConnectionError):
        adapter.get_account()

    assert adapter.connect()
    assert adapter.is_connected()

    account = adapter.get_account()
    assert account.balance == 50000.0
    assert account.equity == 50000.0
    assert account.free_margin == 50000.0

    adapter.disconnect()
    assert not adapter.is_connected()


def test_mock_mt5_market_data_generation() -> None:
    """Test synthetic OHLCV generation from mock adapter."""
    adapter = MockMT5Adapter()
    adapter.connect()

    df = adapter.get_market_data("EURUSD", Timeframe.M1, count=50)
    assert len(df) == 50
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert (df["symbol"] == "EURUSD").all()


def test_mock_mt5_order_fill_and_position_management() -> None:
    """Test full order submission, fill execution, and position closing."""
    adapter = MockMT5Adapter(initial_balance=10000.0, default_spread_pips=2.0)
    adapter.connect()

    order = LiveOrder(
        order_id="ORD_MT5_1",
        signal_id="SIG_MT5_1",
        symbol="EURUSD",
        side="BUY",
        quantity=1.0,
        price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0900,
    )

    filled_order = adapter.submit_order(order)
    assert filled_order.status == LiveOrderState.FILLED
    assert filled_order.broker_ticket is not None
    assert filled_order.filled_quantity == 1.0
    assert filled_order.commission == 3.50

    positions = adapter.get_positions("EURUSD")
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "EURUSD"
    assert pos.side == "BUY"
    assert pos.volume == 1.0

    # Account equity reflects margin and commission
    acc = adapter.get_account()
    assert acc.balance == 9996.50  # 10000 - 3.50 commission
    assert acc.margin > 0

    # Move market price up and check unrealized PnL
    adapter.set_price("EURUSD", 1.0900)
    updated_pos = adapter.get_positions("EURUSD")[0]
    assert updated_pos.unrealized_pnl > 0

    # Close position
    assert adapter.close_position(pos.position_id)
    assert len(adapter.get_positions("EURUSD")) == 0

    # Balance now reflects realized profit
    closed_acc = adapter.get_account()
    assert closed_acc.balance > 10000.0
    assert closed_acc.margin == 0.0


def test_mock_mt5_simulated_rejection() -> None:
    """Test simulated order rejection path."""
    adapter = MockMT5Adapter()
    adapter.connect()
    adapter.simulate_rejection = True
    adapter.rejection_reason = "Simulated: Insufficient free margin"

    order = LiveOrder(
        order_id="ORD_MT5_REJ",
        signal_id="SIG_MT5_REJ",
        symbol="GBPUSD",
        side="SELL",
        quantity=5.0,
        price=1.2700,
    )

    res = adapter.submit_order(order)
    assert res.status == LiveOrderState.REJECTED
    assert len(adapter.get_positions()) == 0


def test_mt5_adapter_import_guard() -> None:
    """Test that real MT5Adapter handles missing MetaTrader5 gracefully."""
    creds = BrokerCredentials(account_id="12345", password="pwd", server="srv")
    adapter = MT5Adapter(credentials=creds)

    # In non-MT5 environment (e.g. CI / Linux / testing), connect raises clear error
    try:
        adapter.connect()
    except (RuntimeError, MT5ConnectionError) as e:
        assert "MetaTrader5" in str(e) or "MT5" in str(e)
