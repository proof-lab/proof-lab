"""Unit tests for prooflab.paper.execution and prooflab.paper.ledger."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prooflab.paper.consumer import LiveBar
from prooflab.paper.execution import PaperExecutionConfig, PaperExecutionEngine
from prooflab.paper.ledger import PaperTradeLedger


def test_paper_execution_buy_order_and_stop_loss_hit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = Path(tmpdir) / "trades.json"
        ledger = PaperTradeLedger(storage_path=ledger_file)
        engine = PaperExecutionEngine(
            config=PaperExecutionConfig(
                initial_capital=100000.0,
                commission_per_unit=0.0,
                slippage_pips=0.0,
            ),
            ledger=ledger,
        )

        t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
        order = engine.execute_order(
            symbol="EURUSD",
            side="BUY",
            quantity=100000.0,
            current_price=1.1000,
            timestamp=t0,
            stop_loss=1.0950,
            take_profit=1.1100,
            spread_pips=0.0,
        )

        assert order.status == "FILLED"
        assert len(engine.open_positions) == 1
        assert engine.open_positions[0].entry_price == 1.1000

        # Incoming bar hitting stop loss at 1.0940
        t1 = datetime(2026, 3, 2, 11, 0, tzinfo=UTC)
        bar = LiveBar(
            symbol="EURUSD",
            timestamp_utc=t1,
            open=1.0980,
            high=1.0990,
            low=1.0940,
            close=1.0945,
            volume=500.0,
            spread=0.0001,
        )

        closed = engine.update_on_bar(bar)
        assert len(closed) == 1
        assert closed[0].exit_reason == "STOP_LOSS"
        assert closed[0].net_pnl == -500.0  # (1.0950 - 1.1000) * 100,000 = -500.0
        assert len(engine.open_positions) == 0
        assert engine.cash == pytest.approx(99500.0)

        # Ledger check
        assert len(ledger.trades) == 1
        assert ledger_file.exists()


def test_paper_trade_ledger_dataframe_export() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = Path(tmpdir) / "trades.json"
        ledger = PaperTradeLedger(storage_path=ledger_file)
        engine = PaperExecutionEngine(ledger=ledger)

        t0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
        engine.execute_order(
            symbol="EURUSD",
            side="SELL",
            quantity=50000.0,
            current_price=1.1000,
            timestamp=t0,
            stop_loss=1.1050,
            take_profit=1.0900,
        )

        # Close manually at profit
        t1 = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
        engine.close_position_manually(
            position_id=engine.open_positions[0].position_id,
            exit_price=1.0920,
            exit_time=t1,
            reason="TAKE_PROFIT",
        )

        df = ledger.get_trades_dataframe()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"
        assert df.iloc[0]["side"] == "SELL"
        assert df.iloc[0]["status"] == "CLOSED"

        # Reload in new ledger instance
        reloaded_ledger = PaperTradeLedger(storage_path=ledger_file)
        assert len(reloaded_ledger.trades) == 1
        assert reloaded_ledger.trades[0].symbol == "EURUSD"
