"""Unit and integration tests for prooflab.backtest.engine (End-to-End Backtesting Engine)."""


import numpy as np
import pandas as pd
import pytest

from prooflab.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    # 50 hourly bars
    idx = pd.date_range("2026-03-02 00:00:00", periods=50, freq="1h", tz="UTC")
    prices = 1.1000 + np.sin(np.linspace(0, 3 * np.pi, 50)) * 0.0100

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0015,
            "low": prices - 0.0015,
            "close": prices + 0.0002,
            "volume": 1000,
            "atr": 0.0020,
        },
        index=idx,
    )
    return df


@pytest.fixture
def sample_predictions(synthetic_ohlcv: pd.DataFrame) -> list[dict]:
    preds = []
    for i, ts in enumerate(synthetic_ohlcv.index):
        if i % 10 == 0:
            pred_dir = "BUY"
            probs = {"BUY": 0.75, "SELL": 0.05, "IGNORE": 0.20}
        elif i % 10 == 5:
            pred_dir = "SELL"
            probs = {"BUY": 0.05, "SELL": 0.70, "IGNORE": 0.25}
        else:
            pred_dir = "IGNORE"
            probs = {"BUY": 0.10, "SELL": 0.10, "IGNORE": 0.80}

        preds.append(
            {
                "timestamp": ts,
                "symbol": "EURUSD",
                "prediction": pred_dir,
                "probabilities": probs,
                "model_votes": {"xgboost": pred_dir, "neural": pred_dir},
            }
        )
    return preds


def test_backtest_engine_end_to_end(
    synthetic_ohlcv: pd.DataFrame,
    sample_predictions: list[dict],
) -> None:
    config = BacktestConfig(
        initial_capital=100000.0,
        risk_per_trade_pct=0.01,
        default_stop_pips=30.0,
        default_target_pips=60.0,
    )
    engine = BacktestEngine(config)

    result = engine.run(synthetic_ohlcv, sample_predictions, symbol="EURUSD")

    assert isinstance(result, BacktestResult)
    assert len(result.trades) > 0
    assert len(result.signals) == 50
    assert len(result.equity_snapshots) == 50

    # Equity curve DataFrame verification
    df_eq = result.get_equity_curve()
    assert len(df_eq) == 50
    assert "net_equity" in df_eq.columns
    assert "gross_equity" in df_eq.columns
    assert "drawdown_net_pct" in df_eq.columns

    # Trades DataFrame verification
    df_trades = result.get_trades_dataframe()
    assert not df_trades.empty
    assert "exit_reason" in df_trades.columns
    assert "net_pnl" in df_trades.columns

    # Metrics validation
    assert result.metrics.initial_capital == 100000.0
    assert result.metrics.trading.trade_count == len(result.trades)
    assert result.metrics.costs.total_execution_costs > 0.0

    # JSON exportability
    json_export = result.to_json()
    assert "metrics" in json_export


def test_backtest_engine_rejection_on_insufficient_margin(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    # Account with only  trying to trade standard lots
    config = BacktestConfig(
        initial_capital=10.0,
        risk_per_trade_pct=0.01,
    )
    engine = BacktestEngine(config)

    predictions = [
        {
            "timestamp": synthetic_ohlcv.index[0],
            "symbol": "EURUSD",
            "prediction": "BUY",
            "probabilities": {"BUY": 0.80, "SELL": 0.10, "IGNORE": 0.10},
        }
    ]

    result = engine.run(synthetic_ohlcv, predictions, symbol="EURUSD")
    # Order should be rejected due to zero quantity / insufficient margin
    assert any(t.status == "REJECTED" for t in result.trades)


def test_backtest_engine_rejects_naive_index(
    synthetic_ohlcv: pd.DataFrame,
    sample_predictions: list[dict],
) -> None:
    # Remove timezone from DataFrame index
    bad_df = synthetic_ohlcv.copy()
    bad_df.index = bad_df.index.tz_localize(None)

    engine = BacktestEngine()
    with pytest.raises(ValueError, match="UTC DatetimeIndex"):
        engine.run(bad_df, sample_predictions)
