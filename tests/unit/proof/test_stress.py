"""Unit tests for prooflab.proof.stress (Execution Friction Stress Testing)."""

import numpy as np
import pandas as pd
import pytest

from prooflab.backtest.engine import BacktestConfig
from prooflab.proof.stress import (
    ExecutionStressAnalyzer,
    ExecutionStressConfig,
    ExecutionStressResult,
)


@pytest.fixture
def stress_test_data() -> tuple[pd.DataFrame, list[dict]]:
    idx = pd.date_range("2026-03-02 00:00:00", periods=40, freq="1h", tz="UTC")
    prices = 1.1000 + np.sin(np.linspace(0, 2 * np.pi, 40)) * 0.0080

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0020,
            "low": prices - 0.0020,
            "close": prices + 0.0001,
            "volume": 1000,
            "atr": 0.0015,
        },
        index=idx,
    )

    preds = []
    for i, ts in enumerate(idx):
        if i % 6 == 0:
            preds.append({
                "timestamp": ts,
                "symbol": "EURUSD",
                "prediction": "BUY",
                "probabilities": {"BUY": 0.80, "SELL": 0.10, "IGNORE": 0.10},
            })
        else:
            preds.append({
                "timestamp": ts,
                "symbol": "EURUSD",
                "prediction": "IGNORE",
                "probabilities": {"BUY": 0.10, "SELL": 0.10, "IGNORE": 0.80},
            })

    return df, preds


def test_execution_stress_analysis(
    stress_test_data: tuple[pd.DataFrame, list[dict]],
) -> None:
    df, preds = stress_test_data

    analyzer = ExecutionStressAnalyzer(ExecutionStressConfig())
    base_bt_config = BacktestConfig(
        initial_capital=100000.0,
        default_stop_pips=25.0,
        default_target_pips=50.0,
    )

    res = analyzer.run_stress_tests(base_bt_config, df, preds, symbol="EURUSD")

    assert isinstance(res, ExecutionStressResult)
    assert len(res.scenarios) == 4
    assert res.scenarios[0].scenario_name == "Normal (1.0x)"
    assert res.scenarios[1].scenario_name == "Conservative (1.5x)"
    assert res.scenarios[2].scenario_name == "Stress (2.5x)"
    assert res.scenarios[3].scenario_name == "Extreme (3.5x)"

    # Costs monotonically increase with stress multiplier
    costs = [s.total_costs_paid for s in res.scenarios]
    assert costs[0] <= costs[1] <= costs[2] <= costs[3]

    # DataFrame export
    df_scen = res.to_dataframe()
    assert len(df_scen) == 4
    assert "spread_multiplier" in df_scen.columns
    assert "total_costs_paid" in df_scen.columns
