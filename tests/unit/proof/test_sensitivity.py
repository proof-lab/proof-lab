"""Unit tests for prooflab.proof.sensitivity (Parameter Sensitivity & Cliff Detection)."""

import numpy as np
import pandas as pd
import pytest

from prooflab.backtest.engine import BacktestConfig
from prooflab.proof.sensitivity import (
    ParameterSensitivityAnalyzer,
    ParameterSensitivityConfig,
    ParameterSensitivityResult,
)


@pytest.fixture
def sensitivity_test_data() -> tuple[pd.DataFrame, list[dict]]:
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
        if i % 8 == 0:
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


def test_parameter_sensitivity_grid_execution(
    sensitivity_test_data: tuple[pd.DataFrame, list[dict]],
) -> None:
    df, preds = sensitivity_test_data

    config = ParameterSensitivityConfig(
        stop_multipliers=[0.9, 1.0, 1.1],
        target_multipliers=[0.9, 1.0, 1.1],
    )
    analyzer = ParameterSensitivityAnalyzer(config)

    base_bt_config = BacktestConfig(
        initial_capital=100000.0,
        default_stop_pips=25.0,
        default_target_pips=50.0,
    )

    result = analyzer.run_sensitivity_grid(base_bt_config, df, preds, symbol="EURUSD")

    assert isinstance(result, ParameterSensitivityResult)
    assert len(result.grid_cells) == 9  # 3x3 grid
    assert result.base_stop_pips == 25.0
    assert result.base_target_pips == 50.0

    # DataFrame and Pivot table
    df_grid = result.to_dataframe()
    assert len(df_grid) == 9
    assert "total_net_return_pct" in df_grid.columns

    pivot = result.to_pivot_table()
    assert pivot.shape == (3, 3)
