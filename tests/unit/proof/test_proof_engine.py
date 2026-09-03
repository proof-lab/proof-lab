"""Unit and integration tests for prooflab.proof.report (ProofEngine & ProofReport)."""

import numpy as np
import pandas as pd
import pytest

from prooflab.backtest.engine import BacktestConfig
from prooflab.proof.report import ProofEngine, ProofReport
from prooflab.proof.scorecard import ProofScorecard
from prooflab.proof.warnings import ResearchWarningCode


@pytest.fixture
def proof_pipeline_data() -> tuple[pd.DataFrame, list[dict]]:
    idx = pd.date_range("2026-03-02 00:00:00", periods=60, freq="1h", tz="UTC")
    prices = 1.1000 + np.sin(np.linspace(0, 3 * np.pi, 60)) * 0.0080

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0025,
            "low": prices - 0.0025,
            "close": prices + 0.0002,
            "volume": 1000,
            "atr": 0.0015,
        },
        index=idx,
    )

    preds = []
    for i, ts in enumerate(idx):
        if i % 5 == 0:
            preds.append({
                "timestamp": ts,
                "symbol": "EURUSD",
                "prediction": "BUY",
                "probabilities": {"BUY": 0.85, "SELL": 0.05, "IGNORE": 0.10},
            })
        else:
            preds.append({
                "timestamp": ts,
                "symbol": "EURUSD",
                "prediction": "IGNORE",
                "probabilities": {"BUY": 0.10, "SELL": 0.10, "IGNORE": 0.80},
            })

    return df, preds


def test_proof_engine_full_evaluation(
    proof_pipeline_data: tuple[pd.DataFrame, list[dict]],
) -> None:
    df, preds = proof_pipeline_data

    engine = ProofEngine()
    bt_config = BacktestConfig(
        initial_capital=100000.0,
        default_stop_pips=20.0,
        default_target_pips=40.0,
    )

    report = engine.evaluate(
        strategy_name="TrendFollowing_EURUSD_v1",
        base_backtest_config=bt_config,
        data=df,
        predictions=preds,
        symbol="EURUSD",
        timeframe="1h",
        has_leakage=False,
        blind_test_completed=True,
    )

    assert isinstance(report, ProofReport)
    assert report.strategy_name == "TrendFollowing_EURUSD_v1"
    assert report.symbol == "EURUSD"
    assert isinstance(report.scorecard, ProofScorecard)
    assert report.proof_status is not None
    assert report.parameter_sensitivity is not None
    assert report.execution_stress is not None
    assert report.monte_carlo is not None
    assert report.regime_analysis is not None

    # Markdown export
    md = report.to_markdown()
    assert "# Proof Lab Research Report" in md
    assert "Proof Status" in md
    assert "Performance Scorecard" in md

    # JSON export
    json_str = report.to_json()
    assert "strategy_name" in json_str
    assert "proof_status" in json_str


def test_research_warning_triggers(
    proof_pipeline_data: tuple[pd.DataFrame, list[dict]],
) -> None:
    df, preds = proof_pipeline_data
    engine = ProofEngine()
    bt_config = BacktestConfig(
        initial_capital=100000.0,
        default_stop_pips=20.0,
        default_target_pips=40.0,
    )

    report = engine.evaluate(
        strategy_name="Fragile_Test_Strategy",
        base_backtest_config=bt_config,
        data=df,
        predictions=preds,
        train_sharpe=3.5,  # High in-sample Sharpe vs lower test Sharpe -> trigger OOS degradation
        class_balance_fraction=0.05,  # Imbalance trigger
    )

    codes = [w.code for w in report.warnings]
    assert ResearchWarningCode.LOW_TRADE_COUNT in codes
    assert ResearchWarningCode.HIGH_CLASS_IMBALANCE in codes
    assert ResearchWarningCode.HIGH_OUT_OF_SAMPLE_DEGRADATION in codes
