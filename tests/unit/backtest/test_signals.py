"""Unit tests for prooflab.backtest.signals (Signal Engine & Filters)."""

import copy
from datetime import UTC, datetime

import pytest

from prooflab.backtest.signals import SignalEngine, SignalFilterConfig, TradeSignal


@pytest.fixture
def valid_prediction() -> dict:
    return {
        "timestamp": datetime(2026, 3, 2, 10, 0, tzinfo=UTC),  # Monday 10:00 UTC
        "symbol": "EURUSD",
        "prediction": "BUY",
        "probabilities": {"BUY": 0.72, "SELL": 0.08, "IGNORE": 0.20},
        "model_votes": {"xgboost": "BUY", "neural_net": "BUY", "svm": "IGNORE"},
    }


def test_signal_engine_happy_path(valid_prediction: dict) -> None:
    engine = SignalEngine(SignalFilterConfig(min_probability=0.60))
    signal = engine.evaluate_prediction(valid_prediction)

    assert isinstance(signal, TradeSignal)
    assert signal.is_actionable is True
    assert signal.direction == "BUY"
    assert signal.calibrated_probability == 0.72
    assert len(signal.rejection_reasons) == 0
    assert signal.filter_audit["min_probability_met"] is True


def test_prediction_immutability(valid_prediction: dict) -> None:
    original_pred = copy.deepcopy(valid_prediction)
    engine = SignalEngine(SignalFilterConfig())

    engine.evaluate_prediction(valid_prediction)
    assert valid_prediction == original_pred

    # Batch test immutability
    batch = [valid_prediction, copy.deepcopy(valid_prediction)]
    engine.evaluate_batch(batch)
    assert batch[0] == original_pred


def test_ignore_prediction_is_never_actionable(valid_prediction: dict) -> None:
    valid_prediction["prediction"] = "IGNORE"
    engine = SignalEngine()
    signal = engine.evaluate_prediction(valid_prediction)

    assert signal.is_actionable is False
    assert signal.direction == "IGNORE"
    assert "Prediction is IGNORE" in signal.rejection_reasons


def test_min_probability_filter(valid_prediction: dict) -> None:
    # Requires 0.80 probability, prediction only has 0.72
    engine = SignalEngine(SignalFilterConfig(min_probability=0.80))
    signal = engine.evaluate_prediction(valid_prediction)

    assert signal.is_actionable is False
    assert signal.direction == "IGNORE"
    assert any("below threshold" in r for r in signal.rejection_reasons)


def test_allowed_directions_filter(valid_prediction: dict) -> None:
    # Only allow SELL signals, input is BUY
    engine = SignalEngine(SignalFilterConfig(allowed_directions=("SELL",)))
    signal = engine.evaluate_prediction(valid_prediction)

    assert signal.is_actionable is False
    assert any("not in allowed_directions" in r for r in signal.rejection_reasons)


def test_voting_and_unanimity_filters(valid_prediction: dict) -> None:
    # Require unanimous agreement (svm voted IGNORE -> fails)
    engine_unanimous = SignalEngine(SignalFilterConfig(require_unanimous_vote=True))
    signal_unanimous = engine_unanimous.evaluate_prediction(valid_prediction)
    assert signal_unanimous.is_actionable is False
    assert any("Unanimous vote required" in r for r in signal_unanimous.rejection_reasons)

    # Require 75% agreement (2/3 = 66.7% -> fails)
    engine_fraction = SignalEngine(SignalFilterConfig(min_agreement_fraction=0.75))
    signal_fraction = engine_fraction.evaluate_prediction(valid_prediction)
    assert signal_fraction.is_actionable is False
    assert any("Agreement fraction" in r for r in signal_fraction.rejection_reasons)


def test_blackout_hours_and_weekdays(valid_prediction: dict) -> None:
    # Prediction is at Monday 10:00 UTC (weekday 0)
    # 1. Hour blackout (10, 12)
    engine_hour = SignalEngine(SignalFilterConfig(blackout_hours_utc=((10, 12),)))
    sig_hour = engine_hour.evaluate_prediction(valid_prediction)
    assert sig_hour.is_actionable is False
    assert any("falls within blackout hours" in r for r in sig_hour.rejection_reasons)

    # 2. Weekday blackout (0 = Monday)
    engine_day = SignalEngine(SignalFilterConfig(blackout_weekdays=(0,)))
    sig_day = engine_day.evaluate_prediction(valid_prediction)
    assert sig_day.is_actionable is False
    assert any("falls within blackout weekdays" in r for r in sig_day.rejection_reasons)


def test_market_context_atr_and_regime_filters(valid_prediction: dict) -> None:
    config = SignalFilterConfig(
        min_atr=0.0010,
        max_atr=0.0050,
        regime_filter="trend_only",
    )
    engine = SignalEngine(config)

    # Context with low ATR (< 0.0010)
    sig_low_atr = engine.evaluate_prediction(
        valid_prediction,
        market_context={"atr": 0.0005, "regime": "trend"},
    )
    assert sig_low_atr.is_actionable is False
    assert any("below minimum threshold" in r for r in sig_low_atr.rejection_reasons)

    # Context with wrong regime ("ranging" instead of "trend")
    sig_wrong_regime = engine.evaluate_prediction(
        valid_prediction,
        market_context={"atr": 0.0020, "regime": "ranging"},
    )
    assert sig_wrong_regime.is_actionable is False
    assert any("Regime filter requires 'trend'" in r for r in sig_wrong_regime.rejection_reasons)

    # Context satisfying all criteria
    sig_ok = engine.evaluate_prediction(
        valid_prediction,
        market_context={"atr": 0.0020, "regime": "trend"},
    )
    assert sig_ok.is_actionable is True
