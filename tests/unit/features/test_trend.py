"""Unit tests for prooflab.features.trend (Trend feature family)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.trend import (
    compute_adx,
    compute_ema,
    compute_ema_distance,
    compute_sma,
    compute_trend_features,
    compute_trend_slope,
)


@pytest.fixture
def strong_uptrend_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    opens = [100.0 + (i * 2.0) for i in range(50)]
    highs = [op + 3.0 for op in opens]
    lows = [op - 0.5 for op in opens]
    closes = [op + 2.5 for op in opens]

    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(50)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


@pytest.fixture
def strong_downtrend_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    opens = [200.0 - (i * 2.0) for i in range(50)]
    highs = [op + 0.5 for op in opens]
    lows = [op - 3.0 for op in opens]
    closes = [op - 2.5 for op in opens]

    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(50)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_ema_and_sma_calculations(strong_uptrend_df: pd.DataFrame) -> None:
    ema_12 = compute_ema(strong_uptrend_df, span=12)
    sma_20 = compute_sma(strong_uptrend_df, window=20)

    assert len(ema_12) == len(strong_uptrend_df)
    assert len(sma_20) == len(strong_uptrend_df)

    # In a strong uptrend, faster EMA is above slower SMA
    assert ema_12.iloc[-1] > sma_20.iloc[-1]


def test_ema_distance(
    strong_uptrend_df: pd.DataFrame,
    strong_downtrend_df: pd.DataFrame,
) -> None:
    dist_up = compute_ema_distance(strong_uptrend_df, fast_span=12, slow_span=26)
    dist_down = compute_ema_distance(strong_downtrend_df, fast_span=12, slow_span=26)

    # Fast > Slow in uptrend -> positive distance
    assert dist_up.iloc[-1] > 0.0
    # Fast < Slow in downtrend -> negative distance
    assert dist_down.iloc[-1] < 0.0


def test_adx_calculations(
    strong_uptrend_df: pd.DataFrame,
    strong_downtrend_df: pd.DataFrame,
) -> None:
    adx_up = compute_adx(strong_uptrend_df, period=14)
    assert "plus_di_14" in adx_up.columns
    assert "minus_di_14" in adx_up.columns
    assert "adx_14" in adx_up.columns

    # Strong uptrend: +DI should exceed -DI
    assert adx_up["plus_di_14"].iloc[-1] > adx_up["minus_di_14"].iloc[-1]
    assert adx_up["adx_14"].iloc[-1] > 20.0  # Strong trend

    # Strong downtrend: -DI should exceed +DI
    adx_down = compute_adx(strong_downtrend_df, period=14)
    assert adx_down["minus_di_14"].iloc[-1] > adx_down["plus_di_14"].iloc[-1]


def test_trend_slope(
    strong_uptrend_df: pd.DataFrame,
    strong_downtrend_df: pd.DataFrame,
) -> None:
    slope_up = compute_trend_slope(strong_uptrend_df, period=14)
    slope_down = compute_trend_slope(strong_downtrend_df, period=14)

    assert slope_up.iloc[-1] > 0.0
    assert slope_down.iloc[-1] < 0.0


def test_compute_trend_features_unified(strong_uptrend_df: pd.DataFrame) -> None:
    trend_df = compute_trend_features(strong_uptrend_df)
    assert "ema_12" in trend_df.columns
    assert "ema_26" in trend_df.columns
    assert "ema_distance_12_26" in trend_df.columns
    assert "sma_20" in trend_df.columns
    assert "sma_50" in trend_df.columns
    assert "adx_14" in trend_df.columns
    assert "trend_slope_14" in trend_df.columns


def test_trend_registry_integration() -> None:
    trend_feats = feature_registry.get_family_features(FeatureFamily.TREND)
    assert "ema_12" in trend_feats
    assert "ema_26" in trend_feats
    assert "ema_distance_12_26" in trend_feats
    assert "sma_20" in trend_feats
    assert "adx_14" in trend_feats
    assert "trend_slope_14" in trend_feats

    meta = feature_registry.get_metadata("adx_14")
    assert meta.lookback_period == 28
    assert meta.uses_future_data is False
