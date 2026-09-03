"""Unit tests for prooflab.features.momentum (Momentum feature family)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.momentum import (
    compute_macd,
    compute_momentum,
    compute_momentum_features,
    compute_roc,
    compute_rsi,
)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 40 bars of strictly upward trending prices
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(40)],
            "close": [100.0 + (i * 1.5) for i in range(40)],
        }
    )


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 40 bars of strictly downward trending prices
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(40)],
            "close": [200.0 - (i * 1.5) for i in range(40)],
        }
    )


def test_rsi_bounds_and_trends(
    trending_up_df: pd.DataFrame,
    trending_down_df: pd.DataFrame,
) -> None:
    rsi_up = compute_rsi(trending_up_df, period=14)
    # Upward trend RSI should be 100 or close to 100 after lookback
    assert rsi_up.iloc[-1] == pytest.approx(100.0)
    assert (rsi_up.dropna() >= 0.0).all()
    assert (rsi_up.dropna() <= 100.0).all()

    rsi_down = compute_rsi(trending_down_df, period=14)
    # Downward trend RSI should be 0.0
    assert rsi_down.iloc[-1] == pytest.approx(0.0)


def test_roc_calculation(trending_up_df: pd.DataFrame) -> None:
    roc = compute_roc(trending_up_df, period=10)
    # Bar 10: close=115.0, prev close at bar 0=100.0 -> ((115-100)/100)*100 = 15.0%
    assert roc.iloc[10] == pytest.approx(15.0)


def test_macd_calculation(trending_up_df: pd.DataFrame) -> None:
    macd_df = compute_macd(trending_up_df, fast_period=12, slow_period=26, signal_period=9)
    assert "macd_line" in macd_df.columns
    assert "macd_signal" in macd_df.columns
    assert "macd_hist" in macd_df.columns

    # macd_hist = macd_line - macd_signal
    diff = macd_df["macd_line"] - macd_df["macd_signal"]
    np.testing.assert_allclose(macd_df["macd_hist"].to_numpy(), diff.to_numpy())

    with pytest.raises(ValueError, match="must be < slow_period"):
        compute_macd(trending_up_df, fast_period=26, slow_period=12)


def test_compute_momentum(trending_up_df: pd.DataFrame) -> None:
    mom = compute_momentum(trending_up_df, lags=[3, 6])
    # Bar 3: close=104.5, bar 0: close=100.0 -> 4.5
    assert mom["momentum_3"].iloc[3] == pytest.approx(4.5)
    # Bar 6: close=109.0, bar 0: close=100.0 -> 9.0
    assert mom["momentum_6"].iloc[6] == pytest.approx(9.0)


def test_compute_momentum_features_unified(trending_up_df: pd.DataFrame) -> None:
    mom_df = compute_momentum_features(trending_up_df)
    assert "rsi_14" in mom_df.columns
    assert "roc_10" in mom_df.columns
    assert "macd_line" in mom_df.columns
    assert "momentum_3" in mom_df.columns
    assert len(mom_df) == len(trending_up_df)


def test_momentum_registry_integration() -> None:
    mom_feats = feature_registry.get_family_features(FeatureFamily.MOMENTUM)
    assert "rsi_14" in mom_feats
    assert "roc_10" in mom_feats
    assert "macd_line" in mom_feats
    assert "momentum_3" in mom_feats

    meta = feature_registry.get_metadata("rsi_14")
    assert meta.lookback_period == 14
    assert meta.uses_future_data is False
