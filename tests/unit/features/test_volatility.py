"""Unit tests for prooflab.features.volatility (Volatility feature family)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.volatility import (
    compute_atr,
    compute_atr_percent,
    compute_rolling_range,
    compute_rolling_std,
    compute_true_range,
    compute_volatility_features,
    compute_volatility_percentile,
)


@pytest.fixture
def sample_vol_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 30 bars of fluctuating prices
    opens = [100.0 + (i % 5) for i in range(30)]
    highs = [op + 2.0 for op in opens]
    lows = [op - 2.0 for op in opens]
    closes = [op + 0.5 for op in opens]

    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(30)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_true_range_calculation(sample_vol_df: pd.DataFrame) -> None:
    tr = compute_true_range(sample_vol_df)
    # First bar: high=102, low=98 -> TR = 4.0
    assert tr.iloc[0] == pytest.approx(4.0)
    assert (tr >= 0.0).all()


def test_atr_and_atr_percent(sample_vol_df: pd.DataFrame) -> None:
    atr = compute_atr(sample_vol_df, period=14)
    assert len(atr) == len(sample_vol_df)
    assert (atr.dropna() > 0.0).all()

    atr_pct = compute_atr_percent(sample_vol_df, period=14)
    # ATR pct = (ATR / close) * 100
    valid_mask = ~atr.isna()
    expected_pct = (atr[valid_mask] / sample_vol_df["close"][valid_mask]) * 100.0
    np.testing.assert_allclose(atr_pct[valid_mask].to_numpy(), expected_pct.to_numpy())


def test_rolling_std(sample_vol_df: pd.DataFrame) -> None:
    r_std = compute_rolling_std(sample_vol_df, period=10)
    assert len(r_std) == len(sample_vol_df)
    assert (r_std.dropna() >= 0.0).all()

    with pytest.raises(ValueError, match="must be >= 2"):
        compute_rolling_std(sample_vol_df, period=1)


def test_rolling_range(sample_vol_df: pd.DataFrame) -> None:
    r_range = compute_rolling_range(sample_vol_df, period=10)
    assert len(r_range) == len(sample_vol_df)
    assert (r_range.dropna() > 0.0).all()


def test_volatility_percentile(sample_vol_df: pd.DataFrame) -> None:
    vol_pct = compute_volatility_percentile(sample_vol_df, atr_period=5, window=10)
    valid = vol_pct.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 100.0).all()


def test_compute_volatility_features_unified(sample_vol_df: pd.DataFrame) -> None:
    vol_df = compute_volatility_features(sample_vol_df)
    assert "true_range" in vol_df.columns
    assert "atr_14" in vol_df.columns
    assert "atr_percent_14" in vol_df.columns
    assert "rolling_std_20" in vol_df.columns
    assert "rolling_range_14" in vol_df.columns
    assert "volatility_percentile_100" in vol_df.columns


def test_volatility_registry_integration() -> None:
    vol_feats = feature_registry.get_family_features(FeatureFamily.VOLATILITY)
    assert "true_range" in vol_feats
    assert "atr_14" in vol_feats
    assert "atr_percent_14" in vol_feats
    assert "rolling_std_20" in vol_feats

    meta = feature_registry.get_metadata("atr_14")
    assert meta.lookback_period == 14
    assert meta.uses_future_data is False
