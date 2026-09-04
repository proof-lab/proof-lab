"""Unit tests for prooflab.features.microstructure (Microstructure feature family)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.microstructure import (
    compute_bid_ask_spread,
    compute_microstructure_features,
    compute_price_acceleration,
    compute_spread_percentile,
    compute_tick_volume_dynamics,
    has_microstructure_data,
)


@pytest.fixture
def sample_micro_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(15)],
            "open": [100.0 + i for i in range(15)],
            "high": [101.0 + i for i in range(15)],
            "low": [99.0 + i for i in range(15)],
            "close": [100.5 + i for i in range(15)],
            "spread": [2.0 + (i % 3) for i in range(15)],
            "tick_volume": [100 + (i * 10) for i in range(15)],
        }
    )


def test_compute_bid_ask_spread(sample_micro_df: pd.DataFrame) -> None:
    spread = compute_bid_ask_spread(sample_micro_df)
    assert len(spread) == len(sample_micro_df)
    assert spread.iloc[0] == 2.0

    # Test bid/ask columns
    ba_df = pd.DataFrame({"bid": [100.0, 101.0], "ask": [100.5, 101.8]})
    spread_ba = compute_bid_ask_spread(ba_df)
    assert spread_ba.iloc[0] == pytest.approx(0.5)
    assert spread_ba.iloc[1] == pytest.approx(0.8)


def test_microstructure_guard_prohibits_faking_spread() -> None:
    ohlcv_only = pd.DataFrame({"open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0]})
    with pytest.raises(ValueError, match="Microstructure Guard Violation"):
        compute_bid_ask_spread(ohlcv_only)


def test_spread_percentile(sample_micro_df: pd.DataFrame) -> None:
    sp = compute_spread_percentile(sample_micro_df, window=10)
    valid = sp.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 100.0).all()


def test_tick_volume_dynamics(sample_micro_df: pd.DataFrame) -> None:
    tv = compute_tick_volume_dynamics(sample_micro_df)
    assert "tick_count" in tv.columns
    assert "tick_volume_change" in tv.columns
    # Bar 1: tv=110, prev=100 -> change = 0.10 (10%)
    assert tv["tick_volume_change"].iloc[1] == pytest.approx(0.10)

    # Missing tick_volume raises
    bad_df = pd.DataFrame({"close": [100.0]})
    with pytest.raises(ValueError, match="Microstructure Guard Violation"):
        compute_tick_volume_dynamics(bad_df)


def test_price_acceleration() -> None:
    df = pd.DataFrame({"close": [10.0, 12.0, 15.0, 19.0]})
    # diff1 = [NaN, 2.0, 3.0, 4.0]
    # diff2 = [NaN, NaN, 1.0, 1.0]
    accel = compute_price_acceleration(df)
    assert np.isnan(accel.iloc[0])
    assert np.isnan(accel.iloc[1])
    assert accel.iloc[2] == pytest.approx(1.0)
    assert accel.iloc[3] == pytest.approx(1.0)


def test_compute_microstructure_features_unified(sample_micro_df: pd.DataFrame) -> None:
    micro_df = compute_microstructure_features(sample_micro_df)
    assert "bid_ask_spread" in micro_df.columns
    assert "spread_percentile_100" in micro_df.columns
    assert "tick_count" in micro_df.columns
    assert "price_acceleration" in micro_df.columns


def test_has_microstructure_data() -> None:
    assert has_microstructure_data(pd.DataFrame({"spread": [1]})) is True
    assert has_microstructure_data(pd.DataFrame({"bid": [1], "ask": [2]})) is True
    assert has_microstructure_data(pd.DataFrame({"tick_volume": [100]})) is True
    assert has_microstructure_data(pd.DataFrame({"open": [100], "close": [101]})) is False


def test_microstructure_registry_integration() -> None:
    micro_feats = feature_registry.get_family_features(FeatureFamily.MICROSTRUCTURE)
    assert "bid_ask_spread" in micro_feats
    assert "spread_percentile_100" in micro_feats
    assert "tick_count" in micro_feats
    assert "price_acceleration" in micro_feats

    meta = feature_registry.get_metadata("bid_ask_spread")
    assert meta.uses_future_data is False
