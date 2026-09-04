"""Unit tests for prooflab.features.price (Price feature family)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.price import (
    compute_bar_geometry,
    compute_high_low_distances,
    compute_price_features,
    compute_ranges,
    compute_returns,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(5)],
            "open": [100.0, 102.0, 101.0, 104.0, 103.0],
            "high": [105.0, 106.0, 103.0, 108.0, 107.0],
            "low": [98.0, 100.0, 99.0, 102.0, 101.0],
            "close": [102.0, 101.0, 103.0, 106.0, 105.0],
            "volume": [1000, 1200, 800, 1500, 1100],
        }
    )


def test_compute_returns(sample_ohlcv: pd.DataFrame) -> None:
    returns = compute_returns(sample_ohlcv, lags=[1, 2])
    assert "return_1" in returns.columns
    assert "return_2" in returns.columns
    assert np.isnan(returns["return_1"].iloc[0])
    # Bar 1 return_1: (101 - 102) / 102
    assert returns["return_1"].iloc[1] == pytest.approx((101.0 - 102.0) / 102.0)
    # Bar 2 return_2: (103 - 102) / 102
    assert returns["return_2"].iloc[2] == pytest.approx((103.0 - 102.0) / 102.0)


def test_compute_ranges(sample_ohlcv: pd.DataFrame) -> None:
    ranges = compute_ranges(sample_ohlcv, windows=[1, 3])
    # range_1 at bar 0: 105 - 98 = 7
    assert ranges["range_1"].iloc[0] == pytest.approx(7.0)
    # range_3 at bar 2: max(105, 106, 103) - min(98, 100, 99) = 106 - 98 = 8
    assert ranges["range_3"].iloc[2] == pytest.approx(8.0)


def test_compute_bar_geometry(sample_ohlcv: pd.DataFrame) -> None:
    geom = compute_bar_geometry(sample_ohlcv)
    # Bar 0: open=100, high=105, low=98, close=102
    # body_size = |102 - 100| = 2
    # upper_wick = 105 - max(100, 102) = 105 - 102 = 3
    # lower_wick = min(100, 102) - 98 = 100 - 98 = 2
    # close_to_open = 102 - 100 = 2
    # close_to_high = 102 - 105 = -3
    # close_to_low = 102 - 98 = 4
    assert geom["body_size"].iloc[0] == pytest.approx(2.0)
    assert geom["upper_wick"].iloc[0] == pytest.approx(3.0)
    assert geom["lower_wick"].iloc[0] == pytest.approx(2.0)
    assert geom["close_to_open"].iloc[0] == pytest.approx(2.0)
    assert geom["close_to_high"].iloc[0] == pytest.approx(-3.0)
    assert geom["close_to_low"].iloc[0] == pytest.approx(4.0)


def test_compute_high_low_distances(sample_ohlcv: pd.DataFrame) -> None:
    dist = compute_high_low_distances(sample_ohlcv, windows=[3])
    # Bar 2: close=103. Rolling 3 High max(105, 106, 103)=106, Low min(98, 100, 99)=98
    # distance_from_high_3 = 106 - 103 = 3
    # distance_from_low_3 = 103 - 98 = 5
    assert dist["distance_from_high_3"].iloc[2] == pytest.approx(3.0)
    assert dist["distance_from_low_3"].iloc[2] == pytest.approx(5.0)


def test_compute_price_features_unified(sample_ohlcv: pd.DataFrame) -> None:
    price_df = compute_price_features(sample_ohlcv)
    assert isinstance(price_df, pd.DataFrame)
    assert len(price_df) == len(sample_ohlcv)
    assert "return_1" in price_df.columns
    assert "range_1" in price_df.columns
    assert "body_size" in price_df.columns
    assert "distance_from_high_12" in price_df.columns


def test_missing_column_errors() -> None:
    bad_df = pd.DataFrame({"dummy": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required column"):
        compute_returns(bad_df)

    with pytest.raises(ValueError, match="Missing required column"):
        compute_ranges(bad_df)

    with pytest.raises(ValueError, match="Missing required column"):
        compute_bar_geometry(bad_df)


def test_price_registry_integration() -> None:
    price_feats = feature_registry.get_family_features(FeatureFamily.PRICE)
    assert "return_1" in price_feats
    assert "range_1" in price_feats
    assert "body_size" in price_feats
    assert "distance_from_high_12" in price_feats

    meta = feature_registry.get_metadata("return_1")
    assert meta.lookback_period == 1
    assert meta.uses_future_data is False
