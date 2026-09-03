"""Unit tests for prooflab.paper.features (Identical Live vs Training Feature Code Path)."""

import numpy as np
import pandas as pd
import pytest

from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset
from prooflab.paper.features import LiveFeatureCalculator, LiveFeatureResult


@pytest.fixture
def sample_ohlcv_feed() -> pd.DataFrame:
    idx = pd.date_range("2026-03-02 00:00:00", periods=200, freq="1h", tz="UTC")
    prices = 1.1000 + np.sin(np.linspace(0, 8 * np.pi, 200)) * 0.0050
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.0010,
            "low": prices - 0.0010,
            "close": prices + 0.0002,
            "volume": 1000.0,
            "spread": 0.0001,
        },
        index=idx,
    )


def test_live_feature_identical_to_batch_training_path(
    sample_ohlcv_feed: pd.DataFrame,
) -> None:
    pipeline = FeaturePipeline(
        features=FeatureSetPreset.PRICE_VOLATILITY_MOMENTUM,
        include_raw_columns=False,
    )

    # 1. Batch training calculation over entire history
    batch_features_df = pipeline.transform(sample_ohlcv_feed, drop_warmup=False)
    feature_cols = pipeline.get_feature_names()

    # 2. Live calculator processing
    live_calc = LiveFeatureCalculator(
        pipeline=pipeline,
        expected_features=feature_cols,
        min_warmup_bars=50,
    )

    for bar_idx in [130, 160, 199]:
        slice_df = sample_ohlcv_feed.iloc[: bar_idx + 1]
        live_res = live_calc.compute_live_features(slice_df)

        assert isinstance(live_res, LiveFeatureResult)
        assert live_res.is_valid is True

        expected_row = batch_features_df.iloc[bar_idx]
        for col in feature_cols:
            val_live = live_res.features[col]
            val_batch = expected_row[col]
            assert np.isclose(val_live, val_batch), f"Mismatch on {col} at bar {bar_idx}"


def test_live_feature_warmup_rejection(sample_ohlcv_feed: pd.DataFrame) -> None:
    pipeline = FeaturePipeline(
        features=FeatureSetPreset.PRICE_ONLY,
        include_raw_columns=False,
    )

    live_calc = LiveFeatureCalculator(pipeline=pipeline, min_warmup_bars=50)

    # Insufficient bars (only 20 bars)
    short_df = sample_ohlcv_feed.iloc[:20]
    res = live_calc.compute_live_features(short_df)

    assert res.is_valid is False
    assert "Insufficient bars" in str(res.rejection_reason)
