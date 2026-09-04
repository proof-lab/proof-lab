"""Unit tests for prooflab.features.pipeline (FeaturePipeline and presets)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from prooflab.features.pipeline import FeaturePipeline, FeatureSetPreset


@pytest.fixture
def ohlcv_pipeline_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 60 bars of synthetic data
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(60)],
            "open": [100.0 + (i * 0.5) for i in range(60)],
            "high": [101.0 + (i * 0.5) for i in range(60)],
            "low": [99.5 + (i * 0.5) for i in range(60)],
            "close": [100.5 + (i * 0.5) for i in range(60)],
            "volume": [1000 + (i * 10) for i in range(60)],
        }
    )


def test_pipeline_presets(ohlcv_pipeline_df: pd.DataFrame) -> None:
    # Price only
    p_price = FeaturePipeline(features=FeatureSetPreset.PRICE_ONLY)
    feats_price = p_price.get_feature_names()
    assert "return_1" in feats_price
    assert "rsi_14" not in feats_price

    # Price + Volatility
    p_pv = FeaturePipeline(features=FeatureSetPreset.PRICE_VOLATILITY)
    feats_pv = p_pv.get_feature_names()
    assert "return_1" in feats_pv
    assert "atr_14" in feats_pv
    assert "rsi_14" not in feats_pv

    # All standard
    p_all = FeaturePipeline(features=FeatureSetPreset.ALL_STANDARD)
    feats_all = p_all.get_feature_names()
    assert "return_1" in feats_all
    assert "rsi_14" in feats_all
    assert "atr_14" in feats_all
    assert "ema_12" in feats_all
    assert "hour_sin" in feats_all


def test_pipeline_transform_and_warmup(ohlcv_pipeline_df: pd.DataFrame) -> None:
    pipeline = FeaturePipeline(features=FeatureSetPreset.PRICE_ONLY, include_raw_columns=False)
    max_lookback = pipeline.get_max_lookback()
    assert max_lookback == 24  # return_24 or distance_from_high_24

    # Transform with drop_warmup=True
    df_clean = pipeline.transform(ohlcv_pipeline_df, drop_warmup=True)
    assert len(df_clean) == len(ohlcv_pipeline_df) - max_lookback
    assert not df_clean["return_1"].isna().any()

    # Transform with drop_warmup=False
    df_raw = pipeline.transform(ohlcv_pipeline_df, drop_warmup=False)
    assert len(df_raw) == len(ohlcv_pipeline_df)
    assert df_raw["return_24"].iloc[0:24].isna().all()


def test_pipeline_custom_features(ohlcv_pipeline_df: pd.DataFrame) -> None:
    pipeline = FeaturePipeline(features=["return_1", "range_1"], include_raw_columns=True)
    assert pipeline.get_max_lookback() == 1
    out = pipeline.transform(ohlcv_pipeline_df, drop_warmup=True)
    assert "return_1" in out.columns
    assert "range_1" in out.columns
    assert "open" in out.columns


def test_pipeline_missing_columns_error() -> None:
    bad_df = pd.DataFrame({"close": [100.0, 101.0]})
    # range_1 requires high and low
    pipeline = FeaturePipeline(features=["range_1"])
    with pytest.raises(ValueError, match="Missing required input column"):
        pipeline.transform(bad_df)


def test_pipeline_metadata_summary() -> None:
    pipeline = FeaturePipeline(features=["return_1", "hour_sin"])
    meta_list = pipeline.get_metadata_summary()
    assert len(meta_list) == 2
    assert meta_list[0].feature_name == "hour_sin" or meta_list[1].feature_name == "hour_sin"
