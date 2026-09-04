"""Volatility feature family implementation.

Computes True Range (TR), Average True Range (ATR), ATR percentage, rolling standard deviation,
normalized rolling range, and historical volatility percentile ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range (TR) across price bars.

    Formula: max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    """
    required = ["high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for True Range calculation.")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()

    # For the first row where prev_close is NaN, use high - low
    tr = np.maximum(hl, np.maximum(hc.fillna(hl), lc.fillna(hl)))
    tr_series = pd.Series(tr, index=df.index, name="true_range")
    return tr_series


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR) using Wilder's exponential smoothing.

    Args:
        df: DataFrame with high, low, close columns.
        period: Smoothing period (default: 14).
    """
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")

    tr = compute_true_range(df)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr.name = f"atr_{period}"
    return atr


def compute_atr_percent(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute ATR as a percentage of the close price.

    Formula: (ATR_t / Close_t) * 100.0
    """
    if "close" not in df.columns:
        raise ValueError("Missing required column 'close' for ATR percent calculation.")

    atr = compute_atr(df, period=period)
    close = df["close"].astype(float)
    atr_pct = (atr / close) * 100.0
    atr_pct.name = f"atr_percent_{period}"
    return atr_pct


def compute_rolling_std(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pd.Series:
    """Compute rolling standard deviation of percentage returns.

    Formula: rolling_std(returns, period)
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for rolling std calculation.")
    if period < 2:
        raise ValueError(f"Period must be >= 2 for standard deviation, got {period}")

    series = df[column].astype(float)
    returns = series.pct_change()
    roll_std = returns.rolling(window=period).std()
    roll_std.name = f"rolling_std_{period}"
    return roll_std


def compute_rolling_range(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Compute normalized rolling range (high-low spread normalized by mean close price).

    Formula: (rolling_max(High, period) - rolling_min(Low, period)) / rolling_mean(Close, period)
    """
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
        raise ValueError("Missing required columns 'high', 'low', 'close' for rolling range.")
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    roll_high = high.rolling(window=period).max()
    roll_low = low.rolling(window=period).min()
    roll_mean = close.rolling(window=period).mean()

    roll_range = (roll_high - roll_low) / roll_mean
    roll_range.name = f"rolling_range_{period}"
    return roll_range


def compute_volatility_percentile(
    df: pd.DataFrame,
    atr_period: int = 14,
    window: int = 100,
) -> pd.Series:
    """Compute percentile ranking of current ATR relative to its historical rolling window.

    Returns:
        pd.Series in range [0, 100].
    """
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")

    atr = compute_atr(df, period=atr_period)

    def _percentile_rank(arr: np.ndarray) -> float:
        val = arr[-1]
        if np.isnan(val):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float((valid < val).sum() / len(valid) * 100.0)

    pct_rank = atr.rolling(window=window).apply(_percentile_rank, raw=True)
    pct_rank.name = f"volatility_percentile_{window}"
    return pct_rank


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard volatility features and return a unified DataFrame."""
    tr = compute_true_range(df).to_frame()
    atr = compute_atr(df, period=14).to_frame()
    atr_pct = compute_atr_percent(df, period=14).to_frame()
    roll_std = compute_rolling_std(df, period=20).to_frame()
    roll_range = compute_rolling_range(df, period=14).to_frame()
    vol_pct = compute_volatility_percentile(df, atr_period=14, window=100).to_frame()

    return pd.concat([tr, atr, atr_pct, roll_std, roll_range, vol_pct], axis=1)


def register_volatility_features() -> None:
    """Register volatility features in the global feature registry."""
    # 1. True Range
    meta_tr = FeatureMetadata(
        feature_name="true_range",
        family=FeatureFamily.VOLATILITY,
        description="Bar True Range (max(H-L, |H-PrevC|, |L-PrevC|))",
        required_columns=["high", "low", "close"],
        lookback_period=1,
    )
    if not feature_registry.has_feature(meta_tr.feature_name):
        feature_registry.register(meta_tr, lambda df: compute_true_range(df).to_frame())

    # 2. ATR
    meta_atr = FeatureMetadata(
        feature_name="atr_14",
        family=FeatureFamily.VOLATILITY,
        description="14-period Average True Range (Wilder's smoothing)",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        lookback_period=14,
    )
    if not feature_registry.has_feature(meta_atr.feature_name):
        feature_registry.register(meta_atr, lambda df: compute_atr(df, period=14).to_frame())

    # 3. ATR Percent
    meta_atrp = FeatureMetadata(
        feature_name="atr_percent_14",
        family=FeatureFamily.VOLATILITY,
        description="14-period ATR as percentage of close price",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        lookback_period=14,
    )
    if not feature_registry.has_feature(meta_atrp.feature_name):
        feature_registry.register(
            meta_atrp,
            lambda df: compute_atr_percent(df, period=14).to_frame(),
        )

    # 4. Rolling Std
    meta_std = FeatureMetadata(
        feature_name="rolling_std_20",
        family=FeatureFamily.VOLATILITY,
        description="20-period rolling standard deviation of returns",
        parameters={"period": 20},
        required_columns=["close"],
        lookback_period=20,
    )
    if not feature_registry.has_feature(meta_std.feature_name):
        feature_registry.register(
            meta_std,
            lambda df: compute_rolling_std(df, period=20).to_frame(),
        )

    # 5. Rolling Range
    meta_rr = FeatureMetadata(
        feature_name="rolling_range_14",
        family=FeatureFamily.VOLATILITY,
        description="14-period rolling range normalized by close price mean",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        lookback_period=14,
    )
    if not feature_registry.has_feature(meta_rr.feature_name):
        feature_registry.register(
            meta_rr,
            lambda df: compute_rolling_range(df, period=14).to_frame(),
        )

    # 6. Volatility Percentile
    meta_vp = FeatureMetadata(
        feature_name="volatility_percentile_100",
        family=FeatureFamily.VOLATILITY,
        description="100-bar rolling percentile rank of 14-period ATR",
        parameters={"atr_period": 14, "window": 100},
        required_columns=["high", "low", "close"],
        lookback_period=114,
    )
    if not feature_registry.has_feature(meta_vp.feature_name):
        feature_registry.register(
            meta_vp,
            lambda df: compute_volatility_percentile(df, atr_period=14, window=100).to_frame(),
        )


# Auto-register volatility features upon import
register_volatility_features()
