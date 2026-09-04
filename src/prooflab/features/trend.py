"""Trend feature family implementation.

Computes Exponential Moving Averages (EMA), Simple Moving Averages (SMA),
EMA distance, Average Directional Index (ADX), and rolling linear regression trend slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)
from prooflab.features.volatility import compute_true_range


def compute_ema(
    df: pd.DataFrame,
    span: int = 12,
    column: str = "close",
) -> pd.Series:
    """Compute Exponential Moving Average (EMA)."""
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for EMA calculation.")
    if span < 1:
        raise ValueError(f"Span must be >= 1, got {span}")

    series = df[column].astype(float)
    ema = series.ewm(span=span, adjust=False).mean()
    ema.name = f"ema_{span}"
    return ema


def compute_sma(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "close",
) -> pd.Series:
    """Compute Simple Moving Average (SMA)."""
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for SMA calculation.")
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")

    series = df[column].astype(float)
    sma = series.rolling(window=window).mean()
    sma.name = f"sma_{window}"
    return sma


def compute_ema_distance(
    df: pd.DataFrame,
    fast_span: int = 12,
    slow_span: int = 26,
    column: str = "close",
) -> pd.Series:
    """Compute normalized distance between fast and slow EMA.

    Formula: (EMA_fast - EMA_slow) / EMA_slow
    """
    if fast_span >= slow_span:
        raise ValueError(f"fast_span ({fast_span}) must be < slow_span ({slow_span})")

    fast_ema = compute_ema(df, span=fast_span, column=column)
    slow_ema = compute_ema(df, span=slow_span, column=column)

    distance = (fast_ema - slow_ema) / slow_ema
    distance.name = f"ema_distance_{fast_span}_{slow_span}"
    return distance


def compute_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) alongside +DI and -DI indicators.

    Returns:
        DataFrame with columns: plus_di, minus_di, adx
    """
    required = ["high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for ADX calculation.")
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = compute_true_range(df)

    # Wilder smoothing (alpha = 1 / period)
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    alpha = 1.0 / period
    tr_smoothed = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smoothed = plus_dm_series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smoothed = minus_dm_series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = (plus_dm_smoothed / tr_smoothed) * 100.0
    minus_di = (minus_dm_smoothed / tr_smoothed) * 100.0

    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = (di_diff / di_sum.replace(0.0, np.nan)) * 100.0
    dx = dx.fillna(0.0)

    # ADX is Wilder smoothed DX
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {
            f"plus_di_{period}": plus_di,
            f"minus_di_{period}": minus_di,
            f"adx_{period}": adx,
        },
        index=df.index,
    )


def compute_trend_slope(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
) -> pd.Series:
    """Compute rolling linear regression slope of prices normalized by price level.

    Formula: slope = cov(x, y) / var(x) / mean(y)
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for trend slope calculation.")
    if period < 2:
        raise ValueError(f"Period must be >= 2 for trend slope, got {period}")

    series = df[column].astype(float)
    x = np.arange(period)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()

    def _calc_slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        if y_mean == 0:
            return 0.0
        y_dev = arr - y_mean
        cov = (x_dev * y_dev).sum()
        slope = cov / x_var
        return float(slope / y_mean)

    slope_series = series.rolling(window=period).apply(_calc_slope, raw=True)
    slope_series.name = f"trend_slope_{period}"
    return slope_series


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard trend features and return a unified DataFrame."""
    ema_fast = compute_ema(df, span=12).to_frame()
    ema_slow = compute_ema(df, span=26).to_frame()
    ema_dist = compute_ema_distance(df, fast_span=12, slow_span=26).to_frame()
    sma_fast = compute_sma(df, window=20).to_frame()
    sma_slow = compute_sma(df, window=50).to_frame()
    adx_df = compute_adx(df, period=14)
    slope = compute_trend_slope(df, period=14).to_frame()

    return pd.concat(
        [ema_fast, ema_slow, ema_dist, sma_fast, sma_slow, adx_df, slope],
        axis=1,
    )


def register_trend_features() -> None:
    """Register trend features in the global feature registry."""
    # 1. EMAs
    for span in [12, 26]:
        meta = FeatureMetadata(
            feature_name=f"ema_{span}",
            family=FeatureFamily.TREND,
            description=f"{span}-period Exponential Moving Average",
            parameters={"span": span},
            required_columns=["close"],
            lookback_period=span,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, span=span: compute_ema(df, span=span).to_frame(),
            )

    # 2. EMA Distance
    meta_dist = FeatureMetadata(
        feature_name="ema_distance_12_26",
        family=FeatureFamily.TREND,
        description="Normalized distance between 12 EMA and 26 EMA ((Fast - Slow) / Slow)",
        parameters={"fast_span": 12, "slow_span": 26},
        required_columns=["close"],
        lookback_period=26,
    )
    if not feature_registry.has_feature(meta_dist.feature_name):
        feature_registry.register(
            meta_dist,
            lambda df: compute_ema_distance(df, fast_span=12, slow_span=26).to_frame(),
        )

    # 3. SMAs
    for window in [20, 50]:
        meta = FeatureMetadata(
            feature_name=f"sma_{window}",
            family=FeatureFamily.TREND,
            description=f"{window}-period Simple Moving Average",
            parameters={"window": window},
            required_columns=["close"],
            lookback_period=window,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, window=window: compute_sma(df, window=window).to_frame(),
            )

    # 4. ADX components
    adx_comps = [
        ("plus_di_14", "14-period Positive Directional Indicator (+DI)", 14),
        ("minus_di_14", "14-period Negative Directional Indicator (-DI)", 14),
        ("adx_14", "14-period Average Directional Movement Index (ADX)", 28),
    ]
    for name, desc, lookback in adx_comps:
        meta = FeatureMetadata(
            feature_name=name,
            family=FeatureFamily.TREND,
            description=desc,
            parameters={"period": 14},
            required_columns=["high", "low", "close"],
            lookback_period=lookback,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, name=name: compute_adx(df, period=14)[[name]],
            )

    # 5. Trend Slope
    meta_slope = FeatureMetadata(
        feature_name="trend_slope_14",
        family=FeatureFamily.TREND,
        description="14-period rolling linear regression normalized slope",
        parameters={"period": 14},
        required_columns=["close"],
        lookback_period=14,
    )
    if not feature_registry.has_feature(meta_slope.feature_name):
        feature_registry.register(
            meta_slope,
            lambda df: compute_trend_slope(df, period=14).to_frame(),
        )


# Auto-register trend features upon import
register_trend_features()
