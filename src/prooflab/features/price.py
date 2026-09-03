"""Price feature family implementation.

Computes multi-lag returns, rolling price ranges, candlestick geometry (body and wicks),
and distances from rolling extreme highs and lows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)

DEFAULT_RETURN_LAGS = [1, 2, 3, 6, 12, 24]
DEFAULT_RANGE_WINDOWS = [1, 3, 6, 12]
DEFAULT_DISTANCE_WINDOWS = [12, 24]


def compute_returns(
    df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Compute multi-period percentage returns from close prices.

    Formula: (Close_t - Close_{t-k}) / Close_{t-k}
    """
    if "close" not in df.columns:
        raise ValueError("Missing required column 'close' for returns calculation.")

    if lags is None:
        lags = DEFAULT_RETURN_LAGS

    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)

    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1, got {lag}")
        result[f"return_{lag}"] = (close - close.shift(lag)) / close.shift(lag)

    return result


def compute_ranges(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling high-low price ranges.

    Formula: rolling_max(High, window) - rolling_min(Low, window)
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("Missing required columns 'high' and 'low' for ranges calculation.")

    if windows is None:
        windows = DEFAULT_RANGE_WINDOWS

    result = pd.DataFrame(index=df.index)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    for window in windows:
        if window < 1:
            raise ValueError(f"Window must be >= 1, got {window}")
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        result[f"range_{window}"] = rolling_high - rolling_low

    return result


def compute_bar_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intra-bar geometric dimensions (body size, upper/lower wicks, relative offsets)."""
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for bar geometry.")

    open_p = df["open"].astype(float)
    high_p = df["high"].astype(float)
    low_p = df["low"].astype(float)
    close_p = df["close"].astype(float)

    result = pd.DataFrame(index=df.index)

    # Candlestick body size
    result["body_size"] = (close_p - open_p).abs()

    # Candlestick wicks
    max_open_close = np.maximum(open_p, close_p)
    min_open_close = np.minimum(open_p, close_p)
    result["upper_wick"] = high_p - max_open_close
    result["lower_wick"] = min_open_close - low_p

    # Relative close offsets
    result["close_to_open"] = close_p - open_p
    result["close_to_high"] = close_p - high_p  # <= 0
    result["close_to_low"] = close_p - low_p    # >= 0

    return result


def compute_high_low_distances(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute distance of current close price from rolling highest high and lowest low."""
    if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
        raise ValueError(
            "Missing required columns 'high', 'low', 'close' for distance calculation."
        )

    if windows is None:
        windows = DEFAULT_DISTANCE_WINDOWS

    result = pd.DataFrame(index=df.index)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    for window in windows:
        if window < 1:
            raise ValueError(f"Window must be >= 1, got {window}")
        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()
        result[f"distance_from_high_{window}"] = rolling_high - close
        result[f"distance_from_low_{window}"] = close - rolling_low

    return result


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard price family features and return a unified DataFrame."""
    returns_df = compute_returns(df)
    ranges_df = compute_ranges(df)
    geom_df = compute_bar_geometry(df)
    dist_df = compute_high_low_distances(df)
    return pd.concat([returns_df, ranges_df, geom_df, dist_df], axis=1)


def register_price_features() -> None:
    """Register all price features in the global feature registry."""
    # 1. Returns
    for lag in DEFAULT_RETURN_LAGS:
        meta = FeatureMetadata(
            feature_name=f"return_{lag}",
            family=FeatureFamily.PRICE,
            description=f"{lag}-bar percentage return from close price",
            parameters={"lag": lag},
            required_columns=["close"],
            lookback_period=lag,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, lag=lag: compute_returns(df, lags=[lag]),
            )

    # 2. Ranges
    for window in DEFAULT_RANGE_WINDOWS:
        meta = FeatureMetadata(
            feature_name=f"range_{window}",
            family=FeatureFamily.PRICE,
            description=f"{window}-bar rolling high-low price range",
            parameters={"window": window},
            required_columns=["high", "low"],
            lookback_period=window,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, window=window: compute_ranges(df, windows=[window]),
            )

    # 3. Bar Geometry
    geom_features = [
        ("body_size", "Candlestick absolute body size (|close - open|)", 0),
        ("upper_wick", "Candlestick upper wick (high - max(open, close))", 0),
        ("lower_wick", "Candlestick lower wick (min(open, close) - low)", 0),
        ("close_to_open", "Signed close to open offset (close - open)", 0),
        ("close_to_high", "Signed close to high offset (close - high)", 0),
        ("close_to_low", "Signed close to low offset (close - low)", 0),
    ]
    for name, desc, lookback in geom_features:
        meta = FeatureMetadata(
            feature_name=name,
            family=FeatureFamily.PRICE,
            description=desc,
            required_columns=["open", "high", "low", "close"],
            lookback_period=lookback,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, name=name: compute_bar_geometry(df)[[name]],
            )

    # 4. Distances
    for window in DEFAULT_DISTANCE_WINDOWS:
        for side in ["high", "low"]:
            feat_name = f"distance_from_{side}_{window}"
            meta = FeatureMetadata(
                feature_name=feat_name,
                family=FeatureFamily.PRICE,
                description=f"Distance from {window}-bar rolling {side} to close",
                parameters={"window": window, "side": side},
                required_columns=["high", "low", "close"],
                lookback_period=window,
            )
            if not feature_registry.has_feature(meta.feature_name):
                feature_registry.register(
                    meta,
                    lambda df, window=window, feat_name=feat_name: (
                        compute_high_low_distances(df, windows=[window])[[feat_name]]
                    ),
                )


# Auto-register price features upon import
register_price_features()
