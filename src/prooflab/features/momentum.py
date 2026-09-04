"""Momentum feature family implementation.

Computes Relative Strength Index (RSI), Rate of Change (ROC), Moving Average
Convergence Divergence (MACD), and multi-horizon directional price momentum.
"""

from __future__ import annotations

import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)

DEFAULT_MOMENTUM_LAGS = [3, 6, 12]


def compute_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's exponential smoothing.

    Args:
        df: Price DataFrame containing target price column.
        period: Smoothing lookback period (default: 14).
        column: Column to compute RSI from (default: 'close').

    Returns:
        pd.Series containing RSI values in [0, 100].
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for RSI calculation.")
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")

    series = df[column].astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing corresponds to ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases (zero loss -> 100, zero gain -> 0)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)

    rsi.name = f"rsi_{period}"
    return rsi


def compute_roc(
    df: pd.DataFrame,
    period: int = 10,
    column: str = "close",
) -> pd.Series:
    """Compute Rate of Change (ROC) percentage indicator.

    Formula: ((Price_t - Price_{t-period}) / Price_{t-period}) * 100
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for ROC calculation.")
    if period < 1:
        raise ValueError(f"Period must be >= 1, got {period}")

    series = df[column].astype(float)
    prev = series.shift(period)
    roc = ((series - prev) / prev) * 100.0
    roc.name = f"roc_{period}"
    return roc


def compute_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """Compute Moving Average Convergence Divergence (MACD).

    Returns:
        DataFrame with columns: macd_line, macd_signal, macd_hist
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for MACD calculation.")
    if fast_period >= slow_period:
        raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")

    series = df[column].astype(float)
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    result = pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        },
        index=df.index,
    )
    return result


def compute_momentum(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    column: str = "close",
) -> pd.DataFrame:
    """Compute directional price momentum (absolute price difference).

    Formula: Price_t - Price_{t-lag}
    """
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for momentum calculation.")

    if lags is None:
        lags = DEFAULT_MOMENTUM_LAGS

    result = pd.DataFrame(index=df.index)
    series = df[column].astype(float)

    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1, got {lag}")
        result[f"momentum_{lag}"] = series - series.shift(lag)

    return result


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all standard momentum features and return a unified DataFrame."""
    rsi_14 = compute_rsi(df, period=14).to_frame()
    roc_10 = compute_roc(df, period=10).to_frame()
    macd_df = compute_macd(df, fast_period=12, slow_period=26, signal_period=9)
    mom_df = compute_momentum(df, lags=DEFAULT_MOMENTUM_LAGS)
    return pd.concat([rsi_14, roc_10, macd_df, mom_df], axis=1)


def register_momentum_features() -> None:
    """Register momentum features in the global feature registry."""
    # 1. RSI
    meta_rsi = FeatureMetadata(
        feature_name="rsi_14",
        family=FeatureFamily.MOMENTUM,
        description="14-period Relative Strength Index (Wilder's smoothing)",
        parameters={"period": 14},
        required_columns=["close"],
        lookback_period=14,
    )
    if not feature_registry.has_feature(meta_rsi.feature_name):
        feature_registry.register(
            meta_rsi,
            lambda df: compute_rsi(df, period=14).to_frame(),
        )

    # 2. ROC
    meta_roc = FeatureMetadata(
        feature_name="roc_10",
        family=FeatureFamily.MOMENTUM,
        description="10-period Rate of Change percentage",
        parameters={"period": 10},
        required_columns=["close"],
        lookback_period=10,
    )
    if not feature_registry.has_feature(meta_roc.feature_name):
        feature_registry.register(
            meta_roc,
            lambda df: compute_roc(df, period=10).to_frame(),
        )

    # 3. MACD
    macd_components = [
        ("macd_line", "MACD Fast-Slow EMA difference line (12, 26)", 26),
        ("macd_signal", "MACD 9-period Signal line", 35),
        ("macd_hist", "MACD Histogram (line - signal)", 35),
    ]
    for comp_name, desc, lookback in macd_components:
        meta = FeatureMetadata(
            feature_name=comp_name,
            family=FeatureFamily.MOMENTUM,
            description=desc,
            parameters={"fast": 12, "slow": 26, "signal": 9},
            required_columns=["close"],
            lookback_period=lookback,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, comp_name=comp_name: compute_macd(df)[[comp_name]],
            )

    # 4. Momentum lags
    for lag in DEFAULT_MOMENTUM_LAGS:
        meta = FeatureMetadata(
            feature_name=f"momentum_{lag}",
            family=FeatureFamily.MOMENTUM,
            description=f"{lag}-bar directional price change",
            parameters={"lag": lag},
            required_columns=["close"],
            lookback_period=lag,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, lag=lag: compute_momentum(df, lags=[lag]),
            )


# Auto-register momentum features upon import
register_momentum_features()
