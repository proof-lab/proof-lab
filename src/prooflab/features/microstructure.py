"""Microstructure feature family implementation.

Computes bid-ask spread analytics, spread percentile rankings, tick volume dynamics,
and price acceleration.

STRICT GUARD: Microstructure features execute only when real spread or tick data
exists in the input DataFrame and are never synthesized from OHLCV alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)


def has_microstructure_data(df: pd.DataFrame) -> bool:
    """Check whether the input DataFrame contains genuine microstructure columns."""
    has_spread = "spread" in df.columns or ("bid" in df.columns and "ask" in df.columns)
    has_ticks = "tick_volume" in df.columns
    return has_spread or has_ticks


def compute_bid_ask_spread(df: pd.DataFrame) -> pd.Series:
    """Extract or compute bid-ask spread.

    Raises:
        ValueError: If neither 'spread' nor ('bid' and 'ask') columns are present.
    """
    if "spread" in df.columns:
        spread = df["spread"].astype(float)
        spread.name = "bid_ask_spread"
        return spread
    elif "bid" in df.columns and "ask" in df.columns:
        spread = df["ask"].astype(float) - df["bid"].astype(float)
        spread.name = "bid_ask_spread"
        return spread
    else:
        raise ValueError(
            "Microstructure Guard Violation: Cannot calculate bid_ask_spread without genuine "
            "'spread' or ('bid', 'ask') columns. "
            "Faking microstructure data from OHLCV is prohibited."
        )


def compute_spread_percentile(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.Series:
    """Compute rolling percentile rank of the bid-ask spread."""
    spread = compute_bid_ask_spread(df)

    def _rank(arr: np.ndarray) -> float:
        val = arr[-1]
        if np.isnan(val):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float((valid < val).sum() / len(valid) * 100.0)

    min_p = max(2, window // 10)
    pct_rank = spread.rolling(window=window, min_periods=min_p).apply(_rank, raw=True)
    pct_rank.name = f"spread_percentile_{window}"
    return pct_rank


def compute_tick_volume_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tick count and percentage change in tick volume."""
    if "tick_volume" not in df.columns:
        raise ValueError(
            "Microstructure Guard Violation: Missing required 'tick_volume' column. "
            "Fabrication of tick counts from volume or OHLC is prohibited."
        )

    tv = df["tick_volume"].astype(float)
    tv_prev = tv.shift(1)
    tv_change = (tv - tv_prev) / tv_prev.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "tick_count": tv,
            "tick_volume_change": tv_change,
        },
        index=df.index,
    )


def compute_price_acceleration(
    df: pd.DataFrame,
    column: str = "close",
) -> pd.Series:
    """Compute price acceleration (second discrete difference: Δ²P_t)."""
    if column not in df.columns:
        raise ValueError(f"Missing required column '{column}' for price acceleration.")

    series = df[column].astype(float)
    accel = series.diff().diff()
    accel.name = "price_acceleration"
    return accel


def compute_microstructure_features(
    df: pd.DataFrame,
    raise_if_missing: bool = True,
) -> pd.DataFrame:
    """Compute all standard microstructure features.

    Args:
        df: Input DataFrame.
        raise_if_missing: If True and microstructure data is missing, raises ValueError.
                          If False, returns empty DataFrame.
    """
    if not has_microstructure_data(df):
        if raise_if_missing:
            raise ValueError(
                "Microstructure Guard: Required spread/tick columns not found in DataFrame."
            )
        return pd.DataFrame(index=df.index)

    dfs: list[pd.DataFrame | pd.Series] = []

    # 1. Spread features
    if "spread" in df.columns or ("bid" in df.columns and "ask" in df.columns):
        dfs.append(compute_bid_ask_spread(df))
        dfs.append(compute_spread_percentile(df, window=100))

    # 2. Tick volume dynamics
    if "tick_volume" in df.columns:
        dfs.append(compute_tick_volume_dynamics(df))

    # 3. Price acceleration
    if "close" in df.columns:
        dfs.append(compute_price_acceleration(df))

    return pd.concat(dfs, axis=1)


def register_microstructure_features() -> None:
    """Register microstructure features in the global feature registry."""
    # 1. Bid Ask Spread
    meta_spread = FeatureMetadata(
        feature_name="bid_ask_spread",
        family=FeatureFamily.MICROSTRUCTURE,
        description="Bid-Ask spread extracted directly from spread or ask/bid market quotes",
        required_columns=["spread"],
        lookback_period=0,
    )
    if not feature_registry.has_feature(meta_spread.feature_name):
        feature_registry.register(
            meta_spread,
            lambda df: compute_bid_ask_spread(df).to_frame(),
        )

    # 2. Spread Percentile
    meta_sp = FeatureMetadata(
        feature_name="spread_percentile_100",
        family=FeatureFamily.MICROSTRUCTURE,
        description="100-bar rolling percentile rank of bid-ask spread",
        parameters={"window": 100},
        required_columns=["spread"],
        lookback_period=100,
    )
    if not feature_registry.has_feature(meta_sp.feature_name):
        feature_registry.register(
            meta_sp,
            lambda df: compute_spread_percentile(df, window=100).to_frame(),
        )

    # 3. Tick Count
    meta_tc = FeatureMetadata(
        feature_name="tick_count",
        family=FeatureFamily.MICROSTRUCTURE,
        description="Bar tick count extracted from tick_volume",
        required_columns=["tick_volume"],
        lookback_period=0,
    )
    if not feature_registry.has_feature(meta_tc.feature_name):
        feature_registry.register(
            meta_tc,
            lambda df: compute_tick_volume_dynamics(df)[["tick_count"]],
        )

    # 4. Tick Volume Change
    meta_tvc = FeatureMetadata(
        feature_name="tick_volume_change",
        family=FeatureFamily.MICROSTRUCTURE,
        description="1-bar percentage change in tick volume",
        required_columns=["tick_volume"],
        lookback_period=1,
    )
    if not feature_registry.has_feature(meta_tvc.feature_name):
        feature_registry.register(
            meta_tvc,
            lambda df: compute_tick_volume_dynamics(df)[["tick_volume_change"]],
        )

    # 5. Price Acceleration
    meta_pa = FeatureMetadata(
        feature_name="price_acceleration",
        family=FeatureFamily.MICROSTRUCTURE,
        description="Second discrete difference of close price (Delta^2 Close)",
        required_columns=["close"],
        lookback_period=2,
    )
    if not feature_registry.has_feature(meta_pa.feature_name):
        feature_registry.register(
            meta_pa,
            lambda df: compute_price_acceleration(df).to_frame(),
        )


# Auto-register microstructure features upon import
register_microstructure_features()
