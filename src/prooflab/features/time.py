"""Cyclical time feature family implementation.

Computes sine and cosine trigonometric encodings for intraday hour and day-of-week
temporal cycles derived strictly from timezone-aware UTC timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)


def compute_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute cyclical sine and cosine time encodings for hour and day-of-week.

    Args:
        df: DataFrame containing UTC timestamp column.
        timestamp_col: Name of the timestamp column (default: 'timestamp').

    Returns:
        DataFrame with columns: hour_sin, hour_cos, dow_sin, dow_cos
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing required timestamp column '{timestamp_col}'.")

    ts_series = pd.to_datetime(df[timestamp_col], utc=True)

    # Hour of day (fractional with minutes and seconds for smooth intraday continuity)
    hours = ts_series.dt.hour + (ts_series.dt.minute / 60.0) + (ts_series.dt.second / 3600.0)
    hour_angle = 2.0 * np.pi * (hours / 24.0)

    # Day of week (0 = Monday, 6 = Sunday)
    dows = ts_series.dt.dayofweek + (hours / 24.0)
    dow_angle = 2.0 * np.pi * (dows / 7.0)

    result = pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle),
            "dow_cos": np.cos(dow_angle),
        },
        index=df.index,
    )
    return result


def register_time_features() -> None:
    """Register time features in the global feature registry."""
    features = [
        ("hour_sin", "Sine encoding of 24-hour daily cycle (sin(2pi * hour / 24))"),
        ("hour_cos", "Cosine encoding of 24-hour daily cycle (cos(2pi * hour / 24))"),
        ("dow_sin", "Sine encoding of 7-day weekly cycle (sin(2pi * day / 7))"),
        ("dow_cos", "Cosine encoding of 7-day weekly cycle (cos(2pi * day / 7))"),
    ]
    for feat_name, desc in features:
        meta = FeatureMetadata(
            feature_name=feat_name,
            family=FeatureFamily.TIME,
            description=desc,
            required_columns=["timestamp"],
            lookback_period=0,
        )
        if not feature_registry.has_feature(meta.feature_name):
            feature_registry.register(
                meta,
                lambda df, feat_name=feat_name: compute_cyclical_time_features(df)[[feat_name]],
            )


# Auto-register time features upon import
register_time_features()
