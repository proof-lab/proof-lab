"""Proof Lab Feature Engine.

Provides causal quantitative feature generation across price, momentum, volatility,
trend, cyclical time, and microstructure families, with explicit lookback tracking,
feature pipeline presets, and leak-free scaling.
"""

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    FeatureRegistry,
    feature_registry,
)
from prooflab.features.microstructure import (
    compute_bid_ask_spread,
    compute_microstructure_features,
    compute_price_acceleration,
    compute_spread_percentile,
    compute_tick_volume_dynamics,
    has_microstructure_data,
)
from prooflab.features.momentum import (
    compute_macd,
    compute_momentum,
    compute_momentum_features,
    compute_roc,
    compute_rsi,
)
from prooflab.features.pipeline import (
    FeaturePipeline,
    FeatureSetPreset,
)
from prooflab.features.price import (
    compute_bar_geometry,
    compute_high_low_distances,
    compute_price_features,
    compute_ranges,
    compute_returns,
)
from prooflab.features.scalers import (
    BaseScaler,
    MinMaxScaler,
    NotFittedError,
    RobustScaler,
    StandardScaler,
)
from prooflab.features.time import compute_cyclical_time_features
from prooflab.features.trend import (
    compute_adx,
    compute_ema,
    compute_ema_distance,
    compute_sma,
    compute_trend_features,
    compute_trend_slope,
)
from prooflab.features.volatility import (
    compute_atr,
    compute_atr_percent,
    compute_rolling_range,
    compute_rolling_std,
    compute_true_range,
    compute_volatility_features,
    compute_volatility_percentile,
)

__all__ = [
    "BaseScaler",
    "FeatureFamily",
    "FeatureMetadata",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureSetPreset",
    "MinMaxScaler",
    "NotFittedError",
    "RobustScaler",
    "StandardScaler",
    "compute_adx",
    "compute_atr",
    "compute_atr_percent",
    "compute_bar_geometry",
    "compute_bid_ask_spread",
    "compute_cyclical_time_features",
    "compute_ema",
    "compute_ema_distance",
    "compute_high_low_distances",
    "compute_macd",
    "compute_microstructure_features",
    "compute_momentum",
    "compute_momentum_features",
    "compute_price_acceleration",
    "compute_price_features",
    "compute_ranges",
    "compute_returns",
    "compute_roc",
    "compute_rolling_range",
    "compute_rolling_std",
    "compute_rsi",
    "compute_sma",
    "compute_spread_percentile",
    "compute_tick_volume_dynamics",
    "compute_trend_features",
    "compute_trend_slope",
    "compute_true_range",
    "compute_volatility_features",
    "compute_volatility_percentile",
    "feature_registry",
    "has_microstructure_data",
]
