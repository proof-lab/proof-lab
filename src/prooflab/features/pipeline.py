"""Feature pipeline orchestrator with explicit lookback tracking and leakage guards.

Supports standardized family comparison presets, automated maximum lookback computation,
warm-up row handling, and seamless reuse between batch research and live inference.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

# Ensure all feature modules are imported to trigger feature registration
import prooflab.features.microstructure  # noqa: F401
import prooflab.features.momentum  # noqa: F401
import prooflab.features.price  # noqa: F401
import prooflab.features.time  # noqa: F401
import prooflab.features.trend  # noqa: F401
import prooflab.features.volatility  # noqa: F401
from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    feature_registry,
)


class FeatureSetPreset(StrEnum):
    """Standardized feature family comparison presets."""

    PRICE_ONLY = "PRICE_ONLY"
    PRICE_VOLATILITY = "PRICE_VOLATILITY"
    PRICE_MOMENTUM = "PRICE_MOMENTUM"
    PRICE_VOLATILITY_MOMENTUM = "PRICE_VOLATILITY_MOMENTUM"
    ALL_STANDARD = "ALL_STANDARD"
    ALL_STANDARD_MICROSTRUCTURE = "ALL_STANDARD_MICROSTRUCTURE"


class FeaturePipeline:
    """Orchestrates multi-family feature generation with causality and lookback management."""

    def __init__(
        self,
        features: (
            FeatureSetPreset | list[str] | list[FeatureFamily]
        ) = FeatureSetPreset.ALL_STANDARD,
        include_raw_columns: bool = True,
    ) -> None:
        self.include_raw_columns = include_raw_columns
        self.feature_names: list[str] = self._resolve_feature_list(features)

    def _resolve_feature_list(
        self,
        features: FeatureSetPreset | list[str] | list[FeatureFamily],
    ) -> list[str]:
        """Resolve a preset, family list, or feature names into a flat list of feature names."""
        if isinstance(features, FeatureSetPreset) or isinstance(features, str):
            preset = FeatureSetPreset(features)
            if preset == FeatureSetPreset.PRICE_ONLY:
                families = [FeatureFamily.PRICE]
            elif preset == FeatureSetPreset.PRICE_VOLATILITY:
                families = [FeatureFamily.PRICE, FeatureFamily.VOLATILITY]
            elif preset == FeatureSetPreset.PRICE_MOMENTUM:
                families = [FeatureFamily.PRICE, FeatureFamily.MOMENTUM]
            elif preset == FeatureSetPreset.PRICE_VOLATILITY_MOMENTUM:
                families = [FeatureFamily.PRICE, FeatureFamily.VOLATILITY, FeatureFamily.MOMENTUM]
            elif preset == FeatureSetPreset.ALL_STANDARD:
                families = [
                    FeatureFamily.PRICE,
                    FeatureFamily.MOMENTUM,
                    FeatureFamily.VOLATILITY,
                    FeatureFamily.TREND,
                    FeatureFamily.TIME,
                ]
            elif preset == FeatureSetPreset.ALL_STANDARD_MICROSTRUCTURE:
                families = [
                    FeatureFamily.PRICE,
                    FeatureFamily.MOMENTUM,
                    FeatureFamily.VOLATILITY,
                    FeatureFamily.TREND,
                    FeatureFamily.TIME,
                    FeatureFamily.MICROSTRUCTURE,
                ]
            else:
                raise ValueError(f"Unsupported preset: {preset}")

            names: list[str] = []
            for fam in families:
                names.extend(feature_registry.get_family_features(fam))
            return sorted(list(dict.fromkeys(names)))

        elif isinstance(features, list):
            resolved: list[str] = []
            for item in features:
                if isinstance(item, FeatureFamily):
                    resolved.extend(feature_registry.get_family_features(item))
                elif isinstance(item, str):
                    if feature_registry.has_feature(item):
                        resolved.append(item)
                    else:
                        raise KeyError(f"Feature '{item}' not found in registry.")
                else:
                    raise TypeError(f"Invalid feature specifier type: {type(item)}")
            return sorted(list(dict.fromkeys(resolved)))

        else:
            raise TypeError(f"Invalid features configuration type: {type(features)}")

    def get_max_lookback(self) -> int:
        """Calculate the maximum lookback period across all included features."""
        return feature_registry.compute_max_lookback(self.feature_names)

    def get_feature_names(self) -> list[str]:
        """Return the list of active feature names."""
        return list(self.feature_names)

    def get_metadata_summary(self) -> list[FeatureMetadata]:
        """Return metadata objects for all active features."""
        return [feature_registry.get_metadata(name) for name in self.feature_names]

    def transform(
        self,
        df: pd.DataFrame,
        drop_warmup: bool = True,
    ) -> pd.DataFrame:
        """Execute the feature pipeline on the input DataFrame.

        Args:
            df: Historical price DataFrame (chronologically sorted).
            drop_warmup: If True, drops the first max_lookback rows where features are warming up.
                         If False, keeps all rows (warmup rows will contain NaNs).

        Returns:
            DataFrame containing calculated features (and optionally raw columns).
        """
        if df.empty:
            return pd.DataFrame()

        # Check required columns
        for name in self.feature_names:
            meta = feature_registry.get_metadata(name)
            for req_col in meta.required_columns:
                if req_col not in df.columns:
                    raise ValueError(
                        f"Missing required input column '{req_col}' for feature '{name}'."
                    )

        feature_series_list: list[pd.DataFrame | pd.Series] = []
        for name in self.feature_names:
            generator = feature_registry.get_generator(name)
            feat_out = generator(df)
            if isinstance(feat_out, pd.Series):
                feat_out.name = name
            feature_series_list.append(feat_out)

        features_df = pd.concat(feature_series_list, axis=1)

        if self.include_raw_columns:
            # Prefix avoid collisions if any
            result_df = pd.concat([df, features_df], axis=1)
        else:
            result_df = features_df

        if drop_warmup:
            max_lookback = self.get_max_lookback()
            if max_lookback > 0:
                result_df = result_df.iloc[max_lookback:].copy()

        return result_df
