"""Live feature computation engine reusing the canonical training FeaturePipeline."""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from prooflab.features.pipeline import FeaturePipeline


class LiveFeatureResult(BaseModel):
    """Result of live single-bar feature extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    features: dict[str, float] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
    warmup_bars_required: int = 50
    bars_available: int = 0


class LiveFeatureCalculator:
    """Computes real-time features delegating to canonical FeaturePipeline."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        expected_features: list[str] | None = None,
        min_warmup_bars: int = 50,
    ) -> None:
        self.pipeline = pipeline
        self.expected_features = list(expected_features) if expected_features else None
        self.min_warmup_bars = max(min_warmup_bars, self.pipeline.get_max_lookback())

    def compute_live_features(self, bars_history: pd.DataFrame) -> LiveFeatureResult:
        """Compute latest feature vector for incoming bars reusing canonical pipeline."""
        bars_count = len(bars_history)
        if bars_count < self.min_warmup_bars:
            return LiveFeatureResult(
                is_valid=False,
                warmup_bars_required=self.min_warmup_bars,
                bars_available=bars_count,
                rejection_reason=(
                    f"Insufficient bars for warmup: {bars_count} < {self.min_warmup_bars}"
                ),
            )

        # Direct call to the canonical training feature pipeline
        features_df = self.pipeline.transform(bars_history, drop_warmup=False)

        if features_df.empty:
            return LiveFeatureResult(
                is_valid=False,
                warmup_bars_required=self.min_warmup_bars,
                bars_available=bars_count,
                rejection_reason="FeaturePipeline returned empty DataFrame",
            )

        # Extract the latest single-row features
        latest_row = features_df.iloc[-1]

        target_features = self.expected_features or list(features_df.columns)
        extracted: dict[str, float] = {}

        for col in target_features:
            if col not in features_df.columns:
                return LiveFeatureResult(
                    is_valid=False,
                    warmup_bars_required=self.min_warmup_bars,
                    bars_available=bars_count,
                    rejection_reason=f"Expected feature '{col}' missing from calculated features",
                )

            val = float(latest_row[col])
            if math.isnan(val) or math.isinf(val):
                return LiveFeatureResult(
                    is_valid=False,
                    warmup_bars_required=self.min_warmup_bars,
                    bars_available=bars_count,
                    rejection_reason=f"Non-finite feature value ({val}) for '{col}'",
                )
            extracted[col] = val

        return LiveFeatureResult(
            is_valid=True,
            features=extracted,
            feature_names=target_features,
            warmup_bars_required=self.min_warmup_bars,
            bars_available=bars_count,
        )
