"""Unit tests for prooflab.features.base (FeatureMetadata and FeatureRegistry)."""

import pytest
from pydantic import ValidationError

from prooflab.features.base import (
    FeatureFamily,
    FeatureMetadata,
    FeatureRegistry,
)


def test_feature_metadata_valid() -> None:
    meta = FeatureMetadata(
        feature_name="return_1",
        family=FeatureFamily.PRICE,
        description="1-bar percentage return",
        parameters={"lag": 1},
        required_columns=["close"],
        lookback_period=1,
        uses_future_data=False,
        version="1.0.0",
    )
    assert meta.feature_name == "return_1"
    assert meta.family == FeatureFamily.PRICE
    assert meta.lookback_period == 1
    assert meta.uses_future_data is False


def test_feature_metadata_negative_lookback_rejected() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        FeatureMetadata(
            feature_name="bad_lookback",
            family=FeatureFamily.PRICE,
            description="Bad lookback",
            lookback_period=-5,
        )


def test_feature_metadata_future_data_strictly_rejected() -> None:
    with pytest.raises(ValidationError, match="must never use future data"):
        FeatureMetadata(
            feature_name="leak_feature",
            family=FeatureFamily.PRICE,
            description="Illegal look-ahead feature",
            uses_future_data=True,  # type: ignore[arg-type]
        )


def test_feature_metadata_immutability() -> None:
    meta = FeatureMetadata(
        feature_name="rsi_14",
        family=FeatureFamily.MOMENTUM,
        description="Relative Strength Index",
        lookback_period=14,
    )
    with pytest.raises(ValidationError):
        meta.lookback_period = 20  # type: ignore[misc]


def test_feature_registry_operations() -> None:
    registry = FeatureRegistry()

    meta1 = FeatureMetadata(
        feature_name="feat_a",
        family=FeatureFamily.PRICE,
        description="Feature A",
        lookback_period=5,
    )
    meta2 = FeatureMetadata(
        feature_name="feat_b",
        family=FeatureFamily.VOLATILITY,
        description="Feature B",
        lookback_period=20,
    )

    def func_a(df: object) -> object:
        return df

    def func_b(df: object) -> object:
        return df

    registry.register(meta1, func_a)
    registry.register(meta2, func_b)

    assert registry.has_feature("feat_a") is True
    assert registry.has_feature("feat_unknown") is False
    assert registry.get_metadata("feat_a") == meta1
    assert registry.get_generator("feat_b") == func_b

    # Duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(meta1, func_a)

    # Missing feature lookup
    with pytest.raises(KeyError, match="not found"):
        registry.get_metadata("non_existent")

    with pytest.raises(KeyError, match="not found"):
        registry.get_generator("non_existent")

    # Family grouping
    price_features = registry.get_family_features(FeatureFamily.PRICE)
    assert price_features == ["feat_a"]
    vol_features = registry.get_family_features(FeatureFamily.VOLATILITY)
    assert vol_features == ["feat_b"]

    # Compute max lookback
    assert registry.compute_max_lookback(["feat_a"]) == 5
    assert registry.compute_max_lookback(["feat_a", "feat_b"]) == 20
    assert registry.compute_max_lookback([]) == 0

    # List all
    assert registry.list_all_features() == ["feat_a", "feat_b"]

    # Clear
    registry.clear()
    assert len(registry.list_all_features()) == 0
