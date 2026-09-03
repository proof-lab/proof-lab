"""Unit tests for prooflab.features.time (Cyclical time feature family)."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from prooflab.features.base import FeatureFamily, feature_registry
from prooflab.features.time import compute_cyclical_time_features


def test_cyclical_time_identities() -> None:
    # Multiple distinct times across hours and days
    timestamps = [
        datetime(2026, 1, 5, 0, 0, tzinfo=UTC),   # Monday 00:00
        datetime(2026, 1, 5, 6, 0, tzinfo=UTC),   # Monday 06:00
        datetime(2026, 1, 5, 12, 0, tzinfo=UTC),  # Monday 12:00
        datetime(2026, 1, 5, 18, 0, tzinfo=UTC),  # Monday 18:00
        datetime(2026, 1, 8, 12, 0, tzinfo=UTC),  # Thursday 12:00
    ]
    df = pd.DataFrame({"timestamp": timestamps})
    result = compute_cyclical_time_features(df)

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "dow_sin" in result.columns
    assert "dow_cos" in result.columns

    # Pythagorean identity sin^2 + cos^2 = 1
    hour_norm = (result["hour_sin"] ** 2) + (result["hour_cos"] ** 2)
    np.testing.assert_allclose(hour_norm.to_numpy(), np.ones(len(df)), atol=1e-12)

    # Midnight 00:00 -> sin = 0, cos = 1
    assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-7)
    assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-7)

    # 06:00 (quarter day) -> sin = 1, cos = 0
    assert result["hour_sin"].iloc[1] == pytest.approx(1.0, abs=1e-7)
    assert result["hour_cos"].iloc[1] == pytest.approx(0.0, abs=1e-7)

    # 12:00 (half day) -> sin = 0, cos = -1
    assert result["hour_sin"].iloc[2] == pytest.approx(0.0, abs=1e-7)
    assert result["hour_cos"].iloc[2] == pytest.approx(-1.0, abs=1e-7)


def test_missing_timestamp_error() -> None:
    bad_df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="Missing required timestamp column"):
        compute_cyclical_time_features(bad_df)


def test_time_registry_integration() -> None:
    time_feats = feature_registry.get_family_features(FeatureFamily.TIME)
    assert "hour_sin" in time_feats
    assert "hour_cos" in time_feats
    assert "dow_sin" in time_feats
    assert "dow_cos" in time_feats

    meta = feature_registry.get_metadata("hour_sin")
    assert meta.lookback_period == 0
    assert meta.uses_future_data is False
