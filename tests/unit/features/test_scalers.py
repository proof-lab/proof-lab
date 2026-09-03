"""Unit tests for prooflab.features.scalers (Fit/Transform leak-free scalers)."""

import pandas as pd
import pytest

from prooflab.features.scalers import (
    MinMaxScaler,
    NotFittedError,
    RobustScaler,
    StandardScaler,
)


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature_b": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    # Test split has higher unseen values
    return pd.DataFrame(
        {
            "feature_a": [60.0, 70.0],
            "feature_b": [600.0, 700.0],
        }
    )


def test_standard_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    scaler = StandardScaler()
    with pytest.raises(NotFittedError):
        scaler.transform(train_df)

    # Fit on train
    scaler.fit(train_df)
    assert scaler.is_fitted is True
    assert scaler.means_["feature_a"] == pytest.approx(30.0)

    # Transform train: mean must be 0, std must be 1
    train_trans = scaler.transform(train_df)
    assert train_trans["feature_a"].mean() == pytest.approx(0.0)
    assert train_trans["feature_a"].std(ddof=0) == pytest.approx(1.0)

    # Transform test without updating means
    test_trans = scaler.transform(test_df)
    # 60.0 in feature_a has z-score: (60 - 30) / sqrt(200)
    expected_z = (60.0 - 30.0) / float(train_df["feature_a"].std(ddof=0))
    assert test_trans["feature_a"].iloc[0] == pytest.approx(expected_z)

    # Inverse transform
    inverted = scaler.inverse_transform(train_trans)
    pd.testing.assert_frame_equal(inverted, train_df)


def test_robust_scaler(train_df: pd.DataFrame) -> None:
    scaler = RobustScaler()
    train_trans = scaler.fit_transform(train_df)
    # Median of feature_a [10, 20, 30, 40, 50] is 30.0
    assert scaler.medians_["feature_a"] == 30.0
    # Median row should be exactly 0
    assert train_trans["feature_a"].iloc[2] == pytest.approx(0.0)

    inverted = scaler.inverse_transform(train_trans)
    pd.testing.assert_frame_equal(inverted, train_df)


def test_min_max_scaler(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_trans = scaler.fit_transform(train_df)

    assert train_trans["feature_a"].min() == pytest.approx(0.0)
    assert train_trans["feature_a"].max() == pytest.approx(1.0)

    test_trans = scaler.transform(test_df)
    # 60.0 in train range [10, 50] -> (60 - 10) / (50 - 10) = 50 / 40 = 1.25
    assert test_trans["feature_a"].iloc[0] == pytest.approx(1.25)

    inverted = scaler.inverse_transform(train_trans)
    pd.testing.assert_frame_equal(inverted, train_df)


def test_constant_column_zero_variance_protection() -> None:
    const_df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
    scaler = StandardScaler()
    trans = scaler.fit_transform(const_df)
    # Does not result in NaN or division by zero
    assert not trans["const"].isna().any()
