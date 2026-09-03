"""Unit tests for prooflab.labels.config."""

import pytest
from pydantic import ValidationError

from prooflab.labels.config import AmbiguityPolicy, Direction, DistanceUnit, SetupConfig


def test_setup_config_valid() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=20.0,
        stop_distance=10.0,
        unit=DistanceUnit.PIPS,
        horizon_bars=15,
        point_value=0.0001,
    )
    assert config.direction == Direction.LONG
    assert config.target_distance == 20.0
    assert config.stop_distance == 10.0
    assert config.ambiguity_policy == AmbiguityPolicy.CONSERVATIVE
    assert config.horizon_bars == 15
    assert config.point_value == 0.0001


def test_setup_config_invalid_distances() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=0.0,
            stop_distance=10.0,
        )

    with pytest.raises(ValidationError, match="strictly positive"):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=10.0,
            stop_distance=-5.0,
        )


def test_setup_config_invalid_horizon() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=10.0,
            stop_distance=10.0,
            horizon_bars=0,
        )


def test_setup_config_invalid_point_value() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=10.0,
            stop_distance=10.0,
            point_value=0.0,
        )


def test_calculate_barriers_long_points() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=20.0,
        stop_distance=10.0,
        unit=DistanceUnit.POINTS,
        point_value=1.0,
    )
    target, stop = config.calculate_barriers(entry_price=100.0)
    assert target == 120.0
    assert stop == 90.0


def test_calculate_barriers_short_pips() -> None:
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=20.0,
        stop_distance=10.0,
        unit=DistanceUnit.PIPS,
        point_value=0.0001,
    )
    target, stop = config.calculate_barriers(entry_price=1.1000)
    assert pytest.approx(target) == 1.0980
    assert pytest.approx(stop) == 1.1010


def test_calculate_barriers_percent() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,  # 2%
        stop_distance=1.0,    # 1%
        unit=DistanceUnit.PERCENT,
    )
    target, stop = config.calculate_barriers(entry_price=200.0)
    assert target == 204.0
    assert stop == 198.0


def test_calculate_barriers_atr() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,  # 2 x ATR
        stop_distance=1.5,    # 1.5 x ATR
        unit=DistanceUnit.ATR,
    )
    target, stop = config.calculate_barriers(entry_price=100.0, atr_value=2.5)
    assert target == 105.0
    assert stop == 96.25

    with pytest.raises(ValueError, match="Positive atr_value is required"):
        config.calculate_barriers(entry_price=100.0, atr_value=None)


def test_setup_config_immutability() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=20.0,
        stop_distance=10.0,
    )
    with pytest.raises(ValidationError):
        config.target_distance = 30.0  # type: ignore[misc]
