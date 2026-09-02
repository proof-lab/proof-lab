"""Unit tests for same-bar ambiguity resolution policies."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import AmbiguityPolicy, Direction, DistanceUnit, SetupConfig
from prooflab.labels.outcome import BarrierOutcome, CanonicalLabel


@pytest.fixture
def volatile_bar_df() -> pd.DataFrame:
    # Bar 0: Entry at 100.0.
    # Bar 1: Extreme spike/dump (High = 110.0, Low = 90.0, Close = 101.0).
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 10, 1, tzinfo=UTC),
            ],
            "open": [100.0, 100.0],
            "high": [100.0, 110.0],
            "low": [100.0, 90.0],
            "close": [100.0, 101.0],
        }
    )


def test_long_ambiguity_conservative(volatile_bar_df: pd.DataFrame) -> None:
    # Long: Target = 105.0 (+5), Stop = 95.0 (-5). Both hit on Bar 1.
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.CONSERVATIVE,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Conservative policy assumes adverse barrier (Stop) hit first
    assert outcome.barrier_outcome == BarrierOutcome.STOP_FIRST
    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.exit_price == pytest.approx(95.0)


def test_long_ambiguity_optimistic(volatile_bar_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.OPTIMISTIC,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Optimistic policy assumes favorable barrier (Target) hit first
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.canonical_label == CanonicalLabel.BUY
    assert outcome.exit_price == pytest.approx(105.0)


def test_long_ambiguity_exclude(volatile_bar_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.EXCLUDE,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Exclude policy flags as EXCLUDED and IGNORE
    assert outcome.barrier_outcome == BarrierOutcome.EXCLUDED
    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.exit_price == volatile_bar_df.loc[1, "close"]


def test_short_ambiguity_conservative(volatile_bar_df: pd.DataFrame) -> None:
    # Short: Target = 95.0 (-5), Stop = 105.0 (+5). Both hit on Bar 1.
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.CONSERVATIVE,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Conservative policy assumes adverse barrier (Stop) hit first
    assert outcome.barrier_outcome == BarrierOutcome.STOP_FIRST
    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.exit_price == pytest.approx(105.0)


def test_short_ambiguity_optimistic(volatile_bar_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.OPTIMISTIC,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Optimistic policy assumes favorable barrier (Target) hit first
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.canonical_label == CanonicalLabel.SELL
    assert outcome.exit_price == pytest.approx(95.0)


def test_short_ambiguity_exclude(volatile_bar_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=5.0,
        stop_distance=5.0,
        unit=DistanceUnit.POINTS,
        ambiguity_policy=AmbiguityPolicy.EXCLUDE,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(volatile_bar_df, idx=0, config=config)

    assert outcome.was_ambiguous is True
    # Exclude policy flags as EXCLUDED
    assert outcome.barrier_outcome == BarrierOutcome.EXCLUDED
    assert outcome.canonical_label == CanonicalLabel.IGNORE
