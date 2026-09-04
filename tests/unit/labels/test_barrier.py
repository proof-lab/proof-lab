"""Unit tests for prooflab.labels.barrier."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import Direction, DistanceUnit, SetupConfig
from prooflab.labels.outcome import BarrierOutcome, CanonicalLabel


@pytest.fixture
def price_df() -> pd.DataFrame:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    timestamps = [base + timedelta(minutes=i) for i in range(10)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.5, 101.0, 102.5, 101.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            "high": [100.5, 101.0, 103.0, 103.5, 101.5, 99.5, 98.5, 97.5, 96.5, 95.5],
            "low": [99.5, 100.0, 100.5, 101.5, 98.5, 97.5, 97.0, 96.0, 95.0, 94.0],
            "close": [100.2, 100.8, 102.5, 103.0, 99.0, 98.0, 97.2, 96.5, 95.5, 94.5],
        }
    )


def test_long_target_first(price_df: pd.DataFrame) -> None:
    # Entry at idx 0: close is 100.2. Target = 102.2 (+2.0), Stop = 98.2 (-2.0).
    # Bar 2 has High = 103.0 (reaches 102.2) and Low = 100.5 (above stop 98.2).
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,
        stop_distance=2.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(price_df, idx=0, config=config)

    assert outcome.canonical_label == CanonicalLabel.BUY
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.exit_price == pytest.approx(102.2)
    assert outcome.bars_held == 2
    assert outcome.was_ambiguous is False


def test_long_stop_first(price_df: pd.DataFrame) -> None:
    # Entry at idx 3: close is 103.0. Target = 106.0 (+3.0), Stop = 101.0 (-2.0).
    # Bar 4 has High = 101.5 (below target), Low = 98.5 (breaches stop 101.0).
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=3.0,
        stop_distance=2.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(price_df, idx=3, config=config)

    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.barrier_outcome == BarrierOutcome.STOP_FIRST
    assert outcome.exit_price == pytest.approx(101.0)
    assert outcome.bars_held == 1


def test_short_target_first(price_df: pd.DataFrame) -> None:
    # Entry at idx 4: close is 99.0. Target = 97.0 (-2.0), Stop = 101.0 (+2.0).
    # Bar 5 has High = 99.5 (below stop), Low = 97.5 (not yet).
    # Bar 6 has High = 98.5, Low = 97.0 (hits target).
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=2.0,
        stop_distance=2.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(price_df, idx=4, config=config)

    assert outcome.canonical_label == CanonicalLabel.SELL
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.exit_price == pytest.approx(97.0)
    assert outcome.bars_held == 2


def test_short_stop_first(price_df: pd.DataFrame) -> None:
    # Entry at idx 0: close is 100.2. Target = 97.0 (-3.2), Stop = 101.2 (+1.0).
    # Bar 2 has High = 103.0 (breaches stop 101.2).
    config = SetupConfig(
        direction=Direction.SHORT,
        target_distance=3.2,
        stop_distance=1.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(price_df, idx=0, config=config)

    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.barrier_outcome == BarrierOutcome.STOP_FIRST
    assert outcome.exit_price == pytest.approx(101.2)


def test_horizon_timeout(price_df: pd.DataFrame) -> None:
    # Very wide target and stop that won't be reached in 2 bars
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=50.0,
        stop_distance=50.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=2,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(price_df, idx=0, config=config)

    assert outcome.canonical_label == CanonicalLabel.IGNORE
    assert outcome.barrier_outcome == BarrierOutcome.TIMEOUT
    assert outcome.bars_held == 2
    assert outcome.exit_price == price_df.loc[2, "close"]


def test_generate_labels_full_series(price_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,
        stop_distance=2.0,
        horizon_bars=3,
    )
    evaluator = BarrierEvaluator()
    matrix = evaluator.generate_labels(price_df, config=config, dataset_id="ds-test")

    assert len(matrix) == len(price_df)
    # The last bar has 0 future bars and should be TIMEOUT
    assert matrix.outcomes[-1].barrier_outcome == BarrierOutcome.TIMEOUT
    assert matrix.outcomes[-1].bars_held == 0


def test_label_determinism(price_df: pd.DataFrame) -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,
        stop_distance=2.0,
        horizon_bars=3,
    )
    evaluator = BarrierEvaluator()
    matrix1 = evaluator.generate_labels(price_df, config=config)
    matrix2 = evaluator.generate_labels(price_df, config=config)

    df1 = matrix1.to_dataframe()
    df2 = matrix2.to_dataframe()
    pd.testing.assert_frame_equal(df1, df2)
