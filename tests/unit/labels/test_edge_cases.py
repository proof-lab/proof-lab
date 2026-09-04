"""Exhaustive edge case tests for the Label Engine."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import Direction, DistanceUnit, SetupConfig
from prooflab.labels.outcome import BarrierOutcome, CanonicalLabel


def test_edge_case_empty_dataframe() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    evaluator = BarrierEvaluator()
    matrix = evaluator.generate_labels(pd.DataFrame(), config=config)
    assert len(matrix) == 0


def test_edge_case_single_row_dataframe() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, 10, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        }
    )
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=5.0,
        stop_distance=5.0,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    matrix = evaluator.generate_labels(df, config=config)
    assert len(matrix) == 1
    assert matrix.outcomes[0].barrier_outcome == BarrierOutcome.TIMEOUT
    assert matrix.outcomes[0].canonical_label == CanonicalLabel.IGNORE
    assert matrix.outcomes[0].bars_held == 0


def test_edge_case_insufficient_future_data() -> None:
    # 3 bars with horizon = 10 -> evaluates available 2 bars
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(3)],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.5, 100.5],
            "low": [100.0, 99.5, 99.5],
            "close": [100.0, 100.0, 100.0],
        }
    )
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=5.0,
        stop_distance=5.0,
        horizon_bars=10,
    )
    evaluator = BarrierEvaluator()
    matrix = evaluator.generate_labels(df, config=config)

    assert len(matrix) == 3
    # First bar evaluated against 2 future bars (neither hits barrier) -> TIMEOUT
    assert matrix.outcomes[0].barrier_outcome == BarrierOutcome.TIMEOUT
    assert matrix.outcomes[0].bars_held == 2


def test_edge_case_exact_price_barrier_touch() -> None:
    # Target = 102.0. Bar 1 High is EXACTLY 102.0 -> target must trigger
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "timestamp": [base, base + timedelta(minutes=1)],
            "open": [100.0, 101.0],
            "high": [100.0, 102.0],
            "low": [100.0, 100.5],
            "close": [100.0, 101.5],
        }
    )
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=2.0,
        stop_distance=2.0,
        unit=DistanceUnit.POINTS,
        horizon_bars=5,
    )
    evaluator = BarrierEvaluator()
    outcome = evaluator.evaluate_bar(df, idx=0, config=config)

    assert outcome.canonical_label == CanonicalLabel.BUY
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.exit_price == pytest.approx(102.0)


def test_edge_case_zero_and_negative_inputs_rejected() -> None:
    with pytest.raises(ValidationError):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=0.0,
            stop_distance=10.0,
        )

    with pytest.raises(ValidationError):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=10.0,
            stop_distance=-1.0,
        )

    with pytest.raises(ValidationError):
        SetupConfig(
            direction=Direction.LONG,
            target_distance=10.0,
            stop_distance=10.0,
            horizon_bars=-1,
        )
