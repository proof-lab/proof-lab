"""Unit tests for prooflab.labels.outcome."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from prooflab.labels.config import Direction, SetupConfig
from prooflab.labels.outcome import (
    BarrierOutcome,
    CanonicalLabel,
    LabelMatrix,
    RichLabelOutcome,
)


def test_canonical_label_values() -> None:
    assert int(CanonicalLabel.BUY) == 1
    assert int(CanonicalLabel.SELL) == -1
    assert int(CanonicalLabel.IGNORE) == 0


def test_barrier_outcome_values() -> None:
    assert BarrierOutcome.TARGET_FIRST == "TARGET_FIRST"
    assert BarrierOutcome.STOP_FIRST == "STOP_FIRST"
    assert BarrierOutcome.TIMEOUT == "TIMEOUT"
    assert BarrierOutcome.AMBIGUOUS == "AMBIGUOUS"
    assert BarrierOutcome.EXCLUDED == "EXCLUDED"


def test_rich_label_outcome_immutability() -> None:
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    outcome = RichLabelOutcome(
        entry_index=0,
        entry_time=now,
        entry_price=100.0,
        target_price=105.0,
        stop_price=95.0,
        canonical_label=CanonicalLabel.BUY,
        barrier_outcome=BarrierOutcome.TARGET_FIRST,
        exit_index=3,
        exit_time=datetime(2026, 1, 1, 10, 3, tzinfo=UTC),
        exit_price=105.2,
        bars_held=3,
        return_at_exit=0.052,
        was_ambiguous=False,
    )
    assert outcome.canonical_label == 1
    assert outcome.barrier_outcome == BarrierOutcome.TARGET_FIRST
    assert outcome.bars_held == 3

    with pytest.raises(ValidationError):
        outcome.entry_price = 110.0  # type: ignore[misc]


def test_label_matrix_to_dataframe() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 10, 1, tzinfo=UTC)

    o1 = RichLabelOutcome(
        entry_index=0,
        entry_time=t1,
        entry_price=100.0,
        target_price=110.0,
        stop_price=90.0,
        canonical_label=CanonicalLabel.BUY,
        barrier_outcome=BarrierOutcome.TARGET_FIRST,
        exit_index=2,
        exit_time=datetime(2026, 1, 1, 10, 2, tzinfo=UTC),
        exit_price=110.5,
        bars_held=2,
        return_at_exit=0.105,
    )
    o2 = RichLabelOutcome(
        entry_index=1,
        entry_time=t2,
        entry_price=101.0,
        target_price=111.0,
        stop_price=91.0,
        canonical_label=CanonicalLabel.IGNORE,
        barrier_outcome=BarrierOutcome.STOP_FIRST,
        exit_index=4,
        exit_time=datetime(2026, 1, 1, 10, 4, tzinfo=UTC),
        exit_price=90.8,
        bars_held=3,
        return_at_exit=-0.101,
    )

    matrix = LabelMatrix(outcomes=[o1, o2], setup_config=config, dataset_id="ds-123")
    assert len(matrix) == 2

    df = matrix.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["canonical_label"]) == [1, 0]
    assert list(df["barrier_outcome"]) == ["TARGET_FIRST", "STOP_FIRST"]

    series = matrix.get_canonical_series()
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert series.iloc[0] == 1
    assert series.iloc[1] == 0


def test_empty_label_matrix() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    matrix = LabelMatrix(outcomes=[], setup_config=config)
    assert len(matrix) == 0

    df = matrix.to_dataframe()
    assert df.empty
    assert "canonical_label" in df.columns

    series = matrix.get_canonical_series()
    assert series.empty
