"""Unit tests for prooflab.labels.quality."""

import json
from datetime import UTC, datetime

from prooflab.labels.config import Direction, SetupConfig
from prooflab.labels.outcome import (
    BarrierOutcome,
    CanonicalLabel,
    LabelMatrix,
    RichLabelOutcome,
)
from prooflab.labels.quality import generate_quality_report


def test_quality_report_balanced() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    outcomes: list[RichLabelOutcome] = []
    # 5 BUYs
    for i in range(5):
        outcomes.append(
            RichLabelOutcome(
                entry_index=i,
                entry_time=now,
                entry_price=100.0,
                target_price=110.0,
                stop_price=90.0,
                canonical_label=CanonicalLabel.BUY,
                barrier_outcome=BarrierOutcome.TARGET_FIRST,
                bars_held=2,
            )
        )
    # 5 IGNOREs
    for i in range(5, 10):
        outcomes.append(
            RichLabelOutcome(
                entry_index=i,
                entry_time=now,
                entry_price=100.0,
                target_price=110.0,
                stop_price=90.0,
                canonical_label=CanonicalLabel.IGNORE,
                barrier_outcome=BarrierOutcome.STOP_FIRST,
                bars_held=3,
            )
        )

    matrix = LabelMatrix(outcomes=outcomes, setup_config=config)
    report = generate_quality_report(matrix)

    assert report.total_samples == 10
    assert report.canonical_counts["BUY"] == 5
    assert report.canonical_counts["IGNORE"] == 5
    assert report.canonical_percentages["BUY"] == 50.0
    assert report.canonical_percentages["IGNORE"] == 50.0
    assert report.is_imbalanced is False
    assert report.imbalance_warning is None
    assert report.average_bars_held == 2.5


def test_quality_report_severe_imbalance() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    outcomes: list[RichLabelOutcome] = []
    # 9 IGNOREs and 1 BUY -> 90% IGNORE
    for i in range(9):
        outcomes.append(
            RichLabelOutcome(
                entry_index=i,
                entry_time=now,
                entry_price=100.0,
                target_price=110.0,
                stop_price=90.0,
                canonical_label=CanonicalLabel.IGNORE,
                barrier_outcome=BarrierOutcome.STOP_FIRST,
                bars_held=1,
            )
        )
    outcomes.append(
        RichLabelOutcome(
            entry_index=9,
            entry_time=now,
            entry_price=100.0,
            target_price=110.0,
            stop_price=90.0,
            canonical_label=CanonicalLabel.BUY,
            barrier_outcome=BarrierOutcome.TARGET_FIRST,
            bars_held=1,
        )
    )

    matrix = LabelMatrix(outcomes=outcomes, setup_config=config)
    report = generate_quality_report(matrix, imbalance_threshold=0.80)

    assert report.total_samples == 10
    assert report.is_imbalanced is True
    assert report.imbalance_warning is not None
    assert "Severe class imbalance detected: 'IGNORE' represents 90.0%" in report.imbalance_warning


def test_quality_report_json_and_summary() -> None:
    config = SetupConfig(
        direction=Direction.LONG,
        target_distance=10.0,
        stop_distance=10.0,
    )
    matrix = LabelMatrix(outcomes=[], setup_config=config)
    report = generate_quality_report(matrix)

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["total_samples"] == 0

    summary_str = report.summary()
    assert "=== Label Quality Report ===" in summary_str
    assert "Total Evaluated Samples: 0" in summary_str
