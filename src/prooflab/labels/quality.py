"""Label quality reporting and class imbalance diagnostics.

Analyzes sample distributions across canonical and rich barrier outcomes and
issues warnings when severe class skew is detected.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from prooflab.labels.outcome import BarrierOutcome, CanonicalLabel, LabelMatrix


class LabelQualityReport(BaseModel):
    """Structured report providing quality and distribution diagnostics for a LabelMatrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_samples: int
    canonical_counts: dict[str, int]
    canonical_percentages: dict[str, float]
    outcome_counts: dict[str, int]
    outcome_percentages: dict[str, float]
    ambiguous_count: int
    ambiguous_percentage: float
    average_bars_held: float
    is_imbalanced: bool
    imbalance_warning: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    def summary(self) -> str:
        """Return a formatted human-readable summary string."""
        lines = [
            "=== Label Quality Report ===",
            f"Total Evaluated Samples: {self.total_samples:,}",
            f"Average Bars Held: {self.average_bars_held:.2f}",
            f"Ambiguous Bar Touches: {self.ambiguous_count:,} ({self.ambiguous_percentage:.2f}%)",
            "",
            "--- Canonical Class Distribution ---",
        ]
        for name, count in self.canonical_counts.items():
            pct = self.canonical_percentages.get(name, 0.0)
            lines.append(f"  {name:8s}: {count:6,d} ({pct:6.2f}%)")

        lines.extend(["", "--- Rich Outcome Breakdown ---"])
        for name, count in self.outcome_counts.items():
            pct = self.outcome_percentages.get(name, 0.0)
            lines.append(f"  {name:18s}: {count:6,d} ({pct:6.2f}%)")

        if self.is_imbalanced:
            lines.extend(["", f"[WARNING] {self.imbalance_warning}"])

        return "\n".join(lines)


def generate_quality_report(
    label_matrix: LabelMatrix,
    imbalance_threshold: float = 0.80,
) -> LabelQualityReport:
    """Generate diagnostic quality report across all evaluated labels in a matrix.

    Args:
        label_matrix: The LabelMatrix containing evaluated outcomes.
        imbalance_threshold: Fraction (0.0 to 1.0) above which a single class is
                             flagged as severely imbalanced (default: 0.80 / 80%).

    Returns:
        LabelQualityReport instance.
    """
    total_samples = len(label_matrix)
    if total_samples == 0:
        return LabelQualityReport(
            total_samples=0,
            canonical_counts={"BUY": 0, "SELL": 0, "IGNORE": 0},
            canonical_percentages={"BUY": 0.0, "SELL": 0.0, "IGNORE": 0.0},
            outcome_counts={b.value: 0 for b in BarrierOutcome},
            outcome_percentages={b.value: 0.0 for b in BarrierOutcome},
            ambiguous_count=0,
            ambiguous_percentage=0.0,
            average_bars_held=0.0,
            is_imbalanced=False,
            imbalance_warning=None,
        )

    # Count canonical labels
    can_counts: dict[str, int] = {"BUY": 0, "SELL": 0, "IGNORE": 0}
    for outcome in label_matrix.outcomes:
        if outcome.canonical_label == CanonicalLabel.BUY:
            can_counts["BUY"] += 1
        elif outcome.canonical_label == CanonicalLabel.SELL:
            can_counts["SELL"] += 1
        else:
            can_counts["IGNORE"] += 1

    can_pcts: dict[str, float] = {
        k: round((v / total_samples) * 100.0, 2) for k, v in can_counts.items()
    }

    # Count rich outcomes
    out_counts: dict[str, int] = {b.value: 0 for b in BarrierOutcome}
    ambiguous_count = 0
    total_bars_held = 0

    for outcome in label_matrix.outcomes:
        out_counts[outcome.barrier_outcome.value] += 1
        if outcome.was_ambiguous:
            ambiguous_count += 1
        total_bars_held += outcome.bars_held

    out_pcts: dict[str, float] = {
        k: round((v / total_samples) * 100.0, 2) for k, v in out_counts.items()
    }
    ambiguous_pct = round((ambiguous_count / total_samples) * 100.0, 2)
    avg_bars_held = round(total_bars_held / total_samples, 2)

    # Check for severe class imbalance
    is_imbalanced = False
    warning_msg: str | None = None
    threshold_pct = imbalance_threshold * 100.0

    for cls_name, pct in can_pcts.items():
        if pct >= threshold_pct and total_samples >= 5:
            is_imbalanced = True
            warning_msg = (
                f"Severe class imbalance detected: '{cls_name}' represents {pct:.1f}% "
                f"of all samples (threshold: {threshold_pct:.1f}%)."
            )
            break

    return LabelQualityReport(
        total_samples=total_samples,
        canonical_counts=can_counts,
        canonical_percentages=can_pcts,
        outcome_counts=out_counts,
        outcome_percentages=out_pcts,
        ambiguous_count=ambiguous_count,
        ambiguous_percentage=ambiguous_pct,
        average_bars_held=avg_bars_held,
        is_imbalanced=is_imbalanced,
        imbalance_warning=warning_msg,
    )
