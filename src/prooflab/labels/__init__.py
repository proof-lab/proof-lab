"""Proof Lab Label Engine.

Implements predictive setup classification via the Triple Barrier Method,
rich outcome tracking, ambiguity resolution policies, and diagnostic quality reporting.
"""

from prooflab.labels.barrier import BarrierEvaluator
from prooflab.labels.config import (
    AmbiguityPolicy,
    Direction,
    DistanceUnit,
    SetupConfig,
)
from prooflab.labels.outcome import (
    BarrierOutcome,
    CanonicalLabel,
    LabelMatrix,
    RichLabelOutcome,
)
from prooflab.labels.quality import (
    LabelQualityReport,
    generate_quality_report,
)

__all__ = [
    "AmbiguityPolicy",
    "BarrierEvaluator",
    "BarrierOutcome",
    "CanonicalLabel",
    "Direction",
    "DistanceUnit",
    "LabelMatrix",
    "LabelQualityReport",
    "RichLabelOutcome",
    "SetupConfig",
    "generate_quality_report",
]
