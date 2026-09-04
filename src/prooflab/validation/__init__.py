"""Validation framework for Proof Lab.

Provides chronological dataset splitting, walk-forward partition generation,
purging, embargo buffers, leakage detection, experiment registry tracking,
and blind test set protection.
"""

from prooflab.validation.blind import (
    BlindAccessViolationError,
    BlindEvaluationAudit,
    BlindEvaluationGate,
    BlindMultipleTestingWarning,
)
from prooflab.validation.calibration import probability_quality
from prooflab.validation.leakage import (
    LeakageReport,
    audit_availability,
    audit_feature_causality,
    audit_fit_rows,
    audit_plan,
)
from prooflab.validation.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    MultipleTestingWarning,
    collect_environment_metadata,
    generate_experiment_id,
)
from prooflab.validation.splits import (
    EmbargoInterval,
    FoldPlan,
    SplitConfig,
    WalkForwardConfig,
    chronological_split,
    validate_timeline,
    walk_forward,
)

__all__ = [
    "BlindAccessViolationError",
    "BlindEvaluationAudit",
    "BlindEvaluationGate",
    "BlindMultipleTestingWarning",
    "EmbargoInterval",
    "ExperimentRecord",
    "ExperimentRegistry",
    "FoldPlan",
    "LeakageReport",
    "MultipleTestingWarning",
    "SplitConfig",
    "WalkForwardConfig",
    "audit_availability",
    "audit_feature_causality",
    "audit_fit_rows",
    "audit_plan",
    "chronological_split",
    "collect_environment_metadata",
    "generate_experiment_id",
    "probability_quality",
    "validate_timeline",
    "walk_forward",
]
