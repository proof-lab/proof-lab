"""Proof Engine and robustness stress-testing framework for Proof Lab."""

from prooflab.proof.importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceEntry,
    FeatureImportanceResult,
)
from prooflab.proof.scorecard import (
    EquityCurveData,
    ProofScorecard,
)
from prooflab.proof.sensitivity import (
    ParameterSensitivityAnalyzer,
    ParameterSensitivityConfig,
    ParameterSensitivityResult,
    SensitivityGridCell,
)

__all__ = [
    "EquityCurveData",
    "FeatureImportanceAnalyzer",
    "FeatureImportanceEntry",
    "FeatureImportanceResult",
    "ParameterSensitivityAnalyzer",
    "ParameterSensitivityConfig",
    "ParameterSensitivityResult",
    "ProofScorecard",
    "SensitivityGridCell",
]
