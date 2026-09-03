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
from prooflab.proof.stress import (
    ExecutionStressAnalyzer,
    ExecutionStressConfig,
    ExecutionStressResult,
    StressScenarioResult,
)

__all__ = [
    "EquityCurveData",
    "ExecutionStressAnalyzer",
    "ExecutionStressConfig",
    "ExecutionStressResult",
    "FeatureImportanceAnalyzer",
    "FeatureImportanceEntry",
    "FeatureImportanceResult",
    "ParameterSensitivityAnalyzer",
    "ParameterSensitivityConfig",
    "ParameterSensitivityResult",
    "ProofScorecard",
    "SensitivityGridCell",
    "StressScenarioResult",
]
